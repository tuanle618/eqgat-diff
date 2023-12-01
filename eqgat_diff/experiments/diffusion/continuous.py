import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments.utils import zero_mean


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def get_beta_schedule(
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    num_diffusion_timesteps: int = 1000,
    kind: str = "cosine",
    nu: float = 1.0,
    plot: bool = False,
    clamp_alpha_min=0.05,
    **kwargs,
):
    if kind == "quad":
        betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=torch.get_default_dtype(),
            )
            ** 2
        )
    elif kind == "sigmoid":
        betas = torch.linspace(-6, 6, num_diffusion_timesteps)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
    elif kind == "linear":
        betas = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif kind == "cosine":
        s = kwargs.get("s")
        if s is None:
            s = 0.008
        steps = num_diffusion_timesteps + 2
        x = torch.linspace(0, num_diffusion_timesteps, steps)
        alphas_cumprod = (
            torch.cos(((x / num_diffusion_timesteps) + s) / (1 + s) * torch.pi * 0.5)
            ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        ### new included
        alphas_cumprod = torch.from_numpy(
            clip_noise_schedule(alphas_cumprod, clip_value=clamp_alpha_min)
        )
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = alphas.clip(min=0.001)
        betas = 1 - alphas
        betas = torch.clip(betas, 0.0, 0.999).float()
    elif kind == "polynomial":
        s = kwargs.get("s")
        p = kwargs.get("p")
        if s is None:
            s = 1e-4
        if p is None:
            p = 3.0
        steps = num_diffusion_timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas2 = (1 - np.power(x / steps, p)) ** 2
        alphas2 = clip_noise_schedule(alphas2, clip_value=clamp_alpha_min)
        precision = 1 - 2 * s
        alphas2 = precision * alphas2 + s
        alphas = np.sqrt(alphas2)
        betas = 1.0 - alphas
        betas = torch.from_numpy(betas).float()
    elif kind == "adaptive":
        s = kwargs.get("s")
        if s is None:
            s = 0.008
        steps = num_diffusion_timesteps + 2
        x = np.linspace(0, steps, steps)
        x = np.expand_dims(x, 0)  # ((1, steps))

        nu_arr = np.array(nu)  # (components, )  # X, charges, E, y, pos
        _steps = steps
        # _steps = num_diffusion_timesteps
        alphas_cumprod = (
            np.cos(0.5 * np.pi * (((x / _steps) ** nu_arr) + s) / (1 + s)) ** 2
        )  # ((components, steps))
        # divide every element of alphas_cumprod by the first element of alphas_cumprod
        alphas_cumprod_new = alphas_cumprod / alphas_cumprod[:, 0]
        ### new included
        alphas_cumprod_new = clip_noise_schedule(
            alphas_cumprod_new.squeeze(), clip_value=clamp_alpha_min
        )[None, ...]
        # remove the first element of alphas_cumprod and then multiply every element by the one before it
        alphas = alphas_cumprod_new[:, 1:] / alphas_cumprod_new[:, :-1]
        # alphas[:, alphas.shape[1]-1] = 0.001
        alphas = alphas.clip(min=0.001)
        betas = 1 - alphas  # ((components, steps)) # X, charges, E, y, pos
        betas = np.swapaxes(betas, 0, 1)
        betas = torch.clip(torch.from_numpy(betas), 0.0, 0.999).squeeze().float()
    elif kind == "linear-time":
        t = np.linspace(1e-6, 1.0, num_diffusion_timesteps + 1)
        alphas_cumprod = 1.0 - t
        alphas_cumprod = clip_noise_schedule(alphas_cumprod, clip_value=0.001)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
        alphas = alphas.clip(min=clamp_alpha_min)
        betas = 1 - alphas
        betas = np.clip(betas, 0.0, 0.999)
        betas = torch.from_numpy(betas).float().squeeze()

    if plot:
        plt.plot(range(len(betas)), betas)
        plt.xlabel("t")
        plt.ylabel("beta")
        plt.show()
        alphas = 1.0 - betas
        signal_coeff = alphas.cumprod(0)
        noise_coeff = torch.sqrt(1.0 - signal_coeff)
        plt.plot(np.arange(len(signal_coeff)), signal_coeff, label="signal")
        plt.plot(np.arange(len(noise_coeff)), noise_coeff, label="noise")
        plt.legend()
        plt.show()

    return betas


class DiscreteDDPM(nn.Module):
    def __init__(
        self,
        beta_min: float = 1e-4,
        beta_max: float = 2e-2,
        scaled_reverse_posterior_sigma: bool = True,
        schedule: str = "cosine",
        nu: float = 1.0,
        enforce_zero_terminal_snr: bool = False,
        T: int = 500,
        param: str = "data",
        clamp_alpha_min=0.05,
        **kwargs,
    ):
        """Constructs discrete Diffusion schedule according to DDPM in Ho et al. (2020).
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          T: number of discretization steps
        """
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.scaled_reverse_posterior_sigma = scaled_reverse_posterior_sigma
        assert param in ["noise", "data"]
        self.param = param

        assert schedule in [
            "linear",
            "quad",
            "cosine",
            "sigmoid",
            "polynomial",
            "adaptive",
            "linear-time",
        ]

        self.schedule = schedule
        self.T = T

        discrete_betas = get_beta_schedule(
            beta_start=beta_min,
            beta_end=beta_max,
            num_diffusion_timesteps=self.T,
            kind=self.schedule,
            nu=nu,
            plot=False,
            alpha_clamp=clamp_alpha_min,
        )

        if enforce_zero_terminal_snr:
            discrete_betas = self.enforce_zero_terminal_snr(betas=discrete_betas)

        self.enforce_zero_terminal_snr = enforce_zero_terminal_snr

        sqrt_betas = torch.sqrt(discrete_betas)
        alphas = 1.0 - discrete_betas

        # is used when using noise parameterization. last entry can be rather small and hence 1/sqrt_alphas approx 31.6230
        sqrt_alphas = torch.sqrt(alphas)

        if schedule == "adaptive":
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            alphas_cumprod = torch.exp(log_alpha_bar)
            log_alpha = torch.log(alphas)
            log_alpha_bar = torch.cumsum(log_alpha, dim=0)
            self._alphas = alphas
            self._log_alpha_bar = log_alpha_bar
            self._alphas_bar = torch.exp(log_alpha_bar)
            self._sigma2_bar = -torch.expm1(2 * log_alpha_bar)
            self._sigma_bar = torch.sqrt(self._sigma2_bar)
            self._gamma = (
                torch.log(-torch.special.expm1(2 * log_alpha_bar)) - 2 * log_alpha_bar
            )
        else:
            alphas_cumprod = torch.cumprod(alphas, dim=0)

        alphas_cumprod_prev = torch.nn.functional.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_1m_alphas_cumprod = sqrt_1m_alphas_cumprod.clamp(min=1e-4)

        if scaled_reverse_posterior_sigma:
            rev_variance = (
                discrete_betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )
            rev_variance[0] = rev_variance[1] / 2.0
            reverse_posterior_sigma = torch.sqrt(rev_variance)
        else:
            reverse_posterior_sigma = torch.sqrt(discrete_betas)

        self.register_buffer("discrete_betas", discrete_betas)
        self.register_buffer("sqrt_betas", sqrt_betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", sqrt_alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_1m_alphas_cumprod", sqrt_1m_alphas_cumprod)
        self.register_buffer("reverse_posterior_sigma", reverse_posterior_sigma)

    def enforce_zero_terminal_snr(self, betas):
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas

        return betas

    def marginal_prob(self, x: Tensor, t: Tensor):
        """_summary_
        Eq. 4 in https://arxiv.org/abs/2006.11239
        Args:
            x (Tensor): _description_ Continuous data feature tensor
            t (Tensor): _description_ Discrete time variable between 1 and T
        Returns:
            _type_: _description_
        """

        assert str(t.dtype) == "torch.int64"
        expand_axis = len(x.size()) - 1

        if self.schedule == "adaptive":
            signal = self.get_alpha_bar(t_int=t)
            std = self.get_sigma_bar(t_int=t)
        else:
            signal = self.sqrt_alphas_cumprod[t]
            std = self.sqrt_1m_alphas_cumprod[t]

        for _ in range(expand_axis):
            signal = signal.unsqueeze(-1)
            std = std.unsqueeze(-1)

        mean = signal * x

        return mean, std

    def plot_signal_to_noise(self):
        t = torch.arange(0, self.T)
        signal = self.alphas_cumprod[t]
        noise = 1.0 - signal
        plt.plot(t, signal, label="signal")
        plt.plot(t, noise, label="noise")
        plt.xlabel("timesteps")
        plt.legend()
        plt.show()

    def get_alpha_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        a = self._alphas_bar.to(t_int.device)[t_int.long()]
        return a.float()

    def get_sigma_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma_bar.to(t_int.device)[t_int]
        return s.float()

    def get_sigma2_bar(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        s = self._sigma2_bar.to(t_int.device)[t_int]
        return s.float()

    def get_gamma(self, t_normalized=None, t_int=None, key=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.T)
        g = self._gamma.to(t_int.device)[t_int]
        return g.float()

    def sigma_pos_ts_sq(self, t_int, s_int):
        gamma_s = self.get_gamma(t_int=s_int)
        gamma_t = self.get_gamma(t_int=t_int)
        delta_soft = F.softplus(gamma_s) - F.softplus(gamma_t)
        sigma_squared = -torch.expm1(delta_soft)
        return sigma_squared

    def get_alpha_pos_ts(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        ratio = torch.exp(log_a_bar[t_int] - log_a_bar[s_int])
        return ratio.float()

    def get_alpha_pos_ts_sq(self, t_int, s_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        ratio = torch.exp(2 * log_a_bar[t_int] - 2 * log_a_bar[s_int])
        return ratio.float()

    def get_sigma_pos_sq_ratio(self, s_int, t_int):
        log_a_bar = self._log_alpha_bar.to(t_int.device)
        s2_s = -torch.expm1(2 * log_a_bar[s_int])
        s2_t = -torch.expm1(2 * log_a_bar[t_int])
        ratio = torch.exp(torch.log(s2_s) - torch.log(s2_t))
        return ratio.float()

    def get_x_pos_prefactor(self, s_int, t_int):
        """a_s (s_t^2 - a_t_s^2 s_s^2) / s_t^2"""
        a_s = self.get_alpha_bar(t_int=s_int)
        alpha_ratio_sq = self.get_alpha_pos_ts_sq(t_int=t_int, s_int=s_int)
        sigma_ratio_sq = self.get_sigma_pos_sq_ratio(s_int=s_int, t_int=t_int)
        prefactor = a_s * (1 - alpha_ratio_sq * sigma_ratio_sq)
        return prefactor.float()

    # from EDM
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))
        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s
        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s

    def sigma(self, gamma):
        """Computes sigma given gamma."""
        return torch.sqrt(torch.sigmoid(gamma))

    def alpha(self, gamma):
        """Computes alpha given gamma."""
        return torch.sqrt(torch.sigmoid(-gamma))

    def sample_reverse(
        self,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_index_global=None,
        edge_attrs=None,
        eta_ddim: float = 1.0,
    ):
        rev_sigma = self.reverse_posterior_sigma[t].unsqueeze(-1)
        noise = torch.randn_like(xt)
        std = rev_sigma[batch]

        if edge_index_global is not None:
            noise = torch.randn_like(edge_attrs)
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            noise = torch.randn_like(xt)
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)

        if self.param == "data":
            sigmast = self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
            sigmas2t = sigmast.pow(2)

            sqrt_alphas = self.sqrt_alphas[t].unsqueeze(-1)
            sqrt_1m_alphas_cumprod_prev = torch.sqrt(
                (1.0 - self.alphas_cumprod_prev[t]).clamp_min(1e-4)
            ).unsqueeze(-1)
            one_m_alphas_cumprod_prev = sqrt_1m_alphas_cumprod_prev.pow(2)
            sqrt_alphas_cumprod_prev = torch.sqrt(
                self.alphas_cumprod_prev[t].unsqueeze(-1)
            )
            one_m_alphas = self.discrete_betas[t].unsqueeze(-1)

            mean = (
                sqrt_alphas[batch] * one_m_alphas_cumprod_prev[batch] * xt
                + sqrt_alphas_cumprod_prev[batch] * one_m_alphas[batch] * model_out
            )
            mean = (1.0 / sigmas2t[batch]) * mean
            xt_m1 = mean + eta_ddim * std * noise
        else:
            a = (1 / self.sqrt_alphas[t].unsqueeze(-1)[batch]).clamp(max=2.0)
            b = self.discrete_betas[t].unsqueeze(-1)[batch]
            c = self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)[batch].clamp_min(1e-4)
            xt_m1 = a * (xt - (b / c) * model_out) + eta_ddim * std * noise

        # added while debugging
        if edge_index_global is None and cog_proj:
            xt_m1 = zero_mean(xt_m1, batch=batch, dim_size=bs, dim=0)

        return xt_m1

    def sample_pos(self, t, pos, data_batch, remove_mean=True):
        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        bs = int(data_batch.max()) + 1

        noise_coords_true = torch.randn_like(pos)
        if remove_mean:
            noise_coords_true = zero_mean(
                noise_coords_true, batch=data_batch, dim_size=bs, dim=0
            )
        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.marginal_prob(x=pos, t=t[data_batch])
        # perturb coords
        pos_perturbed = mean_coords + std_coords * noise_coords_true

        return noise_coords_true, pos_perturbed

    def sample(self, t, feature, data_batch):
        noise_coords_true = torch.randn_like(feature)

        # get signal and noise coefficients for coords
        mean_coords, std_coords = self.marginal_prob(x=feature, t=t[data_batch])
        feature_perturbed = mean_coords + std_coords * noise_coords_true

        return noise_coords_true, feature_perturbed

    def sample_reverse_adaptive(
        self,
        s,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_attrs=None,
        edge_index_global=None,
        eta_ddim: float = 1.0,
        return_signal: bool = False,
    ):
        if edge_index_global is not None:
            noise = torch.randn_like(edge_attrs)
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            noise = torch.randn_like(xt)
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)

        if self.param == "data":
            sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=s, t_int=t)

            prefactor1 = self.get_sigma2_bar(t_int=t)
            prefactor2 = self.get_sigma2_bar(t_int=s) * self.get_alpha_pos_ts_sq(
                t_int=t, s_int=s
            )
            sigma2_t_s = prefactor1 - prefactor2
            noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
            noise_prefactor = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)

            z_t_prefactor = (
                self.get_alpha_pos_ts(t_int=t, s_int=s) * sigma_sq_ratio
            ).unsqueeze(-1)
            x_prefactor = self.get_x_pos_prefactor(s_int=s, t_int=t).unsqueeze(-1)
            mu = z_t_prefactor[batch] * xt + x_prefactor[batch] * model_out
            xt_m1 = mu + eta_ddim * noise_prefactor[batch] * noise

        else:
            gamma_t, gamma_s = self.get_gamma(t_int=t), self.get_gamma(t_int=s)
            (
                sigma2_t_given_s,
                sigma_t_given_s,
                alpha_t_given_s,
            ) = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s)

            alpha_t_given_s = alpha_t_given_s.clamp(min=0.001)
            sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = (
                sigma2_t_given_s.unsqueeze(-1),
                sigma_t_given_s.unsqueeze(-1),
                alpha_t_given_s.unsqueeze(-1),
            )

            a = 1.0 / alpha_t_given_s
            a = a[batch]

            sigma_s, sigma_t = self.sigma(gamma_s).unsqueeze(-1), self.sigma(
                gamma_t
            ).unsqueeze(-1)

            mu = (
                xt * a.clamp(max=2.5)
                - (sigma2_t_given_s[batch] * a.clamp(max=2.5) / sigma_t[batch])
                * model_out
            )
            sigma = sigma_t_given_s * sigma_s / sigma_t
            xt_m1 = mu + eta_ddim * sigma[batch] * noise

            # rev_sigma = self.reverse_posterior_sigma[t].unsqueeze(-1)
            # noise = torch.randn_like(xt)
            # std = rev_sigma[batch]
            # a = 1 / self.sqrt_alphas[t].unsqueeze(-1)[batch].clamp(max=2.0)
            # b = self.discrete_betas[t].unsqueeze(-1)[batch]
            # c = self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)[batch].clamp_min(1e-4)
            # xt_m1 = a * (xt - (b / c) * model_out) + eta_ddim * std * noise

        # added while debugging
        if edge_index_global is None and cog_proj:
            xt_m1 = zero_mean(xt_m1, batch=batch, dim_size=bs, dim=0)

        if return_signal:
            return xt_m1, x_prefactor
        return xt_m1

    def sample_reverse_ddim(
        self,
        t,
        xt,
        model_out,
        batch,
        cog_proj=False,
        edge_index_global=None,
        edge_attrs=None,
        eta_ddim: float = 1.0,
    ):
        assert 0.0 <= eta_ddim <= 1.0

        if self.schedule == "cosine":
            if self.param == "noise":
                # convert noise prediction back to data prediction
                # TODO: Use direct sampling instead going back
                alpha_bar, sigma_bar = self.alphas_cum_prod[t].unsqueeze(
                    -1
                ).sqrt(), self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
                model_out = (1.0 / alpha_bar[batch]) * xt - (
                    sigma_bar[batch] / alpha_bar[batch]
                ) * model_out
            rev_sigma = self.reverse_posterior_sigma[t].unsqueeze(-1)
            rev_sigma_ddim = eta_ddim * rev_sigma

            alphas_cumprod_prev = self.alphas_cumprod_prev[t].unsqueeze(-1)
            sqrt_alphas_cumprod_prev = alphas_cumprod_prev.sqrt()

            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
            sqrt_one_m_alphas_cumprod = self.sqrt_1m_alphas_cumprod[t].unsqueeze(-1)
        elif self.schedule == "adaptive":
            if self.param == "noise":
                # convert noise prediction back to data prediction
                # TODO: Use direct sampling instead going back
                alpha_bar, sigma_bar = self.get_alpha_bar(t_int=t).unsqueeze(
                    -1
                ), self.get_sigma_bar(t_int=t).unsqueeze(-1)
                model_out = (1.0 / alpha_bar[batch]) * xt - (
                    sigma_bar[batch] / alpha_bar[batch]
                ) * model_out

            sigma_sq_ratio = self.get_sigma_pos_sq_ratio(s_int=t - 1, t_int=t)
            prefactor1 = self.get_sigma2_bar(t_int=t)
            prefactor2 = self.get_sigma2_bar(t_int=t - 1) * self.get_alpha_pos_ts_sq(
                t_int=t, s_int=t - 1
            )
            sigma2_t_s = prefactor1 - prefactor2
            noise_prefactor_sq = sigma2_t_s * sigma_sq_ratio
            rev_sigma = torch.sqrt(noise_prefactor_sq).unsqueeze(-1)
            rev_sigma_ddim = eta_ddim * rev_sigma

            alphas_cumprod_prev = self.get_alpha_bar(t_int=t - 1).unsqueeze(-1)
            sqrt_alphas_cumprod_prev = alphas_cumprod_prev.sqrt()
            sqrt_alphas_cumprod = self.get_alpha_bar(t_int=t).sqrt().unsqueeze(-1)
            sqrt_one_m_alphas_cumprod = (
                (1.0 - self.get_alpha_bar(t_int=t).unsqueeze(-1)).clamp_min(0.0).sqrt()
            )

        noise = torch.randn_like(xt)

        mean = sqrt_alphas_cumprod_prev[batch] * model_out + (
            1.0 - alphas_cumprod_prev - rev_sigma_ddim.pow(2)
        ).clamp_min(0.0).sqrt()[batch] * (
            (xt - sqrt_alphas_cumprod[batch] * model_out)
            / sqrt_one_m_alphas_cumprod[batch]
        )

        if edge_index_global is not None:
            noise = torch.randn_like(edge_attrs)
            noise = 0.5 * (noise + noise.permute(1, 0, 2))
            noise = noise[edge_index_global[0, :], edge_index_global[1, :], :]
        else:
            bs = int(batch.max()) + 1
            if cog_proj:
                noise = zero_mean(noise, batch=batch, dim_size=bs, dim=0)

        xt_m1 = mean + rev_sigma_ddim[batch] * noise

        return xt_m1

    def snr_s_t_weighting(self, s, t, device, clamp_min=None, clamp_max=None):
        signal_s = self.alphas_cumprod[s]
        noise_s = 1.0 - signal_s
        snr_s = signal_s / noise_s

        signal_t = self.alphas_cumprod[t]
        noise_t = 1.0 - signal_t
        snr_t = signal_t / noise_t
        weights = snr_s - snr_t
        if clamp_min:
            weights = weights.clamp_min(clamp_min)
        if clamp_max:
            weights = weights.clamp_max(clamp_max)
        return weights.to(device)

    def snr_t_weighting(
        self, t, device, clamp_min: float = 0.05, clamp_max: float = 1.5
    ):
        weights = (
            (self.alphas_cumprod[t] / (1.0 - self.alphas_cumprod[t]))
            .clamp(min=clamp_min, max=clamp_max)
            .to(device)
        )
        return weights

    def exp_t_half_weighting(self, t, device):
        weights = torch.clip(torch.exp(-t / 100 + 1 / 2), min=0.1).to(device)
        return weights

    def exp_t_weighting(self, t, device):
        weights = torch.clip(torch.exp(-t / 200), min=0.1).to(device)
        return weights


if __name__ == "__main__":
    T = 500
    sde = DiscreteDDPM(
        beta_min=1e-4,
        beta_max=2e-2,
        N=T,
        scaled_reverse_posterior_sigma=True,
        schedule="adaptive",
        enforce_zero_terminal_snr=False,
        nu=1.0,
    )
    s = torch.arange(1, T)
    t = s + 1
    signal_s = sde.alphas_cumprod[s]
    noise_s = 1.0 - signal_s
    snr_s = signal_s / noise_s

    signal_t = sde.alphas_cumprod[t]
    noise_t = 1.0 - signal_t
    snr_t = signal_t / noise_t
    w = snr_s - snr_t

    ids = 1
    plt.plot(range(len(w))[ids:], w[ids:].clamp(min=5.0), label="weighting")
    plt.legend()
    plt.show()

    ids = 1
    plt.plot(
        range(len(sde.alphas_cumprod)),
        (sde.alphas_cumprod / (1.0 - sde.alphas_cumprod)).clamp(min=0.05, max=5.0),
        label="weighting",
    )
    plt.legend()
    plt.show()

    logsnr_s = torch.log(snr_s)
    logsnr_t = torch.log(snr_t)

    weights = torch.clip(torch.exp(-t / 200), min=0.1)
    ids = 0
    plt.plot(range(len(weights))[ids:], weights[ids:], label="weighting")
    plt.legend()
    plt.show()

    signal = sde.alphas_cumprod
    noise = 1.0 - signal
    plt.plot(np.arange(len(signal)), signal.sqrt(), label="signal")
    plt.plot(np.arange(len(noise)), noise.sqrt(), label="noise")
    plt.xlabel("timesteps")
    # adaptive
    sde = DiscreteDDPM(
        beta_min=1e-4,
        beta_max=2e-2,
        N=T,
        scaled_reverse_posterior_sigma=True,
        schedule="adaptive",
        enforce_zero_terminal_snr=False,
        nu=2.5,
    )
    t = torch.arange(0, T)
    signal = sde.get_alpha_bar(t_int=t)
    noise = sde.get_sigma_bar(t_int=t)
    plt.plot(t, signal.sqrt(), label="adaptive-signal-sqrt")
    plt.plot(t, noise, label="adaptive-noise")
    plt.xlabel("timesteps")

    t = torch.arange(0, T)
    signal = sde.get_alpha_bar(t_int=t)
    noise = sde.get_sigma_bar(t_int=t)
    plt.plot(t, signal, label="adaptive-signal")
    plt.plot(t, noise, label="adaptive-nois")
    plt.xlabel("timesteps")

    plt.legend()
    plt.show()

    sde.plot_signal_to_noise()

    plt.plot(range(len(sde.discrete_betas)), sde.discrete_betas, label="betas")
    plt.plot(range(len(sde.alphas)), sde.alphas, label="alphas")
    plt.xlabel("t")
    plt.legend()
    plt.show()

    #### SNR
    # equation 1 and 2 in https://arxiv.org/pdf/2107.00630.pdf
    signal = sde.alphas_cumprod  # \alpha_t
    noise = 1.0 - signal  # \sigma_t^2
    snr = signal / noise
    plt.plot(range(len(signal)), snr, label="SNR(t)")
    plt.legend()
    plt.show()

    #### notation from kingma
    gamma = torch.log(snr)
    plt.plot(range(len(signal)), gamma, label="gamam(t) = log_e(SNR(t))")
    plt.legend()
    plt.show()

    ### From https://arxiv.org/pdf/2303.00848.pdf
    # sigmoidal
    plt.plot(gamma, torch.sigmoid(-gamma + 2), label="sigmoidal-weights")
    plt.legend()
    plt.show()

    plt.plot(gamma, torch.sigmoid(-gamma + 2), label="sigmoidal-weights")
    plt.legend()
    plt.show()

    plt.plot(
        torch.arange(len(gamma)), torch.sigmoid(-gamma + 2), label="sigmoidal-weights"
    )
    plt.legend()
    plt.show()
