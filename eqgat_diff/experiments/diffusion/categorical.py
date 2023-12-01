import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import sort_edge_index

from experiments.diffusion.continuous import get_beta_schedule

DEFAULT_BETAS = get_beta_schedule(kind="cosine", num_diffusion_timesteps=500)
DEFAULT_ALPHAS = 1.0 - DEFAULT_BETAS
ALPHAS_BAR = torch.cumprod(DEFAULT_ALPHAS, dim=0)


def get_one_step_transition(alpha_t: float, terminal_distribution: torch.Tensor):
    stay_prob = torch.eye(len(terminal_distribution)) * alpha_t
    diffuse_prob = (1.0 - alpha_t) * (
        torch.ones(1, len(terminal_distribution)) * (terminal_distribution.unsqueeze(0))
    )
    Q_t = stay_prob + diffuse_prob
    return Q_t


class CategoricalDiffusionKernel(torch.nn.Module):
    def __init__(
        self,
        terminal_distribution: torch.Tensor,
        alphas: torch.Tensor = DEFAULT_ALPHAS,
        num_bond_types: int = 5,
        num_atom_types: int = 16,
        num_charge_types: int = 6,
        num_is_in_ring: int = 2,
        num_is_aromatic: int = 2,
        num_hybridization: int = 9,
    ):
        super().__init__()

        self.num_bond_types = num_bond_types
        self.num_atom_types = num_atom_types
        self.num_charge_types = num_charge_types
        self.num_is_in_ring = num_is_in_ring
        self.num_is_aromatic = num_is_aromatic
        self.num_hybridization = num_hybridization

        self.num_classes = len(terminal_distribution)
        assert (terminal_distribution.sum() - 1.0).abs() < 1e-4

        self.register_buffer("eye", torch.eye(self.num_classes))
        self.register_buffer("terminal_distribution", terminal_distribution)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", torch.cumprod(alphas, dim=0))
        self.register_buffer("one_minus_alphas_bar", 1.0 - self.alphas_bar)
        Qt = [
            get_one_step_transition(
                alpha_t=a.item(), terminal_distribution=terminal_distribution
            )
            for a in alphas
        ]
        self.register_buffer("Qt", torch.stack(Qt, dim=0))
        Qt_prev = torch.eye(self.num_classes)
        Qt_bar = []
        for i in range(len(alphas)):
            Qtb = Qt_prev @ Qt[i]
            Qt_bar.append(Qtb)
            Qt_prev = Qtb

        Qt_bar = torch.stack(Qt_bar)
        Qt_bar_prev = Qt_bar[:-1]
        Qt_prev_pad = torch.eye(self.num_classes)
        Qt_bar_prev = torch.concat([Qt_prev_pad.unsqueeze(0), Qt_bar_prev], dim=0)
        self.register_buffer("Qt_bar", Qt_bar)
        self.register_buffer("Qt_bar_prev", Qt_bar_prev)

    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor):
        """_summary_
        Computes the forward categorical posterior q(xt | x0) ~ Cat(xt, p = x0_j . Qt_bar_ji)
        Args:
            x0 (torch.Tensor): _description_ one-hot vectors of shape (n, k)
            t (torch.Tensor): _description_ time variable of shape (n,)

        Returns:
            _type_: _description_
        """

        # Qt_bar (k0, k_t)
        probs = torch.einsum("nj, nji -> ni", [x0, self.Qt_bar[t]])
        check = torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
        assert check

        return probs

    def reverse_posterior_for_every_x0(self, xt: torch.Tensor, t: torch.Tensor):
        """_summary_
        Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
        but for every possible value of x0
        Args:
            xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
            t (torch.Tensor): _description_ time variable of shape (n,)
        Returns:
            _type_: _description_
        """

        # xt: (n, k_t)

        # x0 = torch.eye(self.num_classes, device=xt.device, dtype=xt.dtype).unsqueeze(0)
        # x0 = x0.repeat((xt.size(0), 1, 1))
        # (n, k, k)

        Qt_T = self.Qt[t]  # (n, k_t-1, k_t)
        assert Qt_T.ndim == 3
        Qt_T = Qt_T.permute(0, 2, 1)
        # (n, k_t, k_t-1)

        a = torch.einsum("nj, nji -> ni", [xt, Qt_T])
        # (n, k_t-1)

        a = a.unsqueeze(1)
        # (n, 1, k_t-1)

        # b = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar_prev[t]])
        b = self.Qt_bar_prev[t]
        # (n, k_0, k_t-1)

        p0 = a * b
        # (n, k_0, k_t-1)

        # p1 = torch.einsum('naj, nji -> nai', [x0, self.Qt_bar[t]])
        p1 = self.Qt_bar[t]
        # (n, k_0, k_t)

        ## xt_ = xt.unsqueeze(1)
        # (n, 1, k_t)

        ## p1 = (p1 * xt_).sum(-1, keepdims=True)

        p1 = torch.einsum("nij, nj -> ni", [p1, xt])
        # (n, k_0)

        p1 = p1.unsqueeze(-1)
        # (n, k_0, 1)

        probs = p0 / (p1.clamp(min=1e-5))
        # (n, k_0, k_t-1)

        # check = torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
        # assert check

        return probs

    def reverse_posterior(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
        """_summary_
        Computes the reverse posterior q(x_{t-1} | xt, x0) as described in Austin et al. (2021) https://arxiv.org/abs/2107.03006 in Eq.3
        Args:
            x0 (torch.Tensor): _description_ one specific one-hot vector of shape (n, k)
            xt (torch.Tensor): _description_ a perturbed (noisy) one-hot vector of shape (n, k)
            t (torch.Tensor): _description_ time variable of shape (n,)
        Returns:
            _type_: _description_
        """
        a = torch.einsum("nj, nji -> ni", [xt, self.Qt[t].transpose(-2, -1)])
        b = torch.einsum("nj, nji -> ni", [x0, self.Qt_bar_prev[t]])
        p0 = a * b
        # (n, k)
        p1 = torch.einsum("nj, nji -> ni", [x0, self.Qt_bar[t]])
        p1 = (p1 * xt).sum(-1, keepdims=True)
        # (n, 1)

        probs = p0 / p1
        check = torch.all((probs.sum(-1) - 1.0).abs() < 1e-4)
        assert check

        return probs

    def sample_reverse_categorical(
        self,
        xt: Tensor,
        x0: Tensor,
        t: Tensor,
        num_classes: int,
        eps: float = 1.0e-5,
        local_rank: int = 0,
    ):
        reverse = self.reverse_posterior_for_every_x0(xt=xt, t=t)
        # Eq. 4 in Austin et al. (2023) "Structured Denoising Diffusion Models in Discrete State-Spaces"
        # (N, a_0, a_t-1)
        unweighted_probs = (reverse * x0.unsqueeze(-1)).sum(1)
        unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
        # (N, a_t-1)
        probs = unweighted_probs / (unweighted_probs.sum(-1, keepdims=True) + eps)
        x_tm1 = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=num_classes,
        ).float()

        return x_tm1

    def sample_reverse_edges_categorical(
        self,
        edge_attr_global: Tensor,
        edges_pred: Tensor,
        t: Tensor,
        mask: Tensor,
        mask_i: Tensor,
        batch: Tensor,
        edge_index_global: Tensor,
        num_classes: int,
    ):
        x0 = edges_pred[mask]
        xt = edge_attr_global[mask]
        t = t[batch[mask_i]]

        reverse = self.reverse_posterior_for_every_x0(xt=xt, t=t)
        # Eq. 4 in Austin et al. (2023) "Structured Denoising Diffusion Models in Discrete State-Spaces"
        # (N, a_0, a_t-1)
        unweighted_probs = (reverse * x0.unsqueeze(-1)).sum(1)
        unweighted_probs[unweighted_probs.sum(dim=-1) == 0] = 1e-5
        # (N, a_t-1)
        probs = unweighted_probs / unweighted_probs.sum(-1, keepdims=True)
        edges_triu = F.one_hot(
            probs.multinomial(
                1,
            ).squeeze(),
            num_classes=num_classes,
        ).float()

        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global = torch.stack([j, i], dim=0)
        edge_attr_global = torch.concat([edges_triu, edges_triu], dim=0)
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            sort_by_row=False,
        )

        return edge_attr_global, edge_index_global, mask, mask_i

    def sample_edges_categorical(
        self,
        t,
        edge_index_global,
        edge_attr_global,
        data_batch,
        return_one_hot=True,
    ):
        j, i = edge_index_global
        mask = j < i
        mask_i = i[mask]
        mask_j = j[mask]
        edge_attr_triu = edge_attr_global[mask]
        edge_attr_triu_ohe = F.one_hot(
            edge_attr_triu, num_classes=self.num_bond_types
        ).float()
        t_edge = t[data_batch[mask_i]]
        probs = self.marginal_prob(edge_attr_triu_ohe, t=t_edge)
        edges_t_given_0 = probs.multinomial(
            1,
        ).squeeze()
        j = torch.concat([mask_j, mask_i])
        i = torch.concat([mask_i, mask_j])
        edge_index_global_perturbed = torch.stack([j, i], dim=0)
        edge_attr_global_perturbed = torch.concat(
            [edges_t_given_0, edges_t_given_0], dim=0
        )
        edge_index_global_perturbed, edge_attr_global_perturbed = sort_edge_index(
            edge_index=edge_index_global_perturbed,
            edge_attr=edge_attr_global_perturbed,
            sort_by_row=False,
        )

        edge_attr_global_perturbed = (
            F.one_hot(
                edge_attr_global_perturbed, num_classes=self.num_bond_types
            ).float()
            if return_one_hot
            else edge_attr_global_perturbed
        )

        return edge_attr_global_perturbed

    def sample_categorical(
        self, t, x0, data_batch, dataset_info, num_classes=16, type="atoms"
    ):
        assert type in ["atoms", "charges", "ring", "aromatic", "hybridization"]

        if type == "charges":
            x0 = dataset_info.one_hot_charges(x0)
        else:
            x0 = F.one_hot(x0.squeeze().long(), num_classes=num_classes).float()
        probs = self.marginal_prob(x0.float(), t[data_batch])
        x0_perturbed = probs.multinomial(
            1,
        ).squeeze()
        x0_perturbed = F.one_hot(x0_perturbed, num_classes=num_classes).float()

        return x0, x0_perturbed


def _some_debugging():
    num_classes = 5
    uniform_distribution = (
        torch.ones(
            num_classes,
        )
        / num_classes
    )
    absorbing_distribution = torch.zeros(
        num_classes,
    )
    absorbing_distribution[0] = 1.0
    absorbing_distribution = torch.tensor(
        [9.5523e-01, 3.0681e-02, 2.0021e-03, 4.4172e-05, 1.2045e-02]
    )

    atoms_drugs = [
        4.4119e-01,
        1.0254e-06,
        4.0564e-01,
        6.4677e-02,
        6.6144e-02,
        4.8741e-03,
        0.0000e00,
        9.1150e-07,
        1.0847e-04,
        1.2260e-02,
        4.0306e-03,
        0.0000e00,
        1.0503e-03,
        1.9806e-05,
        0.0000e00,
        7.5958e-08,
    ]

    edges_drugs = [9.5523e-01, 3.0681e-02, 2.0021e-03, 4.4172e-05, 1.2045e-02]

    atoms_qm9 = [0.5122, 0.3526, 0.0562, 0.0777, 0.0013]
    edges_qm9 = [0.8818, 0.1104, 0.0060, 0.0018, 0.0000]

    C0 = CategoricalDiffusionKernel(
        terminal_distribution=uniform_distribution, alphas=DEFAULT_ALPHAS
    )

    C1 = CategoricalDiffusionKernel(
        terminal_distribution=torch.tensor(edges_drugs), alphas=DEFAULT_ALPHAS
    )

    t = 290
    a = C0.Qt_bar[t]
    alphas_bar_t = C0.alphas_bar[t].unsqueeze(-1)
    b = alphas_bar_t * C0.eye + (
        (1.0 - alphas_bar_t) * torch.ones_like(C0.terminal_distribution)
    ).unsqueeze(-1) * C0.terminal_distribution.unsqueeze(0)

    print(a - b)

    alphas_t = C0.alphas[t].unsqueeze(-1)
    a = C0.Qt[t]
    b = alphas_t * C0.eye + (
        (1.0 - alphas_t) * torch.ones_like(C0.terminal_distribution)
    ).unsqueeze(-1) * C0.terminal_distribution.unsqueeze(0)

    print(a - b)
    Qt = C1.Qt[t]
    return None
