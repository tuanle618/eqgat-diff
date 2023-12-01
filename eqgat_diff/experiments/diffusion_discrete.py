import logging
import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from tqdm import tqdm

from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import prepare_context
from experiments.data.utils import (
    write_trajectory_as_xyz,
    write_xyz_file_from_batch,
)
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.utils import (
    bond_guidance,
    energy_guidance,
    initialize_edge_attrs_reverse,
)
from experiments.losses import DiffusionLoss
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (
    coalesce_edges,
    get_molecules,
    load_bond_model,
    load_energy_model,
    load_model,
    zero_mean,
)

logging.getLogger("lightning").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.utilities.rank_zero").addHandler(
    logging.NullHandler()
)
logging.getLogger("pytorch_lightning.accelerators.cuda").addHandler(
    logging.NullHandler()
)

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]


class Trainer(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        dataset_info: AbstractDatasetInfos,
        smiles_list: list,
        prop_dist=None,
        prop_norm=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0
        self.mol_stab = 0.5

        self.dataset_info = dataset_info
        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()

        if (
            "use_absorbing_state" in self.hparams.keys()
        ):  # just as a hack for now for loading older models
            if self.hparams.use_absorbing_state:
                print(
                    "Using absorbing state instead of training distribution as prior!"
                )
                atom_types_distribution = torch.zeros((17,), dtype=torch.float32)
                atom_types_distribution[-1] = 1.0
                charge_types_distribution = torch.zeros((7,), dtype=torch.float32)
                charge_types_distribution[-1] = 1.0
                dataset_info.num_atom_types = 17
                dataset_info.num_charge_classes = 7

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        if "use_absorbing_state" in self.hparams.keys():  # just as a hack for now
            if self.hparams.use_absorbing_state:
                self.hparams.num_atom_types += 1
                self.num_charge_classes += 1

        self.remove_hs = hparams.get("remove_hs")
        if self.remove_hs:
            print("Model without modelling explicit hydrogens")

        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = 5

        self.smiles_list = smiles_list

        empirical_num_nodes = dataset_info.n_nodes
        self.register_buffer(name="empirical_num_nodes", tensor=empirical_num_nodes)

        if self.hparams.load_ckpt_from_pretrained is not None:
            print("Loading from pre-trained model checkpoint...")

            self.model = load_model(
                self.hparams.load_ckpt_from_pretrained, self.num_atom_features
            )
            # num_params = len(self.model.state_dict())
            # for i, param in enumerate(self.model.parameters()):
            #     if i < num_params // 2:
            #         param.requires_grad = False
        # elif self.hparams.load_ckpt:
        #     print("Loading from model checkpoint...")
        #     self.model = load_model(self.hparams.load_ckpt, self.num_atom_features)
        else:
            self.model = DenoisingEdgeNetwork(
                hn_dim=(hparams["sdim"], hparams["vdim"]),
                num_layers=hparams["num_layers"],
                latent_dim=None,
                use_cross_product=hparams["use_cross_product"],
                num_atom_features=self.num_atom_features,
                num_bond_types=self.num_bond_classes,
                edge_dim=hparams["edim"],
                cutoff_local=hparams["cutoff_local"],
                vector_aggr=hparams["vector_aggr"],
                fully_connected=hparams["fully_connected"],
                local_global_model=hparams["local_global_model"],
                recompute_edge_attributes=True,
                recompute_radius_graph=False,
                edge_mp=hparams["edge_mp"],
                context_mapping=hparams["context_mapping"],
                num_context_features=hparams["num_context_features"],
                bond_prediction=hparams["bond_prediction"],
                property_prediction=hparams["property_prediction"],
                coords_param=hparams["continuous_param"],
            )

        self.sde_pos = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=2.5,
            enforce_zero_terminal_snr=False,
            T=self.hparams.timesteps,
            param=self.hparams.continuous_param,
        )
        self.sde_atom_charge = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1,
            enforce_zero_terminal_snr=False,
        )
        self.sde_bonds = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1.5,
            enforce_zero_terminal_snr=False,
        )

        self.cat_atoms = CategoricalDiffusionKernel(
            terminal_distribution=atom_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_distribution,
            alphas=self.sde_bonds.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )

        self.diffusion_loss = DiffusionLoss(
            modalities=["coords", "atoms", "charges", "bonds"],
            param=["data", "data", "data", "data"],
        )

        if self.hparams.bond_model_guidance:
            print("Using bond model guidance...")
            self.bond_model = load_bond_model(
                self.hparams.ckpt_bond_model, dataset_info
            )
            self.bond_model.eval()

    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.hparams.test_interval == 0:
            if self.local_rank == 0:
                print(f"Running evaluation in epoch {self.current_epoch + 1}")
            final_res = self.run_evaluation(
                step=self.i,
                dataset_info=self.dataset_info,
                ngraphs=1000,
                bs=self.hparams.inference_batch_size,
                verbose=True,
                inner_verbose=False,
                eta_ddim=1.0,
                ddpm=True,
                every_k_step=1,
                device="cuda" if self.hparams.gpus > 1 else "cpu",
            )
            self.i += 1
            self.log(
                name="val/validity",
                value=final_res.validity[0],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                name="val/uniqueness",
                value=final_res.uniqueness[0],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                name="val/novelty",
                value=final_res.novelty[0],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                name="val/mol_stable",
                value=final_res.mol_stable[0],
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                name="val/atm_stable",
                value=final_res.atm_stable[0],
                on_epoch=True,
                sync_dist=True,
            )

    def _log(
        self, loss, coords_loss, atoms_loss, charges_loss, bonds_loss, batch_size, stage
    ):
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=False,
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/coords_loss",
            coords_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/atoms_loss",
            atoms_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/charges_loss",
            charges_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

        self.log(
            f"{stage}/bonds_loss",
            bonds_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )

    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1

        t = torch.randint(
            low=1,
            high=self.hparams.timesteps + 1,
            size=(batch_size,),
            dtype=torch.long,
            device=batch.x.device,
        )
        if self.hparams.loss_weighting == "snr_s_t":
            weights = self.sde_atom_charge.snr_s_t_weighting(
                s=t - 1, t=t, device=self.device, clamp_min=0.05, clamp_max=1.5
            )
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.sde_atom_charge.snr_t_weighting(
                t=t,
                device=self.device,
                clamp_min=0.05,
                clamp_max=1.5,
            )
        elif self.hparams.loss_weighting == "exp_t":
            weights = self.sde_atom_charge.exp_t_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "exp_t_half":
            weights = self.sde_atom_charge.exp_t_half_weighting(t=t, device=self.device)
        elif self.hparams.loss_weighting == "uniform":
            weights = None

        if self.hparams.context_mapping:
            context = prepare_context(
                self.hparams["properties_list"],
                self.prop_norm,
                batch,
                self.hparams.dataset,
            )
            batch.context = context

        out_dict = self(batch=batch, t=t)

        true_data = {
            "coords": out_dict["coords_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["coords_noise_true"],
            "atoms": out_dict["atoms_true"],
            "charges": out_dict["charges_true"],
            "bonds": out_dict["bonds_true"],
        }

        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        atoms_pred, charges_pred = atoms_pred.split(
            [self.num_atom_types, self.num_charge_classes], dim=-1
        )
        edges_pred = out_dict["bonds_pred"]

        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
        }

        loss = self.diffusion_loss(
            true_data=true_data,
            pred_data=pred_data,
            batch=batch.batch,
            bond_aggregation_index=out_dict["bond_aggregation_index"],
            weights=weights,
        )

        final_loss = (
            self.hparams.lc_coords * loss["coords"]
            + self.hparams.lc_atoms * loss["atoms"]
            + self.hparams.lc_bonds * loss["bonds"]
            + self.hparams.lc_charges * loss["charges"]
        )

        if torch.any(final_loss.isnan()):
            final_loss = final_loss[~final_loss.isnan()]
            print(f"Detected NaNs. Terminating training at epoch {self.current_epoch}")
            exit()

        self._log(
            final_loss,
            loss["coords"],
            loss["atoms"],
            loss["charges"],
            loss["bonds"],
            batch_size,
            stage,
        )

        return final_loss

    def forward(self, batch: Batch, t: Tensor):
        atom_types: Tensor = batch.x
        pos: Tensor = batch.pos
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        context = batch.context if self.hparams.context_mapping else None
        n = batch.num_nodes
        bs = int(data_batch.max()) + 1
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)

        if not hasattr(batch, "fc_edge_index"):
            edge_index_global = (
                torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.fc_edge_index

        edge_index_global, edge_attr_global = coalesce_edges(
            edge_index=edge_index_global,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=pos.size(0),
        )
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
        )
        batch_edge_global = data_batch[edge_index_global[0]]

        # SAMPLING
        noise_coords_true, pos_perturbed = self.sde_pos.sample_pos(
            t, pos_centered, data_batch
        )
        atom_types, atom_types_perturbed = self.cat_atoms.sample_categorical(
            t,
            atom_types,
            data_batch,
            self.dataset_info,
            num_classes=self.num_atom_types,
            type="atoms",
        )

        charges, charges_perturbed = self.cat_charges.sample_categorical(
            t,
            charges,
            data_batch,
            self.dataset_info,
            num_classes=self.num_charge_classes,
            type="charges",
        )
        edge_attr_global_perturbed = (
            self.cat_bonds.sample_edges_categorical(
                t, edge_index_global, edge_attr_global, data_batch
            )
            if not self.hparams.bond_prediction
            else None
        )
        atom_feats_in_perturbed = torch.cat(
            [atom_types_perturbed, charges_perturbed], dim=-1
        )

        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_perturbed
            if not self.hparams.bond_prediction
            else None,
            batch=data_batch,
            batch_edge_global=batch_edge_global,
            context=context,
        )

        out["coords_perturbed"] = pos_perturbed
        out["atoms_perturbed"] = atom_types_perturbed
        out["charges_perturbed"] = charges_perturbed
        out["bonds_perturbed"] = edge_attr_global_perturbed

        out["coords_true"] = pos_centered
        out["coords_noise_true"] = noise_coords_true
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global
        out["charges_true"] = charges.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global[1]

        return out

    @torch.no_grad()
    def run_evaluation(
        self,
        step: int,
        dataset_info,
        ngraphs: int = 4000,
        bs: int = 500,
        save_dir: str = None,
        return_molecules: bool = False,
        verbose: bool = False,
        inner_verbose=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        run_test_eval: bool = False,
        guidance_scale: float = 1.0e-4,
        use_energy_guidance: bool = False,
        ckpt_energy_model: str = None,
        save_traj: bool = False,
        fix_noise_and_nodes: bool = False,
        guidance_steps: int = 100,
        optimization: str = "minimize",
        relax_sampling: bool = False,
        relax_steps: int = 10,
        device: str = "cpu",
    ):
        energy_model = None
        if use_energy_guidance:
            energy_model = load_energy_model(ckpt_energy_model, self.num_atom_features)
            # for param in self.energy_model.parameters():
            #    param.requires_grad = False
            energy_model.to(self.device)
            energy_model.eval()

        b = ngraphs // bs
        l = [bs] * b
        if sum(l) != ngraphs:
            l.append(ngraphs - sum(l))
        assert sum(l) == ngraphs

        start = datetime.now()
        if verbose:
            if self.local_rank == 0:
                print(f"Creating {ngraphs} graphs in {l} batches")

        molecule_list = []
        for i, num_graphs in enumerate(l):
            molecules = self.reverse_sampling(
                num_graphs=num_graphs,
                verbose=inner_verbose,
                save_traj=save_traj,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                guidance_scale=guidance_scale,
                energy_model=energy_model,
                save_dir=save_dir,
                fix_noise_and_nodes=fix_noise_and_nodes,
                iteration=i,
                guidance_steps=guidance_steps,
                optimization=optimization,
                relax_sampling=relax_sampling,
                relax_steps=relax_steps,
            )

            molecule_list.extend(molecules)
        (
            stability_dict,
            validity_dict,
            statistics_dict,
            all_generated_smiles,
            stable_molecules,
            valid_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=molecule_list,
            dataset_info=dataset_info,
            smiles_train=self.smiles_list,
            local_rank=self.local_rank,
            return_molecules=return_molecules,
            device=device,
        )

        save_cond = (
            self.mol_stab < stability_dict["mol_stable"]
            if self.hparams.dataset != "qm9"
            else (
                self.mol_stab < stability_dict["mol_stable"]
                and validity_dict["novelty"] > 0.75
            )
        )
        if save_cond and not run_test_eval:
            self.mol_stab = stability_dict["mol_stable"]
            save_path = os.path.join(self.hparams.save_dir, "best_mol_stab.ckpt")
            self.trainer.save_checkpoint(save_path)
            # for g in self.optimizers().param_groups:
            #     if g['lr'] > self.hparams.lr_min:
            #         g['lr'] *= 0.9

        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")
        total_res = dict(stability_dict)
        total_res.update(validity_dict)
        total_res.update(statistics_dict)
        if self.local_rank == 0:
            print(total_res)
        total_res = pd.DataFrame.from_dict([total_res])
        if self.local_rank == 0:
            print(total_res)

        total_res["step"] = str(step)
        total_res["epoch"] = str(self.current_epoch)
        total_res["run_time"] = str(run_time)
        total_res["ngraphs"] = str(ngraphs)
        try:
            if save_dir is None:
                save_dir = os.path.join(
                    self.hparams.save_dir,
                    "run" + str(self.hparams.id),
                    "evaluation.csv",
                )
                print(f"Saving evaluation csv file to {save_dir}")
            else:
                save_dir = os.path.join(save_dir, "evaluation.csv")
            if self.local_rank == 0:
                with open(save_dir, "a") as f:
                    total_res.to_csv(f, header=True)
        except Exception as e:
            print(e)
            pass

        if return_molecules:
            return total_res, all_generated_smiles, stable_molecules
        else:
            return total_res

    @torch.no_grad()
    def generate_valid_samples(
        self,
        dataset_info,
        ngraphs: int = 4000,
        bs: int = 500,
        save_dir: str = None,
        return_molecules: bool = False,
        verbose: bool = False,
        inner_verbose=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        guidance_scale: float = 1.0e-4,
        use_energy_guidance: bool = False,
        ckpt_energy_model: str = None,
        save_traj: bool = False,
        fix_noise_and_nodes: bool = False,
        guidance_steps: int = 100,
        optimization: str = "minimize",
        relax_sampling: bool = False,
        relax_steps: int = 10,
        device: str = "cpu",
    ):
        energy_model = None
        if use_energy_guidance:
            energy_model = load_energy_model(ckpt_energy_model, self.num_atom_features)
            # for param in self.energy_model.parameters():
            #    param.requires_grad = False
            energy_model.to(self.device)
            energy_model.eval()

        start = datetime.now()

        i = 0
        n_graphs_remaining = ngraphs
        valid_molecule_list = []
        while len(valid_molecule_list) < ngraphs:
            b = n_graphs_remaining // bs
            l = [bs] * b
            if sum(l) != n_graphs_remaining:
                l.append(n_graphs_remaining - sum(l))
            assert sum(l) == n_graphs_remaining
            if verbose:
                if self.local_rank == 0:
                    print(f"Creating {n_graphs_remaining} graphs in {l} batches")
            molecule_list = []
            for num_graphs in l:
                molecules = self.reverse_sampling(
                    num_graphs=num_graphs,
                    verbose=inner_verbose,
                    save_traj=save_traj,
                    ddpm=ddpm,
                    eta_ddim=eta_ddim,
                    every_k_step=every_k_step,
                    guidance_scale=guidance_scale,
                    energy_model=energy_model,
                    save_dir=save_dir,
                    fix_noise_and_nodes=fix_noise_and_nodes,
                    iteration=i,
                    guidance_steps=guidance_steps,
                    optimization=optimization,
                    relax_sampling=relax_sampling,
                    relax_steps=relax_steps,
                )

                molecule_list.extend(molecules)
                i += 1

            valid_molecules = analyze_stability_for_molecules(
                molecule_list=molecule_list,
                dataset_info=dataset_info,
                smiles_train=self.smiles_list,
                local_rank=self.local_rank,
                return_molecules=return_molecules,
                calculate_statistics=False,
                device=device,
            )
            valid_molecule_list.extend(valid_molecules)

            n_graphs_remaining = (ngraphs - len(valid_molecule_list)) * 2

        (
            stability_dict,
            validity_dict,
            statistics_dict,
            all_generated_smiles,
            stable_molecules,
            valid_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=valid_molecule_list,
            dataset_info=dataset_info,
            smiles_train=self.smiles_list,
            local_rank=self.local_rank,
            return_molecules=return_molecules,
            device=device,
        )

        run_time = datetime.now() - start
        if verbose:
            if self.local_rank == 0:
                print(f"Run time={run_time}")
        total_res = dict(stability_dict)
        total_res.update(validity_dict)
        total_res.update(statistics_dict)
        if self.local_rank == 0:
            print(total_res)
        total_res = pd.DataFrame.from_dict([total_res])
        if self.local_rank == 0:
            print(total_res)

        total_res["run_time"] = str(run_time)
        total_res["ngraphs"] = str(len(valid_molecules))
        try:
            if save_dir is None:
                save_dir = os.path.join(
                    self.hparams.save_dir,
                    "run" + str(self.hparams.id),
                    "evaluation.csv",
                )
                print(f"Saving evaluation csv file to {save_dir}")
            else:
                save_dir = os.path.join(save_dir, "evaluation.csv")
            if self.local_rank == 0:
                with open(save_dir, "a") as f:
                    total_res.to_csv(f, header=True)
        except Exception as e:
            print(e)
            pass

        if return_molecules:
            return total_res, all_generated_smiles, stable_molecules
        else:
            return total_res

    def reverse_sampling(
        self,
        num_graphs: int,
        verbose: bool = False,
        save_traj: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        guidance_scale: float = 1.0e-4,
        energy_model=None,
        save_dir: float = None,
        fix_noise_and_nodes: bool = False,
        iteration: int = 0,
        chain_iterator=None,
        guidance_steps: int = 100,
        optimization: str = "minimize",
        relax_sampling: bool = False,
        relax_steps: int = 10,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        if not fix_noise_and_nodes:
            batch_num_nodes = torch.multinomial(
                input=self.empirical_num_nodes,
                num_samples=num_graphs,
                replacement=True,
            ).to(self.device)
        else:
            seed_value = 42 + iteration
            torch.random.manual_seed(seed_value)
            batch_num_nodes = (
                torch.multinomial(
                    input=self.empirical_num_nodes,
                    num_samples=1,
                    replacement=True,
                )
                .repeat(num_graphs)
                .to(self.device)
            )

        batch_num_nodes = batch_num_nodes.clamp(min=1)
        batch = torch.arange(num_graphs, device=self.device).repeat_interleave(
            batch_num_nodes, dim=0
        )
        bs = int(batch.max()) + 1

        if energy_model is not None:
            t = torch.arange(0, self.hparams.timesteps)
            alphas = self.sde_pos.alphas_cumprod[t]

        # sample context condition
        context = None
        if self.prop_dist is not None:
            context = self.prop_dist.sample_batch(batch_num_nodes).to(self.device)[
                batch
            ]

        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(
            len(batch), 3, device=self.device, dtype=torch.get_default_dtype()
        )

        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)

        n = len(pos)

        # initialize the atom-types
        atom_types = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types = F.one_hot(atom_types, self.num_atom_types).float()

        charge_types = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types = F.one_hot(charge_types, self.num_charge_classes).float()

        # edge_index_local = radius_graph(x=pos,
        #                                r=self.hparams.cutoff_local,
        #                                batch=batch,
        #                                max_num_neighbors=self.hparams.max_num_neighbors)
        edge_index_local = None
        edge_index_global = (
            torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0)
        )
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        if not self.hparams.bond_prediction:
            (
                edge_attr_global,
                edge_index_global,
                mask,
                mask_i,
            ) = initialize_edge_attrs_reverse(
                edge_index_global,
                n,
                self.bonds_prior,
                self.num_bond_classes,
                self.device,
            )
        else:
            edge_attr_global = None
        batch_edge_global = batch[edge_index_global[0]]

        if chain_iterator is None:
            if self.hparams.continuous_param == "data":
                chain = range(0, self.hparams.timesteps)
            elif self.hparams.continuous_param == "noise":
                chain = range(0, self.hparams.timesteps - 1)

            chain = chain[::every_k_step]

            iterator = (
                tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
            )
        else:
            iterator = chain_iterator

        for i, timestep in enumerate(iterator):
            s = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=pos.device
            )
            t = s + 1

            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            node_feats_in = torch.cat([atom_types, charge_types], dim=-1)
            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
                context=context,
            )

            coords_pred = out["coords_pred"].squeeze()
            atoms_pred, charges_pred = out["atoms_pred"].split(
                [self.num_atom_types, self.num_charge_classes], dim=-1
            )
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)

            if ddpm:
                if self.hparams.noise_scheduler == "adaptive":
                    # positions
                    pos = self.sde_pos.sample_reverse_adaptive(
                        s,
                        t,
                        pos,
                        coords_pred,
                        batch,
                        cog_proj=True,
                        eta_ddim=eta_ddim,
                    )
                else:
                    # positions
                    pos = self.sde_pos.sample_reverse(
                        t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
                    )
            else:
                pos = self.sde_pos.sample_reverse_ddim(
                    t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
                )

            # atoms
            atom_types = self.cat_atoms.sample_reverse_categorical(
                xt=atom_types,
                x0=atoms_pred,
                t=t[batch],
                num_classes=self.num_atom_types,
            )
            # charges
            charge_types = self.cat_charges.sample_reverse_categorical(
                xt=charge_types,
                x0=charges_pred,
                t=t[batch],
                num_classes=self.num_charge_classes,
            )
            # edges
            if not self.hparams.bond_prediction:
                (
                    edge_attr_global,
                    edge_index_global,
                    mask,
                    mask_i,
                ) = self.cat_bonds.sample_reverse_edges_categorical(
                    edge_attr_global,
                    edges_pred,
                    t,
                    mask,
                    mask_i,
                    batch=batch,
                    edge_index_global=edge_index_global,
                    num_classes=self.num_bond_classes,
                )
            else:
                edge_attr_global = edges_pred

            if self.hparams.bond_model_guidance:
                pos = bond_guidance(
                    pos,
                    node_feats_in,
                    temb,
                    self.bond_model,
                    batch,
                    batch_edge_global,
                    edge_attr_global,
                    edge_index_local,
                    edge_index_global,
                )
            if energy_model is not None and timestep <= guidance_steps:
                signal = alphas[timestep] / (guidance_scale * 10)
                pos = energy_guidance(
                    pos,
                    node_feats_in,
                    temb,
                    energy_model,
                    batch,
                    batch_size=bs,
                    optimization=optimization,
                    guidance_scale=guidance_scale,
                    signal=signal,
                )

            if save_traj:
                atom_decoder = self.dataset_info.atom_decoder
                write_xyz_file_from_batch(
                    pos,
                    atom_types,
                    batch,
                    atom_decoder=atom_decoder,
                    path=os.path.join(save_dir, f"iter_{iteration}"),
                    i=i,
                )
        if relax_sampling:
            return self.relax_sampling(
                pos,
                atom_types,
                charge_types,
                edge_attr_global,
                edge_index_local,
                edge_index_global,
                batch,
                batch_edge_global,
                mask,
                mask_i,
                batch_size=bs,
                context=context,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                save_traj=save_traj,
                save_dir=save_dir,
                i=i,
                iteration=iteration,
                relax_steps=relax_steps,
            )

        out_dict = {
            "coords_pred": pos,
            "atoms_pred": atom_types,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global,
        }
        molecules = get_molecules(
            out_dict,
            batch,
            edge_index_global,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            device=self.device,
            mol_device="cpu",
            context=context,
            while_train=False,
        )

        if save_traj:
            write_trajectory_as_xyz(
                molecules,
                path=os.path.join(save_dir, f"iter_{iteration}"),
                strict=True,
            )

        return molecules

    def relax_sampling(
        self,
        pos,
        atom_types,
        charge_types,
        edge_attr_global,
        edge_index_local,
        edge_index_global,
        batch,
        batch_edge_global,
        mask,
        mask_i,
        batch_size,
        context,
        ddpm,
        eta_ddim,
        save_dir,
        save_traj,
        i,
        iteration,
        relax_steps=10,
    ):
        timestep = 0
        for j in range(relax_steps):
            s = torch.full(
                size=(batch_size,),
                fill_value=timestep,
                dtype=torch.long,
                device=pos.device,
            )
            t = s + 1

            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            node_feats_in = torch.cat([atom_types, charge_types], dim=-1)
            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=pos,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_attr_global=edge_attr_global,
                batch=batch,
                batch_edge_global=batch_edge_global,
                context=context,
            )

            coords_pred = out["coords_pred"].squeeze()
            atoms_pred, charges_pred = out["atoms_pred"].split(
                [self.num_atom_types, self.num_charge_classes], dim=-1
            )
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)

            if ddpm:
                if self.hparams.noise_scheduler == "adaptive":
                    # positions
                    pos = self.sde_pos.sample_reverse_adaptive(
                        s,
                        t,
                        pos,
                        coords_pred,
                        batch,
                        cog_proj=True,
                        eta_ddim=eta_ddim,
                    )
                else:
                    # positions
                    pos = self.sde_pos.sample_reverse(
                        t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
                    )
            else:
                pos = self.sde_pos.sample_reverse_ddim(
                    t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
                )

            # atoms
            atom_types = self.cat_atoms.sample_reverse_categorical(
                xt=atom_types,
                x0=atoms_pred,
                t=t[batch],
                num_classes=self.num_atom_types,
            )
            # charges
            charge_types = self.cat_charges.sample_reverse_categorical(
                xt=charge_types,
                x0=charges_pred,
                t=t[batch],
                num_classes=self.num_charge_classes,
            )
            # edges
            (
                edge_attr_global,
                edge_index_global,
                mask,
                mask_i,
            ) = self.cat_bonds.sample_reverse_edges_categorical(
                edge_attr_global,
                edges_pred,
                t,
                mask,
                mask_i,
                batch=batch,
                edge_index_global=edge_index_global,
                num_classes=self.num_bond_classes,
            )

            if save_traj:
                atom_decoder = self.dataset_info.atom_decoder
                write_xyz_file_from_batch(
                    pos,
                    atom_types,
                    batch,
                    atom_decoder=atom_decoder,
                    path=os.path.join(save_dir, f"iter_{iteration}"),
                    i=i + j,
                )

        out_dict = {
            "coords_pred": pos,
            "atoms_pred": atom_types,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global,
        }

        molecules = get_molecules(
            out_dict,
            batch,
            edge_index_global,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            device=self.device,
            mol_device="cpu",
            context=context,
            while_train=False,
        )

        if save_traj:
            write_trajectory_as_xyz(
                molecules,
                path=os.path.join(save_dir, f"iter_{iteration}"),
                strict=True,
            )

        return molecules

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.hparams["lr"],
                amsgrad=True,
                weight_decay=1.0e-12,
            )
        elif self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams["lr"],
                momentum=0.9,
                nesterov=True,
            )
        if self.hparams["lr_scheduler"] == "reduce_on_plateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=self.hparams["lr_patience"],
                cooldown=self.hparams["lr_cooldown"],
                factor=self.hparams["lr_factor"],
            )
        elif self.hparams["lr_scheduler"] == "cyclic":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams["lr_min"],
                max_lr=self.hparams["lr"],
                mode="exp_range",
                step_size_up=self.hparams["lr_step_size"],
                cycle_momentum=False,
            )
        elif self.hparams["lr_scheduler"] == "one_cyclic":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams["lr"],
                steps_per_epoch=len(self.trainer.datamodule.train_dataset),
                epochs=self.hparams["num_epochs"],
            )
        elif self.hparams["lr_scheduler"] == "cosine_annealing":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams["lr_patience"],
                eta_min=self.hparams["lr_min"],
            )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": self.mol_stab,
            "strict": False,
        }
        return [optimizer], [scheduler]
