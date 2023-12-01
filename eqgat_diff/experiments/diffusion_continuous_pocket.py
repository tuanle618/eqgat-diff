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
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from tqdm import tqdm

from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import ConditionalDistributionNodes, prepare_context
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.utils import (
    bond_guidance,
    energy_guidance,
    get_joint_edge_attrs,
    initialize_edge_attrs_reverse,
)
from experiments.losses import DiffusionLoss
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import (
    coalesce_edges,
    concat_ligand_pocket,
    get_molecules,
    load_energy_model,
    load_model_ligand,
    remove_mean_pocket,
)

logging.getLogger("lightning").setLevel(logging.WARNING)

BOND_FEATURE_DIMS = get_bond_feature_dims()[0]


class Trainer(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
        dataset_info: AbstractDatasetInfos,
        histogram: dict,
        smiles_list: list,
        prop_dist=None,
        prop_norm=None,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.i = 0
        self.validity = 0.0
        self.connected_components = 0.0
        self.qed = 0.0

        self.dataset_info = dataset_info

        self.prop_norm = prop_norm
        self.prop_dist = prop_dist

        atom_types_distribution = dataset_info.atom_types.float()
        bond_types_distribution = dataset_info.edge_types.float()
        charge_types_distribution = dataset_info.charges_marginals.float()

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = self.num_atom_types + self.num_charge_classes
        self.num_bond_classes = self.hparams.num_bond_classes

        self.remove_hs = hparams.get("remove_hs")
        if self.remove_hs:
            print("Model without modelling explicit hydrogens")

        self.smiles_list = smiles_list

        self.conditional_size_distribution = ConditionalDistributionNodes(histogram)

        if self.hparams.load_ckpt_from_pretrained is not None:
            print("Loading from pre-trained model checkpoint...")

            self.model = load_model_ligand(
                self.hparams.load_ckpt_from_pretrained, dataset_info
            )
            # num_params = len(self.model.state_dict())
            # for i, param in enumerate(self.model.parameters()):
            #     if i < num_params // 2:
            #         param.requires_grad = False
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
                use_pos_norm=hparams["use_pos_norm"],
            )

        self.sde_pos = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=2.5,
            enforce_zero_terminal_snr=False,
            T=self.hparams.timesteps,
            clamp_alpha_min=0.05,
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
            param=self.hparams.continuous_param,
        )
        self.sde_bonds = DiscreteDDPM(
            beta_min=hparams["beta_min"],
            beta_max=hparams["beta_max"],
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=1.0,
            enforce_zero_terminal_snr=False,
            param=self.hparams.continuous_param,
        )

        if self.hparams.atoms_categorical:
            self.cat_atoms = CategoricalDiffusionKernel(
                terminal_distribution=atom_types_distribution,
                alphas=self.sde_atom_charge.alphas.clone(),
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
        if self.hparams.bonds_categorical:
            self.cat_bonds = CategoricalDiffusionKernel(
                terminal_distribution=bond_types_distribution,
                alphas=self.sde_bonds.alphas.clone(),
                num_atom_types=self.num_atom_types,
                num_bond_types=self.num_bond_classes,
                num_charge_types=self.num_charge_classes,
            )

        self.diffusion_loss = DiffusionLoss(
            modalities=["coords", "atoms", "charges", "bonds"],
            param=[
                self.hparams.continuous_param,
                self.hparams.continuous_param,
                self.hparams.continuous_param,
                self.hparams.continuous_param,
            ],
        )

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
                verbose=True,
                inner_verbose=False,
                eta_ddim=1.0,
                ddpm=True,
                every_k_step=1,
                device="cuda",
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
            "atoms": out_dict["atoms_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["atoms_noise_true"],
            "charges": out_dict["charges_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["charges_noise_true"],
            "bonds": out_dict["bonds_true"]
            if self.hparams.continuous_param == "data"
            else out_dict["bonds_noise_true"],
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
        atom_types_pocket: Tensor = batch.x_pocket
        pos: Tensor = batch.pos
        pos_pocket: Tensor = batch.pos_pocket
        charges: Tensor = batch.charges
        data_batch: Tensor = batch.batch
        data_batch_pocket: Tensor = batch.pos_pocket_batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        context = batch.context if self.hparams.context_mapping else None
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )
        n = batch.num_nodes

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        pocket_noise = torch.randn_like(pos_pocket) * self.hparams.pocket_noise_std
        pos_pocket = pos_pocket + pocket_noise

        pos_centered, pos_centered_pocket = remove_mean_pocket(
            pos, pos_pocket, data_batch, data_batch_pocket
        )

        ############NOISING############
        # POSITIONS
        # Coords: point cloud in R^3
        # sample noise for coords and recenter
        # SAMPLING
        noise_coords_true, pos_perturbed = self.sde_pos.sample_pos(
            t,
            pos_centered,
            data_batch,
            remove_mean=False,
        )

        # ATOM-TYPES
        if not self.hparams.atoms_categorical:
            atom_types = F.one_hot(atom_types, num_classes=self.num_atom_types).float()
            if self.hparams.continuous_param == "noise":
                atom_types = 0.25 * atom_types

            # sample noise for OHEs in {0, 1}^NUM_CLASSES
            noise_atom_types = torch.randn_like(atom_types)
            mean_ohes, std_ohes = self.sde_atom_charge.marginal_prob(
                x=atom_types, t=t[data_batch]
            )
            # perturb OHEs
            atom_types_perturbed = mean_ohes + std_ohes * noise_atom_types

            # Charges
            charges = self.dataset_info.one_hot_charges(charges).float()
            # sample noise for OHEs in {0, 1}^NUM_CLASSES
            noise_charges = torch.randn_like(charges)
            mean_ohes, std_ohes = self.sde_atom_charge.marginal_prob(
                x=charges, t=t[data_batch]
            )
            # perturb OHEs
            charges_perturbed = mean_ohes + std_ohes * noise_charges
        else:
            atom_types, atom_types_perturbed = self.cat_atoms.sample_categorical(
                t, atom_types, data_batch, self.dataset_info, type="atoms"
            )
            charges, charges_perturbed = self.cat_charges.sample_categorical(
                t, charges, data_batch, self.dataset_info, type="charges"
            )

        # EDGES
        edge_index_global_lig = (
            torch.eq(data_batch.unsqueeze(0), data_batch.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
        edge_index_global_lig = sort_edge_index(
            edge_index_global_lig, sort_by_row=False
        )
        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=data_batch.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )
        if not self.hparams.bonds_categorical:
            # create block diagonal matrix
            dense_edge = torch.zeros(n, n, device=pos.device, dtype=torch.long)
            # populate entries with integer features
            dense_edge[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :]
            ] = edge_attr_global_lig
            dense_edge_ohe = (
                F.one_hot(dense_edge.view(-1, 1), num_classes=self.num_bond_classes)
                .view(n, n, -1)
                .float()
            )

            assert (
                torch.norm(dense_edge_ohe - dense_edge_ohe.permute(1, 0, 2)).item()
                == 0.0
            )

            # create symmetric noise for edge-attributes
            noise_edges = torch.randn_like(dense_edge_ohe)
            noise_edges = 0.5 * (noise_edges + noise_edges.permute(1, 0, 2))
            assert torch.norm(noise_edges - noise_edges.permute(1, 0, 2)).item() == 0.0

            signal = self.sde_bonds.sqrt_alphas_cumprod[t]
            std = self.sde_bonds.sqrt_1m_alphas_cumprod[t]

            signal_b = signal[data_batch].unsqueeze(-1).unsqueeze(-1)
            std_b = std[data_batch].unsqueeze(-1).unsqueeze(-1)
            dense_edge_ohe_perturbed = dense_edge_ohe * signal_b + noise_edges * std_b

            # retrieve as edge-attributes in PyG Format
            edge_attr_global_perturbed_lig = dense_edge_ohe_perturbed[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
            ]
            edge_attr_global_noise = noise_edges[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
            ]
        else:
            edge_attr_global_perturbed_lig = (
                self.cat_bonds.sample_edges_categorical(
                    t,
                    edge_index_global_lig,
                    edge_attr_global_lig,
                    data_batch,
                    return_one_hot=True,
                )
                if not self.hparams.bond_prediction
                else None
            )

        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charges_pocket = torch.zeros(
            pos_pocket.shape[0], charges_perturbed.shape[1], dtype=torch.float32
        ).to(self.device)

        (
            edge_index_global,
            edge_attr_global_perturbed,
            batch_edge_global,
            edge_mask,
        ) = get_joint_edge_attrs(
            pos_perturbed,
            pos_centered_pocket,
            data_batch,
            data_batch_pocket,
            edge_attr_global_perturbed_lig,
            self.num_bond_classes,
            self.device,
        )
        # Concatenate Ligand-Pocket
        (
            pos_perturbed,
            atom_types_perturbed,
            charges_perturbed,
            batch_full,
            pocket_mask,
        ) = concat_ligand_pocket(
            pos_perturbed,
            pos_centered_pocket,
            atom_types_perturbed,
            atom_types_pocket,
            charges_perturbed,
            charges_pocket,
            data_batch,
            data_batch_pocket,
            sorting=False,
        )

        # Concatenate all node features
        atom_feats_in_perturbed = torch.cat(
            [atom_types_perturbed, charges_perturbed], dim=-1
        )

        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_index_global_lig=edge_index_global_lig,
            edge_attr_global=edge_attr_global_perturbed
            if not self.hparams.bond_prediction
            else None,
            batch=batch_full,
            batch_edge_global=batch_edge_global,
            context=context,
            pocket_mask=pocket_mask.unsqueeze(1),
            edge_mask=edge_mask,
            batch_lig=data_batch,
        )

        out["coords_perturbed"] = pos_perturbed
        out["atoms_perturbed"] = atom_types_perturbed
        out["charges_perturbed"] = charges_perturbed
        out["bonds_perturbed"] = edge_attr_global_perturbed

        out["coords_noise_true"] = noise_coords_true
        if not self.hparams.atoms_categorical:
            out["atoms_noise_true"] = noise_atom_types
            out["charges_noise_true"] = noise_charges
        if not self.hparams.bonds_categorical:
            out["bonds_noise_true"] = edge_attr_global_noise

        out["coords_true"] = pos_centered
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global_lig
        out["charges_true"] = charges.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global_lig[1]

        return out

    @torch.no_grad()
    def run_evaluation(
        self,
        step: int,
        dataset_info,
        save_dir: str = None,
        return_molecules: bool = False,
        verbose: bool = False,
        inner_verbose=False,
        save_traj=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        use_ligand_dataset_sizes: bool = False,
        run_test_eval: bool = False,
        guidance_scale: float = 1.0e-4,
        use_energy_guidance: bool = False,
        ckpt_energy_model: str = None,
        device: str = "cpu",
    ):
        """
        Runs the evaluation on the entire validation dataloader. Generates 1 ligand in 1 receptor structure
        """
        energy_model = None
        if use_energy_guidance:
            energy_model = load_energy_model(ckpt_energy_model, self.num_atom_features)
            # for param in self.energy_model.parameters():
            #    param.requires_grad = False
            energy_model.to(self.device)
            energy_model.eval()

        dataloader = (
            self.trainer.datamodule.val_dataloader()
            if not run_test_eval
            else self.trainer.datamodule.test_dataloader()
        )
        molecule_list = []
        start = datetime.now()
        for _, pocket_data in enumerate(dataloader):
            num_graphs = len(pocket_data.batch.bincount())
            if use_ligand_dataset_sizes:
                num_nodes_lig = pocket_data.batch.bincount().to(self.device)
            else:
                num_nodes_lig = self.conditional_size_distribution.sample_conditional(
                    n1=None, n2=pocket_data.pos_pocket_batch.bincount()
                ).to(self.device)
            (
                out_dict,
                data_batch,
                data_batch_pocket,
                edge_index_global,
                trajs,
                context,
            ) = self.reverse_sampling(
                num_graphs=num_graphs,
                num_nodes_lig=num_nodes_lig,
                pocket_data=pocket_data,
                verbose=inner_verbose,
                save_traj=save_traj,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
                guidance_scale=guidance_scale,
                energy_model=energy_model,
            )

            molecule_list.extend(
                get_molecules(
                    out_dict,
                    data_batch,
                    edge_index_global,
                    self.num_atom_types,
                    self.num_charge_classes,
                    self.dataset_info,
                    data_batch_pocket=data_batch_pocket,
                    device=self.device,
                    mol_device=device,
                    context=context,
                    while_train=False,
                )
            )
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
            remove_hs=self.hparams.remove_hs,
            device=device,
        )

        if not run_test_eval:
            save_cond = (
                self.validity < validity_dict["validity"]
                and self.connected_components < statistics_dict["connected_components"]
            ) or (
                self.validity <= validity_dict["validity"]
                and self.connected_components <= statistics_dict["connected_components"]
                and self.qed < statistics_dict["QED"]
            )
        else:
            save_cond = False
        if save_cond:
            self.validity = validity_dict["validity"]
            self.connected_components = statistics_dict["connected_components"]
            self.qed = statistics_dict["QED"]
            save_path = os.path.join(self.hparams.save_dir, "best_valid.ckpt")
            self.trainer.save_checkpoint(save_path)

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
            return total_res, all_generated_smiles, valid_molecules
        else:
            return total_res

    @torch.no_grad()
    def generate_ligands(
        self,
        pocket_data,
        num_graphs,
        inner_verbose,
        save_traj,
        ddpm,
        eta_ddim,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=False,
        every_k_step=1,
        fix_n_nodes=False,
        mol_device="cpu",
    ):
        if fix_n_nodes:
            num_nodes_lig = pocket_data.batch.bincount().to(self.device)
        else:
            pocket_size = pocket_data.pos_pocket_batch.bincount()[0].unsqueeze(0)
            num_nodes_lig = (
                self.conditional_size_distribution.sample_conditional(
                    n1=None, n2=pocket_size
                )
                .repeat(num_graphs)
                .to(self.device)
            )
        (
            out_dict,
            data_batch,
            data_batch_pocket,
            edge_index_global,
            trajs,
            context,
        ) = self.reverse_sampling(
            num_graphs=num_graphs,
            num_nodes_lig=num_nodes_lig,
            pocket_data=pocket_data,
            verbose=inner_verbose,
            save_traj=save_traj,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            every_k_step=every_k_step,
            guidance_scale=None,
            energy_model=None,
        )
        molecules = get_molecules(
            out_dict,
            data_batch,
            edge_index_global,
            self.num_atom_types,
            self.num_charge_classes,
            self.dataset_info,
            data_batch_pocket=data_batch_pocket,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            device=self.device,
            mol_device=mol_device,
            context=context,
            while_train=False,
        )

        return molecules

    def reverse_sampling(
        self,
        num_graphs: int,
        pocket_data: Tensor,
        num_nodes_lig: int = None,
        verbose: bool = False,
        save_traj: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
        guidance_scale: float = 1.0e-4,
        energy_model=None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List]:
        pos_pocket = pocket_data.pos_pocket.to(self.device)
        batch_pocket = pocket_data.pos_pocket_batch.to(self.device)
        x_pocket = pocket_data.x_pocket.to(self.device)

        batch = torch.arange(num_graphs, device=self.device).repeat_interleave(
            num_nodes_lig, dim=0
        )
        bs = int(batch.max()) + 1

        # sample context condition
        context = None
        if self.prop_dist is not None:
            context = self.prop_dist.sample_batch(num_nodes_lig).to(self.device)[batch]

        # initialize the 0-mean point cloud from N(0, I) centered in the pocket
        pocket_cog = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pocket_cog_batch = pocket_cog[batch]
        pos = pocket_cog_batch + torch.randn_like(pocket_cog_batch)
        # pos = pocket_data.pos.to(self.device)
        # batch = pocket_data.batch.to(self.device)

        # # project to COM-free subspace
        pos, pos_pocket = remove_mean_pocket(pos, pos_pocket, batch, batch_pocket)

        n = len(pos)

        # initialize the atom-types
        if not self.hparams.atoms_categorical:
            atom_types = torch.randn(
                pos.size(0), self.num_atom_types, device=self.device
            )

            # initialize the charges
            charge_types = torch.randn(
                pos.size(0), self.num_charge_classes, device=self.device
            )
        else:
            atom_types = torch.multinomial(
                self.atoms_prior, num_samples=n, replacement=True
            )
            atom_types = F.one_hot(atom_types, self.num_atom_types).float()

            charge_types = torch.multinomial(
                self.charges_prior, num_samples=n, replacement=True
            )
            charge_types = F.one_hot(charge_types, self.num_charge_classes).float()

        atom_types_pocket = F.one_hot(
            x_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        charges_pocket = torch.zeros(
            pos_pocket.shape[0], charge_types.shape[1], dtype=torch.float32
        ).to(self.device)

        edge_index_local = None
        if not self.hparams.bonds_categorical:
            edge_index_global_lig = (
                torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            # sample symmetric edge-attributes
            edge_attrs = torch.randn(
                (
                    edge_index_global_lig.size(0),
                    edge_index_global_lig.size(1),
                    self.num_bond_classes,
                ),
                device=self.device,
                dtype=torch.get_default_dtype(),
            )
            # symmetrize
            edge_attrs = 0.5 * (edge_attrs + edge_attrs.permute(1, 0, 2))
            assert torch.norm(edge_attrs - edge_attrs.permute(1, 0, 2)).item() == 0.0
            # get COO format (2, E)
            edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
            edge_index_global_lig = sort_edge_index(
                edge_index_global_lig, sort_by_row=False
            )
            # select in PyG formt (E, self.hparams.num_bond_types)
            edge_attr_global_lig = edge_attrs[
                edge_index_global_lig[0, :], edge_index_global_lig[1, :], :
            ]
        else:
            edge_index_global = (
                torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
            if not self.hparams.bond_prediction:
                (
                    edge_attr_global_lig,
                    edge_index_global_lig,
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

        batch_edge_lig = batch[edge_index_global_lig[0]]

        (
            edge_index_global,
            edge_attr_global,
            batch_edge_global,
            edge_mask,
        ) = get_joint_edge_attrs(
            pos,
            pos_pocket,
            batch,
            batch_pocket,
            edge_attr_global_lig,
            self.num_bond_classes,
            self.device,
        )
        (
            pos_joint,
            atom_types_joint,
            charge_types_joint,
            batch_full,
            pocket_mask,
        ) = concat_ligand_pocket(
            pos,
            pos_pocket,
            atom_types,
            atom_types_pocket,
            charge_types,
            charges_pocket,
            batch,
            batch_pocket,
            sorting=False,
        )

        pos_traj = []
        atom_type_traj = []
        charge_type_traj = []
        edge_type_traj = []

        if self.hparams.continuous_param == "data":
            chain = range(0, self.hparams.timesteps)
        elif self.hparams.continuous_param == "noise":
            chain = range(0, self.hparams.timesteps - 1)

        chain = chain[::every_k_step]

        iterator = (
            tqdm(reversed(chain), total=len(chain)) if verbose else reversed(chain)
        )
        for timestep in iterator:
            s = torch.full(
                size=(bs,), fill_value=timestep, dtype=torch.long, device=pos.device
            )
            t = s + 1
            temb = t / self.hparams.timesteps
            temb = temb.unsqueeze(dim=1)

            node_feats_in = torch.cat([atom_types_joint, charge_types_joint], dim=-1)
            out = self.model(
                x=node_feats_in,
                t=temb,
                pos=pos_joint,
                edge_index_local=edge_index_local,
                edge_index_global=edge_index_global,
                edge_index_global_lig=edge_index_global_lig,
                edge_attr_global=edge_attr_global,
                batch=batch_full,
                batch_edge_global=batch_edge_global,
                context=context,
                pocket_mask=pocket_mask.unsqueeze(1),
                edge_mask=edge_mask,
                batch_lig=batch,
            )
            coords_pred = out["coords_pred"].squeeze()
            atoms_pred, charges_pred = out["atoms_pred"].split(
                [self.num_atom_types, self.num_charge_classes], dim=-1
            )
            edges_pred = out["bonds_pred"]

            if self.hparams.continuous_param == "data":
                atoms_pred = atoms_pred.softmax(dim=-1)
                # N x a_0
                edges_pred = edges_pred.softmax(dim=-1)
                # E x b_0
                charges_pred = charges_pred.softmax(dim=-1)

            if self.hparams.noise_scheduler == "adaptive":
                pos = self.sde_pos.sample_reverse_adaptive(
                    s,
                    t,
                    pos,
                    coords_pred,
                    batch,
                    cog_proj=True,
                )
                if not self.hparams.atoms_categorical:
                    atom_types = self.sde_atom_charge.sample_reverse_adaptive(
                        s,
                        t,
                        atom_types,
                        atoms_pred,
                        batch,
                    )
                    charge_types = self.sde_atom_charge.sample_reverse_adaptive(
                        s,
                        t,
                        charge_types,
                        charges_pred,
                        batch,
                    )

                if not self.hparams.bonds_categorical:
                    edge_attr_global_lig = self.sde_bonds.sample_reverse_adaptive(
                        s,
                        t,
                        edge_attr_global_lig,
                        edges_pred,
                        batch_edge_lig,
                        edge_attrs=edge_attrs,
                        edge_index_global=edge_index_global_lig,
                    )
            else:
                pos = self.sde_pos.sample_reverse(
                    t,
                    pos,
                    coords_pred,
                    batch,
                    cog_proj=True,
                )
                if not self.hparams.atoms_categorical:
                    atom_types = self.sde_atom_charge.sample_reverse(
                        t,
                        atom_types,
                        atoms_pred,
                        batch,
                    )
                    charge_types = self.sde_atom_charge.sample_reverse(
                        t,
                        charge_types,
                        charges_pred,
                        batch,
                    )
                if not self.hparams.bonds_categorical:
                    edge_attr_global_lig = self.sde_bonds.sample_reverse(
                        t,
                        edge_attr_global_lig,
                        edges_pred,
                        batch_edge_lig,
                        edge_attrs=edge_attrs,
                        edge_index_global=edge_index_global_lig,
                    )
            if self.hparams.atoms_categorical:
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
            if self.hparams.bonds_categorical:
                # edges
                if not self.hparams.bond_prediction:
                    (
                        edge_attr_global_lig,
                        edge_index_global_lig,
                        mask,
                        mask_i,
                    ) = self.cat_bonds.sample_reverse_edges_categorical(
                        edge_attr_global_lig,
                        edges_pred,
                        t,
                        mask,
                        mask_i,
                        batch=batch,
                        edge_index_global=edge_index_global_lig,
                        num_classes=self.num_bond_classes,
                    )
                else:
                    edge_attr_global_lig = edges_pred

            (
                edge_index_global,
                edge_attr_global,
                batch_edge_global,
                edge_mask,
            ) = get_joint_edge_attrs(
                pos,
                pos_pocket,
                batch,
                batch_pocket,
                edge_attr_global_lig,
                self.num_bond_classes,
                self.device,
            )
            # pos, pos_pocket = remove_mean_pocket(pos, pos_pocket, batch, batch_pocket)
            (
                pos_joint,
                atom_types_joint,
                charge_types_joint,
                batch_full,
                pocket_mask,
            ) = concat_ligand_pocket(
                pos,
                pos_pocket,
                atom_types,
                atom_types_pocket,
                charge_types,
                charges_pocket,
                batch,
                batch_pocket,
                sorting=False,
            )

            if save_traj:
                pos_traj.append(pos.detach())
                atom_type_traj.append(atom_types.detach())
                edge_type_traj.append(edge_attr_global.detach())
                charge_type_traj.append(charge_types.detach())

        pos += pocket_cog_batch

        out_dict = {
            "coords_pred": pos,
            "coords_pocket": pos_pocket,
            "atoms_pred": atom_types,
            "atoms_pocket": atom_types_pocket,
            "charges_pred": charge_types,
            "bonds_pred": edge_attr_global_lig,
        }
        return (
            out_dict,
            batch,
            batch_pocket,
            edge_index_global_lig,
            [pos_traj, atom_type_traj, charge_type_traj, edge_type_traj],
            context,
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            amsgrad=True,
            weight_decay=1.0e-12,
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
            "monitor": "val/coords_loss_epoch",
            "strict": False,
        }
        return [optimizer], [scheduler]
