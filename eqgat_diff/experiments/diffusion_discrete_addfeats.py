import logging
import os
from datetime import datetime

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from tqdm import tqdm

from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork
from e3moldiffusion.molfeat import get_bond_feature_dims
from experiments.data.abstract_dataset import AbstractDatasetInfos
from experiments.data.distributions import prepare_context
from experiments.diffusion.categorical import CategoricalDiffusionKernel
from experiments.diffusion.continuous import DiscreteDDPM
from experiments.diffusion.utils import bond_guidance, initialize_edge_attrs_reverse
from experiments.losses import DiffusionLoss
from experiments.molecule_utils import Molecule
from experiments.sampling.analyze_strict import analyze_stability_for_molecules
from experiments.utils import (
    coalesce_edges,
    get_list_of_edge_adjs,
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
        is_aromatic_distribution = dataset_info.is_aromatic.float()
        is_ring_distribution = dataset_info.is_in_ring.float()
        hybridization_distribution = dataset_info.hybridization.float()

        self.register_buffer("atoms_prior", atom_types_distribution.clone())
        self.register_buffer("bonds_prior", bond_types_distribution.clone())
        self.register_buffer("charges_prior", charge_types_distribution.clone())
        self.register_buffer("is_aromatic_prior", is_aromatic_distribution.clone())
        self.register_buffer("is_in_ring_prior", is_ring_distribution.clone())
        self.register_buffer("hybridization_prior", hybridization_distribution.clone())

        self.num_is_aromatic = self.num_is_in_ring = 2
        self.num_hybridization = 9

        self.hparams.num_atom_types = dataset_info.input_dims.X
        self.num_charge_classes = dataset_info.input_dims.C
        self.num_atom_types = self.hparams.num_atom_types
        self.num_atom_features = (
            self.num_atom_types
            + self.num_charge_classes
            + self.num_is_aromatic
            + self.num_is_in_ring
            + self.num_hybridization
        )
        self.num_bond_classes = 5
        self.remove_hs = hparams.get("remove_hs")
        if self.remove_hs:
            print("Model without modelling explicit hydrogens")
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
            N=hparams["timesteps"],
            scaled_reverse_posterior_sigma=True,
            schedule=self.hparams.noise_scheduler,
            nu=2.5,
            enforce_zero_terminal_snr=False,
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
            num_bond_types=self.num_bond_classes - 1
            if self.hparams.use_qm_props
            else self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_bonds = CategoricalDiffusionKernel(
            terminal_distribution=bond_types_distribution,
            alphas=self.sde_bonds.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes
            if self.hparams.use_qm_props
            else self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_charges = CategoricalDiffusionKernel(
            terminal_distribution=charge_types_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_atom_types=self.num_atom_types,
            num_bond_types=self.num_bond_classes
            if self.hparams.use_qm_props
            else self.num_bond_classes,
            num_charge_types=self.num_charge_classes,
        )
        self.cat_aromatic = CategoricalDiffusionKernel(
            terminal_distribution=is_aromatic_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_is_aromatic=self.num_is_aromatic,
        )
        self.cat_ring = CategoricalDiffusionKernel(
            terminal_distribution=is_ring_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_is_in_ring=self.num_is_in_ring,
        )
        self.cat_hybridization = CategoricalDiffusionKernel(
            terminal_distribution=hybridization_distribution,
            alphas=self.sde_atom_charge.alphas.clone(),
            num_hybridization=self.num_hybridization,
        )

        self.diffusion_loss = DiffusionLoss(
            modalities=[
                "coords",
                "atoms",
                "charges",
                "bonds",
                "ring",
                "aromatic",
                "hybridization",
            ],
            param=["data"] * 7,
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
                device="cuda" if self.hparams.gpus > 1 else "cpu",
                dataset_info=self.dataset_info,
                ngraphs=1000,
                bs=self.hparams.inference_batch_size,
                verbose=True,
                inner_verbose=False,
                eta_ddim=1.0,
                ddpm=True,
                every_k_step=1,
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
        self,
        loss,
        coords_loss,
        atoms_loss,
        charges_loss,
        bonds_loss,
        ring_loss,
        aromatic_loss,
        hybridization_loss,
        batch_size,
        stage,
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

        self.log(
            f"{stage}/ring_loss",
            ring_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/aromatic_loss",
            aromatic_loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/hybridization_loss",
            hybridization_loss,
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
            weights = self.sde_bonds.snr_s_t_weighting(
                s=t - 1, t=t, clamp_min=None, clamp_max=None
            ).to(batch.x.device)
        elif self.hparams.loss_weighting == "snr_t":
            weights = self.sde_bonds.snr_t_weighting(
                t=t, device=batch.x.device, clamp_min=0.05, clamp_max=5.0
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
            "ring": out_dict["ring_true"],
            "aromatic": out_dict["aromatic_true"],
            "hybridization": out_dict["hybridization_true"],
        }

        coords_pred = out_dict["coords_pred"]
        atoms_pred = out_dict["atoms_pred"]
        (
            atoms_pred,
            charges_pred,
            ring_pred,
            aromatic_pred,
            hybridization_pred,
        ) = atoms_pred.split(
            [
                self.num_atom_types,
                self.num_charge_classes,
                self.num_is_in_ring,
                self.num_is_aromatic,
                self.num_hybridization,
            ],
            dim=-1,
        )
        edges_pred = out_dict["bonds_pred"]

        pred_data = {
            "coords": coords_pred,
            "atoms": atoms_pred,
            "charges": charges_pred,
            "bonds": edges_pred,
            "ring": ring_pred,
            "aromatic": aromatic_pred,
            "hybridization": hybridization_pred,
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
            + 0.5 * loss["ring"]
            + 0.7 * loss["aromatic"]
            + 1.0 * loss["hybridization"]
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
            loss["ring"],
            loss["aromatic"],
            loss["hybridization"],
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

        ring_feat = batch.is_in_ring
        aromatic_feat = batch.is_aromatic
        hybridization_feat = batch.hybridization

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

        ## tempory: use this version without refactored.
        # ring-feat and perturb
        ring_feat, ring_feat_perturbed = self.cat_ring.sample_categorical(
            t,
            ring_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_is_in_ring,
            type="ring",
        )
        (
            aromatic_feat,
            aromatic_feat_perturbed,
        ) = self.cat_aromatic.sample_categorical(
            t,
            aromatic_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_is_aromatic,
            type="aromatic",
        )
        (
            hybridization_feat,
            hybridization_feat_perturbed,
        ) = self.cat_hybridization.sample_categorical(
            t,
            hybridization_feat,
            data_batch,
            self.dataset_info,
            num_classes=self.num_hybridization,
            type="hybridization",
        )

        atom_feats_in_perturbed = torch.cat(
            [
                atom_types_perturbed,
                charges_perturbed,
                ring_feat_perturbed,
                aromatic_feat_perturbed,
                hybridization_feat_perturbed,
            ],
            dim=-1,
        )

        out = self.model(
            x=atom_feats_in_perturbed,
            t=temb,
            pos=pos_perturbed,
            edge_index_local=None,
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_perturbed,
            batch=data_batch,
            batch_edge_global=batch_edge_global,
            context=context,
        )

        out["coords_perturbed"] = pos_perturbed
        out["atoms_perturbed"] = atom_types_perturbed
        out["charges_perturbed"] = charges_perturbed
        out["bonds_perturbed"] = edge_attr_global_perturbed
        out["ring_perturbed"] = ring_feat_perturbed
        out["aromatic_perturbed"] = aromatic_feat_perturbed
        out["hybridization_perturbed"] = hybridization_feat_perturbed

        out["coords_true"] = pos_centered
        out["coords_noise_true"] = noise_coords_true
        out["atoms_true"] = atom_types.argmax(dim=-1)
        out["bonds_true"] = edge_attr_global
        out["charges_true"] = charges.argmax(dim=-1)
        out["ring_true"] = ring_feat.argmax(dim=-1)
        out["aromatic_true"] = aromatic_feat.argmax(dim=-1)
        out["hybridization_true"] = hybridization_feat.argmax(dim=-1)

        out["bond_aggregation_index"] = edge_index_global[1]

        return out

    @torch.no_grad()
    def generate_graphs(
        self,
        num_graphs: int,
        empirical_distribution_num_nodes: torch.Tensor,
        device: torch.device,
        verbose=False,
        save_traj=False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
    ):
        (
            pos,
            atom_types,
            charge_types,
            ring_feat,
            aromatic_feat,
            hybridization_feat,
            edge_types,
            edge_index_global,
            batch_num_nodes,
            trajs,
            context,
        ) = self.reverse_sampling(
            num_graphs=num_graphs,
            device=device,
            empirical_distribution_num_nodes=empirical_distribution_num_nodes,
            verbose=verbose,
            save_traj=save_traj,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            every_k_step=every_k_step,
        )

        pos_splits = pos.detach().split(batch_num_nodes.cpu().tolist(), dim=0)

        charge_types_integer = torch.argmax(charge_types, dim=-1)
        # offset back
        charge_types_integer = charge_types_integer - self.dataset_info.charge_offset
        charge_types_integer_split = charge_types_integer.detach().split(
            batch_num_nodes.cpu().tolist(), dim=0
        )
        atom_types_integer = torch.argmax(atom_types, dim=-1)
        atom_types_integer_split = atom_types_integer.detach().split(
            batch_num_nodes.cpu().tolist(), dim=0
        )
        # additional feats
        # for rdkit molecule <atom> class, we just select the aromatic and hybridization feats:
        # SetIsAromatic and SetHybridization
        aromatic_feat_integer = torch.argmax(aromatic_feat, dim=-1)
        aromatic_feat_integer_split = aromatic_feat_integer.detach().split(
            batch_num_nodes.cpu().tolist(), dim=0
        )

        hybridization_feat_integer = torch.argmax(hybridization_feat, dim=-1)
        hybridization_feat_integer_split = hybridization_feat_integer.detach().split(
            batch_num_nodes.cpu().tolist(), dim=0
        )
        context_split = (
            context.split(batch_num_nodes.cpu().tolist(), dim=0)
            if context is not None
            else None
        )
        return (
            pos_splits,
            atom_types_integer_split,
            charge_types_integer_split,
            aromatic_feat_integer_split,
            hybridization_feat_integer_split,
            edge_types,
            edge_index_global,
            batch_num_nodes,
            trajs,
            context_split,
        )

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
        save_traj: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        b = ngraphs // bs
        l = [bs] * b
        if sum(l) != ngraphs:
            l.append(ngraphs - sum(l))
        assert sum(l) == ngraphs

        molecule_list = []
        start = datetime.now()
        if verbose:
            if self.local_rank == 0:
                print(f"Creating {ngraphs} graphs in {l} batches")
        for _, num_graphs in enumerate(l):
            (
                pos_splits,
                atom_types_integer_split,
                charge_types_integer_split,
                aromatic_feat_integer_split,
                hybridization_feat_integer_split,
                edge_types,
                edge_index_global,
                batch_num_nodes,
                trajs,
                context_split,
            ) = self.generate_graphs(
                num_graphs=num_graphs,
                verbose=inner_verbose,
                device=self.device,
                empirical_distribution_num_nodes=self.empirical_num_nodes,
                save_traj=save_traj,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                every_k_step=every_k_step,
            )

            n = batch_num_nodes.sum().item()
            edge_attrs_dense = torch.zeros(
                size=(n, n, 5), dtype=edge_types.dtype, device=edge_types.device
            )
            edge_attrs_dense[
                edge_index_global[0, :], edge_index_global[1, :], :
            ] = edge_types
            edge_attrs_dense = edge_attrs_dense.argmax(-1)
            edge_attrs_splits = get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes)

            for i, (
                positions,
                atom_types,
                charges,
                is_aromatic,
                hybridization,
                edges,
            ) in enumerate(
                zip(
                    pos_splits,
                    atom_types_integer_split,
                    charge_types_integer_split,
                    aromatic_feat_integer_split,
                    hybridization_feat_integer_split,
                    edge_attrs_splits,
                )
            ):
                molecule = Molecule(
                    atom_types=atom_types.detach().to(device),
                    positions=positions.detach().to(device),
                    charges=charges.detach().to(device),
                    bond_types=edges.detach().to(device),
                    context=context_split[i][0].detach().to(device)
                    if context_split is not None
                    else None,
                    is_aromatic=is_aromatic.detach().to(device),
                    hybridization=hybridization.detach().to(device),
                    dataset_info=dataset_info,
                    build_mol_with_addfeats=self.hparams.build_mol_with_addfeats,
                )
                molecule_list.append(molecule)

        (
            stability_dict,
            validity_dict,
            statistics_dict,
            all_generated_smiles,
            stable_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=molecule_list,
            dataset_info=dataset_info,
            smiles_train=self.smiles_list,
            local_rank=self.local_rank,
            return_molecules=return_molecules,
            device=device,
        )

        if self.mol_stab < stability_dict["mol_stable"] and not run_test_eval:
            self.mol_stab = stability_dict["mol_stable"]
            save_path = os.path.join(self.hparams.save_dir, "best_mol_stab.ckpt")
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
            return total_res, all_generated_smiles, stable_molecules, trajs
        else:
            return total_res

    def reverse_sampling(
        self,
        num_graphs: int,
        empirical_distribution_num_nodes: Tensor,
        device: torch.device,
        verbose: bool = False,
        save_traj: bool = False,
        ddpm: bool = True,
        eta_ddim: float = 1.0,
        every_k_step: int = 1,
    ):
        batch_num_nodes = torch.multinomial(
            input=empirical_distribution_num_nodes,
            num_samples=num_graphs,
            replacement=True,
        ).to(device)
        batch_num_nodes = batch_num_nodes.clamp(min=1)
        batch = torch.arange(num_graphs, device=device).repeat_interleave(
            batch_num_nodes, dim=0
        )
        bs = int(batch.max()) + 1

        # sample context condition
        context = None
        if self.prop_dist is not None:
            context = self.prop_dist.sample_batch(batch_num_nodes).to(self.device)[
                batch
            ]

        # initialiaze the 0-mean point cloud from N(0, I)
        pos = torch.randn(len(batch), 3, device=device, dtype=torch.get_default_dtype())
        pos = zero_mean(pos, batch=batch, dim_size=bs, dim=0)

        n = len(pos)

        # initialize the atom-types
        atom_types = torch.multinomial(
            self.atoms_prior, num_samples=n, replacement=True
        )
        atom_types = F.one_hot(atom_types, self.num_atom_types).float()

        # charges
        charge_types = torch.multinomial(
            self.charges_prior, num_samples=n, replacement=True
        )
        charge_types = F.one_hot(charge_types, self.num_charge_classes).float()

        # ring
        ring_feat = torch.multinomial(
            self.is_in_ring_prior, num_samples=n, replacement=True
        )
        ring_feat = F.one_hot(ring_feat, self.num_is_in_ring).float()

        # aromatic
        aromatic_feat = torch.multinomial(
            self.is_aromatic_prior, num_samples=n, replacement=True
        )
        aromatic_feat = F.one_hot(aromatic_feat, self.num_is_aromatic).float()

        # hybridization
        hybridization_feat = torch.multinomial(
            self.hybridization_prior, num_samples=n, replacement=True
        )
        hybridization_feat = F.one_hot(
            hybridization_feat, self.num_hybridization
        ).float()

        edge_index_local = None

        # edge types for FC graph
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
                edge_index_global, n, self.bonds_prior, self.num_bond_classes, device
            )
        else:
            edge_attr_global = None

        batch_edge_global = batch[edge_index_global[0]]

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

            node_feats_in = torch.cat(
                [
                    atom_types,
                    charge_types,
                    ring_feat,
                    aromatic_feat,
                    hybridization_feat,
                ],
                dim=-1,
            )
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
            if ddpm:
                if self.hparams.noise_scheduler == "adaptive":
                    # positions
                    pos = self.sde_pos.sample_reverse_adaptive(
                        s, t, pos, coords_pred, batch, cog_proj=True, eta_ddim=eta_ddim
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

            # rest
            (
                atoms_pred,
                charges_pred,
                ring_pred,
                aromatic_pred,
                hybridization_pred,
            ) = out["atoms_pred"].split(
                [
                    self.num_atom_types,
                    self.num_charge_classes,
                    self.num_is_in_ring,
                    self.num_is_aromatic,
                    self.num_hybridization,
                ],
                dim=-1,
            )
            atoms_pred = atoms_pred.softmax(dim=-1)
            # N x a_0
            edges_pred = out["bonds_pred"].softmax(dim=-1)
            # E x b_0
            charges_pred = charges_pred.softmax(dim=-1)
            ring_pred = ring_pred.softmax(dim=-1)
            aromatic_pred = aromatic_pred.softmax(dim=-1)
            hybridization_pred = hybridization_pred.softmax(dim=-1)

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

            # additional feats
            ring_feat = self.cat_ring.sample_reverse_categorical(
                xt=ring_feat,
                x0=ring_pred,
                t=t[batch],
                num_classes=self.num_is_in_ring,
            )

            aromatic_feat = self.cat_aromatic.sample_reverse_categorical(
                xt=aromatic_feat,
                x0=aromatic_pred,
                t=t[batch],
                num_classes=self.num_is_aromatic,
            )

            hybridization_feat = self.cat_hybridization.sample_reverse_categorical(
                xt=hybridization_feat,
                x0=hybridization_pred,
                t=t[batch],
                num_classes=self.num_hybridization,
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

            if save_traj:
                pos_traj.append(pos.detach().cpu())
                atom_type_traj.append(atom_types.argmax(-1).detach().cpu())
                edge_type_traj.append(edge_attr_global.detach().cpu())
                charge_type_traj.append(charge_types.detach().cpu())

        return (
            pos,
            atom_types,
            charge_types,
            ring_feat,
            aromatic_feat,
            hybridization_feat,
            edge_attr_global,
            edge_index_global,
            batch_num_nodes,
            [pos_traj, atom_type_traj, charge_type_traj, edge_type_traj],
            context,
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
