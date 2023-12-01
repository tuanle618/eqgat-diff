from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_mean, scatter_add

from e3moldiffusion.gnn import EQGATEdgeGNN, EQGATLocalGNN, EQGATEnergyGNN
from e3moldiffusion.modules import (
    DenseLayer,
    GatedEquivariantBlock,
    GatedEquivBlock,
    SE3Norm,
)


class PredictionHeadEdge(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        edge_dim: int,
        num_atom_features: int,
        num_bond_types: int = 5,
        coords_param: str = "data",
    ) -> None:
        super(PredictionHeadEdge, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_features = num_atom_features

        self.shared_mapping = DenseLayer(
            self.sdim, self.sdim, bias=True, activation=nn.SiLU()
        )

        self.bond_mapping = DenseLayer(edge_dim, self.sdim, bias=True)

        self.bonds_lin_0 = DenseLayer(
            in_features=self.sdim + 1, out_features=self.sdim, bias=True
        )
        self.bonds_lin_1 = DenseLayer(
            in_features=self.sdim, out_features=num_bond_types, bias=True
        )
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=1, bias=False)
        self.atoms_lin = DenseLayer(
            in_features=self.sdim, out_features=num_atom_features, bias=True
        )

        self.coords_param = coords_param

        self.reset_parameters()

    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.atoms_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()

    def forward(
        self,
        x: Dict,
        batch: Tensor,
        edge_index_global: Tensor,
        edge_index_global_lig: Tensor = None,
        batch_lig: Tensor = None,
        pocket_mask: Tensor = None,
        edge_mask: Tensor = None,
    ) -> Dict:
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]
        s = self.shared_mapping(s)

        coords_pred = self.coords_lin(v).squeeze()
        atoms_pred = self.atoms_lin(s)

        if batch_lig is not None and pocket_mask is not None:
            s = (s * pocket_mask)[pocket_mask.squeeze(), :]
            j, i = edge_index_global_lig
            atoms_pred = (atoms_pred * pocket_mask)[pocket_mask.squeeze(), :]
            coords_pred = (coords_pred * pocket_mask)[pocket_mask.squeeze(), :]
            p = p[pocket_mask.squeeze(), :]
            # coords_pred = (
            #    coords_pred
            #    - scatter_mean(coords_pred, index=batch_lig, dim=0)[batch_lig]
            # )
            d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)

        elif self.coords_param == "data":
            j, i = edge_index_global
            n = s.size(0)
            coords_pred = p + coords_pred
            coords_pred = (
                coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]
            )
            d = (
                (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)
            )  # .sqrt()
        else:
            j, i = edge_index_global
            n = s.size(0)
            d = (p[i] - p[j]).pow(2).sum(-1, keepdim=True)  # .sqrt()
            coords_pred = (
                coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]
            )

        if edge_mask is not None and edge_index_global_lig is not None:
            n = len(batch_lig)
            e = (e * edge_mask.unsqueeze(1))[edge_mask]
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index_global_lig[0], edge_index_global_lig[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index_global_lig[0], edge_index_global_lig[1], :]
        else:
            e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
            e_dense[edge_index_global[0], edge_index_global[1], :] = e
            e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
            e = e_dense[edge_index_global[0], edge_index_global[1], :]

        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)

        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)

        out = {
            "coords_pred": coords_pred,
            "atoms_pred": atoms_pred,
            "bonds_pred": bonds_pred,
        }

        return out


class PropertyPredictionHead(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        num_context_features: int,
    ) -> None:
        super(PropertyPredictionHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_context_features = num_context_features

        self.scalar_mapping = DenseLayer(
            self.sdim, self.sdim, bias=True, activation=nn.SiLU()
        )
        self.vector_mapping = DenseLayer(
            in_features=self.vdim, out_features=self.sdim, bias=False
        )

        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    self.sdim,
                    self.sdim // 2,
                    activation="silu",
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    self.sdim // 2, num_context_features, activation="silu"
                ),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.scalar_mapping.reset_parameters()
        self.vector_mapping.reset_parameters()

    def forward(self, x: Dict, batch: Tensor, edge_index_global: Tensor) -> Dict:
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]

        s = self.scalar_mapping(s)
        v = self.vector_mapping(v) + p.sum() * 0 + e.sum() * 0
        for layer in self.output_network:
            s, v = layer(s, v)
        # include v in output to make sure all parameters have a gradient
        out = s + v.sum() * 0

        return out


class DenoisingEdgeNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction, i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        fully_connected: bool = True,
        local_global_model: bool = False,
        recompute_radius_graph: bool = True,
        recompute_edge_attributes: bool = True,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
        context_mapping: bool = False,
        num_context_features: int = 0,
        property_prediction: bool = False,
        bond_prediction: bool = False,
        coords_param: str = "data",
        ligand_pocket_interaction: bool = False,
    ) -> None:
        super(DenoisingEdgeNetwork, self).__init__()

        self.property_prediction = property_prediction
        self.bond_prediction = bond_prediction
        self.num_bond_types = num_bond_types

        self.ligand_pocket_interaction = ligand_pocket_interaction

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim)

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping or bond_prediction:
            if bond_prediction:
                num_bond_types = 1 * num_atom_features + 1
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)

        self.context_mapping = context_mapping
        if self.context_mapping:
            self.context_mapping = DenseLayer(num_context_features, hn_dim[0])
            self.atom_context_mapping = DenseLayer(hn_dim[0], hn_dim[0])

        assert fully_connected or local_global_model

        self.sdim, self.vdim = hn_dim

        self.local_global_model = local_global_model
        self.fully_connected = fully_connected

        assert fully_connected
        assert not local_global_model

        if latent_dim:
            if context_mapping:
                latent_dim_ = None
            else:
                latent_dim_ = latent_dim
        else:
            latent_dim_ = None

        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            edge_dim=edge_dim,
            latent_dim=latent_dim_,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            local_global_model=local_global_model,
            recompute_radius_graph=recompute_radius_graph,
            recompute_edge_attributes=recompute_edge_attributes,
            edge_mp=edge_mp,
            p1=p1,
            use_pos_norm=use_pos_norm,
            ligand_pocket_interaction=ligand_pocket_interaction,
        )

        if self.property_prediction:
            self.prediction_head = PropertyPredictionHead(
                hn_dim=hn_dim,
                num_context_features=num_context_features,
            )
        else:
            self.prediction_head = PredictionHeadEdge(
                hn_dim=hn_dim,
                edge_dim=edge_dim,
                num_atom_features=num_atom_features,
                num_bond_types=num_bond_types,
                coords_param=coords_param,
            )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.time_mapping_atom.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        if self.context_mapping and hasattr(
            self.atom_context_mapping, "reset_parameters"
        ):
            self.atom_context_mapping.reset_parameters()
        if self.context_mapping and hasattr(self.context_mapping, "reset_parameters"):
            self.context_mapping.reset_parameters()
        self.time_mapping_bond.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()
        if not self.property_prediction:
            self.prediction_head.reset_parameters()

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: OptTensor,
        pos: Tensor,
        sqrt: bool = True,
        batch: Tensor = None,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        if self.ligand_pocket_interaction:
            mask = source != target
            pos[mask] = pos[mask] / torch.norm(pos[mask], dim=1).unsqueeze(1)
            a = pos[target] * pos[source]
        else:
            a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_index_global_lig: OptTensor = None,
        edge_attr_global: OptTensor = None,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None,
        z: OptTensor = None,
        context: OptTensor = None,
        batch_lig: OptTensor = None,
        pocket_mask: OptTensor = None,
        edge_mask: OptTensor = None,
    ) -> Dict:
        if pocket_mask is None:
            pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        # t: (batch_size,)
        ta = self.time_mapping_atom(t)
        tb = self.time_mapping_bond(t)
        tnode = ta[batch]

        # edge_index_global (2, E*)
        tedge_global = tb[batch_edge_global]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)
        cemb = None
        if context is not None and self.context_mapping:
            cemb = self.context_mapping(context)
            s = self.atom_context_mapping(s + cemb)
        s = self.atom_time_mapping(s + tnode)

        if self.bond_prediction:
            # symmetric initial edge-feature
            d = (
                (pos[edge_index_global[1]] - pos[edge_index_global[0]])
                .pow(2)
                .sum(-1, keepdim=True)
                .sqrt()
            )
            edge_attr_global = torch.concat(
                [x[edge_index_global[1]] + x[edge_index_global[0]], d], dim=-1
            )
        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(
            edge_attr_global_transformed + tedge_global
        )

        # edge_dense = torch.zeros(x.size(0), x.size(0), edge_attr_global_transformed.size(-1), device=s.device)
        # edge_dense[edge_index_global[0], edge_index_global[1], :] = edge_attr_global_transformed

        # if not self.fully_connected:
        #    edge_attr_local_transformed = edge_dense[edge_index_local[0], edge_index_local[1], :]
        #    # local
        #    edge_attr_local_transformed = self.calculate_edge_attrs(edge_index=edge_index_local, edge_attr=edge_attr_local_transformed, pos=pos)
        # else:
        #    edge_attr_local_transformed = (None, None, None)

        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_transformed,
            pos=pos,
            sqrt=True,
            batch=batch if self.ligand_pocket_interaction else None,
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            z=z,
            edge_index_local=None,
            edge_attr_local=(None, None, None),
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_transformed,
            batch=batch,
            context=cemb,
            batch_lig=batch_lig,
            pocket_mask=pocket_mask,
        )

        out = self.prediction_head(
            x=out,
            batch=batch,
            edge_index_global=edge_index_global,
            edge_index_global_lig=edge_index_global_lig,
            batch_lig=batch_lig,
            pocket_mask=pocket_mask,
            edge_mask=edge_mask,
        )

        # out['coords_perturbed'] = pos
        # out['atoms_perturbed'] = x
        # out['bonds_perturbed'] = edge_attr_global

        return out


class LatentEncoderNetwork(nn.Module):
    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        intermediate_outs: bool = False,
    ) -> None:
        super(LatentEncoderNetwork, self).__init__()

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping:
            self.bond_mapping = DenseLayer(num_bond_types, edge_dim)
        else:
            self.bond_mapping = nn.Identity()

        self.sdim, self.vdim = hn_dim

        self.gnn = EQGATLocalGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            edge_dim=edge_dim,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            intermediate_outs=intermediate_outs,
        )

        self.intermediate_outs = intermediate_outs

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.gnn.reset_parameters()

    def calculate_edge_attrs(
        self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor, sqrt: bool = True
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (d.unsqueeze(-1) + 1.0))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_attr_local: OptTensor = Tensor,
        batch: OptTensor = None,
    ) -> Dict:
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)

        edge_attr_local_transformed = self.bond_mapping(edge_attr_local)
        edge_attr_local_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_local,
            edge_attr=edge_attr_local_transformed,
            pos=pos,
            sqrt=True,
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        if not self.intermediate_outs:
            out = self.gnn(
                s=s,
                v=v,
                p=pos,
                edge_index_local=edge_index_local,
                edge_attr_local=edge_attr_local_transformed,
                edge_index_global=None,
                edge_attr_global=None,
                batch=batch,
            )
            return out
        else:
            out, scalars = self.gnn(
                s=s,
                v=v,
                p=pos,
                edge_index_local=edge_index_local,
                edge_attr_local=edge_attr_local_transformed,
                edge_index_global=None,
                edge_attr_global=None,
                batch=batch,
            )
            return out, scalars


class SoftMaxAttentionAggregation(nn.Module):
    """
    Softmax Attention Pooling as proposed "Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>
    """

    def __init__(self, dim: int):
        super(SoftMaxAttentionAggregation, self).__init__()

        self.node_net = nn.Sequential(
            DenseLayer(dim, dim, activation=nn.SiLU()), DenseLayer(dim, dim)
        )
        self.gate_net = nn.Sequential(
            DenseLayer(dim, dim, activation=nn.SiLU()), DenseLayer(dim, 1)
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.node_net)
        reset(self.gate_net)

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        if index is None:
            index = torch.zeros(size=(x.size(0)), device=x.device, dtype=torch.long)
        if dim_size is None:
            dim_size = int(index.max()) + 1
        gate = self.gate_net(x)

        gate = softmax(gate, index, dim=0)
        x = self.node_net(x)
        x = gate * x
        x = scatter_add(src=x, index=index, dim=dim, dim_size=dim_size)
        return x


class EdgePredictionHead(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int],
        edge_dim: int,
        num_atom_features: int,
        num_bond_types: int = 5,
    ) -> None:
        super(EdgePredictionHead, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.num_atom_features = num_atom_features

        self.shared_mapping = DenseLayer(
            self.sdim, self.sdim, bias=True, activation=nn.SiLU()
        )

        self.bond_mapping = DenseLayer(edge_dim, self.sdim, bias=True)

        self.bonds_lin_0 = DenseLayer(
            in_features=self.sdim + 1, out_features=self.sdim, bias=True
        )
        self.bonds_lin_1 = DenseLayer(
            in_features=self.sdim, out_features=num_bond_types, bias=True
        )
        self.coords_lin = DenseLayer(in_features=self.vdim, out_features=1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.shared_mapping.reset_parameters()
        self.coords_lin.reset_parameters()
        self.bonds_lin_0.reset_parameters()
        self.bonds_lin_1.reset_parameters()

    def forward(self, x: Dict, batch: Tensor, edge_index_global: Tensor) -> Dict:
        s, v, p, e = x["s"], x["v"], x["p"], x["e"]
        s = self.shared_mapping(s)
        j, i = edge_index_global
        n = s.size(0)

        coords_pred = self.coords_lin(v).squeeze()

        coords_pred = p + coords_pred
        coords_pred = coords_pred - scatter_mean(coords_pred, index=batch, dim=0)[batch]

        e_dense = torch.zeros(n, n, e.size(-1), device=e.device)
        e_dense[edge_index_global[0], edge_index_global[1], :] = e
        e_dense = 0.5 * (e_dense + e_dense.permute(1, 0, 2))
        e = e_dense[edge_index_global[0], edge_index_global[1], :]

        d = (coords_pred[i] - coords_pred[j]).pow(2).sum(-1, keepdim=True)  # .sqrt()
        f = s[i] + s[j] + self.bond_mapping(e)
        edge = torch.cat([f, d], dim=-1)

        bonds_pred = F.silu(self.bonds_lin_0(edge))
        bonds_pred = self.bonds_lin_1(bonds_pred)

        return bonds_pred


class EdgePredictionNetwork(nn.Module):
    """_summary_
    Denoising network that inputs:
        atom features, edge features, position features
    The network is tasked for data prediction, i.e. x0 parameterization as commonly known in the literature:
        atom features, edge features, position features
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_atom_features: int,
        num_bond_types: int = 5,
        hn_dim: Tuple[int, int] = (256, 64),
        edge_dim: int = 32,
        cutoff_local: float = 7.5,
        num_layers: int = 5,
        latent_dim: Optional[int] = None,
        use_cross_product: bool = False,
        fully_connected: bool = True,
        local_global_model: bool = False,
        recompute_radius_graph: bool = True,
        recompute_edge_attributes: bool = True,
        vector_aggr: str = "mean",
        atom_mapping: bool = True,
        bond_mapping: bool = True,
        edge_mp: bool = False,
        p1: bool = True,
        use_pos_norm: bool = True,
    ) -> None:
        super(EdgePredictionNetwork, self).__init__()

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.time_mapping_bond = DenseLayer(1, edge_dim)

        if atom_mapping:
            self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        else:
            self.atom_mapping = nn.Identity()

        if bond_mapping:
            self.bond_mapping = DenseLayer(
                num_atom_features, edge_dim
            )  # BOND PREDICTION
        else:
            self.bond_mapping = nn.Identity()

        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])
        self.bond_time_mapping = DenseLayer(edge_dim, edge_dim)

        assert fully_connected or local_global_model

        self.sdim, self.vdim = hn_dim

        self.local_global_model = local_global_model
        self.fully_connected = fully_connected

        assert fully_connected
        assert not local_global_model

        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff_local=cutoff_local,
            edge_dim=edge_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            fully_connected=fully_connected,
            local_global_model=local_global_model,
            recompute_radius_graph=recompute_radius_graph,
            recompute_edge_attributes=recompute_edge_attributes,
            edge_mp=edge_mp,
            p1=p1,
            use_pos_norm=use_pos_norm,
        )

        self.prediction_head = EdgePredictionHead(
            hn_dim=hn_dim,
            edge_dim=edge_dim,
            num_atom_features=num_atom_features,
            num_bond_types=num_bond_types,
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.atom_mapping, "reset_parameters"):
            self.atom_mapping.reset_parameters()
        if hasattr(self.bond_mapping, "reset_parameters"):
            self.bond_mapping.reset_parameters()
        self.time_mapping_atom.reset_parameters()
        self.atom_time_mapping.reset_parameters()
        self.time_mapping_bond.reset_parameters()
        self.bond_time_mapping.reset_parameters()
        self.gnn.reset_parameters()
        self.prediction_head.reset_parameters()

    def calculate_edge_attrs(
        self, edge_index: Tensor, edge_attr: OptTensor, pos: Tensor, sqrt: bool = True
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        if sqrt:
            d = d.sqrt()
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        edge_index_local: Tensor,
        edge_index_global: Tensor,
        edge_attr_global: OptTensor = Tensor,
        batch: OptTensor = None,
        batch_edge_global: OptTensor = None,
        z: OptTensor = None,
    ) -> Dict:
        pos = pos - scatter_mean(pos, index=batch, dim=0)[batch]
        # t: (batch_size,)
        ta = self.time_mapping_atom(t)
        tb = self.time_mapping_bond(t)
        tnode = ta[batch]

        # edge_index_global (2, E*)
        tedge_global = tb[batch_edge_global]

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long)

        s = self.atom_mapping(x)
        s = self.atom_time_mapping(s + tnode)

        edge_attr_global = x[edge_index_global[1]]

        edge_attr_global_transformed = self.bond_mapping(edge_attr_global)
        edge_attr_global_transformed = self.bond_time_mapping(
            edge_attr_global_transformed + tedge_global
        )

        # global
        edge_attr_global_transformed = self.calculate_edge_attrs(
            edge_index=edge_index_global,
            edge_attr=edge_attr_global_transformed,
            pos=pos,
            sqrt=True,
        )

        v = torch.zeros(size=(x.size(0), 3, self.vdim), device=s.device)

        out = self.gnn(
            s=s,
            v=v,
            p=pos,
            z=z,
            edge_index_local=None,
            edge_attr_local=(None, None, None),
            edge_index_global=edge_index_global,
            edge_attr_global=edge_attr_global_transformed,
            batch=batch,
        )

        out = self.prediction_head(
            x=out, batch=batch, edge_index_global=edge_index_global
        )

        return out


class EQGATEnergyNetwork(nn.Module):
    def __init__(
        self,
        num_atom_features: int,
        hn_dim: Tuple[int, int] = (256, 64),
        num_rbfs: int = 20,
        cutoff_local: float = 5.0,
        num_layers: int = 5,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
    ) -> None:
        super().__init__()
        self.cutoff = cutoff_local
        self.sdim, self.vdim = hn_dim

        self.time_mapping_atom = DenseLayer(1, hn_dim[0])
        self.atom_mapping = DenseLayer(num_atom_features, hn_dim[0])
        self.atom_time_mapping = DenseLayer(hn_dim[0], hn_dim[0])

        self.gnn = EQGATEnergyGNN(
            hn_dim=hn_dim,
            cutoff=cutoff_local,
            num_rbfs=num_rbfs,
            num_layers=num_layers,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
        )
        self.energy_head = GatedEquivBlock(
            in_dims=hn_dim, out_dims=(1, None), use_mlp=True
        )

    def calculate_edge_attrs(self, edge_index: Tensor, pos: Tensor):
        source, target = edge_index
        r = pos[target] - pos[source]
        d = torch.clamp(torch.pow(r, 2).sum(-1), min=1e-6)
        d = d.sqrt()
        # r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        r_norm = torch.div(r, (d.unsqueeze(-1)))
        edge_attr = (d, r_norm)
        return edge_attr

    def forward(
        self, x: Tensor, pos: Tensor, t: Tensor, batch: OptTensor = None
    ) -> Dict:
        edge_index = radius_graph(
            x=pos, r=self.cutoff, batch=batch, max_num_neighbors=128
        )
        ta = self.time_mapping_atom(t)
        tnode = ta[batch]
        s = self.atom_mapping(x)
        s = self.atom_time_mapping(s + tnode)
        v = torch.zeros(
            size=(x.size(0), 3, self.vdim), device=x.device, dtype=pos.dtype
        )

        edge_attr = self.calculate_edge_attrs(edge_index, pos)
        s, v = self.gnn(
            s=s, v=v, edge_index=edge_index, edge_attr=edge_attr, batch=batch
        )
        energy_atoms, v = self.energy_head((s, v))
        energy_atoms = energy_atoms + v.sum() * 0
        bs = len(batch.unique())
        energy_molecule = scatter_add(energy_atoms, index=batch, dim=0, dim_size=bs)
        assert energy_molecule.size(1) == 1
        out = {"property_pred": energy_molecule}
        return out


if __name__ == "__main__":
    pass
