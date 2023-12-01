from typing import Optional, Tuple
import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptTensor
from torch_sparse import SparseTensor
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.nn import knn_graph

from e3moldiffusion.modules import DenseLayer, GatedEquivBlock, SE3Norm


def cross_product(a: Tensor, b: Tensor, dim: int) -> Tensor:
    if a.dtype != torch.float16 and b.dtype != torch.float16:
        return torch.linalg.cross(a, b, dim=dim)
    else:
        s1 = a[:, 1, :] * b[:, -1, :] - a[:, -1, :] * b[:, 1, :]
        s2 = a[:, -1, :] * b[:, 0, :] - a[:, 0, :] * b[:, -1, :]
        s3 = a[:, 0, :] * b[:, 1, :] - a[:, 1, :] * b[:, 0, :]
        cross = torch.stack([s1, s2, s3], dim=dim)
        return cross


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(r: Tensor, rcut: float, p: float = 6.0) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            print(f"Exponent p={p} has to be >= 2.")
            print("Exiting code.")
            exit()

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"


class BesselExpansion(nn.Module):
    def __init__(self, max_value: float, K: int = 20):
        super(BesselExpansion, self).__init__()
        self.max_value = max_value
        self.K = K
        frequency = math.pi * torch.arange(start=1, end=K + 1)
        self.register_buffer("frequency", frequency)
        self.reset_parameters()

    def reset_parameters(self):
        self.frequency.data = math.pi * torch.arange(start=1, end=self.K + 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Bessel RBF, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        ax = x.unsqueeze(-1) / self.max_value
        ax = ax * self.frequency
        sinax = torch.sin(ax)
        norm = torch.where(
            x == 0.0, torch.tensor(1.0, dtype=x.dtype, device=x.device), x
        )
        out = sinax / norm[..., None]
        out *= math.sqrt(2 / self.max_value)
        return out


class EQGATConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        num_rbfs: int,
        cutoff: float,
        eps: float = 1e-6,
        has_v_in: bool = True,
        use_cross_product: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
    ):
        super(EQGATConv, self).__init__(node_dim=0, aggr=None, flow="source_to_target")

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.use_cross_product = use_cross_product

        self.has_v_in = has_v_in
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.cutoff = cutoff
        self.num_rbfs = num_rbfs

        self.distance_expansion = BesselExpansion(
            max_value=cutoff,
            K=num_rbfs,
        )
        self.cutoff_fnc = PolynomialCutoff(cutoff, p=6)
        self.edge_net = nn.Sequential(
            DenseLayer(
                2 * self.si + self.num_rbfs, self.si, bias=True, activation=nn.SiLU()
            ),
            DenseLayer(self.si, self.v_mul * self.vi + self.si, bias=True),
        )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(
            in_dims=(self.si, self.vi),
            hs_dim=self.si,
            hv_dim=self.vi,
            out_dims=(self.si, self.vi),
            norm_eps=eps,
            use_mlp=use_mlp_update,
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        batch: Optional[Tensor],
    ):
        s, v = x
        d, r = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, r),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor],
        index: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(
            inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size
        )
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        d, r = edge_attr

        de = self.distance_expansion(d)
        dc = self.cutoff_fnc(d)
        de = dc.view(-1, 1) * de

        aij = torch.cat([sa_i, sa_j, de], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul * self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            if self.use_cross_product:
                vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
            else:
                vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            if self.use_cross_product:
                v_ij_cross = cross_product(va_i, va_j, dim=1)
                nv2_j = vij2 * v_ij_cross
                nv_j = nv0_j + nv1_j + nv2_j
            else:
                nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


########### With Edge Features ###########
class EQGATGlobalEdgeConvFinal(MessagePassing):
    """
    Slightly modified SO(3) equivariant graph attention convolution described in
    @inproceedings{
        le2022representation,
        title={Representation Learning on Biomolecular Structures using Equivariant Graph Attention},
        author={Tuan Le and Frank Noe and Djork-Arn{\'e} Clevert},
        booktitle={The First Learning on Graphs Conference},
        year={2022},
        url={https://openreview.net/forum?id=kv4xUo5Pu6}
    }

    Intention for this layer is to be used as a global fully-connected message passing layer.
    """

    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        edge_dim: int,
        eps: float = 1e-6,
        has_v_in: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
        use_cross_product: bool = False,
        edge_mp: bool = False,
        use_pos_norm: bool = True,
    ):
        super(EQGATGlobalEdgeConvFinal, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        assert edge_dim is not None

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.use_cross_product = use_cross_product
        self.silu = nn.SiLU()
        self.use_pos_norm = use_pos_norm
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        if use_pos_norm:
            self.posnorm = SE3Norm()
        else:
            self.posnorm = None

        self.edge_pre = DenseLayer(edge_dim, edge_dim)
        self.edge_dim = edge_dim
        # input_edge_dim = (
        #     2 * self.si + edge_dim + 2 + 2
        #     if self.use_pos_norm
        #     else 2 * self.si + edge_dim + 2
        # )
        input_edge_dim = 2 * self.si + edge_dim + 2 + 2

        self.edge_net = nn.Sequential(
            DenseLayer(input_edge_dim, self.si, bias=True, activation=nn.SiLU()),
            DenseLayer(
                self.si, self.v_mul * self.vi + self.si + 1 + edge_dim, bias=True
            ),
        )
        self.edge_post = DenseLayer(edge_dim, edge_dim)

        self.edge_mp = edge_mp

        emlp = False

        if edge_mp:
            if emlp:
                self.edge_lin = nn.Sequential(
                    DenseLayer(2 * edge_dim + 3, edge_dim, activation=nn.SiLU()),
                    DenseLayer(edge_dim, edge_dim),
                )
            else:
                self.edge_lin = DenseLayer(2 * edge_dim + 3, edge_dim)
        # previously, still keep in case old model checkpoints are loaded
        else:
            self.edge_lin = None

        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(
            in_dims=(self.si, self.vi),
            hs_dim=self.si,
            hv_dim=self.vi,
            out_dims=(self.so, self.vo),
            norm_eps=eps,
            use_mlp=use_mlp_update,
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        if self.has_v_in:
            reset(self.vector_net)
        reset(self.scalar_net)
        reset(self.update_net)
        if self.posnorm:
            self.posnorm.reset_parameters()

    @staticmethod
    def get_triplet(edge_index: torch.Tensor, num_nodes: int):
        assert edge_index.size(0) == 2
        input_edge_index = edge_index.clone()
        source, target = edge_index  # j->i
        # create identifiers for edges based on (source, target)
        value = torch.arange(source.size(0), device=source.device)
        # as row-index select the target (column) nodes --> transpose
        # create neighbours from j
        adj_t = SparseTensor(
            row=target, col=source, value=value, sparse_sizes=(num_nodes, num_nodes)
        )
        # get neighbours from j
        adj_t_row = adj_t[source]
        # returns the target nodes (k) that include the source (j)
        # note there can be path i->j->k where k is i
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)
        # print(num_triplets)
        # Node indices (k->j->i) for triplets.
        idx_i = target.repeat_interleave(num_triplets)
        idx_j = source.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()  # get index for k
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]
        # print(idx_i); print(idx_j); print(idx_k)
        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return input_edge_index, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def edge_message_passing(
        self,
        p: Tensor,
        batch: Tensor,
        k: int,
        edge_index_full: Tensor,
        edge_attr_full: Tensor,
    ):
        num_nodes = p.size(0)

        E_full = torch.zeros(
            size=(num_nodes, num_nodes, edge_attr_full.size(-1)),
            device=edge_attr_full.device,
            dtype=edge_attr_full.dtype,
        )
        E_full[edge_index_full[0], edge_index_full[1], :] = edge_attr_full

        # create kNN graph
        edge_index_knn = knn_graph(x=p, k=k, batch=batch, flow="source_to_target")
        j, i = edge_index_knn

        p_ij = p[j] - p[i]
        p_ij_n = F.normalize(p_ij, p=2, dim=-1)
        d_ij = torch.pow(p_ij, 2).sum(-1, keepdim=True).sqrt()

        edge_ij = E_full[j, i, :]

        edge_index_knn, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.get_triplet(
            edge_index_knn, num_nodes=num_nodes
        )

        p_jk = -1.0 * p_ij_n[idx_kj]
        p_ji = p_ij_n[idx_ji]

        theta_ijk = torch.sum(p_jk * p_ji, -1, keepdim=True).clamp_(
            -1.0 + 1e-7, 1.0 - 1e-7
        )
        theta_ijk = torch.arccos(theta_ijk)
        d_ji = d_ij[idx_ji]
        d_jk = d_ij[idx_kj]
        edge_0 = edge_ij[idx_ji]
        edge_1 = edge_ij[idx_kj]
        f_ijk = torch.cat([edge_0, edge_1, theta_ijk, d_ji, d_jk], dim=-1)
        f_ijk = self.edge_lin(f_ijk)
        aggr_edges = scatter(
            src=f_ijk,
            index=idx_ji,
            dim=0,
            reduce="mean",
            dim_size=edge_index_knn.size(-1),
        )
        E_aggr = torch.zeros_like(E_full)
        E_aggr[edge_index_knn[0], edge_index_knn[1], :] = aggr_edges

        E_out = E_full + E_aggr

        edge_attr_full = E_out[edge_index_full[0], edge_index_full[1], :]

        return edge_attr_full

    def forward(
        self,
        x: Tuple[Tensor, Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch: Tensor,
        batch_lig: Tensor = None,
        pocket_mask: Tensor = None,
    ):
        s, v, p = x
        d, a, r, e = edge_attr

        e = self.edge_pre(e)

        if self.edge_mp:
            e = self.edge_message_passing(
                p=p, batch=batch, k=4, edge_index_full=edge_index, edge_attr_full=e
            )

        ms, mv, mp, me = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            p=p,
            edge_attr=(d, a, r, e),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        if self.posnorm:
            p = (
                p + self.posnorm(mp, batch, batch_lig, pocket_mask) * pocket_mask
                if pocket_mask is not None
                else p + self.posnorm(mp, batch)
            )
        else:
            p = p + mp * pocket_mask if pocket_mask is not None else p + mp
        e = F.silu(me + e)
        e = self.edge_post(e)

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        out = {"s": s, "v": v, "p": p, "e": e}
        return out

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor, Tensor, Tensor],
        index: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(
            inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size
        )
        p = scatter(
            inputs[2], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size
        )
        edge = inputs[3]
        return s, v, p, edge

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        p_i: Tensor,
        p_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        d, a, r, e = edge_attr

        de0 = d.view(-1, 1)
        a0 = a.view(-1, 1)

        if self.use_pos_norm:
            d_i, d_j = (
                torch.pow(p_i, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
                torch.pow(p_j, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
            )
        else:
            d_i, d_j = torch.zeros_like(a0).to(a.device), torch.zeros_like(a0).to(
                a.device
            )
        aij = torch.cat([torch.cat([sa_i, sa_j], dim=-1), de0, a0, e, d_i, d_j], dim=-1)
        # else:
        #     aij = torch.cat([torch.cat([sa_i, sa_j], dim=-1), de0, a0, e], dim=-1)
        aij = self.edge_net(aij)

        fdim = aij.shape[-1]
        aij, gij = aij.split([fdim - 1, 1], dim=-1)
        fdim = aij.shape[-1]
        aij, edge = aij.split([fdim - self.edge_dim, self.edge_dim], dim=-1)
        pj = gij * r

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul * self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            if self.use_cross_product:
                vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
            else:
                vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            if self.use_cross_product:
                v_ij_cross = cross_product(va_i, va_j, dim=1)
                nv2_j = vij2 * v_ij_cross
                nv_j = nv0_j + nv1_j + nv2_j
            else:
                nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j, pj, edge


########### Local Without Edge Features ###########
class EQGATLocalConvFinal(MessagePassing):
    """
    Slightly modified SO(3) equivariant graph attention convolution described in
    @inproceedings{
        le2022representation,
        title={Representation Learning on Biomolecular Structures using Equivariant Graph Attention},
        author={Tuan Le and Frank Noe and Djork-Arn{\'e} Clevert},
        booktitle={The First Learning on Graphs Conference},
        year={2022},
        url={https://openreview.net/forum?id=kv4xUo5Pu6}
    }

    Intention for this layer is to be used as a local message passing layer.
    """

    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        edge_dim: int,
        eps: float = 1e-6,
        has_v_in: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
        use_cross_product: bool = False,
    ):
        super(EQGATLocalConvFinal, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        assert edge_dim is not None

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.use_cross_product = use_cross_product
        self.silu = nn.SiLU()
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.edge_net = nn.Sequential(
            DenseLayer(
                2 * self.si + edge_dim + 2 + 2, self.si, bias=True, activation=nn.SiLU()
            ),
            DenseLayer(self.si, self.v_mul * self.vi + self.si, bias=True),
        )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(
            in_dims=(self.si, self.vi),
            hs_dim=self.si,
            hv_dim=self.vi,
            out_dims=(self.so, self.vo),
            norm_eps=eps,
            use_mlp=use_mlp_update,
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        if self.has_v_in:
            reset(self.vector_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch: Tensor,
    ):
        s, v, p = x
        d, a, r, e = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            p=p,
            edge_attr=(d, a, r, e),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        out = {"s": s, "v": v, "p": p, "e": e}
        return out

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor],
        index: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(
            inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size
        )
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        p_i: Tensor,
        p_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        d, a, r, e = edge_attr

        de0 = d.view(-1, 1)
        a0 = a.view(-1, 1)

        d_i, d_j = (
            torch.pow(p_i, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
            torch.pow(p_j, 2).sum(-1, keepdim=True).clamp(min=1e-6).sqrt(),
        )
        aij = torch.cat([torch.cat([sa_i, sa_j], dim=-1), de0, a0, e, d_i, d_j], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul * self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            if self.use_cross_product:
                vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
            else:
                vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            if self.use_cross_product:
                v_ij_cross = cross_product(va_i, va_j, dim=1)
                nv2_j = vij2 * v_ij_cross
                nv_j = nv0_j + nv1_j + nv2_j
            else:
                nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


# Topological Conv without 3d coords
class TopoEdgeConvLayer(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, aggr: str = "mean"):
        super(TopoEdgeConvLayer, self).__init__(aggr=None)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_pre = DenseLayer(edge_dim, edge_dim)
        self.edge_dim = edge_dim
        self.edge_post = DenseLayer(edge_dim, edge_dim)
        self._aggr = aggr

        self.neighbour_lin = DenseLayer(in_dim, in_dim)
        self.msg_mlp = nn.Sequential(
            DenseLayer(2 * in_dim + edge_dim, in_dim, activation=nn.SiLU()),
            DenseLayer(in_dim, in_dim + edge_dim),
        )
        self.update_mlp = nn.Sequential(
            DenseLayer(in_dim, in_dim, activation=nn.SiLU()),
            DenseLayer(in_dim, out_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.edge_pre.reset_parameters()
        self.edge_post.reset_parameters()
        self.neighbour_lin.reset_parameters()
        reset(self.msg_mlp)
        reset(self.update_mlp)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        xn = self.neighbour_lin(x)

        e = self.edge_pre(edge_attr)
        mx, me = self.propagate(x=x, xu=xn, edge_index=edge_index, edge_attr=e)
        x = mx + x
        e = F.silu(me + e)
        e = self.edge_post(e)

        ox = self.update_mlp(x)
        x = ox + x
        return x, e

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor],
        index: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce=self._aggr, dim_size=dim_size)
        edge = inputs[1]
        return s, edge

    def message(
        self, x_i: Tensor, x_j: Tensor, xu_j: Tensor, edge_attr: Tensor
    ) -> Tensor:
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg, e = self.msg_mlp(msg).split([self.in_dim, self.edge_dim], dim=-1)
        x_j = msg * xu_j
        return x_j, e
