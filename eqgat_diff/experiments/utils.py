import argparse
import math
from os.path import dirname, exists, join
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.utilities import rank_zero_warn
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sort_edge_index import sort_edge_index
from torch_geometric.utils.subgraph import subgraph
from torch_scatter import scatter_add, scatter_mean
from torch_sparse import coalesce

from e3moldiffusion.molfeat import atom_type_config
from experiments.molecule_utils import Molecule

# fmt: off
# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after N
atomic_masses = np.array([
    1.0, 1.008, 4.002602, 6.94, 9.0121831,
    10.81, 12.011, 14.007, 15.999, 18.998403163,
    20.1797, 22.98976928, 24.305, 26.9815385, 28.085,
    30.973761998, 32.06, 35.45, 39.948, 39.0983,
    40.078, 44.955908, 47.867, 50.9415, 51.9961,
    54.938044, 55.845, 58.933194, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.921595, 78.971,
    79.904, 83.798, 85.4678, 87.62, 88.90584,
    91.224, 92.90637, 95.95, 97.90721, 101.07,
    102.9055, 106.42, 107.8682, 112.414, 114.818,
    118.71, 121.76, 127.6, 126.90447, 131.293,
    132.90545196, 137.327, 138.90547, 140.116, 140.90766,
    144.242, 144.91276, 150.36, 151.964, 157.25,
    158.92535, 162.5, 164.93033, 167.259, 168.93422,
    173.054, 174.9668, 178.49, 180.94788, 183.84,
    186.207, 190.23, 192.217, 195.084, 196.966569,
    200.592, 204.38, 207.2, 208.9804, 208.98243,
    209.98715, 222.01758, 223.01974, 226.02541, 227.02775,
    232.0377, 231.03588, 238.02891, 237.04817, 244.06421,
    243.06138, 247.07035, 247.07031, 251.07959, 252.083,
    257.09511, 258.09843, 259.101, 262.11, 267.122,
    268.126, 271.134, 270.133, 269.1338, 278.156,
    281.165, 281.166, 285.177, 286.182, 289.19,
    289.194, 293.204, 293.208, 294.214,
])
# fmt: on


class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            if (
                "load_model" in config
                and namespace.load_model is not None
                and config["load_model"] != namespace.load_model
            ):
                rank_zero_warn(
                    f"The load model argument was specified as a command line argument "
                    f"({namespace.load_model}) and in the config file ({config['load_model']}). "
                    f"Ignoring 'load_model' from the config file and loading {namespace.load_model}."
                )
                del config["load_model"]
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


class LoadFromCheckpoint(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        hparams_path = join(dirname(values), "hparams.yaml")
        if not exists(hparams_path):
            print(
                "Failed to locate the checkpoint's hparams.yaml file. Relying on command line args."
            )
            return
        with open(hparams_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config.keys():
            if key not in namespace and key != "prior_args":
                raise ValueError(f"Unknown argument in the model checkpoint: {key}")
        namespace.__dict__.update(config)
        namespace.__dict__.update(load_model=values)


def save_argparse(args, filename, exclude=None):
    import json

    if filename.endswith("yaml") or filename.endswith("yml"):
        if isinstance(exclude, str):
            exclude = [exclude]
        args = args.__dict__.copy()
        for exl in exclude:
            del args[exl]

        ds_arg = args.get("dataset_arg")
        if ds_arg is not None and isinstance(ds_arg, str):
            args["dataset_arg"] = json.loads(args["dataset_arg"])
        yaml.dump(args, open(filename, "w"))
    else:
        raise ValueError("Configuration file should end with yaml or yml")


def number(text):
    if text is None or text == "None":
        return None

    try:
        num_int = int(text)
    except ValueError:
        num_int = None
    num_float = float(text)

    if num_int == num_float:
        return num_int
    return num_float


class MissingEnergyException(Exception):
    pass


mol_properties = [
    "DIP",
    "HLgap",
    "eAT",
    "eC",
    "eEE",
    "eH",
    "eKIN",
    "eKSE",
    "eL",
    "eNE",
    "eNN",
    "eMBD",
    "eTS",
    "eX",
    "eXC",
    "eXX",
    "mPOL",
]


class MultiTaskLoss(torch.nn.Module):
    """https://arxiv.org/abs/1705.07115"""

    def __init__(self, reduction="mean"):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = torch.Tensor([True, True, True, True])
        self.n_tasks = len(self.is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, preds, targets, loss_fn):
        dtype = preds.dtype
        device = preds.device
        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)

        coeffs = 1 / ((self.is_regression + 1) * (stds**2))

        loss_0 = loss_fn(preds[:, 0], targets[:, 0])
        loss_1 = loss_fn(preds[:, 1], targets[:, 1])
        loss_2 = loss_fn(preds[:, 2], targets[:, 2])
        loss_3 = loss_fn(preds[:, 3], targets[:, 3])
        losses = torch.stack([loss_0, loss_1, loss_2, loss_3])

        multi_task_losses = coeffs * losses + torch.log(stds)

        if self.reduction == "sum":
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == "mean":
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


"""
usage
is_regression = torch.Tensor([True, True, False]) # True: Regression/MeanSquaredErrorLoss, False: Classification/CrossEntropyLoss
multitaskloss_instance = MultiTaskLoss(is_regression)

params = list(model.parameters()) + list(multitaskloss_instance.parameters())
torch.optim.Adam(params, lr=1e-3)

model.train()
multitaskloss.train()

losses = torch.stack(loss0, loss1, loss3)
multitaskloss = multitaskloss_instance(losses)
"""

from typing import Optional

from torch import Tensor


def one_hot(
    index: Tensor,
    num_classes: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Taskes a one-dimensional :obj:`index` tensor and returns a one-hot
    encoded representation of it with shape :obj:`[*, num_classes]` that has
    zeros everywhere except where the index of last dimension matches the
    corresponding value of the input tensor, in which case it will be :obj:`1`.

    .. note::
        This is a more memory-efficient version of
        :meth:`torch.nn.functional.one_hot` as you can customize the output
        :obj:`dtype`.

    Args:
        index (torch.Tensor): The one-dimensional input tensor.
        num_classes (int, optional): The total number of classes. If set to
            :obj:`None`, the number of classes will be inferred as one greater
            than the largest class value in the input tensor.
            (default: :obj:`None`)
        dtype (torch.dtype, optional): The :obj:`dtype` of the output tensor.
    """
    if index.dim() != 1:
        raise ValueError("'index' tensor needs to be one-dimensional")

    if num_classes is None:
        num_classes = int(index.max()) + 1

    out = torch.zeros((index.size(0), num_classes), dtype=dtype, device=index.device)
    return out.scatter_(1, index.unsqueeze(1), 1)


def coalesce_edges(edge_index, bond_edge_index, bond_edge_attr, n):
    edge_attr = torch.full(
        size=(edge_index.size(-1),),
        fill_value=0,
        device=edge_index.device,
        dtype=torch.long,
    )
    edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
    edge_attr = torch.cat([edge_attr, bond_edge_attr], dim=0)
    edge_index, edge_attr = coalesce(
        index=edge_index, value=edge_attr, m=n, n=n, op="max"
    )
    return edge_index, edge_attr


def get_global_ligand_pocket_index(
    pos_perturbed,
    batch,
    batch_pocket,
    batch_full,
    edge_index_global_lig,
    edge_attr_global_perturbed_lig,
    cutoff=5,
    max_num_neighbors=128,
    num_bond_classes=5,
    device="cuda",
):
    # Global interaction Ligand-Pocket
    edge_index_global = (
        torch.eq(batch_full.unsqueeze(0), batch_full.unsqueeze(-1))
        .int()
        .fill_diagonal_(0)
    )
    edge_index_global, _ = dense_to_sparse(edge_index_global)
    edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
    edge_index_global, edge_attr_global_perturbed = coalesce_edges(
        edge_index=edge_index_global,
        bond_edge_index=edge_index_global_lig,
        bond_edge_attr=edge_attr_global_perturbed_lig,
        n=batch_full.size(0),
    )
    edge_index_global, edge_attr_global_perturbed = sort_edge_index(
        edge_index=edge_index_global,
        edge_attr=edge_attr_global_perturbed,
        sort_by_row=False,
    )
    # Local interaction Ligand-Pocket
    edge_index_local = radius_graph(
        x=pos_perturbed,
        r=cutoff,
        batch=batch_full,
        max_num_neighbors=max_num_neighbors,
    )
    edge_index_global, _ = coalesce_edges(
        edge_index=edge_index_global,
        bond_edge_index=edge_index_local,
        bond_edge_attr=torch.zeros_like(edge_index_local[0]),
        n=batch_full.size(0),
    )
    edge_index_global, _ = sort_edge_index(
        edge_index=edge_index_global,
        edge_attr=edge_attr_global_perturbed,
        sort_by_row=False,
    )
    lig_n = batch.bincount()
    pocket_n = batch_pocket.bincount()
    lig_ids = torch.cat(
        [
            torch.arange(
                lig_n[:i].sum() + pocket_n[:i].sum(),
                lig_n[:i].sum() + pocket_n[:i].sum() + lig_n[i],
            )
            for i in range(len(lig_n))
        ]
    ).to(device)
    i = (
        torch.tensor([1 if i in lig_ids else 0 for i in edge_index_global[0]])
        .bool()
        .to(device)
    )
    j = (
        torch.tensor([1 if j in lig_ids else 0 for j in edge_index_global[1]])
        .bool()
        .to(device)
    )
    edge_mask = i & j

    edge_attr_global_perturbed = F.one_hot(
        edge_attr_global_perturbed, num_classes=num_bond_classes
    ).float() * (edge_mask).unsqueeze(1)

    batch_edge_global = batch_full[edge_index_global[0]]

    return batch_edge_global, edge_index_global, edge_attr_global_perturbed, edge_mask


def get_empirical_num_nodes(dataset_info):
    num_nodes_dict = dataset_info.get("n_nodes")
    max_num_nodes = max(num_nodes_dict.keys())
    empirical_distribution_num_nodes = {
        i: num_nodes_dict.get(i) for i in range(max_num_nodes)
    }
    empirical_distribution_num_nodes_tensor = {}

    for key, value in empirical_distribution_num_nodes.items():
        if value is None:
            value = 0
        empirical_distribution_num_nodes_tensor[key] = value
    empirical_distribution_num_nodes_tensor = torch.tensor(
        list(empirical_distribution_num_nodes_tensor.values())
    ).float()
    return empirical_distribution_num_nodes_tensor


def get_list_of_edge_adjs(edge_attrs_dense, batch_num_nodes):
    ptr = torch.cat(
        [
            torch.zeros(1, device=batch_num_nodes.device, dtype=torch.long),
            batch_num_nodes.cumsum(0),
        ]
    )
    edge_tensor_lists = []
    for i in range(len(ptr) - 1):
        select_slice = slice(ptr[i].item(), ptr[i + 1].item())
        e = edge_attrs_dense[select_slice, select_slice]
        edge_tensor_lists.append(e)
    return edge_tensor_lists


def get_num_atom_types_geom(dataset: str):
    assert dataset in ["qm9", "drugs"]
    return len(atom_type_config(dataset=dataset))


def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out


def remove_mean_ligand(x_lig, x_pocket, batch, batch_pocket):
    # Just subtract the center of mass of the sampled part
    mean = scatter_mean(x_lig, batch, dim=0)

    x_lig = x_lig - mean[batch]
    x_pocket = x_pocket - mean[batch_pocket]
    return x_lig, x_pocket


def remove_mean_pocket(pos_lig, pos_pocket, batch, batch_pocket):
    mean = scatter_mean(pos_pocket, batch_pocket, dim=0)
    pos_lig = pos_lig - mean[batch]
    pos_pocket = pos_pocket - mean[batch_pocket]
    return pos_lig, pos_pocket


def concat_ligand_pocket(
    pos_lig,
    pos_pocket,
    x_lig,
    x_pocket,
    c_lig,
    c_pocket,
    batch_lig,
    batch_pocket,
    sorting=False,
):
    batch_ctx = torch.cat([batch_lig, batch_pocket], dim=0)

    mask_ligand = torch.cat(
        [
            torch.ones([batch_lig.size(0)], device=batch_lig.device).bool(),
            torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        ],
        dim=0,
    )
    pos_ctx = torch.cat([pos_lig, pos_pocket], dim=0)
    x_ctx = torch.cat([x_lig, x_pocket], dim=0)
    c_ctx = torch.cat([c_lig, c_pocket], dim=0)

    if sorting:
        sort_idx = torch.sort(batch_ctx, stable=True).indices
        mask_ligand = mask_ligand[sort_idx]
        batch_ctx = batch_ctx[sort_idx]
        pos_ctx = pos_ctx[sort_idx]
        x_ctx = x_ctx[sort_idx]
        c_ctx = c_ctx[sort_idx]

    return pos_ctx, x_ctx, c_ctx, batch_ctx, mask_ligand


def concat_ligand_pocket_addfeats(
    pos_lig,
    pos_pocket,
    atom_features_ligand,
    atom_features_pocket,
    batch_lig,
    batch_pocket,
    sorting=False,
):
    batch_ctx = torch.cat([batch_lig, batch_pocket], dim=0)

    mask_ligand = torch.cat(
        [
            torch.ones([batch_lig.size(0)], device=batch_lig.device).bool(),
            torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        ],
        dim=0,
    )
    pos_ctx = torch.cat([pos_lig, pos_pocket], dim=0)
    atom_feats = torch.cat([atom_features_ligand, atom_features_pocket], dim=0)

    if sorting:
        sort_idx = torch.sort(batch_ctx, stable=True).indices
        mask_ligand = mask_ligand[sort_idx]
        batch_ctx = batch_ctx[sort_idx]
        pos_ctx = pos_ctx[sort_idx]
        atom_feats = atom_feats[sort_idx]

    return pos_ctx, atom_feats, batch_ctx, mask_ligand


def assert_zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0, eps: float = 1e-6):
    out = scatter_mean(x, index=batch, dim=dim, dim_size=dim_size).mean()
    return abs(out) < eps


def load_model(filepath, num_atom_features, device="cpu", **kwargs):
    import re

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]

    args["use_pos_norm"] = True

    model = create_model(args, num_atom_features)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def load_model_ligand(
    filepath, num_atom_features, num_bond_classes=5, device="cpu", **kwargs
):
    import re

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]

    args["use_pos_norm"] = False

    model = create_model(args, num_atom_features, num_bond_classes)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat", "posnorm", "se3norm"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_model(hparams, num_atom_features, num_bond_classes):
    from e3moldiffusion.coordsatomsbonds import DenoisingEdgeNetwork

    model = DenoisingEdgeNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        latent_dim=None,
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        num_bond_types=num_bond_classes,
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
    return model


def load_energy_model(filepath, num_atom_features, device="cpu"):
    import re

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_energy_model(args, num_atom_features)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_energy_model(hparams, num_atom_features):
    from e3moldiffusion.coordsatomsbonds import EQGATEnergyNetwork

    model = EQGATEnergyNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        num_rbfs=hparams["rbf_dim"],
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
    )
    return model


def load_bond_model(filepath, dataset_statistics, device="cpu", **kwargs):
    import re

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_bond_model(args, dataset_statistics)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_bond_model(hparams, dataset_statistics):
    from e3moldiffusion.coordsatomsbonds import EdgePredictionNetwork

    num_atom_types = dataset_statistics.input_dims.X
    num_charge_classes = dataset_statistics.input_dims.C
    num_atom_features = num_atom_types + num_charge_classes
    model = EdgePredictionNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        latent_dim=None,
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        num_bond_types=5,
        edge_dim=hparams["edim"],
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
        fully_connected=hparams["fully_connected"],
        local_global_model=hparams["local_global_model"],
        recompute_edge_attributes=True,
        recompute_radius_graph=False,
        edge_mp=hparams["edge_mp"],
    )
    return model


def truncated_exp_distribution(theta, x):
    return theta * torch.exp(-theta * x) / (1 - math.exp(-theta))


def cdf_exp_distribution(theta, x):
    return 1 - torch.exp(-theta * x)


def cdf_truncated_exp_distribution(theta, x):
    return cdf_exp_distribution(theta, x) / cdf_exp_distribution(
        theta, torch.ones_like(x)
    )


def quantile_func_truncated_exp_distribution(theta, p):
    return -1 / theta * torch.log(1 - p * (1 - math.exp(-theta)))


def sample_from_truncated_exp(theta, num_samples, device="cpu"):
    t = torch.rand((num_samples,), device=device)
    w = quantile_func_truncated_exp_distribution(theta=theta, p=t)
    return w


def t_int_to_frac(t, tmin, tmax):
    return (t - tmin) / tmax


def t_frac_to_int(t, tmin, tmax):
    return (t * tmax + tmin).long()


def dropout_node(
    edge_index: Tensor,
    batch: Tensor,
    p: float = 0.5,
    num_nodes: Optional[int] = None,
    training: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Randomly drops nodes from the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained. (3) the node mask indicating
    which nodes were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Dropout probability. (default: :obj:`0.5`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`, :class:`BoolTensor`)

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
        >>> edge_index
        tensor([[0, 1],
                [1, 0]])
        >>> edge_mask
        tensor([ True,  True, False, False, False, False])
        >>> node_mask
        tensor([ True,  True, False, False])
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"Dropout probability has to be between 0 and 1 " f"(got {p}")

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p

    bs = len(batch.unique())
    nSelect = scatter_add(node_mask.float(), index=batch, dim=0, dim_size=bs).float()
    batch_mask = ~(nSelect == 0.0)

    deleted_graphs = torch.where(~batch_mask)[0]
    if len(deleted_graphs) > 0:
        take_again = torch.zeros_like(batch)
        for graph_idx in deleted_graphs.cpu().numpy():
            graph_ids = graph_idx == batch
            take_again += graph_ids
        node_mask = (node_mask + take_again).bool()

    nSelect = scatter_add(node_mask.float(), index=batch, dim=0, dim_size=bs).float()
    batch_mask = ~(nSelect == 0.0)

    edge_index, _, edge_mask = subgraph(
        node_mask, edge_index, num_nodes=num_nodes, return_edge_mask=True
    )

    return edge_index, edge_mask, node_mask, batch_mask


def load_force_model(filepath, num_atom_features, device="cpu"):
    import re

    ckpt = torch.load(filepath, map_location="cpu")
    args = ckpt["hyper_parameters"]
    model = create_force_model(args, num_atom_features)

    state_dict = ckpt["state_dict"]
    state_dict = {
        re.sub(r"^model\.", "", k): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model")
    }
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if not any(x in k for x in ["prior", "sde", "cat"])
    }
    model.load_state_dict(state_dict)
    return model.to(device)


def create_force_model(hparams, num_atom_features):
    from e3moldiffusion.coordsatomsbonds import EQGATForceNetwork

    model = EQGATForceNetwork(
        hn_dim=(hparams["sdim"], hparams["vdim"]),
        num_layers=hparams["num_layers"],
        num_rbfs=hparams["rbf_dim"],
        use_cross_product=hparams["use_cross_product"],
        num_atom_features=num_atom_features,
        cutoff_local=hparams["cutoff_local"],
        vector_aggr=hparams["vector_aggr"],
    )
    return model


def effective_batch_size(
    max_size, reference_batch_size, reference_size=20, sampling=False
):
    x = reference_batch_size * (reference_size / max_size) ** 2
    return math.floor(1.8 * x) if sampling else math.floor(x)


def get_edges(
    batch_mask_lig, batch_mask_pocket, pos_lig, pos_pocket, cutoff_p, cutoff_lp
):
    adj_ligand = batch_mask_lig[:, None] == batch_mask_lig[None, :]
    adj_pocket = batch_mask_pocket[:, None] == batch_mask_pocket[None, :]
    adj_cross = batch_mask_lig[:, None] == batch_mask_pocket[None, :]

    if cutoff_p is not None:
        adj_pocket = adj_pocket & (torch.cdist(pos_pocket, pos_pocket) <= cutoff_p)

    if cutoff_lp is not None:
        adj_cross = adj_cross & (torch.cdist(pos_lig, pos_pocket) <= cutoff_lp)

    adj = torch.cat(
        (
            torch.cat((adj_ligand, adj_cross), dim=1),
            torch.cat((adj_cross.T, adj_pocket), dim=1),
        ),
        dim=0,
    )
    edges = torch.stack(torch.where(adj), dim=0)

    return edges


def get_molecules(
    out,
    data_batch,
    edge_index_global_lig,
    num_atom_types,
    num_charge_classes,
    dataset_info,
    device,
    data_batch_pocket=None,
    relax_mol=False,
    max_relax_iter=200,
    sanitize=False,
    mol_device="cpu",
    context=None,
    check_validity=False,
    build_mol_with_addfeats=False,
    build_obabel_mol=False,
    while_train=False,
):
    if while_train:
        atoms_pred = out["atoms_pred"]
        atoms_pred, charges_pred = atoms_pred.split(
            [num_atom_types, num_charge_classes], dim=-1
        )
    else:
        atoms_pred = out["atoms_pred"]
        charges_pred = out["charges_pred"]

    if out["bonds_pred"].shape[-1] > 5:
        out["bonds_pred"] = out["bonds_pred"][:, :5]
    n = data_batch.bincount().sum().item()
    edge_attrs_dense = torch.zeros(size=(n, n, 5), device=device).float()
    edge_attrs_dense[edge_index_global_lig[0, :], edge_index_global_lig[1, :], :] = out[
        "bonds_pred"
    ]
    edge_attrs_dense = edge_attrs_dense.argmax(-1)
    edge_attrs_splits = get_list_of_edge_adjs(edge_attrs_dense, data_batch.bincount())

    batch_num_nodes = data_batch.bincount().cpu().tolist()
    pos_splits = out["coords_pred"].detach().split(batch_num_nodes, dim=0)
    if data_batch_pocket is not None:
        batch_num_nodes_pocket = data_batch_pocket.bincount().cpu().tolist()
        pos_pocket_splits = (
            out["coords_pocket"].detach().split(batch_num_nodes_pocket, dim=0)
        )
        atom_types_integer_pocket = torch.argmax(out["atoms_pocket"], dim=-1)
        atom_types_integer_split_pocket = atom_types_integer_pocket.detach().split(
            batch_num_nodes_pocket, dim=0
        )

    atom_types_integer = torch.argmax(atoms_pred, dim=-1)
    atom_types_integer_split = atom_types_integer.detach().split(batch_num_nodes, dim=0)
    charge_types_integer = torch.argmax(charges_pred, dim=-1)
    # offset back
    charge_types_integer = charge_types_integer - dataset_info.charge_offset
    charge_types_integer_split = charge_types_integer.detach().split(
        batch_num_nodes, dim=0
    )
    if "aromatic_pred" in out.keys() and "hybridization_pred" in out.keys():
        add_feats = True
        aromatic_feat = out["aromatic_pred"]
        hybridization_feat = out["hybridization_pred"]
        aromatic_feat_integer = torch.argmax(aromatic_feat, dim=-1)
        aromatic_feat_integer_split = aromatic_feat_integer.detach().split(
            batch_num_nodes, dim=0
        )

        hybridization_feat_integer = torch.argmax(hybridization_feat, dim=-1)
        hybridization_feat_integer_split = hybridization_feat_integer.detach().split(
            batch_num_nodes, dim=0
        )
    else:
        add_feats = False
    context_split = (
        context.split(batch_num_nodes, dim=0) if context is not None else None
    )

    molecule_list = []
    for i, (
        positions,
        atom_types,
        charges,
        edges,
    ) in enumerate(
        zip(
            pos_splits,
            atom_types_integer_split,
            charge_types_integer_split,
            edge_attrs_splits,
        )
    ):
        molecule = Molecule(
            atom_types=atom_types.detach().to(mol_device),
            positions=positions.detach().to(mol_device),
            charges=charges.detach().to(mol_device),
            bond_types=edges.detach().to(mol_device),
            positions_pocket=pos_pocket_splits[i].detach().to(mol_device)
            if data_batch_pocket is not None
            else None,
            atom_types_pocket=atom_types_integer_split_pocket[i].detach().to(mol_device)
            if data_batch_pocket is not None
            else None,
            context=context_split[i][0].detach().to(mol_device)
            if context_split is not None
            else None,
            is_aromatic=aromatic_feat_integer_split[i].detach().to(mol_device)
            if add_feats
            else None,
            hybridization=hybridization_feat_integer_split[i].detach().to(mol_device)
            if add_feats
            else None,
            build_mol_with_addfeats=build_mol_with_addfeats,
            dataset_info=dataset_info,
            relax_mol=relax_mol,
            max_relax_iter=max_relax_iter,
            sanitize=sanitize,
            check_validity=check_validity,
            build_obabel_mol=build_obabel_mol,
        )
        molecule_list.append(molecule)

    return molecule_list


def get_inp_molecules(
    inp,
    data_batch,
    edge_index_global_lig,
    dataset_info,
    device,
    mol_device="cpu",
    context=None,
):
    atoms = inp["atoms"]
    charges = inp["charges"]

    n = data_batch.bincount().sum().item()
    edge_attrs_dense = torch.zeros(size=(n, n), device=device).long()
    edge_attrs_dense[edge_index_global_lig[0, :], edge_index_global_lig[1, :]] = inp[
        "edges"
    ]
    edge_attrs_splits = get_list_of_edge_adjs(edge_attrs_dense, data_batch.bincount())

    pos_splits = (
        inp["coords"].detach().split(data_batch.bincount().cpu().tolist(), dim=0)
    )
    atom_types_integer = torch.argmax(atoms, dim=-1)
    atom_types_integer_split = atom_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )
    charge_types_integer = torch.argmax(charges, dim=-1)
    # offset back
    charge_types_integer = charge_types_integer - dataset_info.charge_offset
    charge_types_integer_split = charge_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )
    context_split = (
        context.split(data_batch.bincount().cpu().tolist(), dim=0)
        if context is not None
        else None
    )

    molecule_list = []
    for i, (positions, atom_types, charges, edges) in enumerate(
        zip(
            pos_splits,
            atom_types_integer_split,
            charge_types_integer_split,
            edge_attrs_splits,
        )
    ):
        molecule = Molecule(
            atom_types=atom_types.detach().to(mol_device),
            positions=positions.detach().to(mol_device),
            charges=charges.detach().to(mol_device),
            bond_types=edges.detach().to(mol_device),
            context=context_split[i][0].detach().to(mol_device)
            if context_split is not None
            else None,
            dataset_info=dataset_info,
        )
        molecule_list.append(molecule)

    return molecule_list


def write_sdf_file(sdf_path, molecules):
    # NOTE Changed to be compatitble with more versions of rdkit
    # with Chem.SDWriter(str(sdf_path)) as w:
    #    for mol in molecules:
    #        w.write(mol)

    w = Chem.SDWriter(str(sdf_path))
    w.SetKekulize(False)
    for m in molecules:
        if m.rdkit_mol is not None:
            w.write(m.rdkit_mol)

    # print(f'Wrote SDF file to {sdf_path}')


def unbatch_data(
    pos,
    atom_types,
    charges,
    data_batch,
    dataset_info,
):
    pos_splits = pos.detach().split(data_batch.bincount().cpu().tolist(), dim=0)

    atom_types_integer = torch.argmax(atom_types, dim=-1)
    atom_types_integer_split = atom_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )
    charge_types_integer = torch.argmax(charges, dim=-1)
    # offset back
    charge_types_integer = charge_types_integer - dataset_info.charge_offset
    charge_types_integer_split = charge_types_integer.detach().split(
        data_batch.bincount().cpu().tolist(), dim=0
    )

    return pos_splits, atom_types_integer_split, charge_types_integer_split


def prepare_pocket(
    biopython_residues, full_atom_encoder, no_H=True, repeats=1, device="cuda"
):
    pocket_atoms = [
        a
        for res in biopython_residues
        for a in res.get_atoms()
        if (a.element.capitalize() in full_atom_encoder)
    ]

    pocket_coord = torch.tensor(
        np.array([a.get_coord() for a in pocket_atoms]),
        device=device,
        dtype=torch.float32,
    )
    pocket_atoms = [a.element.capitalize() for a in pocket_atoms]
    if no_H:
        indices_H = np.where(pocket_atoms == "H")
        if indices_H[0].size > 0:
            mask = np.ones(pocket_atoms.size, dtype=bool)
            mask[indices_H] = False
            pocket_atoms = pocket_atoms[mask]
            pocket_coord = pocket_coord[mask]

    pocket_types = (
        torch.tensor([full_atom_encoder[a] for a in pocket_atoms]).long().to(device)
    )
    pocket_mask = torch.repeat_interleave(
        torch.arange(repeats, device=device), len(pocket_coord)
    ).long()

    pocket = Data(
        x_pocket=pocket_types.repeat(repeats),
        pos_pocket=pocket_coord.repeat(repeats, 1),
        pos_pocket_batch=pocket_mask,
    )

    return pocket
