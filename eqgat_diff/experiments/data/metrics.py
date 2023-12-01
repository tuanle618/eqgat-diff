import math
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torchmetrics import (
    KLDivergence,
    MeanAbsoluteError,
    MeanSquaredError,
    Metric,
    MetricCollection,
)

from experiments.data.utils import Statistics
from experiments.molecule_utils import Molecule

allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {
        0: [2, 3],
        1: [2, 3, 4],
        -1: 2,
    },  # In QM9, N+ seems to be present in the form NH+ and NH2+
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}
bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


def compute_all_statistics(
    data_list,
    atom_encoder,
    charges_dic,
    additional_feats: bool = True,
    include_force_norms: bool = False,
):
    num_nodes = node_counts(data_list)
    atom_types = atom_type_counts(data_list, num_classes=len(atom_encoder))
    print(f"Atom types: {atom_types}")
    bond_types = edge_counts(data_list)
    print(f"Bond types: {bond_types}")
    charge_types = charge_counts(
        data_list, num_classes=len(atom_encoder), charges_dic=charges_dic
    )
    print(f"Charge types: {charge_types}")
    valency = valency_count(data_list, atom_encoder)
    print("Valency: ", valency)

    bond_lengths = bond_lengths_counts(data_list)
    print("Bond lengths: ", bond_lengths)
    angles = bond_angles(data_list, atom_encoder)
    dihedrals = dihedral_angles(data_list, normalize=normalize)

    feats = {}
    if include_force_norms:
        feats["force_norms"] = per_atom_force_norm(data_list)
        print(feats["force_norms"])

    if additional_feats:
        feats.update(additional_feat_counts(data_list=data_list))

    return Statistics(
        num_nodes=num_nodes,
        atom_types=atom_types,
        bond_types=bond_types,
        charge_types=charge_types,
        valencies=valency,
        bond_lengths=bond_lengths,
        bond_angles=angles,
        dihedrals=dihedrals,
        **feats,
    )


def per_atom_force_norm(data_list):
    # make list of all force norms
    forces = []
    for data in data_list:
        forces.extend(list(np.linalg.norm(data.grad, axis=1)))
    forces = torch.tensor(forces)
    # calculate histograms with bin width 1e-5
    bin_width = 1e-5
    bins = torch.arange(0, 0.2, bin_width)
    counts, _ = torch.histogram(forces, bins=bins)
    return counts


def additional_feat_counts(
    data_list, keys: list = ["is_aromatic", "is_in_ring", "hybridization"]
):
    print(f"Computing node counts for features = {str(keys)}")
    from experiments.data.utils import x_map

    num_classes_list = [len(x_map.get(key)) for key in keys]
    counts_list = [np.zeros(num_classes) for num_classes in num_classes_list]

    for data in data_list:
        for i, key, num_classes in zip(range(len(keys)), keys, num_classes_list):
            x = torch.nn.functional.one_hot(data.get(key), num_classes=num_classes)
            counts_list[i] += x.sum(dim=0).numpy()

    for i in range(len(counts_list)):
        counts_list[i] = counts_list[i] / counts_list[i].sum()
    print("Done")

    results = dict()
    for key, count in zip(keys, counts_list):
        results[key] = count

    print(results)

    return results


def node_counts(data_list):
    print("Computing node counts...")
    all_node_counts = Counter()
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        all_node_counts[num_nodes] += 1
    print("Done.")
    return all_node_counts


def atom_type_counts(data_list, num_classes):
    print("Computing node types distribution...")
    counts = np.zeros(num_classes)
    for data in data_list:
        x = torch.nn.functional.one_hot(data.x, num_classes=num_classes)
        counts += x.sum(dim=0).numpy()

    counts = counts / counts.sum()
    print("Done.")
    return counts


def edge_counts(data_list, num_bond_types=5):
    print("Computing edge counts...")
    d = np.zeros(num_bond_types)

    for data in data_list:
        total_pairs = data.num_nodes * (data.num_nodes - 1)

        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = (
            torch.nn.functional.one_hot(
                data.edge_attr - 1, num_classes=num_bond_types - 1
            )
            .sum(dim=0)
            .numpy()
        )
        d[0] += num_non_edges
        d[1:] += edge_types

    d = d / d.sum()
    return d


def charge_counts(data_list, num_classes, charges_dic):
    print("Computing charge counts...")
    d = np.zeros((num_classes, len(charges_dic)))

    for data in data_list:
        for atom, charge in zip(data.x, data.charges):
            assert charge in [-2, -1, 0, 1, 2, 3]
            d[atom.item(), charges_dic[charge.item()]] += 1

    s = np.sum(d, axis=1, keepdims=True)
    s[s == 0] = 1
    d = d / s
    print("Done.")
    return d


def valency_count(data_list, atom_encoder):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing valency counts...")
    valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}

    for data in data_list:
        edge_attr = data.edge_attr
        edge_attr[edge_attr == 4] = 1.5
        bond_orders = edge_attr

        for atom in range(data.num_nodes):
            edges = bond_orders[data.edge_index[0] == atom]
            valency = edges.sum(dim=0)
            valencies[atom_decoder[data.x[atom].item()]][valency.item()] += 1

    # Normalizing the valency counts
    for atom_type in valencies.keys():
        s = sum(valencies[atom_type].values())
        for valency, count in valencies[atom_type].items():
            valencies[atom_type][valency] = count / s
    print("Done.")
    return valencies


def bond_lengths_counts(data_list, num_bond_types=5):
    """Compute the bond lenghts separetely for each bond type."""
    print("Computing bond lengths...")
    all_bond_lenghts = {1: Counter(), 2: Counter(), 3: Counter(), 4: Counter()}
    for data in data_list:
        cdists = torch.cdist(data.pos.unsqueeze(0), data.pos.unsqueeze(0)).squeeze(0)
        bond_distances = cdists[data.edge_index[0], data.edge_index[1]]
        for bond_type in range(1, num_bond_types):
            bond_type_mask = data.edge_attr == bond_type
            distances_to_consider = bond_distances[bond_type_mask]
            distances_to_consider = torch.round(distances_to_consider, decimals=2)
            for d in distances_to_consider:
                all_bond_lenghts[bond_type][d.item()] += 1

    # Normalizing the bond lenghts
    for bond_type in range(1, num_bond_types):
        s = sum(all_bond_lenghts[bond_type].values())
        for d, count in all_bond_lenghts[bond_type].items():
            all_bond_lenghts[bond_type][d] = count / s
    print("Done.")
    return all_bond_lenghts


def bond_angles(data_list, atom_encoder):
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing bond angles...")
    all_bond_angles = np.zeros((len(atom_encoder.keys()), 180 * 10 + 1))
    for data in data_list:
        assert not torch.isnan(data.pos).any()
        for i in range(data.num_nodes):
            neighbors = data.edge_index[1][data.edge_index[0] == i]
            for j in neighbors:
                for k in neighbors:
                    if j == k:
                        continue
                    assert i != j and i != k and j != k, "i, j, k: {}, {}, {}".format(
                        i, j, k
                    )
                    a = data.pos[j] - data.pos[i]
                    b = data.pos[k] - data.pos[i]

                    # print(a, b, torch.norm(a) * torch.norm(b))
                    angle = torch.acos(
                        torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-6)
                    )
                    angle = angle * 180 / math.pi

                    bin = int(torch.round(angle, decimals=1) * 10)
                    all_bond_angles[data.x[i].item(), bin] += 1

    # Normalizing the angles
    s = all_bond_angles.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    all_bond_angles = all_bond_angles / s
    print("Done.")
    return all_bond_angles


def dihedral_angles(data_list, normalize=True):
    def calculate_dihedral_angles(mol):
        def find_dihedrals(mol):
            torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
            torsionQuery = Chem.MolFromSmarts(torsionSmarts)
            matches = mol.GetSubstructMatches(torsionQuery)
            torsionList = []
            btype = []
            for match in matches:
                idx2 = match[0]
                idx3 = match[1]
                bond = mol.GetBondBetweenAtoms(idx2, idx3)
                jAtom = mol.GetAtomWithIdx(idx2)
                kAtom = mol.GetAtomWithIdx(idx3)
                if (
                    (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ) or (
                    (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
                    and (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
                ):
                    continue
                for b1 in jAtom.GetBonds():
                    if b1.GetIdx() == bond.GetIdx():
                        continue
                    idx1 = b1.GetOtherAtomIdx(idx2)
                    for b2 in kAtom.GetBonds():
                        if (b2.GetIdx() == bond.GetIdx()) or (
                            b2.GetIdx() == b1.GetIdx()
                        ):
                            continue
                        idx4 = b2.GetOtherAtomIdx(idx3)
                        # skip 3-membered rings
                        if idx4 == idx1:
                            continue
                        bt = bond.GetBondTypeAsDouble()
                        # bt = str(bond.GetBondType())
                        # if bond.IsInRing():
                        #     bt += '_R'
                        btype.append(bt)
                        torsionList.append((idx1, idx2, idx3, idx4))
            return np.asarray(torsionList), np.asarray(btype)

        dihedral_idx, dihedral_types = find_dihedrals(mol)

        coords = mol.GetConformer().GetPositions()
        t_angles = []
        for t in dihedral_idx:
            u1, u2, u3, u4 = coords[torch.tensor(t)]

            a1 = u2 - u1
            a2 = u3 - u2
            a3 = u4 - u3

            v1 = np.cross(a1, a2)
            v1 = v1 / (v1 * v1).sum(-1) ** 0.5
            v2 = np.cross(a2, a3)
            v2 = v2 / (v2 * v2).sum(-1) ** 0.5
            porm = np.sign((v1 * a3).sum(-1))
            rad = np.arccos(
                (v1 * v2).sum(-1)
                / ((v1**2).sum(-1) * (v2**2).sum(-1) + 1e-9) ** 0.5
            )
            if not porm == 0:
                rad = rad * porm
            t_angles.append(rad * 180 / torch.pi)

        return np.asarray(t_angles), dihedral_types

    generated_dihedrals = torch.zeros(5, 180 * 10 + 1)
    for d in data_list:
        mol = d.mol
        angles, types = calculate_dihedral_angles(mol)
        # transform types to idx
        types[types == 1.5] = 4
        types = types.astype(int)
        for a, t in zip(np.abs(angles), types):
            if np.isnan(a):
                continue
            generated_dihedrals[t, int(np.round(a, decimals=1) * 10)] += 1

    if normalize:
        s = generated_dihedrals.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        generated_dihedrals = generated_dihedrals.float() / s

    return generated_dihedrals


def counter_to_tensor(c: Counter):
    max_key = max(c.keys())
    assert type(max_key) == int
    arr = torch.zeros(max_key + 1, dtype=torch.float)
    for k, v in c.items():
        arr[k] = v
    arr / torch.sum(arr)
    return arr


def wasserstein1d(preds, target, step_size=1):
    """preds and target are 1d tensors. They contain histograms for bins that are regularly spaced"""
    target = normalize(target) / step_size
    preds = normalize(preds) / step_size
    max_len = max(len(preds), len(target))
    preds = F.pad(preds, (0, max_len - len(preds)))
    target = F.pad(target, (0, max_len - len(target)))

    cs_target = torch.cumsum(target, dim=0)
    cs_preds = torch.cumsum(preds, dim=0)
    return torch.sum(torch.abs(cs_preds - cs_target)).item()


def total_variation1d(preds, target):
    assert (
        target.dim() == 1 and preds.shape == target.shape
    ), f"preds: {preds.shape}, target: {target.shape}"
    target = normalize(target)
    preds = normalize(preds)
    return torch.sum(torch.abs(preds - target)).item(), torch.abs(preds - target)


def normalize(tensor):
    s = tensor.sum()
    assert s > 0
    return tensor / s


class NoSyncMetricCollection(MetricCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncMetric(Metric):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncKL(KLDivergence):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncMSE(MeanSquaredError):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching


class NoSyncMAE(MeanAbsoluteError):
    def __init__(self):
        super().__init__(
            sync_on_compute=False, dist_sync_on_step=False
        )  # disabling syncs since it messes up DDP sub-batching>>>>>>> main:utils.py


def molecules_to_datalist(molecules):
    data_list = []
    for molecule in molecules:
        x = molecule.atom_types.long()
        bonds = molecule.bond_types.long()
        positions = molecule.positions
        charges = molecule.charges
        edge_index = bonds.nonzero().contiguous().T
        bond_types = bonds[edge_index[0], edge_index[1]]
        edge_attr = bond_types.long()
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            charges=charges,
        )
        data_list.append(data)

    return data_list


def batch_to_list(one_hot, positions, charges, batch, dataset_info, bonds=None):
    atomsxmol = batch.bincount()

    num_atoms_prev = 0
    molecule_list = []
    for num_atoms in atomsxmol:
        z = one_hot[num_atoms_prev : num_atoms_prev + num_atoms]
        pos = positions[num_atoms_prev : num_atoms_prev + num_atoms]
        q = charges[num_atoms_prev : num_atoms_prev + num_atoms]
        b = (
            bonds[
                num_atoms_prev : num_atoms_prev + num_atoms,
                num_atoms_prev : num_atoms_prev + num_atoms,
            ]
            if bonds is not None
            else None
        )

        molecule = Molecule(
            atom_types=z,
            positions=pos,
            charges=q,
            bond_types=b,
            dataset_info=dataset_info,
        )
        molecule_list.append(molecule)

        num_atoms_prev += num_atoms
    return molecule_list
