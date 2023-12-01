import argparse
import os

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data.collate import collate
from tqdm import tqdm

from experiments.xtb_energy import calculate_dipole


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Energy calculation')
    parser.add_argument('--dataset', type=str, help='Which dataset')
    parser.add_argument('--split', type=str, help='Which data split train/val/test')
    parser.add_argument('--idx', type=int, default=None, help='Which part of the dataset (pubchem only)')

    args = parser.parse_args()
    return args


atom_encoder = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
}
atom_decoder = {v: k for k, v in atom_encoder.items()}
atom_reference = {
    "H": -0.393482763936,
    "B": -0.952436614164,
    "C": -1.795110518041,
    "N": -2.60945245463,
    "O": -3.769421097051,
    "F": -4.619339964238,
    "Al": -0.905328611479,
    "Si": -1.571424085131,
    "P": -2.377807088084,
    "S": -3.148271017078,
    "Cl": -4.482525134961,
    "As": -2.239425948594,
    "Br": -4.048339371234,
    "I": -3.77963026339,
    "Hg": -0.848032246708,
    "Bi": -2.26665341636,
}


def process(dataset, split, idx):
    if dataset == "drugs":
        from experiments.data.geom.geom_dataset_adaptive import (
            GeomDrugsDataset as DataModule,
        )

        root_path = "----"
    elif dataset == "qm9":
        from experiments.data.qm9.qm9_dataset import QM9Dataset as DataModule

        root_path = "----"
    elif dataset == "aqm":
        from experiments.data.aqm.aqm_dataset_nonadaptive import (
            AQMDataset as DataModule,
        )

        root_path = "----"
    elif dataset == "pubchem":
        from experiments.data.pubchem.pubchem_dataset_nonadaptive import (
            PubChemLMDBDataset as DataModule,
        )

        root_path = "----"
    else:
        raise ValueError("Dataset not found")

    remove_hs = False

    datamodule = DataModule(split=split, root=root_path, remove_h=remove_hs)

    if dataset == "pubchem":
        split_len = len(datamodule) // 500
        rng = np.arange(0, len(datamodule))
        rng = rng[idx * split_len : (idx + 1) * split_len]
        datamodule = Subset(datamodule, rng)

    # elif dataset == "drugs":
    #     split_len = len(datamodule) // 50
    #     rng = np.arange(0, len(datamodule))
    #     rng = rng[idx * split_len : (idx + 1) * split_len]
    #     datamodule = Subset(datamodule, rng)

    mols = []
    for i, mol in tqdm(enumerate(datamodule), total=len(datamodule)):
        atom_types = [atom_decoder[int(a)] for a in mol.x]
        try:
            d = calculate_dipole(mol.pos, atom_types)
            mol.dipole_classic = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
            mols.append(mol)
        except:
            print(f"Molecule with id {i} failed...")
            continue

    print(f"Collate the data...")
    data, slices = _collate(mols)

    print(f"Saving the data...")
    torch.save(
        (data, slices),
        (os.path.join(root_path, f"processed/{split}_{idx}_data_energy.pt")),
    )


def _collate(data_list):
    r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
    to the internal storage format of
    :class:`~torch_geometric.data.InMemoryDataset`."""
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices


if __name__ == "__main__":
    args = get_args()
    process(args.dataset, args.split, args.idx)
