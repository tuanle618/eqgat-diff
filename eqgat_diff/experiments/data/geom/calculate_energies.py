import argparse
import os
import pickle

import numpy as np
import torch
from torch_geometric.data.collate import collate
from tqdm import tqdm

from experiments.xtb_energy import calculate_xtb_energy


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Energy calculation')
    parser.add_argument('--dataset', type=str, help='Which dataset')
    parser.add_argument('--split', type=str, help='Which data split train/val/test')
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


def process(dataset, split):
    if dataset == "drugs":
        from experiments.data.geom.geom_dataset_adaptive import (
            GeomDrugsDataset as DataModule,
        )

        root_path = "/scratch1/cremej01/data/geom"
    elif dataset == "qm9":
        from experiments.data.qm9.qm9_dataset import GeomDrugsDataset as DataModule

        root_path = "/scratch1/cremej01/data/qm9"
    else:
        raise ValueError("Dataset not found")

    remove_hs = False

    dataset = DataModule(split=split, root=root_path, remove_h=remove_hs)

    failed_ids = []
    mols = []
    for i, mol in tqdm(enumerate(dataset)):
        atom_types = [atom_decoder[int(a)] for a in mol.x]
        try:
            e_ref = np.sum(
                [atom_reference[a] for a in atom_types]
            )  # * 27.2114 #Hartree to eV
            e, _ = calculate_xtb_energy(mol.pos, atom_types)
            e *= 0.0367493  # eV to Hartree
            mol.energy = torch.tensor(e - e_ref, dtype=torch.float32).unsqueeze(0)
        except:
            print(f"Molecule with id {i} failed...")
            failed_ids.append(i)
            continue
        mols.append(mol)

    print("Collate the data...")
    data, slices = _collate(mols)

    print("Saving the data...")
    torch.save(
        (data, slices), (os.path.join(root_path, f"processed/{split}_data_energy.pt"))
    )

    with open(os.path.join(root_path, f"failed_ids_{split}.pickle"), "wb") as f:
        pickle.dump(failed_ids, f)


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
    process(args.dataset, args.split)
