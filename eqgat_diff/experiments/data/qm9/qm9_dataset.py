import os
import os.path as osp
from typing import Any, Sequence
from torch.utils.data import Subset
import numpy as np
import pandas as pd
import torch
from experiments.data.abstract_dataset import AbstractDataModule
from experiments.data.metrics import compute_all_statistics
from experiments.data.utils import (
    Statistics,
    load_pickle,
    mol_to_torch_geometric,
    remove_hydrogens,
    save_pickle,
    train_subset,
)
from os.path import join

from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from tqdm import tqdm
from torch_geometric.data import DataLoader


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


full_atom_encoder = {
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


class QM9Dataset(InMemoryDataset):
    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(
        self,
        split,
        root,
        remove_h: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        only_stats=False
    ):
        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        super().__init__(root, transform, pre_transform, pre_filter)
        if not only_stats:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])).float(),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])).float(),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[10])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[11])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[8])

    @property
    def raw_file_names(self):
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        if self.split == "train":
            return [
                f"train_{h}.pt",
                f"train_n_{h}.pickle",
                f"train_atom_types_{h}.npy",
                f"train_bond_types_{h}.npy",
                f"train_charges_{h}.npy",
                f"train_valency_{h}.pickle",
                f"train_bond_lengths_{h}.pickle",
                f"train_angles_{h}.npy",
                "train_smiles.pickle",
                f"train_is_aromatic_{h}.npy",
                f"train_is_in_ring_{h}.npy",
                f"train_hybridization_{h}.npy",
            ]
        elif self.split == "val":
            return [
                f"val_{h}.pt",
                f"val_n_{h}.pickle",
                f"val_atom_types_{h}.npy",
                f"val_bond_types_{h}.npy",
                f"val_charges_{h}.npy",
                f"val_valency_{h}.pickle",
                f"val_bond_lengths_{h}.pickle",
                f"val_angles_{h}.npy",
                "val_smiles.pickle",
                f"val_is_aromatic_{h}.npy",
                f"val_is_in_ring_{h}.npy",
                f"val_hybridization_{h}.npy",
            ]
        else:
            return [
                f"test_{h}.pt",
                f"test_n_{h}.pickle",
                f"test_atom_types_{h}.npy",
                f"test_bond_types_{h}.npy",
                f"test_charges_{h}.npy",
                f"test_valency_{h}.pickle",
                f"test_bond_lengths_{h}.pickle",
                f"test_angles_{h}.npy",
                "test_smiles.pickle",
                f"test_is_aromatic_{h}.npy",
                f"test_is_in_ring_{h}.npy",
                f"test_hybridization_{h}.npy",
            ]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, "3195404"),
                osp.join(self.raw_dir, "uncharacterized.txt"),
            )
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
        )

        train.to_csv(os.path.join(self.raw_dir, "train.csv"))
        val.to_csv(os.path.join(self.raw_dir, "val.csv"))
        test.to_csv(os.path.join(self.raw_dir, "test.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=["mol_id"], inplace=True)

        with open(self.raw_paths[-1]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        data_list = []
        all_smiles = []
        num_errors = 0
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smiles is None:
                num_errors += 1
            else:
                all_smiles.append(smiles)

            data = mol_to_torch_geometric(mol, full_atom_encoder, smiles)
            if self.remove_h:
                data = remove_hydrogens(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

        statistics = compute_all_statistics(
            data_list,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
            additional_feats=True,
        )

        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)

        np.save(self.processed_paths[9], statistics.is_aromatic)
        np.save(self.processed_paths[10], statistics.is_in_ring)
        np.save(self.processed_paths[11], statistics.hybridization)

        print("Number of molecules that could not be mapped to smiles: ", num_errors)
        save_pickle(set(all_smiles), self.processed_paths[8])
        torch.save(self.collate(data_list), self.processed_paths[0])


class QM9DataModule(AbstractDataModule):
    def __init__(self, cfg, only_stats: bool = False):
        self.datadir = cfg.dataset_root
        root_path = self.datadir

        train_dataset = QM9Dataset(
            split="train", root=root_path, remove_h=cfg.remove_hs, only_stats=only_stats
        )
        val_dataset = QM9Dataset(split="val", root=root_path, remove_h=cfg.remove_hs, only_stats=only_stats)
        test_dataset = QM9Dataset(split="test", root=root_path, remove_h=cfg.remove_hs, only_stats=only_stats)

        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }
        if not only_stats:
            if cfg.select_train_subset:
                self.idx_train = train_subset(
                    dset_len=len(train_dataset),
                    train_size=cfg.train_size,
                    seed=cfg.seed,
                    filename=join(cfg.save_dir, "splits.npz"),
                )
                self.train_smiles = train_dataset.smiles
                train_dataset = Subset(train_dataset, self.idx_train)

        self.remove_h = cfg.remove_hs
        super().__init__(
            cfg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )

    def get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.cfg.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.cfg.inference_batch_size
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

        return dl