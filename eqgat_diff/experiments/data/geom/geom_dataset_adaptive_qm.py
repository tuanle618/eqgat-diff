import warnings
from os.path import join

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, InMemoryDataset
from tqdm import tqdm

import experiments.data.utils as dataset_utils
from experiments.data.abstract_dataset import AbstractAdaptiveDataModule
from experiments.data.metrics import compute_all_statistics
from experiments.data.utils import load_pickle, save_pickle, train_subset

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
atom_decoder = {v: k for k, v in full_atom_encoder.items()}


PROPS = [
    "dipole_norm",
    "total_energy",
    "HOMO-LUMO_gap",
    "dispersion",
    "atomisation_energy",
    "polarizability",
]


class GeomQMDataset(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        remove_h,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        n_confs=5,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.remove_h = remove_h
        self.n_confs = n_confs

        self.atom_encoder = full_atom_encoder
        self.label2idx = {k: i for i, k in enumerate(PROPS)}

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.post_processing()

    def post_processing(self):
        """load statistics and smiles"""
        self.statistics = dataset_utils.Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])),
            dihedrals=torch.from_numpy(np.load(self.processed_paths[8])).float(),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[10])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[11])).float(),
            force_norms=torch.from_numpy(np.load(self.processed_paths[13])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[12])

    @property
    def raw_file_names(self):
        if self.split == "train":
            return ["train_data_qm.pickle"]
        elif self.split == "val":
            return ["val_data_qm.pickle"]
        else:
            return ["test_data_qm.pickle"]

    def download(self):
        raise ValueError(
            "Download and preprocessing is manual. If the data is already downloaded, "
            f"check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}"
        )

    def process(self):
        all_data = load_pickle(self.raw_paths[0])
        data_list = []
        all_smiles = []
        for data in tqdm(all_data):
            smiles, all_conformers, all_properties = data
            all_smiles.append(smiles)
            for i, (conformer, properties) in enumerate(
                zip(all_conformers, all_properties)
            ):
                if (i >= self.n_confs) & (self.n_confs > 0):
                    break
                data = dataset_utils.mol_to_torch_geometric(
                    conformer,
                    self.atom_encoder,
                    smiles,
                    remove_hydrogens=self.remove_h,
                    wbo=properties["wbo"],
                    mulliken=properties["mulliken"],
                    grad=properties["grad"],
                )
                if not properties["normal_termination"]:
                    warnings.warn("Abnormal termination of xtb, skipping data point.")
                    continue
                if data.has_isolated_nodes():
                    warnings.warn("Found disconnected graph, skipping data point.")
                    continue
                data.y = torch.tensor([properties[pname] for pname in PROPS]).unsqueeze(
                    0
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        statistics = compute_all_statistics(
            data_list,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
            include_force_norms=True,
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        np.save(self.processed_paths[8], statistics.dihedrals)
        np.save(self.processed_paths[9], statistics.is_aromatic)
        np.save(self.processed_paths[10], statistics.is_in_ring)
        np.save(self.processed_paths[11], statistics.hybridization)
        save_pickle(set(all_smiles), self.processed_paths[12])
        np.save(self.processed_paths[13], statistics.force_norms)

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        confs = "all_confs" if self.n_confs < 1 else f"{self.n_confs}_confs"
        if self.split == "train":
            return [
                f"train_{h}_{confs}.pt",
                f"train_n_{h}_{confs}.pickle",
                f"train_atom_types_{h}_{confs}.npy",
                f"train_bond_types_{h}_{confs}.npy",
                f"train_charges_{h}_{confs}.npy",
                f"train_valency_{h}_{confs}.pickle",
                f"train_bond_lengths_{h}_{confs}.pickle",
                f"train_angles_{h}_{confs}.npy",
                f"train_dihedrals_{h}_{confs}.npy",
                f"train_is_aromatic_{h}_{confs}.npy",
                f"train_is_in_ring_{h}_{confs}.npy",
                f"train_hybridization_{h}_{confs}.npy",
                f"train_smiles_{confs}.pickle",
                f"train_fnorms_{confs}.npy",
            ]
        elif self.split == "val":
            return [
                f"val_{h}_{confs}.pt",
                f"val_n_{h}_{confs}.pickle",
                f"val_atom_types_{h}_{confs}.npy",
                f"val_bond_types_{h}_{confs}.npy",
                f"val_charges_{h}_{confs}.npy",
                f"val_valency_{h}_{confs}.pickle",
                f"val_bond_lengths_{h}_{confs}.pickle",
                f"val_angles_{h}_{confs}.npy",
                f"val_dihedrals_{h}_{confs}.npy",
                f"val_is_aromatic_{h}_{confs}.npy",
                f"val_is_in_ring_{h}_{confs}.npy",
                f"val_hybridization_{h}_{confs}.npy",
                f"val_smiles_{confs}.pickle",
                f"val_fnorms_{confs}.npy",
            ]
        else:
            return [
                f"test_{h}_{confs}.pt",
                f"test_n_{h}_{confs}.pickle",
                f"test_atom_types_{h}_{confs}.npy",
                f"test_bond_types_{h}_{confs}.npy",
                f"test_charges_{h}_{confs}.npy",
                f"test_valency_{h}_{confs}.pickle",
                f"test_bond_lengths_{h}_{confs}.pickle",
                f"test_angles_{h}_{confs}.npy",
                f"test_dihedrals_{h}_{confs}.npy",
                f"test_is_aromatic_{h}_{confs}.npy",
                f"test_is_in_ring_{h}_{confs}.npy",
                f"test_hybridization_{h}_{confs}.npy",
                f"test_smiles_{confs}.pickle",
                f"test_fnorms_{confs}.npy",
            ]


class GeomQMDataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg):
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.pin_memory = True

        train_dataset = GeomQMDataset(
            split="train",
            root=root_path,
            remove_h=cfg.remove_hs,
            n_confs=cfg.max_num_conformers,
        )
        val_dataset = GeomQMDataset(
            split="val",
            root=root_path,
            remove_h=cfg.remove_hs,
            n_confs=cfg.max_num_conformers,
        )
        test_dataset = GeomQMDataset(
            split="test",
            root=root_path,
            remove_h=cfg.remove_hs,
            n_confs=cfg.max_num_conformers,
        )

        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }

        self.label2idx = {k: i for i, k in enumerate(PROPS)}

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

        super().__init__(cfg, train_dataset, val_dataset, test_dataset)

    def _train_dataloader(self, shuffle=True):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def _val_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def _test_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def compute_mean_mad(self, properties_list):
        if (
            self.cfg.dataset == "qm9"
            or self.cfg.dataset == "drugs"
            or self.cfg.dataset == "geomqm"
        ):
            dataloader = self.get_dataloader(self.train_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        elif self.cfg.dataset == "qm9_1half" or self.cfg.dataset == "qm9_2half":
            dataloader = self.get_dataloader(self.val_dataset, "val")
            return self.compute_mean_mad_from_dataloader(dataloader, properties_list)
        else:
            raise Exception("Wrong dataset name")

    def compute_mean_mad_from_dataloader(self, dataloader, properties_list):
        property_norms = {}
        for property_key in properties_list:
            try:
                property_name = property_key + "_mm"
                values = getattr(dataloader.dataset[:], property_name)
            except:
                property_name = property_key
                idx = dataloader.dataset[:].label2idx[property_name]
                values = torch.tensor(
                    [data.y[:, idx] for data in dataloader.dataset[:]]
                )

            mean = torch.mean(values)
            ma = torch.abs(values - mean)
            mad = torch.mean(ma)
            property_norms[property_key] = {}
            property_norms[property_key]["mean"] = mean
            property_norms[property_key]["mad"] = mad
            del values
        return property_norms

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
