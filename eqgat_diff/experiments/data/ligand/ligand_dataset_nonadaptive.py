import tempfile

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import DataLoader, InMemoryDataset

import experiments.data.utils as dataset_utils
from experiments.data.abstract_dataset import (
    AbstractDataModuleLigand,
)
from experiments.data.metrics import compute_all_statistics
from experiments.data.utils import (
    get_rdkit_mol,
    load_pickle,
    save_pickle,
    write_xyz_file,
)


def get_mol_babel(coords, atom_types):
    with tempfile.NamedTemporaryFile() as tmp:
        tmp_file = tmp.name
        # Write xyz file
        write_xyz_file(coords=coords, atom_types=atom_types, filename=tmp_file)
        rdkit_mol = get_rdkit_mol(tmp_file)
        smiles = Chem.MolToSmiles(rdkit_mol, isomericSmiles=False)
    return smiles, rdkit_mol


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


class LigandPocketDataset(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        remove_hs=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.remove_hs = remove_hs

        self.compute_bond_distance_angles = True
        self.atom_encoder = full_atom_encoder

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = dataset_utils.Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[8])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[10])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[11])

    @property
    def raw_file_names(self):
        if self.split == "train":
            return ["train.npz"]
        elif self.split == "val":
            return ["val.npz"]
        else:
            return ["test.npz"]

    def processed_file_names(self):
        h = "noh" if self.remove_hs else "h"
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
                f"train_is_aromatic_{h}.npy",
                f"train_is_in_ring_{h}.npy",
                f"train_hybridization_{h}.npy",
                "train_smiles.pickle",
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
                f"val_is_aromatic_{h}.npy",
                f"val_is_in_ring_{h}.npy",
                f"val_hybridization_{h}.npy",
                "val_smiles.pickle",
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
                f"test_is_aromatic_{h}.npy",
                f"test_is_in_ring_{h}.npy",
                f"test_hybridization_{h}.npy",
                "test_smiles.pickle",
            ]

    def download(self):
        raise ValueError(
            "Download and preprocessing is manual. If the data is already downloaded, "
            f"check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}"
        )

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        data_list_lig = []
        data_list_pocket = []
        all_smiles = []

        with np.load(self.raw_paths[0], allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        mol_data = {}
        for k, v in data.items():
            if k == "names" or k == "receptors" or k == "lig_mol":
                mol_data[k] = v
                continue

            sections = (
                np.where(np.diff(data["lig_mask"]))[0] + 1
                if "lig" in k
                else np.where(np.diff(data["pocket_mask"]))[0] + 1
            )
            if k == "lig_atom" or k == "pocket_atom":
                mol_data[k] = [
                    torch.tensor([full_atom_encoder[a] for a in atoms])
                    for atoms in np.split(v, sections)
                ]
            else:
                mol_data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]
            # add number of nodes for convenience
            if k == "lig_mask":
                mol_data["num_lig_atoms"] = torch.tensor(
                    [len(x) for x in mol_data["lig_mask"]]
                )
            elif k == "pocket_mask":
                mol_data["num_pocket_nodes"] = torch.tensor(
                    [len(x) for x in mol_data["pocket_mask"]]
                )

        for i, (
            mol_lig,
            coords_lig,
            atoms_lig,
            mask_lig,
            coords_pocket,
            atoms_pocket,
            mask_pocket,
        ) in enumerate(
            zip(
                mol_data["lig_mol"],
                mol_data["lig_coords"],
                mol_data["lig_atom"],
                mol_data["lig_mask"],
                mol_data["pocket_coords"],
                mol_data["pocket_atom"],
                mol_data["pocket_mask"],
            )
        ):
            try:
                # atom_types = [atom_decoder[int(a)] for a in atoms_lig]
                # smiles_lig, conformer_lig = get_mol_babel(coords_lig, atom_types)
                smiles_lig = Chem.MolToSmiles(mol_lig)
                data = dataset_utils.mol_to_torch_geometric(
                    mol_lig,
                    full_atom_encoder,
                    smiles_lig,
                    remove_hydrogens=self.remove_hs,
                    cog_proj=False,
                )
            except:
                print(f"Ligand {i} failed")
                continue
            data.pos_lig = coords_lig
            data.x_lig = atoms_lig
            data.pos_pocket = coords_pocket
            data.x_pocket = atoms_pocket
            data.lig_mask = mask_lig
            data.pocket_mask = mask_pocket

            all_smiles.append(smiles_lig)
            data_list_lig.append(data)

        center = False
        if center:
            for i in range(len(data_list_lig)):
                mean = (
                    data_list_lig[i].pos.sum(0) + data_list_lig[i].pos_pocket.sum(0)
                ) / (len(data_list_lig[i].pos) + len(data_list_lig[i].pos_pocket))
                data_list_lig[i].pos = data_list_lig[i].pos - mean
                data_list_lig[i].pos_pocket = data_list_lig[i].pos_pocket - mean

        torch.save(self.collate(data_list_lig), self.processed_paths[0])

        statistics = compute_all_statistics(
            data_list_lig,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
            additional_feats=True,
            # do not compute bond distance and bond angle statistics due to time and we do not use it anyways currently
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        np.save(self.processed_paths[8], statistics.is_aromatic)
        np.save(self.processed_paths[9], statistics.is_in_ring)
        np.save(self.processed_paths[10], statistics.hybridization)

        save_pickle(set(all_smiles), self.processed_paths[11])


class LigandPocketDataModule(AbstractDataModuleLigand):
    def __init__(self, cfg):
        self.datadir = cfg.dataset_root
        root_path = cfg.dataset_root
        self.pin_memory = True

        train_dataset = LigandPocketDataset(
            split="train", root=root_path, remove_hs=cfg.remove_hs
        )
        val_dataset = LigandPocketDataset(
            split="val", root=root_path, remove_hs=cfg.remove_hs
        )
        test_dataset = LigandPocketDataset(
            split="test", root=root_path, remove_hs=cfg.remove_hs
        )
        self.remove_hs = cfg.remove_hs
        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }
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
        if self.cfg.dataset == "qm9" or self.cfg.dataset == "drugs":
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
