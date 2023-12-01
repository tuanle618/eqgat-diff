from rdkit import RDLogger
import torch
import numpy as np
from os.path import join
from torch_geometric.data import Dataset, DataLoader
from experiments.data.utils import load_pickle, make_splits
from torch.utils.data import Subset
import lmdb
import pickle
import gzip
import io
from pytorch_lightning import LightningDataModule
import os
import experiments.data.utils as dataset_utils

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
GEOM_DATADIR = "----"


class PubChemLMDBDataset(Dataset):
    def __init__(
        self,
        root: str,
        remove_hs: bool,
        evaluation: bool = False,
    ):
        """
        Constructor
        """
        self.data_file = root
        self._num_graphs = 95173200 if remove_hs else 95173300  # 94980241 for noH!!
        if remove_hs:
            assert "_noh" in root
            self.stats_dir = "----"
        else:
            assert "_h" in root
            self.stats_dir = "----"
        super().__init__(root)

        self.remove_hs = remove_hs
        self.statistics = dataset_utils.Statistics(
            num_nodes=load_pickle(
                os.path.join(self.stats_dir, self.processed_files[0])
            ),
            atom_types=torch.from_numpy(
                np.load(os.path.join(self.stats_dir, self.processed_files[1]))
            ),
            bond_types=torch.from_numpy(
                np.load(os.path.join(self.stats_dir, self.processed_files[2]))
            ),
            charge_types=torch.from_numpy(
                np.load(os.path.join(self.stats_dir, self.processed_files[3]))
            ),
            valencies=None #load_pickle(
                #os.path.join(self.stats_dir, self.processed_files[4])
            #),
            ,
            bond_lengths=load_pickle(
                os.path.join(self.stats_dir, self.processed_files[5])
            ),
            bond_angles=torch.from_numpy(
                np.load(os.path.join(self.stats_dir, self.processed_files[6]))
            ),
            is_aromatic=torch.from_numpy(
                np.load(os.path.join(self.stats_dir, self.processed_files[7]))
            ).float(),
            is_in_ring=torch.from_numpy(
                np.load(os.path.join(self.stats_dir, self.processed_files[8]))
            ).float(),
            hybridization=torch.from_numpy(
                np.load(os.path.join(self.stats_dir, self.processed_files[9]))
            ).float(),
        )
        self.smiles = None # load_pickle(
            #os.path.join(self.stats_dir, self.processed_files[10])
        #)

    def _init_db(self):
        self._env = lmdb.open(
            str(self.data_file),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            create=False,
        )

    def get(self, index: int):
        self._init_db()

        with self._env.begin(write=False) as txn:
            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            try:
                item = pickle.loads(serialized)["data"]
            except:
                return None

        return item

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return self._num_graphs

    def __len__(self) -> int:
        return self._num_graphs

    @property
    def processed_files(self):
        h = "noh" if self.remove_hs else "h"
        return [
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


class PubChemDataModule(LightningDataModule):
    def __init__(self, hparams, evaluation=False):
        super(PubChemDataModule, self).__init__()
        self.save_hyperparameters(hparams)
        self.datadir = hparams.dataset_root
        self.pin_memory = True

        self.remove_hs = hparams.remove_hs
        if self.remove_hs:
            print("Pre-Training on dataset with implicit hydrogens")
        self.dataset = PubChemLMDBDataset(
            root=self.datadir, remove_hs=self.remove_hs, evaluation=evaluation
        )

        self.train_smiles = self.dataset.smiles

        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            train_size=hparams.train_size,
            val_size=hparams.val_size,
            test_size=hparams.test_size,
            seed=hparams.seed,
            filename=join(self.hparams["save_dir"], "splits.npz"),
            splits=None,
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )
        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

        self.statistics = {
            "train": self.dataset.statistics,
            "val": self.dataset.statistics,
            "test": self.dataset.statistics,
        }

    def train_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=False,
        )
        return dataloader

    def val_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=False,
        )
        return dataloader

    def test_dataloader(self, shuffle=False):
        dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
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
