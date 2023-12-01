from rdkit import Chem
import gzip
from glob import glob
import os
from multiprocessing import cpu_count, Pool
import pubchem.data.dataset_utils as dataset_utils
from tqdm import tqdm
import pickle
import argparse


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--index', default=0, type=int, help='Which part of the splitted dataset')
    args = parser.parse_args()
    return args


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


full_atom_encoder = {
    "H": 0,
    "C": 1,
    "N": 2,
    "O": 3,
    "F": 4,
    "Si": 5,
    "P": 6,
    "S": 7,
    "Cl": 8,
    "Br": 9,
    "I": 10,
}


def process(file):
    inf = gzip.open(file)
    with Chem.ForwardSDMolSupplier(inf) as gzsuppl:
        molecules = [x for x in gzsuppl if x is not None]
    for mol in molecules:
        try:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
            data = dataset_utils.mol_to_torch_geometric(mol, full_atom_encoder, smiles)
            if data.pos.shape[0] != data.x.shape[0]:
                print(f"Molecule {smiles} does not have 3D information!")
                continue
            if data.pos.ndim != 2:
                print(f"Molecule {smiles} does not have 3D information!")
                continue
            if len(data.pos) < 2:
                print(f"Molecule {smiles} does not have 3D information!")
                continue
            data_list.append(data)
        except:
            continue
    # pbar.update(1)

    return


if __name__ == "__main__":
    args = get_args()

    data_list = []
    smiles_list = []

    files = f"----files_{args.index}.pickle"
    with open(files, "rb") as f:
        files = pickle.load(f)

    pbar = tqdm(total=len(files))

    for file in tqdm(files):
        process(file)

    with open(f"data_list_{args.index}.pickle", "wb") as f:
        pickle.dump(data_list, f)
    with open(f"smiles_list_{args.index}.pickle", "wb") as f:
        pickle.dump(smiles_list, f)
