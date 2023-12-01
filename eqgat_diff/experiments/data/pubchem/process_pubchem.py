from glob import glob
import os
from tqdm import tqdm
import lmdb
from rdkit import Chem, RDLogger
import experiments.data.utils as dataset_utils
import gzip
import io
import multiprocessing as mp
import pickle

RDLogger.DisableLog("rdApp.*")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


DATA_PATH = "---"
FULL_ATOM_ENCODER = {
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


def process_files(
    processes: int = 36, chunk_size: int = 1024, subchunk: int = 128, removeHs=False
):
    """
    :param dataset:
    :param max_conformers:
    :param processes:
    :param chunk_size:
    :param subchunk:
    :return:
    """

    data_list = glob(os.path.join(DATA_PATH, "raw/*.gz"))
    h = "noh" if removeHs else "h"
    print(f"Process without hydrogens: {removeHs}")

    save_path = os.path.join(DATA_PATH, f"database_{h}")
    if os.path.exists(save_path):
        print("FYI: Output directory has been created already.")
    chunked_list = list(chunks(data_list, chunk_size))
    chunked_list = [list(chunks(l, subchunk)) for l in chunked_list]

    print(f"Total number of molecules {len(data_list)}.")
    print(f"Processing {len(chunked_list)} chunks each of size {chunk_size}.")

    env = lmdb.open(str(save_path), map_size=int(1e13))
    global_id = 0
    with env.begin(write=True) as txn:
        for chunklist in tqdm(chunked_list, total=len(chunked_list), desc="Chunks"):
            chunkresult = []
            for datachunk in tqdm(chunklist, total=len(chunklist), desc="Datachunks"):
                removeHs_list = [removeHs] * len(datachunk)
                with mp.Pool(processes=processes) as pool:
                    res = pool.starmap(
                        func=db_sample_helper, iterable=zip(datachunk, removeHs_list)
                    )
                    res = [r for r in res if r is not None]
                chunkresult.append(res)

            confs_sub = []
            smiles_sub = []
            for cr in chunkresult:
                subconfs = [a["confs"] for a in cr]
                subconfs = [item for sublist in subconfs for item in sublist]
                subsmiles = [a["smiles"] for a in cr]
                subsmiles = [item for sublist in subsmiles for item in sublist]
                confs_sub.append(subconfs)
                smiles_sub.append(subsmiles)

            confs_sub = [item for sublist in confs_sub for item in sublist]
            smiles_sub = [item for sublist in smiles_sub for item in sublist]

            assert len(confs_sub) == len(smiles_sub)
            # save
            for conf in confs_sub:
                result = txn.put(str(global_id).encode(), conf, overwrite=False)
                if not result:
                    raise RuntimeError(
                        f"LMDB entry {global_id} in {str(save_path)} " "already exists"
                    )
                global_id += 1

        print(f"{global_id} molecules have been processed!")
        print("Finished!")


def db_sample_helper(file, removeHs=False):
    saved_confs_list = []
    smiles_list = []

    inf = gzip.open(file)
    with Chem.ForwardSDMolSupplier(inf, removeHs=removeHs) as gzsuppl:
        molecules = [x for x in gzsuppl if x is not None]
    for mol in molecules:
        try:
            smiles = Chem.MolToSmiles(mol)
            data = dataset_utils.mol_to_torch_geometric(
                mol, FULL_ATOM_ENCODER, smiles, remove_hydrogens=removeHs
            )
            if data.pos.shape[0] != data.x.shape[0]:
                continue
            if data.pos.ndim != 2:
                continue
            if len(data.pos) < 2:
                continue
        except:
            continue
        # create binary object to be saved
        buf = io.BytesIO()
        saves = {
            "mol": mol,
            "data": data,
        }
        with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=6) as f:
            f.write(pickle.dumps(saves))
        compressed = buf.getvalue()
        saved_confs_list.append(compressed)
        smiles_list.append(smiles)
    return {
        "confs": saved_confs_list,
        "smiles": smiles_list,
    }


if __name__ == "__main__":
    process_files(removeHs=False)
    # process_files(removeHs=True)
