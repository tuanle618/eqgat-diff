import numpy as np
import ase.units
import ase
from ase import Atoms
from ase.io import write, read
import ase.units as units
import logging
import os
import subprocess
from rdkit import rdBase
import argparse
import pickle
import shutil
from tqdm import tqdm

rdBase.DisableLog("rdApp.error")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def make_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print(f"\nDirectory {path} has been created!")
    else:
        print(f"\nDirectory {path} exists already!")


def parse_xtb_xyz(filename):
    num_atoms = 0
    energy = 0
    pos = []
    with open(filename, "r") as f:
        for line_num, line in enumerate(f):
            if line_num == 0:
                num_atoms = int(line)
            elif line_num == 1:
                # xTB outputs energy in Hartrees: Hartree to eV
                energy = np.array(
                    float(line.split(" ")[2]) * units.Hartree, dtype=np.float32
                )
            elif line_num >= 2:
                _, x, y, z = line.split()
                pos.append([parse_float(x), parse_float(y), parse_float(z)])

    result = {
        "num_atoms": num_atoms,
        "energy": energy,
        "pos": np.array(pos, dtype=np.float32),
    }
    return result


def parse_float(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        base, power = s.split("*^")
        return float(base) * 10 ** float(power)


def xtb_optimization(data, file_path):
    zs = []
    positions = []
    energies = []
    failed_mol_ids = []

    for i, mol in tqdm(enumerate(data)):
        xtb_temp_dir = os.path.join(file_path, "xtb_tmp")
        make_dir(xtb_temp_dir)

        mol_size = mol.GetNumAtoms()
        opt = "lax" if mol_size > 60 else "normal"

        pos = np.array(mol.GetConformer().GetPositions(), dtype=np.float32)
        atomic_number = []
        for atom in mol.GetAtoms():
            atomic_number.append(atom.GetAtomicNum())
        z = np.array(atomic_number, dtype=np.int64)
        mol = Atoms(numbers=z, positions=pos)

        mol_path = os.path.join(xtb_temp_dir, f"xtb_conformer.xyz")
        write(mol_path, images=mol)

        os.chdir(xtb_temp_dir)
        try:
            subprocess.call(
                # ["xtb", mol_path, "--opt", opt, "--cycles", "2000", "--gbsa", "water"],
                ["xtb", mol_path, "--opt", opt],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except:
            print(f"Molecule with id {i} failed!")
            failed_mol_ids.append(i)
            continue
        result = parse_xtb_xyz(os.path.join(xtb_temp_dir, f"xtbopt.xyz"))
        atom = read(filename=os.path.join(xtb_temp_dir, f"xtbopt.xyz"))

        os.chdir(file_path)
        shutil.rmtree(xtb_temp_dir)

        z = atom.get_atomic_numbers()
        energy = result["energy"]
        pos = atom.get_positions()

        zs.append(z)
        energies.append(energy)
        positions.append(pos)

    with open(os.path.join(file_path, "energies.pickle"), "wb") as f:
        pickle.dump(energies, f)
    with open(os.path.join(file_path, "failed_mol_ids.pickle"), "wb") as f:
        pickle.dump(failed_mol_ids, f)

    return zs, positions, energies, failed_mol_ids


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--output-path', default=None, type=str,
                        help='If set, saves the energy to a text file to the given path.')
    parser.add_argument('--data', type=list, help='Input data as a list of RDkit molecules')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    _ = xtb_optimization(args.data, args.output_path)
