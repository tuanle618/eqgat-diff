import tempfile
import warnings

import numpy as np
import openbabel
import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

from experiments.data.utils import write_xyz_file

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

from rdkit.Chem.rdForceFieldHelpers import UFFHasAllMoleculeParams, UFFOptimizeMolecule

from experiments.data.utils import x_map as additional_node_map


class Molecule:
    def __init__(
        self,
        atom_types,
        bond_types,
        positions,
        charges,
        dataset_info,
        atom_types_pocket=None,
        positions_pocket=None,
        context=None,
        is_aromatic=None,
        hybridization=None,
        build_mol_with_addfeats=False,
        relax_mol=False,
        max_relax_iter=200,
        sanitize=True,
        check_validity=False,
        build_obabel_mol=False,
    ):
        """
        atom_types: n      LongTensor
        charges: n         LongTensor
        bond_types: n x n  LongTensor
        positions: n x 3   FloatTensor
        atom_decoder: extracted from dataset_infos.
        """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, (
            f"shape of atoms {atom_types.shape} " f"and dtype {atom_types.dtype}"
        )
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, (
            f"shape of bonds {bond_types.shape} --" f" {bond_types.dtype}"
        )
        assert len(atom_types.shape) == 1
        assert len(bond_types.shape) == 2
        assert len(positions.shape) == 2

        self.relax_mol = relax_mol
        self.max_relax_iter = max_relax_iter
        self.sanitize = sanitize
        self.check_validity = check_validity

        self.dataset_info = dataset_info
        self.atom_decoder = (
            dataset_info["atom_decoder"]
            if isinstance(dataset_info, dict)
            else self.dataset_info.atom_decoder
        )

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()
        self.positions = positions
        self.positions_pocket = positions_pocket
        self.atom_types_pocket = atom_types_pocket
        self.charges = charges
        self.context = context

        if isinstance(is_aromatic, torch.Tensor):
            assert len(is_aromatic.shape) == 1
            assert (
                is_aromatic.max().item() <= len(additional_node_map["is_aromatic"]) - 1
            )
            self.is_aromatic = is_aromatic
        else:
            self.is_aromatic = None

        if isinstance(hybridization, torch.Tensor):
            assert len(hybridization.shape) == 1
            assert (
                hybridization.max().item()
                <= len(additional_node_map["hybridization"]) - 1
            )
            self.hybridization = hybridization
        else:
            self.hybridization = None

        self.additional_feats = isinstance(
            self.is_aromatic, torch.Tensor
        ) and isinstance(self.hybridization, torch.Tensor)
        self.build_mol_with_addfeats = build_mol_with_addfeats

        self.rdkit_mol = (
            self.build_molecule_openbabel()
            if build_obabel_mol
            else self.build_molecule()
        )
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(self.atom_decoder)

    def build_molecule(self, verbose=False):
        """If positions is None,"""
        if verbose:
            print("building new molecule")

        mol = Chem.RWMol()

        if self.additional_feats and self.build_mol_with_addfeats:
            for atom, charge, is_aromatic, sp_hybridization in zip(
                self.atom_types, self.charges, self.is_aromatic, self.hybridization
            ):
                if atom == -1:
                    continue
                try:
                    a = Chem.Atom(self.atom_decoder[int(atom.item())])
                except:
                    continue
                if charge.item() != 0:
                    a.SetFormalCharge(charge.item())
                a.SetIsAromatic(additional_node_map["is_aromatic"][is_aromatic.item()])
                a.SetHybridization(
                    additional_node_map["hybridization"][sp_hybridization.item()]
                )
                mol.AddAtom(a)
                if verbose:
                    print("Atom added: ", atom.item(), self.atom_decoder[atom.item()])
        else:
            for atom, charge in zip(self.atom_types, self.charges):
                if atom == -1:
                    continue
                try:
                    a = Chem.Atom(self.atom_decoder[int(atom.item())])
                except:
                    a = Chem.Atom("H")
                if charge.item() != 0:
                    a.SetFormalCharge(charge.item())
                mol.AddAtom(a)
                if verbose:
                    print("Atom added: ", atom.item(), self.atom_decoder[atom.item()])

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(
                    bond[0].item(),
                    bond[1].item(),
                    bond_dict[edge_types[bond[0], bond[1]].item()],
                )
                if verbose:
                    print(
                        "bond added:",
                        bond[0].item(),
                        bond[1].item(),
                        edge_types[bond[0], bond[1]].item(),
                        bond_dict[edge_types[bond[0], bond[1]].item()],
                    )

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    positions[i][0].item(),
                    positions[i][1].item(),
                    positions[i][2].item(),
                ),
            )
        mol.AddConformer(conf)

        if self.relax_mol:
            mol_uff = mol
            try:
                if self.sanitize:
                    Chem.SanitizeMol(mol_uff)
                self.uff_relax(mol_uff, self.max_relax_iter)
                if self.sanitize:
                    Chem.SanitizeMol(mol_uff)
                return mol_uff
            except (RuntimeError, ValueError) as e:
                if self.check_validity:
                    return self.compute_validity(mol)
                else:
                    return mol
        else:
            if self.check_validity:
                return self.compute_validity(mol)
            else:
                return mol

    def build_molecule_openbabel(self):
        """
        Build an RDKit molecule using openbabel for creating bonds
        Args:
            positions: N x 3
            atom_types: N
            atom_decoder: maps indices to atom types
        Returns:
            rdkit molecule
        """
        atom_types = [self.atom_decoder[a] for a in self.atom_types]

        try:
            with tempfile.NamedTemporaryFile() as tmp:
                tmp_file = tmp.name

                # Write xyz file
                write_xyz_file(self.positions, atom_types, tmp_file)

                # Convert to sdf file with openbabel
                # openbabel will add bonds
                obConversion = openbabel.OBConversion()
                obConversion.SetInAndOutFormats("xyz", "sdf")
                ob_mol = openbabel.OBMol()
                obConversion.ReadFile(ob_mol, tmp_file)

                obConversion.WriteFile(ob_mol, tmp_file)

                # Read sdf file with RDKit
                tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

            # Build new molecule. This is a workaround to remove radicals.
            mol = Chem.RWMol()
            for atom in tmp_mol.GetAtoms():
                mol.AddAtom(Chem.Atom(atom.GetSymbol()))
            mol.AddConformer(tmp_mol.GetConformer(0))

            for bond in tmp_mol.GetBonds():
                mol.AddBond(
                    bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
                )
            mol = self.process_obabel_molecule(mol, sanitize=True, largest_frag=True)
        except:
            return None

        return mol

    def process_obabel_molecule(
        self,
        rdmol,
        add_hydrogens=False,
        sanitize=False,
        relax_iter=0,
        largest_frag=False,
    ):
        """
        Apply filters to an RDKit molecule. Makes a copy first.
        Args:
            rdmol: rdkit molecule
            add_hydrogens
            sanitize
            relax_iter: maximum number of UFF optimization iterations
            largest_frag: filter out the largest fragment in a set of disjoint
                molecules
        Returns:
            RDKit molecule or None if it does not pass the filters
        """

        # Create a copy
        mol = Chem.Mol(rdmol)

        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                warnings.warn("Sanitization failed. Returning None.")
                return None

        if add_hydrogens:
            mol = Chem.AddHs(mol, addCoords=(len(mol.GetConformers()) > 0))

        if largest_frag:
            mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if sanitize:
                # sanitize the updated molecule
                try:
                    Chem.SanitizeMol(mol)
                except ValueError:
                    return None

        if relax_iter > 0:
            if not UFFHasAllMoleculeParams(mol):
                warnings.warn(
                    "UFF parameters not available for all atoms. " "Returning None."
                )
                return None

            try:
                self.uff_relax(mol, relax_iter)
                if sanitize:
                    # sanitize the updated molecule
                    Chem.SanitizeMol(mol)
            except (RuntimeError, ValueError) as e:
                return None

        return mol

    def uff_relax(self, mol, max_iter=200):
        """
        Uses RDKit's universal force field (UFF) implementation to optimize a
        molecule.
        """
        more_iterations_required = UFFOptimizeMolecule(mol, maxIters=max_iter)
        if more_iterations_required:
            warnings.warn(
                f"Maximum number of FF iterations reached. "
                f"Returning molecule after {max_iter} relaxation steps."
            )
        return more_iterations_required

    def compute_validity(self, mol, strict=False):
        if mol is not None:
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(
                    mol, asMols=True, sanitizeFrags=False
                )
                if len(mol_frags) > 1:
                    return None
                else:
                    largest_mol = max(
                        mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                    )
                    initial_adj = Chem.GetAdjacencyMatrix(
                        largest_mol, useBO=True, force=True
                    )
                    Chem.SanitizeMol(largest_mol)
                    smiles = Chem.MolToSmiles(largest_mol)

                    if sum([a.GetNumImplicitHs() for a in largest_mol.GetAtoms()]) > 0:
                        return None
                    if strict:
                        # sanitization changes bond order without throwing exceptions for certain cases
                        # https://github.com/rdkit/rdkit/blob/master/Docs/Book/RDKit_Book.rst#molecular-sanitization
                        # only consider change in BO to be wrong when difference is > 0.5 (not just kekulization difference)
                        adj2 = Chem.GetAdjacencyMatrix(
                            largest_mol, useBO=True, force=True
                        )
                        if not np.all(np.abs(initial_adj - adj2) < 1):
                            return None
                        # atom valencies are only correct when unpaired electrons are added
                        # when training data does not contain open shell systems, this should be considered an error
                        if (
                            sum(
                                [
                                    a.GetNumRadicalElectrons()
                                    for a in largest_mol.GetAtoms()
                                ]
                            )
                            > 0
                        ):
                            return None
            except:
                return None

        return mol

    def build_molecule_edm(self, positions, atom_types, dataset_info):
        atom_decoder = dataset_info["atom_decoder"]
        X, A, E = self.build_xae_molecule(positions, atom_types, dataset_info)
        mol = Chem.RWMol()
        for atom in X:
            a = Chem.Atom(atom_decoder[atom.item()])
            mol.AddAtom(a)

        all_bonds = torch.nonzero(A)
        for bond in all_bonds:
            mol.AddBond(
                bond[0].item(), bond[1].item(), bond_dict[E[bond[0], bond[1]].item()]
            )
        return mol

    def build_xae_molecule(self, positions, atom_types, dataset_info):
        """Returns a triplet (X, A, E): atom_types, adjacency matrix, edge_types
        args:
        positions: N x 3  (already masked to keep final number nodes)
        atom_types: N
        returns:
        X: N         (int)
        A: N x N     (bool)                  (binary adjacency matrix)
        E: N x N     (int)  (bond type, 0 if no bond) such that A = E.bool()
        """
        atom_decoder = dataset_info["atom_decoder"]
        n = positions.shape[0]
        X = atom_types
        A = torch.zeros((n, n), dtype=torch.bool)
        E = torch.zeros((n, n), dtype=torch.int)

        pos = positions.unsqueeze(0)
        dists = torch.cdist(pos, pos, p=2).squeeze(0)
        for i in range(n):
            for j in range(i):
                pair = sorted([atom_types[i], atom_types[j]])
                if (
                    dataset_info["name"] == "qm9"
                    or dataset_info["name"] == "qm9_second_half"
                    or dataset_info["name"] == "qm9_first_half"
                ):
                    order = get_bond_order(
                        atom_decoder[pair[0]], atom_decoder[pair[1]], dists[i, j]
                    )
                elif dataset_info["name"] == "drugs" or dataset_info["name"] == "aqm":
                    order = geom_predictor(
                        (atom_decoder[pair[0]], atom_decoder[pair[1]]),
                        dists[i, j],
                        limit_bonds_to_one=True,
                    )
                # TODO: a batched version of get_bond_order to avoid the for loop
                if order > 0:
                    # Warning: the graph should be DIRECTED
                    A[i, j] = 1
                    E[i, j] = order
        return X, A, E


class PlaceHolder:
    def __init__(self, pos, X, C, E, y, t_int=None, t=None, node_mask=None):
        self.pos = pos
        self.X = X
        self.C = C
        self.E = E
        self.y = y
        self.t_int = t_int
        self.t = t
        self.node_mask = node_mask

    def device_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.pos = self.pos.to(x.device) if self.pos is not None else None
        self.X = self.X.to(x.device) if self.X is not None else None
        self.C = self.C.to(x.device) if self.C is not None else None
        self.E = self.E.to(x.device) if self.E is not None else None
        self.y = self.y.to(x.device) if self.y is not None else None
        return self

    def mask(self, node_mask=None):
        if node_mask is None:
            assert self.node_mask is not None
            node_mask = self.node_mask
        bs, n = node_mask.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        diag_mask = (
            ~torch.eye(n, dtype=torch.bool, device=node_mask.device)
            .unsqueeze(0)
            .expand(bs, -1, -1)
            .unsqueeze(-1)
        )  # bs, n, n, 1

        if self.X is not None:
            self.X = self.X * x_mask
        if self.C is not None:
            self.C = self.C * x_mask
        if self.E is not None:
            self.E = self.E * e_mask1 * e_mask2 * diag_mask
        if self.pos is not None:
            self.pos = self.pos - self.pos.mean(dim=1, keepdim=True)
            self.pos = self.pos * x_mask
        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

    def collapse(self, collapse_charges):
        copy = self.copy()
        copy.X = torch.argmax(self.X, dim=-1)
        copy.C = collapse_charges.to(self.C.device)[torch.argmax(self.C, dim=-1)]
        copy.E = torch.argmax(self.E, dim=-1)
        x_mask = self.node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
        copy.X[self.node_mask == 0] = -1
        copy.C[self.node_mask == 0] = 1000
        copy.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        return copy

    def __repr__(self):
        return (
            f"pos: {self.pos.shape if type(self.pos) == torch.Tensor else self.pos} -- "
            + f"X: {self.X.shape if type(self.X) == torch.Tensor else self.X} -- "
            + f"C: {self.C.shape if type(self.C) == torch.Tensor else self.C} -- "
            + f"E: {self.E.shape if type(self.E) == torch.Tensor else self.E} -- "
            + f"y: {self.y.shape if type(self.y) == torch.Tensor else self.y}"
        )

    def copy(self):
        return PlaceHolder(
            X=self.X,
            C=self.C,
            E=self.E,
            y=self.y,
            pos=self.pos,
            t_int=self.t_int,
            t=self.t,
            node_mask=self.node_mask,
        )


# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {
    "H": {
        "H": 74,
        "C": 109,
        "N": 101,
        "O": 96,
        "F": 92,
        "B": 119,
        "Si": 148,
        "P": 144,
        "As": 152,
        "S": 134,
        "Cl": 127,
        "Br": 141,
        "I": 161,
    },
    "C": {
        "H": 109,
        "C": 154,
        "N": 147,
        "O": 143,
        "F": 135,
        "Si": 185,
        "P": 184,
        "S": 182,
        "Cl": 177,
        "Br": 194,
        "I": 214,
    },
    "N": {
        "H": 101,
        "C": 147,
        "N": 145,
        "O": 140,
        "F": 136,
        "Cl": 175,
        "Br": 214,
        "S": 168,
        "I": 222,
        "P": 177,
    },
    "O": {
        "H": 96,
        "C": 143,
        "N": 140,
        "O": 148,
        "F": 142,
        "Br": 172,
        "S": 151,
        "P": 163,
        "Si": 163,
        "Cl": 164,
        "I": 194,
    },
    "F": {
        "H": 92,
        "C": 135,
        "N": 136,
        "O": 142,
        "F": 142,
        "S": 158,
        "Si": 160,
        "Cl": 166,
        "Br": 178,
        "P": 156,
        "I": 187,
    },
    "B": {"H": 119, "Cl": 175},
    "Si": {
        "Si": 233,
        "H": 148,
        "C": 185,
        "O": 163,
        "S": 200,
        "F": 160,
        "Cl": 202,
        "Br": 215,
        "I": 243,
    },
    "Cl": {
        "Cl": 199,
        "H": 127,
        "C": 177,
        "N": 175,
        "O": 164,
        "P": 203,
        "S": 207,
        "B": 175,
        "Si": 202,
        "F": 166,
        "Br": 214,
    },
    "S": {
        "H": 134,
        "C": 182,
        "N": 168,
        "O": 151,
        "S": 204,
        "F": 158,
        "Cl": 207,
        "Br": 225,
        "Si": 200,
        "P": 210,
        "I": 234,
    },
    "Br": {
        "Br": 228,
        "H": 141,
        "C": 194,
        "O": 172,
        "N": 214,
        "Si": 215,
        "S": 225,
        "F": 178,
        "Cl": 214,
        "P": 222,
    },
    "P": {
        "P": 221,
        "H": 144,
        "C": 184,
        "O": 163,
        "Cl": 203,
        "S": 210,
        "F": 156,
        "N": 177,
        "Br": 222,
    },
    "I": {
        "H": 161,
        "C": 214,
        "Si": 243,
        "N": 222,
        "O": 194,
        "S": 234,
        "F": 187,
        "I": 266,
    },
    "As": {"H": 152},
}

bonds2 = {
    "C": {"C": 134, "N": 129, "O": 120, "S": 160},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121, "P": 150},
    "P": {"O": 150, "S": 186},
    "S": {"P": 186},
}


bonds3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}


def print_table(bonds_dict):
    letters = ["H", "C", "O", "N", "P", "S", "F", "Si", "Cl", "Br", "I"]

    new_letters = []
    for key in letters + list(bonds_dict.keys()):
        if key in bonds_dict.keys():
            if key not in new_letters:
                new_letters.append(key)

    letters = new_letters

    for j, y in enumerate(letters):
        if j == 0:
            for x in letters:
                print(f"{x} & ", end="")
            print()
        for i, x in enumerate(letters):
            if i == 0:
                print(f"{y} & ", end="")
            if x in bonds_dict[y]:
                print(f"{bonds_dict[y][x]} & ", end="")
            else:
                print("- & ", end="")
        print()


# print_table(bonds3)
def check_consistency_bond_dictionaries():
    for bonds_dict in [bonds1, bonds2, bonds3]:
        for atom1 in bonds1:
            for atom2 in bonds_dict[atom1]:
                bond = bonds_dict[atom1][atom2]
                try:
                    bond_check = bonds_dict[atom2][atom1]
                except KeyError:
                    raise ValueError("Not in dict " + str((atom1, atom2)))

                assert (
                    bond == bond_check
                ), f"{bond} != {bond_check} for {atom1}, {atom2}"


stdv = {"H": 5, "C": 1, "N": 1, "O": 2, "F": 3}

margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
}


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance  # We change the metric

    # Check exists for large molecules where some atom pairs do not have a
    # typical bond length.
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    # margin1, margin2 and margin3 have been tuned to maximize the stability of
    # the QM9 true samples.
    if distance < bonds1[atom1][atom2] + margin1:
        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3  # Triple
                return 2  # Double
        return 1  # Single
    return 0  # No bond


def single_bond_only(threshold, length, margin1=5):
    if length < threshold + margin1:
        return 1
    return 0


def geom_predictor(p, l, margin1=5, limit_bonds_to_one=False):
    """p: atom pair (couple of str)
    l: bond length (float)"""
    bond_order = get_bond_order(p[0], p[1], l, check_exists=True)

    # If limit_bonds_to_one is enabled, every bond type will return 1.
    if limit_bonds_to_one:
        return 1 if bond_order > 0 else 0
    else:
        return bond_order
