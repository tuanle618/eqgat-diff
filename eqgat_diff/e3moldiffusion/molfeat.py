from typing import Union

import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import GetPeriodicTable
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import Data, Batch


PERIODIC_TABLE = GetPeriodicTable()

# allowable multiple choice node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC'#,
    #    'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}

def atom_type_config(dataset: str = "qm9"):
    if dataset == "qm9":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    elif dataset == "drugs":
        mapping =  {
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
    return mapping

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing())
            ]
    return atom_feature


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType()))
            ]
    return bond_feature


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        ]))


def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
    degree_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }
    return feature_dict


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx]
    }

    return feature_dict


def smiles_or_mol_to_graph(smol: Union[str, Chem.Mol], create_bond_graph: bool = True):
    if isinstance(smol, str):
        mol = Chem.MolFromSmiles(smol)
    else:
        mol = smol

    # atoms
    atom_features_list = []
    atom_element_name_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
        atom_element_name_list.append(PERIODIC_TABLE.GetElementSymbol(atom.GetAtomicNum()))

    
    x = torch.tensor(atom_features_list, dtype=torch.int64)
    assert x.size(-1) == 5
    # only take atom element
    # x = x[:, 0].view(-1, 1)

    if create_bond_graph:
        # bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature[0])
            edges_list.append((j, i))
            edge_features_list.append(edge_feature[0])

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(edges_list, dtype=torch.int64).T
        # data.edge_attr: Edge feature matrix with shape [num_edges]
        edge_attr = torch.tensor(edge_features_list, dtype=torch.int64)

        if edge_index.numel() > 0:  # Sort indices.
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
    else:
        edge_index = edge_attr = None
    
    data = Data(x=x, atom_elements = atom_element_name_list, edge_index=edge_index, edge_attr=edge_attr)
    return data


class AtomEncoder(nn.Module):
    def __init__(self, emb_dim, max_norm: float = 10.0,
                 use_all_atom_features: bool = False):
        super(AtomEncoder, self).__init__()
        # before: richer input featurization that also consists information about topology of graph like degree etc.
        FULL_ATOM_FEATURE_DIMS = get_atom_feature_dims()
        if not use_all_atom_features:
            # now: only atom type
            FULL_ATOM_FEATURE_DIMS = [FULL_ATOM_FEATURE_DIMS[0]]
        self.atom_embedding_list = nn.ModuleList()
        for dim in FULL_ATOM_FEATURE_DIMS:
            emb = nn.Embedding(dim, emb_dim, max_norm=max_norm)
            self.atom_embedding_list.append(emb)
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.atom_embedding_list:
            nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x):
        x_embedding = 0
        for i in range(len(self.atom_embedding_list)):
            x_embedding += self.atom_embedding_list[i](x[:, i])
        return x_embedding

class BondEncoderOHE(nn.Module):
    def __init__(self, emb_dim, max_num_classes: int):
        super(BondEncoderOHE, self).__init__()
        self.linear = nn.Linear(max_num_classes, emb_dim)
        self.reset_parameters()
    def reset_parameters(self):
        self.linear.reset_parameters
    def forward(self, edge_attr):
        bond_embedding = self.linear(edge_attr)
        return bond_embedding
    
    
class BondEncoder(nn.Module):
    def __init__(self, emb_dim, max_norm: float = 10.0):
        super(BondEncoder, self).__init__()
        FULL_BOND_FEATURE_DIMS = get_bond_feature_dims()
        self.bond_embedding = nn.Embedding(FULL_BOND_FEATURE_DIMS[0] + 3, emb_dim, max_norm=max_norm)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bond_embedding.weight.data)
    def forward(self, edge_attr):
        bond_embedding = self.bond_embedding(edge_attr)
        return bond_embedding
    
    
if __name__ == '__main__':
    FULL_BOND_FEATURE_DIMS = get_bond_feature_dims()
    bond_attrs = FULL_BOND_FEATURE_DIMS[0] + 3
    from torch_sparse import coalesce
    from torch_cluster import radius_graph
    smol = "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5"
    data = smiles_or_mol_to_graph(smol)
    print(data)
    print(data.x)
    print(data.atom_elements)
    
    smol1 = "CC(=O)Oc1ccccc1C(=O)O"
    smol2 = "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12"

    datalist = [smiles_or_mol_to_graph(s) for s in [smol, smol1, smol2]]
    data = Batch.from_data_list(datalist)
    bond_edge_index, bond_edge_attr, batch = data.edge_index, data.edge_attr, data.batch
    
    
    atomencoder = AtomEncoder(emb_dim=16)
    bondencoder = BondEncoder(emb_dim=16)
    
    x = atomencoder(data.x)
    edge_attr = bondencoder(data.edge_attr)
    
    pos = torch.randn(data.x.size(0), 3)
    bond_edge_index, bond_edge_attr = data.edge_index, data.edge_attr
    radius_edge_index = radius_graph(pos, r=5.0, max_num_neighbors=64, flow="source_to_target")
    # fromr radius_edge_index remove that are in bond_edge_index
    
    radius_feat = FULL_BOND_FEATURE_DIMS[0] + 1
    radius_edge_attr = torch.full((radius_edge_index.size(1), ), fill_value=radius_feat, device=pos.device, dtype=torch.long)
    # need to combine radius-edge-index with graph-edge-index

    nbonds = bond_edge_index.size(1)
    nradius = radius_edge_index.size(1)
    
    combined_edge_index = torch.cat([bond_edge_index, radius_edge_index], dim=-1)
    combined_edge_attr = torch.cat([bond_edge_attr, radius_edge_attr], dim=0)
    
    nbefore = combined_edge_index.size(1) 
    # coalesce
    combined_edge_index, combined_edge_attr = coalesce(index=combined_edge_index, value=combined_edge_attr, m=pos.size(0), n=pos.size(0), op="min")
    print(combined_edge_index[:, :30])
    print()
    print(combined_edge_attr[:30])