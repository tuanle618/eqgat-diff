import itertools
from collections import Counter
from multiprocessing import Pool

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from rdkit.Chem.QED import qed
from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
from torchmetrics import MaxMetric, MeanMetric
from tqdm import tqdm

from experiments.sampling.utils import *
from experiments.sampling.utils import calculateScore

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class BasicMolecularMetrics(object):
    def __init__(self, dataset_info, smiles_train=None, test=False, device="cpu"):
        self.atom_decoder = (
            dataset_info["atom_decoder"]
            if isinstance(dataset_info, dict)
            else dataset_info.atom_decoder
        )
        self.dataset_info = dataset_info

        self.number_samples = 0  # update based on unique generated smiles
        self.train_smiles, _ = canonicalize_list(smiles_train)

        self.train_fps = get_fingerprints_from_smileslist(self.train_smiles)
        self.test = test

        self.atom_stable = MeanMetric().to(device)
        self.mol_stable = MeanMetric().to(device)

        # Retrieve dataset smiles.
        self.validity_metric = MeanMetric().to(device)
        self.uniqueness = MeanMetric().to(device)
        self.novelty = MeanMetric().to(device)
        self.mean_components = MeanMetric().to(device)
        self.max_components = MaxMetric().to(device)
        self.num_nodes_w1 = MeanMetric().to(device)
        self.atom_types_tv = MeanMetric().to(device)
        self.edge_types_tv = MeanMetric().to(device)
        self.charge_w1 = MeanMetric().to(device)
        self.valency_w1 = MeanMetric().to(device)
        self.bond_lengths_w1 = MeanMetric().to(device)
        self.angles_w1 = MeanMetric().to(device)

        self.pc_descriptor_subset = [
            "BertzCT",
            "MolLogP",
            "MolWt",
            "TPSA",
            "NumHAcceptors",
            "NumHDonors",
            "NumRotatableBonds",
            "NumAliphaticRings",
            "NumAromaticRings",
        ]

    def reset(self):
        for metric in [
            self.atom_stable,
            self.mol_stable,
            self.validity_metric,
            self.uniqueness,
            self.novelty,
            self.mean_components,
            self.max_components,
            self.num_nodes_w1,
            self.atom_types_tv,
            self.edge_types_tv,
            self.charge_w1,
            self.valency_w1,
            self.bond_lengths_w1,
            self.angles_w1,
        ]:
            metric.reset()

    def compute_validity(self, generated, local_rank=0):
        """generated: list of couples (positions, atom_types)"""
        valid_smiles = []
        valid_ids = []
        valid_molecules = []
        num_components = []
        error_message = Counter()
        for i, mol in enumerate(generated):
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(
                        rdmol, asMols=True, sanitizeFrags=False
                    )
                    num_components.append(len(mol_frags))
                    if len(mol_frags) > 1:
                        error_message[4] += 1
                    else:
                        largest_mol = max(
                            mol_frags, default=mol, key=lambda m: m.GetNumAtoms()
                        )
                        Chem.SanitizeMol(largest_mol)
                        smiles = Chem.MolToSmiles(largest_mol)
                        valid_molecules.append(generated[i])
                        valid_smiles.append(smiles)
                        valid_ids.append(i)
                        error_message[-1] += 1
                except Chem.rdchem.AtomValenceException:
                    error_message[1] += 1
                    # print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    error_message[2] += 1
                    # print("Can't kekulize molecule")
                except Chem.rdchem.AtomKekulizeException or ValueError:
                    error_message[3] += 1
        if local_rank == 0:
            print(
                f"Error messages: AtomValence {error_message[1]}, Kekulize {error_message[2]}, other {error_message[3]}, "
                f" -- No error {error_message[-1]}"
            )
        self.validity_metric.update(
            value=len(valid_smiles) / len(generated), weight=len(generated)
        )
        num_components = torch.tensor(
            num_components, device=self.mean_components.device
        )
        self.mean_components.update(num_components)
        self.max_components.update(num_components)
        not_connected = 100.0 * error_message[4] / len(generated)
        connected_components = 100.0 - not_connected

        valid_smiles, duplicate_ids = canonicalize_list(valid_smiles)
        valid_molecules = [
            mol for i, mol in enumerate(valid_molecules) if i not in duplicate_ids
        ]

        return valid_smiles, valid_molecules, connected_components, error_message

    def compute_sanitize_validity(self, generated):
        if len(generated) < 1:
            return -1.0

        valid = []
        for mol in generated:
            rdmol = mol.rdkit_mol
            if rdmol is not None:
                try:
                    Chem.SanitizeMol(rdmol)
                except ValueError:
                    continue

                valid.append(rdmol)

        return len(valid) / len(generated)

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.train_smiles is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.train_smiles:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated, local_rank):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        # Validity
        (
            valid_smiles,
            valid_molecules,
            connected_components,
            error_message,
        ) = self.compute_validity(generated, local_rank=local_rank)

        validity = self.validity_metric.compute()
        uniqueness, novelty = 0, 0
        mean_components = self.mean_components.compute()
        max_components = self.max_components.compute()

        # Uniqueness
        if len(valid_smiles) > 0:
            unique = list(set(valid_smiles))
            self.uniqueness.update(
                value=len(unique) / len(valid_smiles), weight=len(valid_smiles)
            )
            uniqueness = self.uniqueness.compute()

            if self.train_smiles is not None:
                novel = []
                for smiles in unique:
                    if smiles not in self.train_smiles:
                        novel.append(smiles)
                self.novelty.update(value=len(novel) / len(unique), weight=len(unique))
            novelty = self.novelty.compute()

        num_molecules = int(self.validity_metric.weight.item())
        if local_rank == 0:
            print(
                f"Validity over {num_molecules} molecules:" f" {validity * 100 :.2f}%"
            )
            print(
                f"Number of connected components of {num_molecules} molecules: "
                f"mean:{mean_components:.2f} max:{max_components:.2f}"
            )
            print(
                f"Connected components of {num_molecules} molecules: "
                f"{connected_components:.2f}"
            )

        return (
            valid_smiles,
            valid_molecules,
            validity,
            novelty,
            uniqueness,
            connected_components,
        )

    def __call__(
        self,
        molecules: list,
        local_rank=0,
        remove_hs=False,
        return_molecules=False,
        return_stats_per_molecule=False,
        calculate_statistics=True,
    ):
        stable_molecules = []
        if not remove_hs:
            # Atom and molecule stability
            if local_rank == 0:
                print(f"Analyzing molecule stability")
            for i, mol in enumerate(molecules):
                mol_stable, at_stable, num_bonds = check_stability(
                    mol, self.dataset_info
                )
                self.mol_stable.update(value=mol_stable)
                self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)
                if mol_stable:
                    stable_molecules.append(mol)
            stability_dict = {
                "mol_stable": self.mol_stable.compute().item(),
                "atm_stable": self.atom_stable.compute().item(),
            }
        else:
            stability_dict = {}
            if local_rank == 0:
                print(f"No explicit hydrogens - skipping molecule stability metric")

        # Validity, uniqueness, novelty
        (
            valid_smiles,
            valid_molecules,
            validity,
            novelty,
            uniqueness,
            connected_components,
        ) = self.evaluate(molecules, local_rank=local_rank)
        # Save in any case in the graphs folder

        sanitize_validity = self.compute_sanitize_validity(molecules)

        novelty = novelty if isinstance(novelty, int) else novelty.item()
        uniqueness = uniqueness if isinstance(uniqueness, int) else uniqueness.item()

        validity_dict = {
            "validity": validity.item(),
            "sanitize_validity": sanitize_validity,
            "novelty": novelty,
            "uniqueness": uniqueness,
        }

        if calculate_statistics:
            statistics_dict = self.compute_statistics(molecules, local_rank)
            statistics_dict["connected_components"] = connected_components

            self.number_samples = len(valid_smiles)

            self.train_subset = (
                get_random_subset(self.train_smiles, self.number_samples, seed=42)
                if len(valid_smiles) <= len(self.train_smiles)
                else self.train_smiles
            )
            similarity = self.get_bulk_similarity_with_train(valid_smiles)
            diversity = self.get_bulk_diversity(valid_smiles)
            if len(valid_smiles) > 0:
                try:
                    kl_score = self.get_kl_divergence(valid_smiles)
                except:
                    print("kl_score could not be calculated. Setting kl_score to -1")
                    kl_score = -1.0
            else:
                print("No valid smiles have been generated. Setting kl_score to -1")
                kl_score = -1.0
            statistics_dict["bulk_similarity"] = similarity
            statistics_dict["bulk_diversity"] = diversity
            statistics_dict["kl_score"] = kl_score

            if len(valid_smiles) > 0:
                mols = get_mols_list(valid_smiles)
                # rings = np.mean([num_rings(mol) for mol in mols])
                # aromatic_rings = np.mean([num_aromatic_rings(mol) for mol in mols])
                qed, sa, logp, lipinski, diversity = self.evaluate_mean(mols)
            else:
                print("No valid smiles have been generated. Setting scores to -1")
                qed = -1.0
                sa = -1.0
                logp = -1.0
                lipinski = -1.0
                diversity = -1.0
            statistics_dict["QED"] = qed
            statistics_dict["SA"] = sa
            statistics_dict["LogP"] = logp
            statistics_dict["Lipinski"] = lipinski
            statistics_dict["Diversity"] = diversity

            if return_stats_per_molecule:
                qed, sa, logp, lipinski = self.evaluate_per_mol(mols)
                statistics_dict["QEDs"] = qed
                statistics_dict["SAs"] = sa
                statistics_dict["LogPs"] = logp
                statistics_dict["Lipinskis"] = lipinski
        else:
            statistics_dict = None

        self.reset()

        if not return_molecules:
            valid_smiles = None
            stable_molecules = None
            valid_molecules = None
        return (
            stability_dict,
            validity_dict,
            statistics_dict,
            valid_smiles,
            stable_molecules,
            valid_molecules,
        )

    def compute_statistics(self, molecules, local_rank):
        # Compute statistics
        stat = (
            self.dataset_info.statistics["test"]
            if self.test
            else self.dataset_info.statistics["val"]
        )

        self.num_nodes_w1(number_nodes_distance(molecules, stat.num_nodes))

        atom_types_tv, atom_tv_per_class = atom_types_distance(
            molecules, stat.atom_types, save_histogram=self.test
        )
        self.atom_types_tv(atom_types_tv)
        edge_types_tv, bond_tv_per_class, sparsity_level = bond_types_distance(
            molecules, stat.bond_types, save_histogram=self.test
        )
        print(
            f"Sparsity level on local rank {local_rank}: {int(100 * sparsity_level)} %"
        )
        self.edge_types_tv(edge_types_tv)
        charge_w1, charge_w1_per_class = charge_distance(
            molecules, stat.charge_types, stat.atom_types, self.dataset_info
        )
        self.charge_w1(charge_w1)
        valency_w1, valency_w1_per_class = valency_distance(
            molecules, stat.valencies, stat.atom_types, self.dataset_info.atom_encoder
        )
        self.valency_w1(valency_w1)
        bond_lengths_w1, bond_lengths_w1_per_type = bond_length_distance(
            molecules, stat.bond_lengths, stat.bond_types
        )
        self.bond_lengths_w1(bond_lengths_w1)
        if sparsity_level < 0.7:
            if local_rank == 0:
                print(f"Too many edges, skipping angle distance computation.")
            angles_w1 = 0
            angles_w1_per_type = [-1] * len(self.dataset_info.atom_decoder)
        else:
            angles_w1, angles_w1_per_type = angle_distance(
                molecules,
                stat.bond_angles,
                stat.atom_types,
                stat.valencies,
                atom_decoder=self.dataset_info.atom_decoder,
                save_histogram=self.test,
            )
        self.angles_w1(angles_w1)
        statistics_log = {
            "sampling/NumNodesW1": self.num_nodes_w1.compute().item(),
            "sampling/AtomTypesTV": self.atom_types_tv.compute().item(),
            "sampling/EdgeTypesTV": self.edge_types_tv.compute().item(),
            "sampling/ChargeW1": self.charge_w1.compute().item(),
            "sampling/ValencyW1": self.valency_w1.compute().item(),
            "sampling/BondLengthsW1": self.bond_lengths_w1.compute().item(),
            "sampling/AnglesW1": self.angles_w1.compute().item(),
        }
        # if local_rank == 0:
        #     print(
        #         f"Sampling metrics",
        #         {key: round(val.item(), 3) for key, val in statistics_log},
        #     )

        sampling_per_class = False
        if sampling_per_class:
            for i, atom_type in enumerate(self.dataset_info.atom_decoder):
                statistics_log[
                    f"sampling_per_class/{atom_type}_TV"
                ] = atom_tv_per_class[i].item()
                statistics_log[
                    f"sampling_per_class/{atom_type}_ValencyW1"
                ] = valency_w1_per_class[i].item()
                statistics_log[f"sampling_per_class/{atom_type}_BondAnglesW1"] = (
                    angles_w1_per_type[i].item() if angles_w1_per_type[i] != -1 else -1
                )
                statistics_log[
                    f"sampling_per_class/{atom_type}_ChargesW1"
                ] = charge_w1_per_class[i].item()

            for j, bond_type in enumerate(
                ["No bond", "Single", "Double", "Triple", "Aromatic"]
            ):
                statistics_log[
                    f"sampling_per_class/{bond_type}_TV"
                ] = bond_tv_per_class[j].item()
                if j > 0:
                    statistics_log[
                        f"sampling_per_class/{bond_type}_BondLengthsW1"
                    ] = bond_lengths_w1_per_type[j - 1].item()

        return statistics_log

    def get_similarity_with_train(self, generated_smiles, parallel=False):
        fps = get_fingerprints_from_smileslist(generated_smiles)
        fp_pair = list(itertools.product(fps, self.train_fps))
        if not parallel:
            similarity_list = []
            for fg1, fg2 in tqdm(fp_pair, desc="Calculate similarity with train"):
                similarity_list.append(get_similarity((fg1, fg2)))
        else:
            with Pool(102) as pool:
                similarity_list = list(
                    tqdm(
                        pool.imap(get_similarity, fp_pair),
                        total=len(fps) * len(self.train_fps),
                    )
                )
        # calculate the max similarity of each mol with train data
        similarity_max = np.reshape(similarity_list, (len(generated_smiles), -1)).max(
            axis=1
        )
        return np.mean(similarity_max)

    def get_diversity(self, generated_smiles, parallel=False):
        fps = get_fingerprints_from_smileslist(generated_smiles)
        all_fp_pairs = list(itertools.combinations(fps, 2))
        if not parallel:
            similarity_list = []
            for fg1, fg2 in tqdm(all_fp_pairs, desc="Calculate diversity"):
                similarity_list.append(TanimotoSimilarity(fg1, fg2))
        else:
            with Pool(102) as pool:
                similarity_list = pool.imap_unordered(TanimotoSimilarity, all_fp_pairs)
        return 1 - np.mean(similarity_list)

    def get_bulk_similarity_with_train(self, generated_smiles):
        fps = get_fingerprints_from_smileslist(generated_smiles)
        scores = []

        for fp in fps:
            scores.append(BulkTanimotoSimilarity(fp, self.train_fps))
        return np.mean(scores)

    def get_bulk_diversity(self, generated_smiles):
        fps = get_fingerprints_from_smileslist(generated_smiles)
        scores = []
        for i, fp in enumerate(fps):
            fps_tmp = fps.copy()
            del fps_tmp[i]
            scores.append(BulkTanimotoSimilarity(fp, fps_tmp))
        return 1 - np.mean(scores)

    def get_kl_divergence(self, generated_smiles):
        # canonicalize_list in order to remove stereo information (also removes duplicates and invalid molecules, but there shouldn't be any)
        unique_molecules = set(
            canonicalize_list(generated_smiles, include_stereocenters=False)[0]
        )

        # first we calculate the descriptors, which are np.arrays of size n_samples x n_descriptors
        d_sampled = calculate_pc_descriptors(
            unique_molecules, self.pc_descriptor_subset
        )
        d_chembl = calculate_pc_descriptors(
            self.train_subset, self.pc_descriptor_subset
        )

        kldivs = {}

        # now we calculate the kl divergence for the float valued descriptors ...
        for i in range(4):
            kldiv = continuous_kldiv(
                X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i]
            )
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # ... and for the int valued ones.
        for i in range(4, 9):
            kldiv = discrete_kldiv(X_baseline=d_chembl[:, i], X_sampled=d_sampled[:, i])
            kldivs[self.pc_descriptor_subset[i]] = kldiv

        # pairwise similarity

        chembl_sim = calculate_internal_pairwise_similarities(self.train_subset)
        chembl_sim = chembl_sim.max(axis=1)

        sampled_sim = calculate_internal_pairwise_similarities(unique_molecules)
        sampled_sim = sampled_sim.max(axis=1)

        kldiv_int_int = continuous_kldiv(X_baseline=chembl_sim, X_sampled=sampled_sim)
        kldivs["internal_similarity"] = kldiv_int_int

        # Each KL divergence value is transformed to be in [0, 1].
        # Then their average delivers the final score.
        partial_scores = [np.exp(-score) for score in kldivs.values()]
        score = sum(partial_scores) / len(partial_scores)

        return score

    def calculate_qed(self, rdmol):
        return QED.qed(rdmol)

    def calculate_sa(self, rdmol):
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)  # from pocket2mol

    def calculate_logp(self, rdmol):
        return Crippen.MolLogP(rdmol)

    def calculate_lipinski(self, rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    def calculate_diversity(self, pocket_mols):
        if len(pocket_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(pocket_mols)):
            for j in range(i + 1, len(pocket_mols)):
                div += 1 - self.similarity(pocket_mols[i], pocket_mols[j])
                total += 1
        return div / total

    def similarity(self, mol_a, mol_b):
        # fp1 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_a, 2, nBits=2048, useChirality=False)
        # fp2 = AllChem.GetMorganFingerprintAsBitVect(
        #     mol_b, 2, nBits=2048, useChirality=False)
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def evaluate_mean(self, rdmols):
        """
        Run full evaluation and return mean of each property
        Args:
            rdmols: list of RDKit molecules
        Returns:
            QED, SA, LogP, Lipinski, and Diversity
        """

        if len(rdmols) < 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        for mol in rdmols:
            Chem.SanitizeMol(mol)
            assert mol is not None, "only evaluate valid molecules"

        qed = np.mean([self.calculate_qed(mol) for mol in rdmols])
        sa = np.mean([self.calculate_sa(mol) for mol in rdmols])
        logp = np.mean([self.calculate_logp(mol) for mol in rdmols])
        lipinski = np.mean([self.calculate_lipinski(mol) for mol in rdmols])
        diversity = self.calculate_diversity(rdmols)

        return qed, sa, logp, lipinski, diversity

    def evaluate_per_mol(self, rdmols):
        """
        Run full evaluation and return mean of each property
        Args:
            rdmols: list of RDKit molecules
        Returns:
            QED, SA, LogP, Lipinski, and Diversity
        """

        if len(rdmols) < 1:
            return -1.0, -1.0, -1.0, -1.0

        qed = [self.calculate_qed(mol) for mol in rdmols]
        sa = [self.calculate_sa(mol) for mol in rdmols]
        logp = [self.calculate_logp(mol) for mol in rdmols]
        lipinski = [self.calculate_lipinski(mol) for mol in rdmols]

        return qed, sa, logp, lipinski


def analyze_stability_for_molecules(
    molecule_list,
    dataset_info,
    smiles_train,
    local_rank,
    return_molecules=False,
    return_stats_per_molecule=False,
    remove_hs=False,
    device="cpu",
    calculate_statistics=True,
):
    metrics = BasicMolecularMetrics(
        dataset_info,
        smiles_train=smiles_train,
        device=device,
    )
    (
        stability_dict,
        validity_dict,
        statistics_dict,
        sampled_smiles,
        stable_molecules,
        valid_molecules,
    ) = metrics(
        molecule_list,
        local_rank=local_rank,
        remove_hs=remove_hs,
        return_stats_per_molecule=return_stats_per_molecule,
        return_molecules=return_molecules,
        calculate_statistics=calculate_statistics,
    )

    if calculate_statistics:
        return (
            stability_dict,
            validity_dict,
            statistics_dict,
            sampled_smiles,
            stable_molecules,
            valid_molecules,
        )
    else:
        return valid_molecules
