import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser
from tqdm import tqdm

from experiments.data.distributions import DistributionProperty
from experiments.docking import calculate_qvina2_score
from experiments.sampling.analyze import analyze_stability_for_molecules
from experiments.utils import prepare_pocket, write_sdf_file

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def evaluate(
    model_path,
    save_dir,
    test_dir,
    test_list,
    skip_existing,
    fix_n_nodes,
    num_ligands_per_pocket,
    build_obabel_mol,
    batch_size,
    ddpm,
    eta_ddim,
    relax_mol,
    max_relax_iter,
    sanitize,
    write_dict,
    write_csv,
    pdbqt_dir,
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams["num_charge_classes"] = 6
    hparams = dotdict(hparams)

    hparams.load_ckpt_from_pretrained = None
    hparams.load_ckpt = None
    hparams.gpus = 1

    print(f"Loading {hparams.dataset} Datamodule.")
    dataset = "crossdocked"
    if hparams.use_adaptive_loader:
        print("Using adaptive dataloader")
        from experiments.data.ligand.ligand_dataset_adaptive import (
            LigandPocketDataModule as DataModule,
        )
    else:
        print("Using non-adaptive dataloader")
        from experiments.data.ligand.ligand_dataset_nonadaptive import (
            LigandPocketDataModule as DataModule,
        )

    datamodule = DataModule(hparams)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)
    histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
    histogram = np.load(histogram).tolist()
    train_smiles = list(datamodule.train_dataset.smiles)

    prop_norm, prop_dist = None, None
    if len(hparams.properties_list) > 0 and hparams.context_mapping:
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    from experiments.diffusion_discrete_pocket import Trainer

    # if you want bond_model_guidance, flag this here in the Trainer
    device = "cuda"
    model = Trainer.load_from_checkpoint(
        model_path,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        histogram=histogram,
        prop_norm=prop_norm,
        prop_dist=prop_dist,
        load_ckpt_from_pretrained=None,
        load_ckpt=None,
        # energy_model_guidance=True if use_energy_guidance else False,
        # ckpt_energy_model=ckpt_energy_model,
        run_evaluation=True,
        strict=False,
    ).to(device)
    model = model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir.mkdir(exist_ok=skip_existing)
    raw_sdf_dir = Path(save_dir, "raw")
    raw_sdf_dir.mkdir(exist_ok=skip_existing)
    processed_sdf_dir = Path(save_dir, "processed")
    processed_sdf_dir.mkdir(exist_ok=skip_existing)
    times_dir = Path(save_dir, "pocket_times")
    times_dir.mkdir(exist_ok=skip_existing)

    test_files = list(test_dir.glob("[!.]*.sdf"))
    if test_list is not None:
        with open(test_list, "r") as f:
            test_list = set(f.read().split(","))
        test_files = [x for x in test_files if x.stem in test_list]

    pbar = tqdm(test_files)
    time_per_pocket = {}

    statistics_dict = {"QED": [], "SA": [], "Lipinski": [], "Diversity": []}
    sdf_files = []

    if build_obabel_mol:
        print(
            "Sampled molecules will be built with OpenBabel (without bond information)!"
        )
    print("\nStarting sampling...\n")
    for sdf_file in pbar:
        ligand_name = sdf_file.stem

        pdb_name, pocket_id, *suffix = ligand_name.split("_")
        pdb_file = Path(sdf_file.parent, f"{pdb_name}.pdb")
        txt_file = Path(sdf_file.parent, f"{ligand_name}.txt")
        sdf_out_file_raw = Path(raw_sdf_dir, f"{ligand_name}_gen.sdf")
        sdf_out_file_processed = Path(processed_sdf_dir, f"{ligand_name}_gen.sdf")
        time_file = Path(times_dir, f"{ligand_name}.txt")

        if (
            skip_existing
            and time_file.exists()
            and sdf_out_file_processed.exists()
            and sdf_out_file_raw.exists()
        ):
            with open(time_file, "r") as f:
                time_per_pocket[str(sdf_file)] = float(f.read().split()[1])

            continue

        t_pocket_start = time()

        with open(txt_file, "r") as f:
            resi_list = f.read().split()

        pdb_struct = PDBParser(QUIET=True).get_structure("", pdb_file)[0]
        if resi_list is not None:
            # define pocket with list of residues
            residues = [
                pdb_struct[x.split(":")[0]][(" ", int(x.split(":")[1]), " ")]
                for x in resi_list
            ]

        all_molecules = 0
        tmp_molecules = []

        pocket_data = prepare_pocket(
            residues,
            dataset_info.atom_encoder,
            no_H=True,
            repeats=batch_size,
            device=device,
        )
        start = datetime.now()
        while len(tmp_molecules) < num_ligands_per_pocket:
            molecules = model.generate_ligands(
                pocket_data,
                num_graphs=batch_size,
                fix_n_nodes=fix_n_nodes,
                build_obabel_mol=build_obabel_mol,
                inner_verbose=False,
                save_traj=False,
                ddpm=ddpm,
                eta_ddim=eta_ddim,
                relax_mol=relax_mol,
                max_relax_iter=max_relax_iter,
                sanitize=sanitize,
            )
            all_molecules += len(molecules)
            valid_molecules = analyze_stability_for_molecules(
                molecule_list=molecules,
                dataset_info=dataset_info,
                smiles_train=train_smiles,
                local_rank=0,
                return_molecules=True,
                calculate_statistics=False,
                remove_hs=hparams.remove_hs,
                device="cpu",
            )
            tmp_molecules.extend(valid_molecules)

        run_time = datetime.now() - start
        print(f"\n Run time={run_time} for 100 valid molecules \n")

        (
            _,
            _,
            statistics,
            _,
            _,
            valid_molecules,
        ) = analyze_stability_for_molecules(
            molecule_list=tmp_molecules,
            dataset_info=dataset_info,
            smiles_train=train_smiles,
            local_rank=0,
            return_molecules=True,
            return_stats_per_molecule=True,
            remove_hs=hparams.remove_hs,
            device="cpu",
        )
        statistics_dict["QED"].append(statistics["QED"])
        statistics_dict["SA"].append(statistics["SA"])
        statistics_dict["Lipinski"].append(statistics["Lipinski"])
        statistics_dict["Diversity"].append(statistics["Diversity"])

        valid_molecules = valid_molecules[
            :num_ligands_per_pocket
        ]  # we could sort them by QED, SA or whatever

        write_sdf_file(sdf_out_file_raw, valid_molecules)
        sdf_files.append(sdf_out_file_raw)

        # Time the sampling process
        time_per_pocket[str(sdf_file)] = time() - t_pocket_start
        with open(time_file, "w") as f:
            f.write(f"{str(sdf_file)} {time_per_pocket[str(sdf_file)]}")

        pbar.set_description(
            f"Last processed: {ligand_name}. "
            f"{(time() - t_pocket_start) / all_molecules:.2f} "
            f"sec/mol."
        )

    statistics_dict["QED"] = np.mean(statistics_dict["QED"])
    statistics_dict["SA"] = np.mean(statistics_dict["SA"])
    statistics_dict["Lipinski"] = np.mean(statistics_dict["Lipinski"])
    statistics_dict["Diversity"] = np.mean(statistics_dict["Diversity"])

    with open(Path(save_dir, "pocket_times.txt"), "w") as f:
        for k, v in time_per_pocket.items():
            f.write(f"{k} {v}\n")

    times_arr = torch.tensor([x for x in time_per_pocket.values()])
    print(
        f"Time per pocket: {times_arr.mean():.3f} \pm "
        f"{times_arr.std(unbiased=False):.2f}"
    )
    print("Sampling finished.")

    ############## DOCKING ##############
    print("Starting docking...")
    results = {"receptor": [], "ligand": [], "scores": []}
    results_dict = {}

    pbar = tqdm(sdf_files)

    for sdf_file in pbar:
        pbar.set_description(f"Processing {sdf_file.name}")

        if dataset == "moad":
            """
            Ligand file names should be of the following form:
            <receptor-name>_<pocket-id>_<some-suffix>.sdf
            where <receptor-name> and <pocket-id> cannot contain any
            underscores, e.g.: 1abc-bio1_pocket0_gen.sdf
            """
            ligand_name = sdf_file.stem
            receptor_name, pocket_id, *suffix = ligand_name.split("_")
            suffix = "_".join(suffix)
            receptor_file = Path(pdbqt_dir, receptor_name + ".pdbqt")
        elif dataset == "crossdocked":
            ligand_name = sdf_file.stem
            receptor_name = ligand_name.split("_")[0]
            receptor_file = Path(pdbqt_dir, receptor_name + ".pdbqt")

        # try:
        scores, rdmols = calculate_qvina2_score(
            receptor_file, sdf_file, save_dir, return_rdmol=True
        )
        # except AttributeError as e:
        #     print(e)
        #     continue
        results["receptor"].append(str(receptor_file))
        results["ligand"].append(str(sdf_file))
        results["scores"].append(scores)

        if write_dict:
            results_dict[ligand_name] = {
                "receptor": str(receptor_file),
                "ligand": str(sdf_file),
                "scores": scores,
                "rmdols": rdmols,
            }

    if write_csv:
        df = pd.DataFrame.from_dict(results)
        df.to_csv(Path(save_dir, "qvina2_scores.csv"))

    if write_dict:
        torch.save(results_dict, Path(save_dir, "qvina2_scores.pt"))

    scores_fl = [r[0] for r in results["scores"] if len(r) >= 1]

    print(f"Mean statistics across all sampled ligands: {statistics_dict}")

    missing = len(results["scores"]) - len(scores_fl)
    print(f"Number of dockings evaluated with NaN: {missing}")

    mean_score = np.mean(scores_fl)
    std_score = np.std(scores_fl)
    print(f"Mean score: {mean_score}")
    print(f"Standard deviation: {std_score}")

    scores_fl.sort(reverse=False)
    mean_top10_score = np.mean(scores_fl[:10])
    print(f"Top-10 mean score: {mean_top10_score}")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="/hpfs/userws/cremej01/workspace/logs/aqm_qm7x/x0_t_weighting_dip_mpol/best_mol_stab.ckpt", type=str,
                        help='Path to trained model')
    parser.add_argument("--use-energy-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument('--guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-energy', default=False, action="store_true",
                        help='Whether or not to calculate xTB energies and forces')
    parser.add_argument('--num-ligands-per-pocket', default=100, type=int,
                            help='How many ligands per pocket to sample. Defaults to 10')
    parser.add_argument("--build-obabel-mol", default=False, action="store_true")
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--ddim', default=False, action="store_true",
                        help='If DDIM sampling should be used. Defaults to False')
    parser.add_argument('--eta-ddim', default=1.0, type=float,
                        help='How to scale the std of noise in the reverse posterior. \
                            Can also be used for DDPM to track a deterministic trajectory. \
                            Defaults to 1.0')
    parser.add_argument("--relax-mol", default=False, action="store_true")
    parser.add_argument("--sanitize", default=False, action="store_true")
    parser.add_argument('--max-relax-iter', default=200, type=int,
                            help='How many iteration steps for UFF optimization')
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--test-list", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--fix-n-nodes", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--write-dict", action="store_true")
    parser.add_argument("--pdbqt-dir", type=Path, help="Receptor files in pdbqt format")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Evaluate negative log-likelihood for the test partitions
    evaluate(
        model_path=args.model_path,
        test_dir=args.test_dir,
        test_list=args.test_list,
        save_dir=args.save_dir,
        skip_existing=args.skip_existing,
        fix_n_nodes=args.fix_n_nodes,
        batch_size=args.batch_size,
        num_ligands_per_pocket=args.num_ligands_per_pocket,
        build_obabel_mol=args.build_obabel_mol,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
        write_dict=args.write_dict,
        write_csv=args.write_csv,
        pdbqt_dir=args.pdbqt_dir,
        relax_mol=args.relax_mol,
        max_relax_iter=args.max_relax_iter,
        sanitize=args.sanitize,
    )
