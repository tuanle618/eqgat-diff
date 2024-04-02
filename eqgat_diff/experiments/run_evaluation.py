import argparse
import os
import pickle
import warnings

import numpy as np
import torch
from tqdm import tqdm

from experiments.data.distributions import DistributionProperty
from experiments.data.utils import write_xyz_file
from experiments.xtb_energy import calculate_xtb_energy
from experiments.xtb_wrapper import xtb_calculate

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
    dataset,
    dataset_root,
    save_xyz=False,
    calculate_energy=False,
    calculate_props=False,
    use_energy_guidance=False,
    ckpt_energy_model=None,
    guidance_scale=1.0e-4,
    ngraphs=5000,
    save_traj=False,
    batch_size=80,
    step=0,
    ddpm=True,
    eta_ddim=1.0,
    fix_noise_and_nodes=False,
    guidance_steps=100,
    optimization="minimize",
    relax_sampling=False,
    relax_steps=10,
    sample_only_valid=False,
):
    # load hyperparameter
    hparams = torch.load(model_path)["hyper_parameters"]
    hparams["select_train_subset"] = False
    hparams["diffusion_pretraining"] = False
    hparams["num_charge_classes"] = 6
    hparams = dotdict(hparams)
    
    hparams.dataset = dataset
    hparams.dataset_root = dataset_root

    hparams.load_ckpt_from_pretrained = None
    hparams.load_ckpt = None
    hparams.gpus = 1

    print(f"Loading {hparams.dataset} Datamodule.")
    non_adaptive = True
    if hparams.dataset == "drugs":
        dataset = "drugs"
        if hparams.use_adaptive_loader:
            print("Using adaptive dataloader")
            non_adaptive = False
            from experiments.data.geom.geom_dataset_adaptive import (
                GeomDataModule as DataModule,
            )
        else:
            print("Using non-adaptive dataloader")
            from experiments.data.geom.geom_dataset_nonadaptive import (
                GeomDataModule as DataModule,
            )
    elif hparams.dataset == "qm9":
        dataset = "qm9"
        from experiments.data.qm9.qm9_dataset import QM9DataModule as DataModule

    elif hparams.dataset == "pubchem":
        dataset = "pubchem"  # take dataset infos from GEOM for simplicity
        if hparams.use_adaptive_loader:
            print("Using adaptive dataloader")
            non_adaptive = False
            from experiments.data.pubchem.pubchem_dataset_adaptive import (
                PubChemDataModule as DataModule,
            )
        else:
            print("Using non-adaptive dataloader")
            from experiments.data.pubchem.pubchem_dataset_nonadaptive import (
                PubChemDataModule as DataModule,
            )

    if dataset == "pubchem":
        datamodule = DataModule(hparams, evaluation=True)
    else:
        datamodule = DataModule(hparams, only_stats=True)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)

    train_smiles = (
        list(datamodule.train_dataset.smiles)
        if hparams.dataset != "pubchem"
        else datamodule.train_smiles
    )
    prop_norm, prop_dist = None, None
    if len(hparams.properties_list) > 0 and hparams.context_mapping:
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    if hparams.continuous:
        print("Using continuous diffusion")
        from experiments.diffusion_continuous import Trainer
    else:
        print("Using discrete diffusion")
        if hparams.additional_feats:
            print("Using additional features")
            from experiments.diffusion_discrete_addfeats import Trainer
        else:
            from experiments.diffusion_discrete import Trainer

    # if you want bond_model_guidance, flag this here in the Trainer
    device = "cuda"
    model = Trainer.load_from_checkpoint(
        model_path,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
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

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if sample_only_valid:
        print("\nStarting sampling of only valid molecules...\n")
        results_dict, generated_smiles, stable_molecules = model.generate_valid_samples(
            dataset_info=model.dataset_info,
            ngraphs=ngraphs,
            bs=batch_size,
            return_molecules=True,
            verbose=True,
            inner_verbose=True,
            save_dir=save_dir,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            save_traj=save_traj,
            guidance_scale=guidance_scale,
            use_energy_guidance=use_energy_guidance,
            ckpt_energy_model=ckpt_energy_model,
            fix_noise_and_nodes=fix_noise_and_nodes,
            guidance_steps=guidance_steps,
            optimization=optimization,
            relax_sampling=relax_sampling,
            relax_steps=relax_steps,
            device="cpu",
        )
    else:
        print("\nStarting sampling...\n")
        results_dict, generated_smiles, stable_molecules = model.run_evaluation(
            step=step,
            dataset_info=model.dataset_info,
            ngraphs=ngraphs,
            bs=batch_size,
            return_molecules=True,
            verbose=True,
            inner_verbose=True,
            save_dir=save_dir,
            ddpm=ddpm,
            eta_ddim=eta_ddim,
            run_test_eval=True,
            save_traj=save_traj,
            guidance_scale=guidance_scale,
            use_energy_guidance=use_energy_guidance,
            ckpt_energy_model=ckpt_energy_model,
            fix_noise_and_nodes=fix_noise_and_nodes,
            guidance_steps=guidance_steps,
            optimization=optimization,
            relax_sampling=relax_sampling,
            relax_steps=relax_steps,
            device="cpu",
        )

    print("\nFinished sampling!\n")
    atom_decoder = stable_molecules[0].atom_decoder

    if calculate_energy:
        energies = []
        forces_norms = []
        print("Calculating energies...")
        for i in range(len(stable_molecules)):
            atom_types = [atom_decoder[int(a)] for a in stable_molecules[i].atom_types]
            try:
                e, f = calculate_xtb_energy(stable_molecules[i].positions, atom_types)
            except:
                continue
            stable_molecules[i].energy = e
            stable_molecules[i].forces_norm = f
            energies.append(e)
            forces_norms.append(f)

        print(f"Mean energies: {np.mean(energies)}")
        print(f"Mean force norms: {np.mean(forces_norms)}")

    if calculate_props:
        polarizabilities = []
        sm = stable_molecules.copy()
        stable_molecules = []
        print("Calculating properties...")
        for mol in tqdm(sm):
            atom_types = [atom_decoder[int(a)] for a in mol.atom_types]

            if prop_dist is not None:
                for j, key in enumerate(hparams.properties_list):
                    mean, mad = (
                        prop_dist.normalizer[key]["mean"],
                        prop_dist.normalizer[key]["mad"],
                    )
                    prop = mol.context[j] * mad + mean
                    mol.context = float(prop)
            try:
                charge = mol.charges.sum().item()
                results = xtb_calculate(
                    atoms=atom_types,
                    coords=mol.positions.tolist(),
                    charge=charge,
                    options={"grad": True},
                )
                for key, value in results.items():
                    mol.__setattr__(key, value)

                polarizabilities.append(mol.polarizability)
                stable_molecules.append(mol)
            except Exception as e:
                print(e)
                continue
        print(f"Mean polarizability: {np.mean(polarizabilities)}")

    if save_xyz:
        context = []
        for i in range(len(stable_molecules)):
            types = [atom_decoder[int(a)] for a in stable_molecules[i].atom_types]
            write_xyz_file(
                stable_molecules[i].positions,
                types,
                os.path.join(save_dir, f"mol_{i}.xyz"),
            )
            if prop_dist is not None:
                tmp = []
                for j, key in enumerate(hparams.properties_list):
                    mean, mad = (
                        prop_dist.normalizer[key]["mean"],
                        prop_dist.normalizer[key]["mad"],
                    )
                    prop = stable_molecules[i].context[j] * mad + mean
                    tmp.append(float(prop))
                context.append(tmp)

    if prop_dist is not None and save_xyz:
        with open(os.path.join(save_dir, "context.pickle"), "wb") as f:
            pickle.dump(context, f)
    if calculate_energy:
        with open(os.path.join(save_dir, "energies.pickle"), "wb") as f:
            pickle.dump(energies, f)
        with open(os.path.join(save_dir, "forces_norms.pickle"), "wb") as f:
            pickle.dump(forces_norms, f)
    with open(os.path.join(save_dir, "generated_smiles.pickle"), "wb") as f:
        pickle.dump(generated_smiles, f)
    with open(os.path.join(save_dir, "stable_molecules.pickle"), "wb") as f:
        pickle.dump(stable_molecules, f)
    with open(os.path.join(save_dir, "evaluation.pickle"), "wb") as f:
        pickle.dump(results_dict, f)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="----", type=str,
                        help='Path to trained model')
    parser.add_argument('--dataset-root', default="----", type=str,
                        help='Path to dataset root model')
    parser.add_argument('--dataset', default="qm9", type=str,
                        help='Dataset to use')
    parser.add_argument("--sample-only-valid", default=False, action="store_true")
    parser.add_argument("--use-energy-guidance", default=False, action="store_true")
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument("--optimization", default="minimize", type=str, choices=["minimize", "maximize"])
    parser.add_argument('--guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--guidance-steps', default=100, type=int,
                        help='How many guidance steps')
    parser.add_argument('--save-dir', default="----", type=str,
                        help='Path to test output')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-energy', default=False, action="store_true",
                        help='Whether or not to calculate xTB energies and forces')
    parser.add_argument('--calculate-props', default=False, action="store_true",
                        help='Whether or not to calculate xTB properties')
    parser.add_argument('--save-traj', default=False, action="store_true",
                        help='Whether or not to save whole trajectory')
    parser.add_argument('--fix-noise-and-nodes', default=False, action="store_true",
                        help='Whether or not to fix noise, e.g., for interpolation or guidance')
    parser.add_argument('--relax-sampling', default=False, action="store_true",
                        help='Whether or not to relax using denoising with timestep 0')
    parser.add_argument('--relax-steps', default=10, type=int, help='How many denoising relaxation steps')
    parser.add_argument('--ngraphs', default=5000, type=int,
                            help='How many graphs to sample. Defaults to 5000')
    parser.add_argument('--batch-size', default=80, type=int,
                            help='Batch-size to generate the selected ngraphs. Defaults to 80.')
    parser.add_argument('--ddim', default=False, action="store_true",
                        help='If DDIM sampling should be used. Defaults to False')
    parser.add_argument('--eta-ddim', default=1.0, type=float,
                        help='How to scale the std of noise in the reverse posterior. \
                            Can also be used for DDPM to track a deterministic trajectory. \
                            Defaults to 1.0')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    evaluate(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        model_path=args.model_path,
        save_dir=args.save_dir,
        ngraphs=args.ngraphs,
        batch_size=args.batch_size,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
        save_xyz=args.save_xyz,
        save_traj=args.save_traj,
        calculate_energy=args.calculate_energy,
        calculate_props=args.calculate_props,
        use_energy_guidance=args.use_energy_guidance,
        ckpt_energy_model=args.ckpt_energy_model,
        guidance_steps=args.guidance_steps,
        guidance_scale=args.guidance_scale,
        fix_noise_and_nodes=args.fix_noise_and_nodes,
        optimization=args.optimization,
        relax_sampling=args.relax_sampling,
        relax_steps=args.relax_steps,
        sample_only_valid=args.sample_only_valid,
    )
