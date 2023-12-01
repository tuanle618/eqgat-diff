import argparse
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from experiments.data.distributions import DistributionProperty

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)
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
    save_xyz=True,
    calculate_energy=False,
    batch_size=2,
    use_ligand_dataset_sizes=False,
    build_obabel_mol=False,
    save_traj=False,
    use_energy_guidance=False,
    ckpt_energy_model=None,
    guidance_scale=1.0e-4,
    ddpm=True,
    eta_ddim=1.0,
):
    print("Loading from checkpoint; adapting hyperparameters to specified args")

    # load model
    ckpt = torch.load(model_path)
    ckpt["hyper_parameters"]["load_ckpt"] = None
    ckpt["hyper_parameters"]["load_ckpt_from_pretrained"] = None
    ckpt["hyper_parameters"]["test_save_dir"] = save_dir
    ckpt["hyper_parameters"]["calculate_energy"] = calculate_energy
    ckpt["hyper_parameters"]["save_xyz"] = save_xyz
    ckpt["hyper_parameters"]["batch_size"] = batch_size
    ckpt["hyper_parameters"]["select_train_subset"] = False
    ckpt["hyper_parameters"]["diffusion_pretraining"] = False
    ckpt["hyper_parameters"]["gpus"] = 1
    ckpt["hyper_parameters"]["use_ligand_dataset_sizes"] = use_ligand_dataset_sizes
    ckpt["hyper_parameters"]["build_obabel_mol"] = build_obabel_mol
    ckpt["hyper_parameters"]["save_traj"] = save_traj
    ckpt["hyper_parameters"]["num_charge_classes"] = 6

    ckpt_path = os.path.join(save_dir, f"test_model.ckpt")
    if not os.path.exists(ckpt_path):
        torch.save(ckpt, ckpt_path)

    hparams = ckpt["hyper_parameters"]
    hparams = dotdict(hparams)

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
        if hparams.diffusion_pretraining:
            print("Starting pre-training")
            if hparams.additional_feats:
                from experiments.diffusion_pretrain_discrete_addfeats import (
                    Trainer,
                )
            else:
                from experiments.diffusion_pretrain_discrete import Trainer
        elif hparams.additional_feats:
            if dataset == "crossdocked":
                print("Ligand-pocket testing using additional features")
                from experiments.diffusion_discrete_moreFeats_ligand import Trainer
            else:
                print("Using additional features")
                from experiments.diffusion_discrete_moreFeats import Trainer
        else:
            if dataset == "crossdocked":
                print("Ligand-pocket testing")
                histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
                histogram = np.load(histogram).tolist()
                from experiments.diffusion_discrete_pocket import Trainer
            else:
                from experiments.diffusion_discrete import Trainer

    if build_obabel_mol:
        print(
            "Sampled molecules will be built with OpenBabel (without bond information)!"
        )

    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=1,
        strategy="auto",
        num_nodes=1,
        precision=hparams.precision,
    )
    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)

    model = Trainer(
        hparams=hparams,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        prop_dist=prop_dist,
        prop_norm=prop_norm,
        histogram=histogram,
    )

    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Data generation')
    parser.add_argument('--model-path', default="----", type=str,
                        help='Path to trained model')
    parser.add_argument("--use-energy-guidance", default=False, action="store_true")
    parser.add_argument("--use-ligand-dataset-sizes", default=False, action="store_true")
    parser.add_argument("--build-obabel-mol", default=False, action="store_true")
    parser.add_argument("--save-traj", default=False, action="store_true")
    parser.add_argument("--ckpt-energy-model", default=None, type=str)
    parser.add_argument('--guidance-scale', default=1.0e-4, type=float,
                        help='How to scale the guidance shift')
    parser.add_argument('--save-dir', default="----", type=str,
                        help='Path to test output')
    parser.add_argument('--save-xyz', default=False, action="store_true",
                        help='Whether or not to store generated molecules in xyz files')
    parser.add_argument('--calculate-energy', default=False, action="store_true",
                        help='Whether or not to calculate xTB energies and forces')
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
    # Evaluate negative log-likelihood for the test partitions
    evaluate(
        model_path=args.model_path,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        ddpm=not args.ddim,
        eta_ddim=args.eta_ddim,
        save_xyz=args.save_xyz,
        save_traj=args.save_traj,
        calculate_energy=args.calculate_energy,
        use_energy_guidance=args.use_energy_guidance,
        use_ligand_dataset_sizes=args.use_ligand_dataset_sizes,
        build_obabel_mol=args.build_obabel_mol,
        ckpt_energy_model=args.ckpt_energy_model,
        guidance_scale=args.guidance_scale,
    )
