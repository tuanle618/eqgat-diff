import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks.ema import ExponentialMovingAverage

warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

from experiments.data.distributions import DistributionProperty
from experiments.hparams import add_arguments

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = add_arguments(parser)
    hparams = parser.parse_args()

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    if not os.path.isdir(hparams.save_dir + f"/run{hparams.id}/"):
        print("Creating directory")
        os.mkdir(hparams.save_dir + f"/run{hparams.id}/")
    print(f"Starting Run {hparams.id}")
    ema_callback = ExponentialMovingAverage(decay=hparams.ema_decay)
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.save_dir + f"/run{hparams.id}/",
        save_top_k=3,
        monitor="val/loss",
        save_last=True,
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger(
        hparams.save_dir + f"/run{hparams.id}/", default_hp_metric=False
    )

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
    elif hparams.dataset == "crossdocked":
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
    else:
        raise ValueError(f"Unknown dataset: {hparams.dataset}")

    datamodule = DataModule(hparams)

    from experiments.data.data_info import GeneralInfos as DataInfos

    dataset_info = DataInfos(datamodule, hparams)

    train_smiles = (
        (
            list(datamodule.train_dataset.smiles)
            if hparams.dataset != "pubchem"
            else None
        )
        if not hparams.select_train_subset
        else datamodule.train_smiles
    )
    prop_norm, prop_dist = None, None
    if (
        len(hparams.properties_list) > 0 and hparams.context_mapping
    ) or hparams.property_training:
        prop_norm = datamodule.compute_mean_mad(hparams.properties_list)
        prop_dist = DistributionProperty(datamodule, hparams.properties_list)
        prop_dist.set_normalizer(prop_norm)

    histogram = None

    if hparams.continuous and dataset != "crossdocked":
        print("Using continuous diffusion")
        if hparams.diffusion_pretraining:
            print("Starting pre-training")
            from experiments.diffusion_pretrain_continuous import Trainer
        else:
            from experiments.diffusion_continuous import Trainer
    else:
        print("Using discrete diffusion")
        if hparams.diffusion_pretraining:
            if hparams.additional_feats:
                print("Starting pre-training on PubChem3D with additional features")
                from experiments.diffusion_pretrain_discrete_addfeats import (
                    Trainer,
                )
            else:
                print("Starting pre-training on PubChem3D")
                from experiments.diffusion_pretrain_discrete import Trainer
        elif (
            dataset == "crossdocked"
            and hparams.additional_feats
            and not hparams.use_qm_props
        ):
            histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
            histogram = np.load(histogram).tolist()
            print("Ligand-pocket training using additional features")
            from experiments.diffusion_discrete_pocket_addfeats import (
                Trainer,
            )
        else:
            if dataset == "crossdocked":
                histogram = os.path.join(hparams.dataset_root, "size_distribution.npy")
                histogram = np.load(histogram).tolist()
                if hparams.continuous:
                    print("Continuous ligand-pocket training")
                    from experiments.diffusion_continuous_pocket import Trainer
                else:
                    print("Discrete ligand-pocket training")
                    from experiments.diffusion_discrete_pocket import (
                        Trainer,
                    )
            else:
                if hparams.additional_feats:
                    from experiments.diffusion_discrete_addfeats import Trainer
                else:
                    from experiments.diffusion_discrete import Trainer

    model = Trainer(
        hparams=hparams.__dict__,
        dataset_info=dataset_info,
        smiles_list=train_smiles,
        histogram=histogram,
        prop_dist=prop_dist,
        prop_norm=prop_norm,
    )

    from pytorch_lightning.plugins.environments import LightningEnvironment

    strategy = "ddp" if hparams.gpus > 1 else "auto"
    # strategy = 'ddp_find_unused_parameters_true'
    callbacks = [
        ema_callback,
        lr_logger,
        checkpoint_callback,
        TQDMProgressBar(refresh_rate=5),
        ModelSummary(max_depth=2),
    ]

    if hparams.ema_decay == 1.0:
        callbacks = callbacks[1:]

    trainer = pl.Trainer(
        accelerator="gpu" if hparams.gpus else "cpu",
        devices=hparams.gpus if hparams.gpus else 1,
        strategy=strategy,
        plugins=LightningEnvironment(),
        num_nodes=1,
        logger=tb_logger,
        enable_checkpointing=True,
        accumulate_grad_batches=hparams.accum_batch,
        val_check_interval=hparams.eval_freq,
        gradient_clip_val=hparams.grad_clip_val,
        callbacks=callbacks,
        precision=hparams.precision,
        num_sanity_val_steps=2,
        max_epochs=hparams.num_epochs,
        detect_anomaly=hparams.detect_anomaly,
    )

    pl.seed_everything(seed=hparams.seed, workers=hparams.gpus > 1)

    ckpt_path = None
    if hparams.load_ckpt is not None:
        print("Loading from checkpoint ...")
        import torch

        ckpt_path = hparams.load_ckpt
        ckpt = torch.load(ckpt_path)
        if ckpt["optimizer_states"][0]["param_groups"][0]["lr"] != hparams.lr:
            print("Changing learning rate ...")
            ckpt["optimizer_states"][0]["param_groups"][0]["lr"] = hparams.lr
            ckpt["optimizer_states"][0]["param_groups"][0]["initial_lr"] = hparams.lr
            ckpt_path = (
                "lr" + "_" + str(hparams.lr) + "_" + os.path.basename(hparams.load_ckpt)
            )
            ckpt_path = os.path.join(
                os.path.dirname(hparams.load_ckpt),
                f"retraining_with_lr{hparams.lr}.ckpt",
            )
            if not os.path.exists(ckpt_path):
                torch.save(ckpt, ckpt_path)
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=ckpt_path if hparams.load_ckpt is not None else None,
    )
