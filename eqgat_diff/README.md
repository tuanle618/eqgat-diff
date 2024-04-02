# E(3) Equivariant Diffusion for Molecules

Research repository exploring the capabalities of (continuous and discrete) denoising diffusion probabilistic models applied on molecular data.

## Installation
Best installed using mamba.
```bash
mamba env create -f environment.yml
```

## Experiments
 We will update the repository with the corresponding datasets to run the experiments.  

 Generally, in `experiments/run_train.py` the main training script is executed, while in `configs/*.yaml` the configuration files for each dataset are stored.

 An example training run can be executed with the following command

 ```bash
 mamba activate eqgatdiff
 export PYTHONPATH="YOUR_ABSOLUTE_PATH/eqgat-diff/eqgat_diff"
 python experiments/run_train --conf configs/my_config_file.yaml
 ```

 Note that currently we do not provide the datasets, hence the code is currently just for reviewing and how the model and training runs are implemented.

 For example, an EQGAT-diff model that leverages Gaussian diffusion on atomic coordinates, but discrete diffusion for atom- and bond-types is implemented in `experiments/diffusion_discrete.py`.

 The same model, that leverages Gaussian diffusion for atomic coordinates, atom- and bond-types is implemented in `experiments/diffusion_continuous.py`.

 All configurable hyperparameters are listed in `experiments/hparams.py`

 ## Inference and Weights

 Currently we are still in the progress of publishing the code. Upon request, we provide model weights of the QM9 and Geom-Drugs models.
 Please look into `inference/` and `weights/` subdirectory for more details.