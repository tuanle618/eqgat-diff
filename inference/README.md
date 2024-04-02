## Inference

We provide an exemplary jupyter notebook as well as bash script to generate ligands for QM9 and Geom-Drugs.
Notice that the model weights for each dataset are not provided.
As of now, we share the model weights upon request.

To run the bash script, execute the following:

```bash
mamba activate eqgatdiff
bash run_eval_geom.sh
bash run_eval_qm9.sh
```

In each bash script, the paths to dataset-root, model-dir and save-dir have to included.
E.g. in case of geom-drugs:

````
export PYTHONPATH="YOUR_ABSOLUTE_PATH/eqgat-diff/eqgat_diff"
dataset_root="YOUR_ABSOLUTE_PATH/eqgat-diff/data/geom"
model_path="YOUR_ABSOLUTE_PATH/eqgat-diff/weights/geom/best_mol_stab.ckpt"
save_dir="YOUR_ABSOLUTE_PATH/eqgat-diff/inference/tmp/geom/gen_samples"

python YOUR_ABSOLUTE_PATH/eqgat_diff/experiments/run_evaluation.py \
    --dataset drugs \
    --dataset-root $dataset_root \
    --model-path $model_path \
    --save-dir $save_dir \
    --batch-size 50 \
    --ngraphs 100 \
````

Depending on your GPU, you can increase the batch-size to have more molecules in a generated batch.