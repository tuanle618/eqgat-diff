#!/bin/sh

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