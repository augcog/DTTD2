#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 eval_dttd_gt.py --dataset_root ../../datasets/dttd/DTTD_Dataset\
                    --model ../../out/out_abc_dttd/checkpoints/\
                    --output eval_results\
                    --visualize