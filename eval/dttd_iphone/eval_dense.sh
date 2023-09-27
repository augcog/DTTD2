#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 eval_iphone_densefusion.py --dataset_root /ssd/zixun/Documents/DigitalTwin-6DPose_cleaned/estimation/dataset/dttd_iphone/DTTD_IPhone_Dataset/root\
                    --model /home/zixun/Documents/DigitalTwin-6DPose_cleaned/estimation/result/densefusion_iphone_continuing/checkpoints\
                    --output eval_results_dense\
                    --visualize