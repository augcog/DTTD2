#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 eval.py --dataset_root /ssd/zixun/Documents/DigitalTwin-6DPose_cleaned/estimation/dataset/dttd_iphone/DTTD_IPhone_Dataset/root\
                    --model /home/zixun/Documents/Robust-Digital-Twin-Tracking/checkpoints/m8p4_filter.pth\
                    --base_latent 256 --embed_dim 512 --fusion_block_num 1 --layer_num_m 8 --layer_num_p 4\
                    --visualize --output eval_results_m8p4_model_filtered_best\
                    --filter #--debug