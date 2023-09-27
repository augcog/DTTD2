#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

if [ ! -d YCB_Video_toolbox ];then
    echo 'Downloading the YCB_Video_toolbox...'
    git clone https://github.com/yuxng/YCB_Video_toolbox.git
    cd YCB_Video_toolbox
    unzip results_PoseCNN_RSS2018.zip
    cd ..
    cp replace_ycb_toolbox/*.m YCB_Video_toolbox/
fi

# python3 eval_ycb_posecnn.py --dataset_root ../../dataset/ycb/YCB_Video_Dataset\
#                     --model ../../result/train_densefusion_adds/checkpoints/\
#                     --output eval_results\
#                     --visualize 

python3 eval_ycb_gt.py --dataset_root ../../dataset/ycb/YCB_Video_Dataset\
                    --model ../../result/train_densefusion_adds/checkpoints/\
                    --output eval_results\
                    --visualize 
