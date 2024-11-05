# DTTDNet: Robust Digital-Twin Localization via An RGBD-based Transformer Network and A Comprehensive Evaluation on a Mobile Dataset
This repository is the implementation code of the 2024 ICML workshop paper "Robust Digital-Twin Localization via An RGBD-based Transformer Network and A Comprehensive Evaluation on a Mobile Dataset" [[arXiv](https://arxiv.org/abs/2309.13570), [project page](https://openark-berkeley.github.io/DTTDNet/), [dataset](https://drive.google.com/drive/folders/1U7YJKSrlWOY5h2MJRc_cwJPkQ8600jbd), [poster](https://icml.cc/media/PosterPDFs/ICML%202024/36411.png?t=1721891810.2840796)]. 

![Group 162](https://github.com/OpenARK-Berkeley/Robust-Digital-Twin-Tracking/assets/106426767/faea5212-f400-48b6-bbec-312b7887d2a1)

In this work, we bridge the existing gap towards mobile AR object tracking scenarios in a dual approach. At the algorithm level, we introduced a novel Transformer-based 6DoF  pose estimator, specifically designed to navigate the complexities introduced by noisy depth data, which is a common issue in mobile AR environments. At the dataset level, on the other hand, we expanded the scope of our previous work [DTTD](https://arxiv.org/abs/2302.05991) by introducing an innovative RGBD dataset captured using the iPhone 14 Pro, thus broadening the applicability of our approach to include iPhone sensor data. 

### Updates:
- [11/05/24] The DTTD-Mobile dataset will be migrated to huggingface due to our Google Drive storage limitation, check [here](https://huggingface.co/datasets/ZixunH/DTTD2-IPhone/tree/main).
- [09/10/24] Our **MoCap data pipeline** has been released, check [here](https://github.com/OpenARK-Berkeley/DTTDv2-IPhoneLiDAR) (iPhone-ARKit-based version) for your customized data collection and annotation. For the release of our **data capture app** for iPhone, check [here](https://github.com/OpenARK-Berkeley/iphone-capture-app). For our previous released Azure-based version, check [here](https://github.com/augcog/DTTDv1).
- [06/17/24] Our work has been accepted at 2024 ICML workshop: Data-centric Machine Learning Research. [demo video](https://icml.cc/virtual/2024/36411), [openreview](https://openreview.net/forum?id=X7lBl0CPdw)
- [x] [09/28/23] Our trained **checkpoints** for pose estimator are released [here](https://drive.google.com/drive/folders/128yIostfVzvbTQzoW3GO2MKEm62uTplp?usp=drive_link).
- [x] [09/27/23] Our **dataset**: [DTTD-Mobile](https://drive.google.com/drive/folders/1U7YJKSrlWOY5h2MJRc_cwJPkQ8600jbd) is released.

### Citation
If our work is useful or relevant to your research, please kindly recognize our contributions by citing our papers:
```
@misc{huang2024robust6dofposeestimation,
      title={Robust 6DoF Pose Estimation Against Depth Noise and a Comprehensive Evaluation on a Mobile Dataset}, 
      author={Zixun Huang and Keling Yao and Seth Z. Zhao and Chuanyu Pan and Chenfeng Xu and Kathy Zhuang and Tianjian Xu and Weiyu Feng and Allen Y. Yang},
      year={2024},
      eprint={2309.13570},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2309.13570}, 
}

@InProceedings{DTTDv1,
    author    = {Feng, Weiyu and Zhao, Seth Z. and Pan, Chuanyu and Chang, Adam and Chen, Yichen and Wang, Zekun and Yang, Allen Y.},
    title     = {Digital Twin Tracking Dataset (DTTD): A New RGB+Depth 3D Dataset for Longer-Range Object Tracking Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3288-3297}
}
```

### Dependencies:

Before running our pose estimation pipeline, you can activate a __conda__ environment where Python Version >= 3.8:
```
conda create --name [YOUR ENVIR NAME] python = [PYTHON VERSION]
conda activate [YOUR ENVIR NAME]
```

then install all necessary packages:
```
torch
torchvision
torchaudio
numpy
eniops
pillow
scipy
opencv_python
tensorboard
tqdm
```

For knn module used in ADD-S loss, install KNN_CUDA: https://github.com/pptrick/KNN_CUDA. (Install KNN_CUDA requires CUDA environment, ensure that your CUDA version >= 10.2. Also, It only supports torch v1.0+.)

### Load Dataset and Checkpoints:
Download our [checkpoints](https://drive.google.com/drive/folders/128yIostfVzvbTQzoW3GO2MKEm62uTplp?usp=drive_link) and [datasets](https://drive.google.com/drive/folders/1U7YJKSrlWOY5h2MJRc_cwJPkQ8600jbd), then organize the file structure:
```
Robust-Digital-Twin-Tracking
├── checkpoints
│   ├── m8p4.pth
│   ├── m8p4_filter.pth
│   └── ...
|   ...
└── dataset
    └── dttd_iphone
        ├── dataset_config
        ├── dataset.py
        └── DTTD_IPhone_Dataset
            └── root
                ├── cameras
                │   ├── az_camera1 (if you want to train our algorithm with DTTD v1)
                │   ├── iphone14pro_camera1
                │   └── ZED2 (to be released...)
                ├── data
                │   ├── scene1
                │   │   └── data
                │   │   │   ├── 00001_color.jpg
                │   │   │   ├── 00001_depth.png
                │   │   │   └── ...
                |   │   └── scene_meta.yaml
                │   ├── scene2
                │   │   └── data
                |   │   └── scene_meta.yaml
                │   ...
                └── objects
                    ├── apple
                    │   ├── apple.mtl
                    │   ├── apple.obj
                    │   ├── front.xyz
                    │   ├── points.xyz
                    │   ├── ...
                    │   └── textured.obj.mtl
                    ├── black_expo_marker
                    └── ...
```

### Run Estimation:
This repository contains scripts for 6dof object pose estimation (end-to-end coarse estimation). To run estimation, please make sure you have installed all the dependencies.

![Group 169](https://github.com/OpenARK-Berkeley/Robust-Digital-Twin-Tracking/assets/106426767/446c0f53-ab63-4260-9ef0-ac1e02755d92)

To run dttd-net (either training or evaluation), first download the dataset. It is recommended to create a soft link to `dataset/dttd_iphone/` folder using:
```bash
ln -s <path to dataset>/DTTD_IPhone_Dataset ./dataset/dttd_iphone/
```
To run trained estimator with test dataset, move to `./eval/`. For example, to evaluate on dttd v2 dataset:
```bash
cd eval/dttd_iphone/
bash eval.sh
```
You can customize your own eval command, for example:
```bash
python eval.py --dataset_root ./dataset/dttd_iphone/DTTD_IPhone_Dataset/root\
                --model ./checkpoints/m2p1.pth\
                --base_latent 256 --embed_dim 512 --fusion_block_num 1 --layer_num_m 2 --layer_num_p 1\
                --visualize --output eval_results_m8p4_model_filtered_best\
```
To load model with filter-enhanced MLP, please add flag `--filter`.
To visualize the attention map or/and the reduced geometric embeddings' distribution, you can add flag `--debug`.

### Eval:
This is the [ToolBox](https://github.com/yuxng/YCB_Video_toolbox) that we used for the experiment result evaluation and comparison.

### Train:
To run training of our method, you can use:
```bash
python train.py --device 0 \
    --dataset iphone --dataset_root ./dataset/dttd_iphone/DTTD_IPhone_Dataset/root --dataset_config ./dataset/dttd_iphone/dataset_config \
    --output_dir ./result/result \
    --base_latent 256 --embed_dim 512 --fusion_block_num 1 --layer_num_m 8 --layer_num_p 4 \
    --recon_w 0.3 --recon_choice depth \
    --loss adds --optim_batch 4 \
    --start_epoch 0 \
    --lr 1e-5 --min_lr 1e-6 --lr_rate 0.3 --decay_margin 0.033 --decay_rate 0.77 --nepoch 60 --warm_epoch 1 \
    --filter_enhance \
```
To train a smaller model, you can set flags `--layer_num_m 2 --layer_num_p 1`.
To enable our method with depth robustifying modules, you can add flags `--filter_enhance` or/and `--recon_choice model`.

To adjust the weight of Chamfer Distance Loss to 0.5, you can set flags `--reon_w 0.5`.

Our model is applicable on YCBV_Dataset and DTTD_v1 as well, please try following commands to run training of our method with other datasets (please ensure you download the dataset that you want to train on):
```bash
python train.py --dataset ycb --output_dir ./result/train_result --device 0 --batch_size 1 --lr 8e-5 --min_lr 8e-6 --warm_epoch 1
python train.py --dataset dttd --output_dir ./result/train_result --device 0 --batch_size 1 --lr 1e-5 --min_lr 1e-6 --warm_epoch 1
```



