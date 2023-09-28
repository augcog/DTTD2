# Towards Robust Mobile Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset
This repository is the implementation code of the paper "Towards Robust Mobile Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset" [ [arxiv](https://arxiv.org/abs/2309.13570), website, [code](https://github.com/OpenARK-Berkeley/DigitalTwin-6DPose/edit/), [dataset](https://github.com/OpenARK-Berkeley/DTTDv2-IPhoneLiDAR) ]. 

![ModelArch](https://github.com/OpenARK-Berkeley/DigitalTwin-6DPose/assets/106426767/3f78f335-2801-4822-934c-55bac10c543d)

In this work, we bridge the existing gap towards mobile AR object tracking scenarios in a dual approach. At the algorithm level, we introduced a novel Transformer-based 6DoF  pose estimator, specifically designed to navigate the complexities introduced by noisy depth data, which is a common issue in mobile AR environments. At the dataset level, on the other hand, we expanded the scope of our previous work [DTTD](https://arxiv.org/abs/2302.05991) by introducing an innovative RGBD dataset captured using the iPhone 14 Pro, thus broadening the applicability of our approach to include iPhone sensor data. 

### Updates:
- [x] 09/27/23 Our dataset: **DTTDv2-IPhoneLiDAR** is released, please check our [offical repository](https://github.com/OpenARK-Berkeley/DTTDv2-IPhoneLiDAR) for data collection and annotation.
- [x] 09/27/23 Our trained **checkpoints** for pose estimator are released [here](https://drive.google.com/drive/folders/18laguqXN7b-WTFrHlRpbteqmE8oRF_8H?usp=drive_link).

### Installation
```
python=3.8
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

### Load Dataset and Checkpoints
Download our checkpoints and datasets, then organize the file structure:
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

### Run Estimation
This repository contains scripts for 6dof object pose estimation (end-to-end coarse estimation). To run estimation, please make sure you have installed all the dependencies.

To run dttd v2 (either training or evaluation), first download the dataset. It is recommended to create a soft link to `dataset/dttd_iphone/` folder using:
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
python3 eval_ycb_gt.py --dataset_root ../../dataset/ycb/YCB_Video_Dataset --model ../../result/train_m8p4_adds/checkpoints/ --output eval_results --visualize 
```

### Eval
This is the [ToolBox](https://github.com/yuxng/YCB_Video_toolbox) that we used for the experiment result evaluation and comparison.

### Train
To run training of our method, you can use:
```bash
python train.py --dataset dttd_iphone --output_dir ./result/train_result --device 0 --batch_size 1 --lr 8e-5 --min_lr 3e-5 --warm_epoch 3 --pretrain ./checkpoints/m8p4_filter_modelrecon.pth
```
Our model is applible on YCBV_Dataset and DTTD v1 as well, please try following commands to run training of our method with other dataset:
```bash
```

### Citation
If our work is useful or relevant to your research, please kindly recognize our contributions by citing our papers:
```
@InProceedings{DTTDv1,
    author    = {Feng, Weiyu and Zhao, Seth Z. and Pan, Chuanyu and Chang, Adam and Chen, Yichen and Wang, Zekun and Yang, Allen Y.},
    title     = {Digital Twin Tracking Dataset (DTTD): A New RGB+Depth 3D Dataset for Longer-Range Object Tracking Applications},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3288-3297}
}

@misc{DTTDv2,
      title={Towards Subcentimeter Accuracy Digital-Twin Tracking via An RGBD-based Transformer Model and A Comprehensive Mobile Dataset}, 
      author={Zixun Huang and Keling Yao and Seth Z. Zhao and Chuanyu Pan and Tianjian Xu and Weiyu Feng and Allen Y. Yang},
      year={2023},
      eprint={2309.13570},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


