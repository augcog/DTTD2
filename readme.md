# Readme for pose estimation

This is a directory that contains scripts for 6dof object pose estimation (coarse estimation). The model is implemented based on densefusion. please make sure you have installed all the dependencies in `../README.md`.

To run ycb (either training or evaluation), first download the dataset. It is recommended to create a soft link to `dataset/ycb/` folder using:
```bash
ln -s <path to dataset>/YCB_Video_Dataset ./dataset/ycb/
ln -s <path to dataset>/DTTD_Dataset ./dataset/dttd/
ln -s <path to dataset>/DTTD_IPhone_Dataset ./dataset/dttd_iphone/
```

To run training of DenseFusion, you can use:
```bash
python train_densefusion.py --dataset ycb --output_dir ./result/train_result --device 0 --batch_size 1 --lr 0.00008
```

To run training of our method (with pretrained checkpoint), you can use:
```bash
python train_poseformer.py --dataset ycb --output_dir ./result/train_result --device 0 --batch_size 1 --lr 8e-5 --min_lr 3e-5 --warm_epoch 3 --pretrain ./result/poseformer_4layer_largerlatent/checkpoints/epoch_16_dist_0.0113256146830752.pth
```

Or with our DTTD-IPhone Dataset:
```bash
python train_densefusion.py --dataset dttd_iphone --output_dir ./result/train_result --device 0 --batch_size 1 --lr 0.00008 --dataset_root ./dataset/dttd_iphone/DTTD_IPhone_Dataset/root
```
```bash
python train_poseformer.py --dataset dttd_iphone --output_dir ./result/train_result --device 0 --batch_size 1 --lr 9e-6 --min_lr 9e-7 --warm_epoch 1 --decay_margin 0.045 --dataset_root ./dataset/dttd_iphone/DTTD_IPhone_Dataset/root
```

See `./run/` for more running command.

To evaluate trained model, move to `./eval/`. For example, to evaluate on ycb dataset:
```bash
cd eval/ycb/
bash eval.sh
```
You can customize your own eval command, for example:
```bash
python3 eval_ycb_gt.py --dataset_root ../../dataset/ycb/YCB_Video_Dataset --model ../../result/train_densefusion_adds/checkpoints/ --output eval_results --visualize 
```