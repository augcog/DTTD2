python train.py --device 0 \
    --dataset iphone --dataset_root ./dataset/dttd_iphone/DTTD_IPhone_Dataset/root --dataset_config ./dataset/dttd_iphone/dataset_config \
    --output_dir ./result/result \
    --base_latent 256 --embed_dim 512 --fusion_block_num 1 --layer_num_m 8 --layer_num_p 4 \
    --recon_w 0.6 --recon_choice model \
    --loss adds --optim_batch 4 \
    --start_epoch 0 \
    --lr 1e-5 --min_lr 1e-6 --lr_rate 0.3 --decay_margin 0.033 --decay_rate 0.82 --nepoch 60 --warm_epoch 1 \
    --filter_enhance \
    --pretrain /checkpoints/m8p4_filter.pth \