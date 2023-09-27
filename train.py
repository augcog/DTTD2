import argparse
import os
from datetime import datetime

import torch
import torch.optim as optim
from loss.add_s import ADDS_Loss
# from loss.gadd import GADD_Loss
from loss.abc import ABC_Loss

from model.posefusion import PoseNet
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.log import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default = 0, help='device number, not valid when set data parallel')
parser.add_argument('--dataset', type=str, default = 'iphone', choices=['ycb', 'dttd', 'iphone'], help='Dataset name: ycb, dttd, ...')
parser.add_argument('--dataset_root', type=str, default = './dataset/dttd_iphone/DTTD_IPhone_Dataset/root', help='dataset root dir (''YCB_Video_Dataset'')')
parser.add_argument('--dataset_config', type=str, default = './dataset/dttd_iphone/dataset_config', help='dataset config dir')

parser.add_argument('--loss', type=str, default = 'adds', choices=['adds', 'abc'])
parser.add_argument('--recon_choice', type=str, default = 'depth', choices=['depth', 'model'])
parser.add_argument('--recon_w', default=0.3, type=float, help='recon_weight')
parser.add_argument('--filter_enhance', action='store_true')
parser.add_argument('--attn_diverse', action='store_true')

parser.add_argument('--base_latent', type=int, default = 128, help='base latent dim for unimodal encoder')
parser.add_argument('--embed_dim', type=int, default = 512, help='embedding dim for transformer encoder')
parser.add_argument('--fusion_block_num', type=int, default = 1, help='number of fusion block')
parser.add_argument('--layer_num_m', type=int, default = 3, help='layer num for modality fusion per block')
parser.add_argument('--layer_num_p', type=int, default = 3, help='layer num for point-to-point fuison per block')

parser.add_argument('--pretrain', type=str, default = None, help='path to pretrain model')
parser.add_argument('--frozen', type=str, default = None, choices=['ptnet', 'filter'])
parser.add_argument('--output_dir', type=str, default = './out', help='output directory, store output info')
parser.add_argument('--optim_batch', type=int, default = 4, help='number of batch per optim')
parser.add_argument('--workers', type=int, default = 8, help='number of data loading workers')
parser.add_argument('--data_parallel', action='store_true')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')

parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--min_lr', default=1e-6, type=float, help='min learning rate')
parser.add_argument('--warm_epoch', type=int, default=1, help='epoch for warmup')
parser.add_argument('--warm_iters', type=int, default=0, help='iters for warmup')
parser.add_argument('--lr_rate', default=0.3, type=float, help='learning rate decay rate')
parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
# parser.add_argument('--clip_grad', default=None, type=float, help='clip grad norm')

parser.add_argument('--w', default=0.015, type=float, help='balancing rate')
parser.add_argument('--w_rate', default=0.3, type=float, help='balancing rate decay rate')
parser.add_argument('--decay_margin', default=0.033, type=float, help='margin to decay lr & w')
parser.add_argument('--decay_rate', default=0.77, type=float, help='decay rate for decay margin')
parser.add_argument('--noise_trans', default=0.03, type=float, help='range of the random noise of translation added to the training data')
parser.add_argument('--nepoch', type=int, default=120, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--start_epoch', type=int, default = 0, help='which epoch to start')

parser.add_argument('--single_obj_test', type=int, default = -1, help='train based on a single obj')
parser.add_argument('--exclude_single_obj', type=int, default = -1, help='exclude a single obj')

opt = parser.parse_args()

import math

import numpy as np


def cosine_scheduler(lr, min_lr, epochs, steps_per_epoch, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1, cur_epoch=None):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * steps_per_epoch
    # set up warmup iterations
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    if warmup_iters > 0:
        warmup_schedule = np.linspace(start_warmup_value, lr, warmup_iters)
    iters = np.arange(epochs * steps_per_epoch - warmup_iters)
    # set up schedule
    if cur_epoch is None:
        schedule = np.array([min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * steps_per_epoch
        return schedule
    elif cur_epoch*steps_per_epoch<warmup_iters:
        schedule = np.array([min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * i / (len(iters)))) for i in np.arange(steps_per_epoch)])    
        warmup_schedule = np.concatenate((warmup_schedule, schedule))
        return warmup_schedule[cur_epoch*steps_per_epoch : cur_epoch*steps_per_epoch+steps_per_epoch]
    else:
        return np.array(
            [min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * (i - warmup_iters) / (len(iters)))) for i in np.arange(cur_epoch*steps_per_epoch, cur_epoch*steps_per_epoch+steps_per_epoch)])    
    
def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def main():
    # set device
    if not opt.data_parallel:
        torch.cuda.set_device(opt.device)

    # set output
    opt.output_dir = opt.output_dir + '_m' + str(opt.layer_num_m) + 'p' + str(opt.layer_num_p)\
        + '_' +opt.dataset + '_' + datetime.now().strftime('%m%d-%H%M')
    out_dir = opt.output_dir
    cp_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if not os.path.isdir(cp_dir):
        os.makedirs(cp_dir)
    logger = Logger(os.path.join(out_dir, "log_train.txt"))
    writer = SummaryWriter(os.path.join(out_dir, "train_event"))

    # set dataset
    if opt.dataset == "ycb":
        from dataset.ycb.dataset import YCBDataset
        train_dataset = YCBDataset(root=opt.dataset_root, config_path=opt.dataset_config, mode='train', add_noise=True)
        test_dataset = YCBDataset(root=opt.dataset_root, config_path=opt.dataset_config, mode='test', add_noise=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    
    elif opt.dataset == "dttd":
        # from dataset.dttd.dataset import DTTDDataset
        raise NotImplementedError

    elif opt.dataset == "iphone":
        from dataset.dttd_iphone.dataset import DTTDDataset
        train_dataset = DTTDDataset(root=opt.dataset_root, mode='train', add_noise=True)
        test_dataset = DTTDDataset(root=opt.dataset_root, mode='test', add_noise=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    else:
        raise NotImplementedError
    
    # set model
    estimator = PoseNet(num_points=train_dataset.get_2d_sample_num(), \
                        num_obj = train_dataset.get_object_num(),\
                        base_latent=opt.base_latent, \
                        embedding_dim=opt.embed_dim, \
                        fusion_block_num=opt.fusion_block_num, \
                        layer_num_m=opt.layer_num_m, layer_num_p=opt.layer_num_p, \
                        filter_enhance = opt.filter_enhance,\
                        require_adl = opt.attn_diverse)
    if opt.pretrain:
        tar_params = estimator.state_dict()
        src_params = torch.load(opt.pretrain, map_location=torch.device('cpu'))
        diff_params = {k:v for k,v in tar_params.items() if k not in src_params.keys()}
        logger.log(f"Warnning: Params NOT be Found. Details: {diff_params.keys()}")
        estimator.load_state_dict(src_params, strict=False)
        if opt.frozen:
            if opt.frozen == 'filter':
                for param in estimator.filter_enhance.parameters():
                    param.requires_grad = False
            if opt.frozen == 'ptnet':
                for param in estimator.ptnet.parameters():
                    param.requires_grad = False
    estimator = estimator.cuda()
    adds_loss = ADDS_Loss(num_points_mesh=train_dataset.get_model_point_num(), sym_list=train_dataset.get_sym_list())
    abc_loss = ABC_Loss(num_points_mesh=train_dataset.get_model_point_num())

    # gadd_loss = GADD_Loss(torch.device("cuda"))
    optimizer = optim.AdamW(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    
    logger.log(f"=========Object 6D Pose Estimation Training Log========")
    logger.log(f"training start date and time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    logger.log(f"dataset: {opt.dataset}")
    logger.log(f"data root: {opt.dataset_root}")
    logger.log(f"train set size: {len(train_dataset)}, test set size: {len(test_dataset)}")
    logger.log(f"output directory: {opt.output_dir}")
    logger.log(f"pretrain: {opt.pretrain}")
    logger.log(f"param frozen: {opt.frozen}")
    logger.log(f"resume posenet path: {opt.resume_posenet}")
    logger.log(f"device: {opt.device}")
    logger.log(f"network parallel: {opt.data_parallel}")
    logger.log(f"training on single obj: {opt.single_obj_test} exclude: {opt.exclude_single_obj}")

    logger.log(f"\n=========Object 6D Pose Estimation Model Architecture========")
    logger.log(f"base latent dim for unimodal encoder: {opt.base_latent}")
    logger.log(f"embedding dim for transformer encoder: {opt.embed_dim}")
    logger.log(f"number of fusion block: {opt.fusion_block_num}")
    logger.log(f"layer num for modality fusion per block: {opt.layer_num_m}")
    logger.log(f"layer num for point-to-point fuison per block: {opt.layer_num_p}")
    logger.log(f"loss type: {opt.loss}")
    logger.log(f"recon weight: {opt.recon_w}")
    logger.log(f"recon choice: {opt.recon_choice}")
    logger.log(f"filter enhance: {opt.filter_enhance}")
    logger.log(f"attn diverse: {opt.attn_diverse}")

    logger.log(f"\n=========Object 6D Pose Estimation Model Hyper Params========")
    logger.log(f"learning rate: {opt.lr}")
    logger.log(f"min learning rate: {opt.min_lr}")
    logger.log(f"warm lr: epoch {opt.warm_epoch} iters {opt.warm_iters}")
    # logger.log(f"clip grad: {opt.clip_grad}")
    logger.log(f"weight decay: {opt.weight_decay}")
    logger.log(f"decay margin: {opt.decay_margin}")
    logger.log(f"decay rate: {opt.decay_rate}")
    logger.log(f"optim batch: {opt.optim_batch}")
    logger.log(f"balancing rate: {opt.w}")
    logger.log(f"epoch num: {opt.nepoch}")
    logger.log(f"start epoch: {opt.start_epoch}")

    logger.log("\n\n")
    
    # start training
    best_test_dist = 1e10
    best_dist_2cm = 0
    for epoch in range(opt.start_epoch, opt.nepoch):
        # train
        logger.log(f"start training epoch {epoch}....")
        estimator.train()
        loss_list = []
        robust_loss_list = []
        dist_list = []
        # lr planning
        lr_schedule = cosine_scheduler(opt.lr, opt.min_lr, opt.nepoch, 
                                       len(train_dataset)//opt.optim_batch+1, 
                                       opt.warm_epoch, 0, opt.warm_iters, epoch)
        logger.log(f"learning rate planning: start {lr_schedule[0]} end {lr_schedule[-1]}\n")
        with tqdm(train_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                try:
                    if opt.single_obj_test>0:
                        assert data["obj_id"]==opt.single_obj_test
                    if opt.exclude_single_obj>0:
                        assert not data["obj_id"]==opt.exclude_single_obj
                except:
                    continue
                rgb, pt_cld, sample_2d, obj_id = data["img_crop"].cuda(), data["point_cloud"].cuda(), data["sample_2d"].cuda(), data["obj_id"].cuda()
                target, model_points = data["sampled_model_pt_camera"].cuda(), data["sampled_model_pt"].cuda()
                tar_r, tar_t = data["R"].cuda(), data["T"].cuda()

                if opt.recon_choice=='model':
                    recon_ref = model_points
                elif opt.recon_choice=='depth':
                    recon_ref = pt_cld
                else:
                    recon_ref = None

                rx, tx, cx, _, _, robust_loss = estimator(rgb, pt_cld, sample_2d, obj_id, recon_ref)
                
                if opt.loss == 'adds':
                    loss, dist = adds_loss(rx, tx, cx, target, model_points, obj_id, pt_cld, opt.w)
                elif opt.loss == 'abc':
                    loss, dist = abc_loss(rx, tx, cx, target, model_points, pt_cld, opt.w)
                else:
                    raise NotImplementedError
                    # loss, _ = gadd_loss(rx, tx, cx, tar_r, tar_t, train_dataset.prim_groups, obj_id, pt_cld, opt.w)
                
                loss_list.append(loss.item())
                dist_list.append(dist.item())
                robust_loss_list.append(robust_loss.item())
                
                # optimize
                loss = (loss + opt.recon_w*robust_loss)

                loss.backward()
                    
                if i % opt.optim_batch == 0:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_schedule[i//opt.optim_batch]
                    grad_norm = get_grad_norm_(estimator.parameters())
                    optimizer.step()
                    # renew
                    tepoch.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())
                    optimizer.zero_grad()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
        # save loss logs
        if len(loss_list) > 0:
            logger.log(f"Epoch: {epoch}, Avg Loss: {sum(loss_list)/len(loss_list)}, Avg Recon Loss: {sum(robust_loss_list)/len(robust_loss_list)}")
            writer.add_scalar("Loss/train", sum(loss_list)/len(loss_list), epoch)
        
        # test
        logger.log(f"start testing for epoch {epoch}....")
        estimator.eval()
        dist_list = []
        robust_loss_list = []
        torch.cuda.empty_cache()
        for i, data in enumerate(test_loader):
            try:
                if opt.single_obj_test>0:
                    assert data["obj_id"]==opt.single_obj_test
                if opt.exclude_single_obj>0:
                    assert not data["obj_id"]==opt.exclude_single_obj
            except:
                continue
            rgb, pt_cld, sample_2d, obj_id = data["img_crop"].cuda(), data["point_cloud"].cuda(), data["sample_2d"].cuda(), data["obj_id"].cuda()
            target, model_points = data["sampled_model_pt_camera"].cuda(), data["sampled_model_pt"].cuda()
            if opt.recon_choice=='model':
                recon_ref = model_points
            elif opt.recon_choice=='depth':
                recon_ref = pt_cld
            else:
                recon_ref = None
            rx, tx, cx, _, _, robust_loss = estimator(rgb, pt_cld, sample_2d, obj_id, recon_ref)
            _, dist = adds_loss(rx, tx, cx, target, model_points, obj_id, pt_cld, opt.w)
            dist_list.append(dist.item())
            robust_loss_list.append(robust_loss.item())
        dist_value = sum(dist_list)/len(dist_list)
        robust_loss = sum(robust_loss_list)/len(robust_loss_list)
        dist_2cm = 100.0 * len([d for d in dist_list if d<=0.02])/len(dist_list)
        dist_1cm = 100.0 * len([d for d in dist_list if d<=0.01])/len(dist_list)
        logger.log(f"testing done >>>> \n Average distance value: {dist_value}. \n Percentage of distance<1cm: {dist_1cm}.\n Percentage of distance<2cm: {dist_2cm}. \n Max dist value: {max(dist_list)}. \n Robust loss: {robust_loss} <<<<")
        writer.add_scalar("Distance/validation", dist_value, epoch)
        writer.add_scalar("Dist<1cm/validation", dist_1cm, epoch)
        writer.add_scalar("Dist<2cm/validation", dist_2cm, epoch)
        
        # save checkpoints
        saved = False
        if dist_value < best_test_dist:
            logger.log(f"Reach best test performance, save checkpoint to {cp_dir}")
            best_test_dist = dist_value
            saved = True
            torch.save(estimator.state_dict(), os.path.join(cp_dir, f"epoch_{epoch}_dist_{dist_value}.pth"))
        if dist_2cm > best_dist_2cm:
            logger.log(f"Reach best test performance, save checkpoint to {cp_dir}")
            best_dist_2cm = dist_2cm
            if not saved:
                torch.save(estimator.state_dict(), os.path.join(cp_dir, f"epoch_{epoch}_dist_{dist_value}.pth"))
            
        # update training settings
        if best_test_dist < opt.decay_margin:
            opt.lr *= opt.lr_rate
            opt.min_lr *= opt.lr_rate
            opt.w *= opt.w_rate
            opt.decay_margin *= opt.decay_rate
            logger.log(f"learning rate decay to {opt.lr}, minimum end learning rate decay to {opt.min_lr}, w decay to {opt.w}, decay_margin decay to {opt.decay_margin}")
            optimizer = optim.AdamW(estimator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # end epoch
        logger.log(f"epoch {epoch} ends. \n\n")
    
    writer.close()
            

if __name__ == "__main__":
    main()
    
    
