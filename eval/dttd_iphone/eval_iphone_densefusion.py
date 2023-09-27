import os
import sys
import cv2
import json
import argparse
from datetime import datetime
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

sys.path.append("../../")

from dataset.dttd_iphone.dataset import DTTDDataset, Borderlist
from model.posenet import PoseNet
from utils.log import Logger
from utils.file import get_checkpoint
from utils.visualizer import visualize
from utils.transformations import quaternion_matrix
from utils.image import get_discrete_width_bbox

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str, default = '../../dataset/dttd_iphone/DTTD_IPhone_Dataset/root', help='dataset root dir')
    parser.add_argument('--output', type=str, default = './eval_results', help='output directory to save results')
    parser.add_argument('--model', type=str, default = '',  help='path to resume model file')
    parser.add_argument('--result', type=str, default = 'eval_results',  help='Directory to save results')
    parser.add_argument('--visualize', action='store_true')
    return parser.parse_args()

def eval():
    opt = parse()
    # get meta parameters
    dataset = DTTDDataset(opt.dataset_root, "test", add_noise=False)
    # set dirs
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
        os.mkdir(os.path.join(opt.output, "mats"))
    # set visualize
    if opt.visualize:
        if not os.path.exists(os.path.join(opt.output, "visualize")):
            os.mkdir(os.path.join(opt.output, "visualize"))
        color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (123, 10, 265), (245, 163, 101), (100, 100, 178)]
    # set model
    estimator = PoseNet(num_points = dataset.sample_2d_pt_num, num_obj = dataset.num_obj) # pt_num:1000, num_obj: 20
    estimator.cuda()
    opt.model = get_checkpoint(opt.model)
    estimator.load_state_dict(torch.load(opt.model, map_location=torch.device('cuda')))
    estimator.eval()
    # set logger
    logger = Logger(os.path.join(opt.output, "log.txt"))
    logger.log("evaluation of DTTDDataset...")
    logger.log(f"eval date and time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    logger.log(f"mode: use groudtruth label")
    logger.log(f"dataset root: {opt.dataset_root}")
    logger.log(f"load model: {opt.model}")
    logger.log(f"output directory: {opt.output}")
    logger.log(f"if visualize: {opt.visualize}\n")
    # set data
    testlist = dataset.all_data_dirs
    logger.log(f"len of evaluation data list: {len(testlist)}")
    
    # start eval
    for i in range(len(testlist)):
        img = np.array(Image.open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_color.jpg")))
        depth = np.array(Image.open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_depth.png")), dtype=np.uint16)
        label = np.array(Image.open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_label.png")))
        
        with open(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_meta.json"), "r") as f:
            meta = json.load(f) # dict_keys(['objects', 'object_poses', 'intrinsic', 'distortion'])
        
        lst = np.array(meta['objects']).flatten().astype(np.int32) # id of objects that appear in the image
        results = []
        
        for idx in range(len(lst)): # for each object in the scene
            itemid = lst[idx]
            try:
                data = process_data(itemid, img, depth, label, meta, Borderlist, num_points=dataset.sample_2d_pt_num, norm=dataset.norm)
                
                # put data to cuda
                cloud = Variable(data["cloud"]).cuda()
                sample = Variable(data["sample"]).cuda()
                img_crop = Variable(data["img_crop"]).cuda()
                index = Variable(data["index"]).cuda()
                
                # add batch dim
                cloud = cloud.view(1, dataset.sample_2d_pt_num, 3)
                img_crop = img_crop.view(1, 3, img_crop.size()[1], img_crop.size()[2])
                
                # predict
                pred_r, pred_t, pred_c, _ = estimator(img_crop, cloud, sample, index)
                
                pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, dataset.sample_2d_pt_num, 1)

                pred_c = pred_c.view(1, dataset.sample_2d_pt_num)
                pred_t = pred_t.view(1 * dataset.sample_2d_pt_num, 1, 3)
                points = cloud.view(1 * dataset.sample_2d_pt_num, 1, 3)
                
                how_max, which_max = torch.max(pred_c, 1)

                pred_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
                pred_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
                pred_concat = np.append(pred_r, pred_t)
                results.append(pred_concat.tolist())
            except Exception as e:
                logger.log(f"Detector Lost {itemid} at No.{i} keyframe. Error message: {e}")
                results.append([0.0 for i in range(7)])
        
        scio.savemat(os.path.join(opt.output, "mats", '%04d.mat' % i), {'poses':results})
        
        # visualize
        if opt.visualize:
            try:
                img = cv2.imread(os.path.join(opt.dataset_root, dataset.prefix, f"{testlist[i]}_color.jpg"))
                for idx in range(len(lst)):
                    itemid = lst[idx]
                    model_pts = np.array(dataset.model_points[itemid])
                    R, T = results[idx][0:4], results[idx][4:7]
                    R = quaternion_matrix(R)[0:3, 0:3]
                    T = np.array(T).reshape(1,3)
                    img = visualize(img=img, model_pts=model_pts, R=R, T=T, intrinsics=np.array(meta['intrinsic']), color=color_list[idx])
                cv2.imwrite(os.path.join(opt.output, "visualize", '%04d.png' % i), img)
            except Exception as e:
                logger.log(f"Visualization fail at No.{i} keyframe. Error message: {e}")
                
        logger.log("Finish No.{0} keyframe".format(i))
        
            
def process_data(itemid, img, depth, label, meta, border_list, num_points=1000, norm=None):
    img_h, img_w, _ = img.shape
    xmap = np.array([[j for i in range(img_w)] for j in range(img_h)])
    ymap = np.array([[i for i in range(img_w)] for j in range(img_h)])
    
    # mask valid depth (!= 0) and valid label (== itemid)
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
    mask = mask_label * mask_depth
    
    # get object's discretized bounding box
    rmin, rmax, cmin, cmax = get_discrete_width_bbox(mask_label, border_list, img_w, img_h)
    
    # set sample points (2D) on depth/point cloud
    sample = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # non-zero positions on flattened mask, 1-D array
    if len(sample) >= num_points:
        sample = np.array(sorted(np.random.choice(sample, num_points))) # randomly choose pt_num points (idx)
    elif len(sample) == 0:
        sample = np.pad(sample, (0, num_points - len(sample)), 'constant')
    else:
        sample = np.pad(sample, (0, num_points - len(sample)), 'wrap')
        
    # crop image, depth and xy map with bbox, take the sample points
    img_crop = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
    depth_crop = depth[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, )
    xmap_crop = xmap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store y for sample points
    ymap_crop = ymap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store x for sample points
    
    # set camera focus and center
    cam = np.array(meta['intrinsic'])
    cam_cx, cam_cy, cam_fx, cam_fy =  cam[0][2], cam[1][2], cam[0][0], cam[1][1]
    
    # get point cloud [[px, py, pz], ...]
    cam_scale = 1000 # uint16 * 1000
    pz = depth_crop / cam_scale
    px = (ymap_crop - cam_cx) * pz / cam_fx
    py = (xmap_crop - cam_cy) * pz / cam_fy
    point_cloud = np.concatenate((px, py, pz), axis=1) # (pt_num, 3) store XYZ point cloud value for sample points
    
    # trans to torch tensor
    if norm == None:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return {
        "cloud": torch.from_numpy(point_cloud.astype(np.float32)),
        "sample": torch.LongTensor(sample.astype(np.int32)),
        "img_crop": norm(torch.from_numpy(img_crop.astype(np.float32))),
        "index": torch.LongTensor([itemid - 1])
    }
    
if __name__ == "__main__":
    eval()