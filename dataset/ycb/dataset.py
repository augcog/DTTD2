import os
import cv2
import time
import json
import random
import numpy as np
import numpy.ma as ma
from PIL import Image
import scipy.io as scio

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

Borderlist = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

class YCBDataset(data.Dataset):
    def __init__(self, root, mode, add_noise=True, config_path="dataset/ycb/dataset_config"):
        self.config_path = config_path
        self.Trainlist = os.path.join(config_path, 'train_data_list.txt')
        self.Testlist = os.path.join(config_path, 'test_data_list.txt')
        self.Classlist = os.path.join(config_path, 'classes.txt')
        self.mode = mode
        self.root = root
        self.add_noise = add_noise
        self._init_config()
        self._init_data_info()
        self.model_points = self.load_points(self.root, self.classes) # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}
        self.prim_groups = self.load_primitives()

    def __len__(self):
        return len(self.all_data_dirs)

    def __getitem__(self, index):
        # load raw data
        img = Image.open(os.path.join(self.root, f"{self.all_data_dirs[index]}-color.png"))   # PIL, size: (640, 480)
        depth = np.array(Image.open(os.path.join(self.root, f"{self.all_data_dirs[index]}-depth.png"))) # shape: (480, 640)
        label = np.array(Image.open(os.path.join(self.root, f"{self.all_data_dirs[index]}-label.png"))) # shape: (480, 640)
        meta = scio.loadmat(os.path.join(self.root, f"{self.all_data_dirs[index]}-meta.mat"))
        
        # set camera focus and center
        cam = np.array(meta['intrinsic_matrix'])
        cam_scale = meta["factor_depth"][0][0]
        cam_cx, cam_cy, cam_fx, cam_fy =  cam[0][2], cam[1][2], cam[0][0], cam[1][1]
        
        # color jittor (noise) for img
        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img) # shape: (480, 640, 3)

        # Get id of objects that appear in the image
        objs = meta['cls_indexes'].flatten().astype(np.int32)
        
        # randomly choose one object and get its mask
        obj_indices = list(range(len(objs)))
        random.shuffle(obj_indices)
        for obj_idx in obj_indices:
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, objs[obj_idx]))
            mask = mask_depth # consider both label (where = objs[obj_idx]) and valid depth (where > 0)
            if len((mask_label * mask_depth).nonzero()[0]) > self.minimum_px_num:
                break
        
        # get ground truth rotation and translation
        R_gt = np.array(meta['poses'][:, :, obj_idx][:, 0:3]) # (3, 3)
        T_gt = np.array([meta['poses'][:, :, obj_idx][:, 3:4].flatten()]) # (1, 3)
        
        # get sample points (3D) on model
        model_sample_list = list(range(len(self.model_points[objs[obj_idx]])))
        model_sample_list = sorted(random.sample(model_sample_list, self.sample_model_pt_num))
        sampled_model_pt = np.array(self.model_points[objs[obj_idx]][model_sample_list, :]) # (sample_model_pt_num, 3) 

        # get model points in the camera coordinate
        sampled_model_pt_camera = np.add(np.dot(sampled_model_pt, R_gt.T), T_gt) # (sample_model_pt_num, 3)
        
        # projection and get bbox
        proj_x = sampled_model_pt_camera[:, 0] * cam_fx / sampled_model_pt_camera[:, 2] + cam_cx
        proj_y = sampled_model_pt_camera[:, 1] * cam_fy / sampled_model_pt_camera[:, 2] + cam_cy
        cmin, cmax = min(proj_x), max(proj_x)
        rmin, rmax = min(proj_y), max(proj_y)
        img_h, img_w = label.shape
        rmin, rmax, cmin, cmax = discretize_bbox(rmin, rmax, cmin, cmax, Borderlist, img_w, img_h)
        
        # set sample points (2D) on depth/point cloud
        sample2D = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # non-zero positions on flattened mask, 1-D array
        if len(sample2D) >= self.sample_2d_pt_num:
            sample2D = np.array(sorted(np.random.choice(sample2D, self.sample_2d_pt_num))) # randomly choose pt_num points (idx)
        elif len(sample2D) == 0:
            sample2D = np.pad(sample2D, (0, self.sample_2d_pt_num - len(sample2D)), 'constant')
        else:
            sample2D = np.pad(sample2D, (0, self.sample_2d_pt_num - len(sample2D)), 'wrap')
            
        depth_crop = depth[rmin:rmax, cmin:cmax].flatten()[sample2D][:, np.newaxis].astype(np.float32) # (pt_num, )
        xmap_crop = self.xmap[rmin:rmax, cmin:cmax].flatten()[sample2D][:, np.newaxis].astype(np.float32) # (pt_num, ) store y for sample points
        ymap_crop = self.ymap[rmin:rmax, cmin:cmax].flatten()[sample2D][:, np.newaxis].astype(np.float32) # (pt_num, ) store x for sample points
        
        # get point cloud [[px, py, pz], ...]
        cam_scale = meta["factor_depth"][0][0]
        pz = depth_crop / cam_scale
        px = (xmap_crop - cam_cx) * pz / cam_fx
        py = (ymap_crop - cam_cy) * pz / cam_fy
        point_cloud = np.concatenate((px, py, pz), axis=1) # (pt_num, 3) store XYZ point cloud value for sample points
        
        # get rgb image
        img_crop = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax] # shape (3, H, W)

        return {"img": torch.from_numpy(img),
               "label": torch.from_numpy(label),
               "depth": torch.from_numpy(depth.astype(np.float32)),
               "img_crop": self.norm(torch.from_numpy(img_crop.astype(np.float32))), \
               "point_cloud": torch.from_numpy(point_cloud.astype(np.float32)), \
               "sample_2d": torch.LongTensor(sample2D.astype(np.int32)), \
               "sampled_model_pt_camera": torch.from_numpy(sampled_model_pt_camera.astype(np.float32)), \
               "sampled_model_pt": torch.from_numpy(sampled_model_pt.astype(np.float32)), \
               "obj_id": torch.LongTensor([int(objs[obj_idx]) - 1]), \
               "R": torch.from_numpy(R_gt.astype(np.float32)), \
               "T": torch.from_numpy(T_gt.astype(np.float32))} 

    def _init_config(self):
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        
        self.noise_trans = 0.02
        self.minimum_px_num = 50
        self.symmetry_obj_cls = [12, 15, 18, 19, 20]
        self.sample_model_pt_num = 500
        self.sample_2d_pt_num = 1000 # pt_num
        self.num_obj = 21
    
    def _init_data_info(self):
        # set data list, store in local path
        if self.mode == 'train':
            self.data_list_path = self.Trainlist
        elif self.mode == 'test':
            self.data_list_path = self.Testlist
        else:
            raise NotImplementedError
        self.class_path = self.Classlist
        
        # read data information from files
        with open(self.data_list_path, 'r') as f:
            self.all_data_dirs = f.read().splitlines()
            self.real_data_dirs = [d for d in self.all_data_dirs if d.startswith('data/')] # real-world data path list
            self.syn_data_dirs = [d for d in self.all_data_dirs if d.startswith('data_syn/')] # synthetic data path list
            
        with open(self.class_path, 'r') as f:
            self.classes = f.read().splitlines()
            
    def load_points(self, root, classes):
        ''' load points of each model from points.xyz and save as dict(key: class id, value: points array) '''
        model_points = {} # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}
        class_id = 1
        for cls in classes:
            cls_filepath = os.path.join(root, 'models', cls, 'points.xyz')
            if os.path.isfile(cls_filepath):
                model_points[class_id] = np.loadtxt(cls_filepath) # (2621, 3) float
            else:
                print(f"[Warning] {cls} doesn't exist, load model points for {cls} failed")
            class_id += 1
        return model_points
    
    def load_primitives(self):
        obj_radius = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
        primitives = []
        with open(os.path.join(self.config_path, "ycb_gp.json"), 'r') as f:
            prim_groups = json.load(f)
            for i, prim in enumerate(prim_groups):
                tmp = []
                for grp in prim['groups']:
                    tmp.append(torch.tensor(grp, dtype=torch.float).permute(1, 0).contiguous() / obj_radius[i])
                primitives.append(tmp)
        return primitives
    
    def get_object_num(self):
        return self.num_obj
    
    def get_sym_list(self):
        return self.symmetry_obj_cls
    
    def get_model_point_num(self):
        return self.sample_model_pt_num
    
    def get_2d_sample_num(self):
        return self.sample_2d_pt_num
    
    def get_models_xyz(self):
        return self.model_points


######################## dataset utils #############################

def get_discrete_width_bbox(label, border_list, img_w, img_h):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt
    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt
    return rmin, rmax, cmin, cmax

def discretize_bbox(rmin, rmax, cmin, cmax, border_list, img_w, img_h):
    rmax += 1
    cmax += 1
    r_b = border_list[binary_search(border_list, rmax - rmin)]
    c_b = border_list[binary_search(border_list, cmax - cmin)]
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_h:
        delt = rmax - img_h
        rmax = img_h
        rmin -= delt
    if cmax > img_w:
        delt = cmax - img_w
        cmax = img_w
        cmin -= delt
    return rmin, rmax, cmin, cmax
    
def binary_search(sorted_list, target):
    l = 0
    r = len(sorted_list)-1
    while l!=r:
        mid = (l+r)>>1
        if sorted_list[mid] > target:
            r = mid
        elif sorted_list[mid] < target:
            l = mid + 1
        else:
            return mid
    return l
    
    
if __name__ == "__main__":
    dataset = YCBDataset(root="./YCB_Video_Dataset", mode="train", config_path="./dataset_config")
    dt = dataset[0]
    print("img crop shape: ", dt["img_crop"].size())
    print("xyzmap crop shape: ", dt["xyzmap_crop"].size())