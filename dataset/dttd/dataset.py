from cgi import test
import os
import cv2
import time
import random
import json
import numpy as np
import numpy.ma as ma
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.image import get_discrete_width_bbox

# TODO: add synthetic noise ??

Borderlist = [-1] + list(range(40, 1320, 40))

class DTTDDataset(data.Dataset):
    def __init__(self, root, mode, refine=False, add_noise=True):
        self.Trainlist = os.path.join(root, 'train_data_list.txt')
        self.Testlist = os.path.join(root, 'test_data_list.txt')
        self.Classlist = os.path.join(root, 'classes.txt')
        self.mode = mode
        self.root = root
        self.refine = refine
        self.add_noise = add_noise
        self._init_config()
        self._init_data_info()
        self.model_points = self.load_points(self.root, self.classes) # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}
        self.model_frontv = self.load_frontv(self.root, self.classes) # # {1:[vx1, vy1, vz1], 2:[vx2, vy2, vz2], 3:[...], ...}

    def __len__(self):
        return len(self.all_data_dirs)

    def __getitem__(self, index):
        # load raw data
        img = Image.open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_color.jpg"))   # PIL, size: (1280, 720)
        depth = np.array(Image.open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_depth.png")), dtype=np.uint16) # shape: (720, 1280)
        label = np.array(Image.open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_label.png"))) # shape: (720, 1280)
       
        with open(os.path.join(self.root, self.prefix, f"{self.all_data_dirs[index]}_meta.json"), "r") as f:
            meta = json.load(f) # dict_keys(['objects', 'object_poses', 'intrinsic', 'distortion'])
        
        # color jittor (noise) for img
        if self.add_noise:
            img = self.trancolor(img)
        img = np.array(img) # shape: (720, 1280, 3)

        objs = np.array(meta['objects']).flatten().astype(np.int32) # id of objects that appear in the image
        
        if len(objs) == 0:
            print(self.all_data_dirs[index])
        # randomly choose one object and get its mask
        obj_indices = list(range(len(objs)))
        random.shuffle(obj_indices)
        for obj_idx in obj_indices:
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, objs[obj_idx]))
            mask = mask_label * mask_depth # consider both label (where = objs[obj_idx]) and valid depth (where > 0)
            # cv2.imwrite(f"test_{objs[obj_idx]}.png", mask.astype(np.int32)*255) # for debug
            if len(mask.nonzero()[0]) > self.minimum_px_num:
                break

        # get object's bounding box
        img_h, img_w = label.shape
        rmin, rmax, cmin, cmax = get_discrete_width_bbox(mask_label, Borderlist, img_w, img_h)
        
        # set sample points (2D) on depth/point cloud
        sample = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # non-zero positions on flattened mask, 1-D array
        if len(sample) >= self.pt_num:
            sample = np.array(sorted(np.random.choice(sample, self.pt_num))) # randomly choose pt_num points (idx)
        elif len(sample) == 0:
            sample = np.pad(sample, (0, self.pt_num - len(sample)), 'constant')
        else:
            sample = np.pad(sample, (0, self.pt_num - len(sample)), 'wrap')
        
        # crop image, depth and xy map with bbox, take the sample points
        img_crop = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        depth_crop = depth[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, )
        xmap_crop = self.xmap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store y for sample points
        ymap_crop = self.ymap[rmin:rmax, cmin:cmax].flatten()[sample][:, np.newaxis].astype(np.float32) # (pt_num, ) store x for sample points
        
        # set camera focus and center
        cam = np.array(meta['intrinsic'])
        cam_cx, cam_cy, cam_fx, cam_fy =  cam[0][2], cam[1][2], cam[0][0], cam[1][1]
        
        # get point cloud [[px, py, pz], ...]
        cam_scale = 1000 # uint16 * 1000
        pz = depth_crop / cam_scale
        px = (ymap_crop - cam_cx) * pz / cam_fx
        py = (xmap_crop - cam_cy) * pz / cam_fy
        point_cloud = np.concatenate((px, py, pz), axis=1) # (pt_num, 3) store XYZ point cloud value for sample points
        
        # get ground truth rotation and translation
        R_gt = np.array(meta['object_poses'][str(objs[obj_idx])])[0:3, 0:3] # (3, 3)
        T_gt = np.array(meta['object_poses'][str(objs[obj_idx])])[0:3, 3:4].T # (1, 3)
        
        # get sample points (3D) on model
        model_sample_list = list(range(len(self.model_points[objs[obj_idx]])))
        if self.refine:
            model_sample_list = sorted(random.sample(model_sample_list, self.pt_num_mesh_large))
        else:
            model_sample_list = sorted(random.sample(model_sample_list, self.pt_num_mesh_small))
        sampled_model_pt = np.array(self.model_points[objs[obj_idx]][model_sample_list, :]) # (pt_num_mesh*, 3) 

        # get model points in the world coordinate
        sampled_model_pt_world = np.add(np.dot(sampled_model_pt, R_gt.T), T_gt) # (pt_num_mesh*, 3)
        
        # add noise for pointcloud and model points
        if self.add_noise:
            add_noise_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
            point_cloud = np.add(point_cloud, add_noise_t)
            sampled_model_pt_world = np.add(sampled_model_pt_world, add_noise_t)

        # get model front vector in the world coordinate
        frontv = np.array(self.model_frontv[objs[obj_idx]]) # (3,)
        frontv_world = np.dot(frontv, R_gt.T) # (3,)
        
        return {"img": torch.from_numpy(img),
               "label": torch.from_numpy(label),
               "depth": torch.from_numpy(depth.astype(np.float32)),
               "point_cloud": torch.from_numpy(point_cloud.astype(np.float32)), \
               "sample": torch.LongTensor(sample.astype(np.int32)), \
               "img_crop": self.norm(torch.from_numpy(img_crop.astype(np.float32))), \
               "sampled_model_pt_world": torch.from_numpy(sampled_model_pt_world.astype(np.float32)), \
               "sampled_model_pt": torch.from_numpy(sampled_model_pt.astype(np.float32)), \
               "obj_id": torch.LongTensor([int(objs[obj_idx]) - 1]), \
               "R": torch.from_numpy(R_gt.astype(np.float32)), \
               "T": torch.from_numpy(T_gt.astype(np.float32)), \
               "frontv": torch.from_numpy(frontv.astype(np.float32)), \
               "frontv_world": torch.from_numpy(frontv_world.astype(np.float32))} 

    def _init_config(self):
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.xmap = np.array([[j for i in range(1280)] for j in range(720)])
        self.ymap = np.array([[i for i in range(1280)] for j in range(720)])
        
        # self.noise_img_loc = 0.0
        # self.noise_img_scale = 7.0
        self.noise_trans = 0.02
        self.minimum_px_num = 50
        self.symmetry_obj_cls = []
        self.pt_num = 1000
        self.pt_num_mesh_small = 500
        self.pt_num_mesh_large = 2600
        self.num_obj = 20
        self.prefix = "data"
    
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
            self.real_data_dirs = self.all_data_dirs # real-world data path list
            self.syn_data_dirs = [d for d in self.all_data_dirs if d.startswith('data_syn/')] # synthetic data path list
            
        with open(self.class_path, 'r') as f:
            self.classes = f.read().splitlines()
            
    def load_points(self, root, classes):
        ''' load points of each model from points.xyz and save as dict(key: class id, value: points array) '''
        model_points = {} # {1:[[x1, y1, z1], [x2, y2, z2], ...], 2:[...], 3:[...], ...}
        class_id = 1
        for cls in classes:
            cls_filepath = os.path.join(root, 'objects', cls, 'points.xyz')
            if os.path.isfile(cls_filepath):
                model_points[class_id] = np.loadtxt(cls_filepath) # (2621, 3) float
            else:
                print(f"[Warning] {cls} doesn't exist, load model points for {cls} failed")
            class_id += 1
        return model_points
    
    def load_frontv(self, root, classes):
        ''' load the front vector of each model from front.xyz and save as dict(key: class id, value: vector array) '''
        model_frontv = {} # {1:[vx1, vy1, vz1], 2:[vx2, vy2, vz2], 3:[...], ...}
        class_id = 1
        for cls in classes:
            cls_filepath = os.path.join(root, 'objects', cls, 'front.xyz')
            if os.path.isfile(cls_filepath):
                model_frontv[class_id] = np.loadtxt(cls_filepath) # (3, ) float
            else:
                print(f"[Warning] {cls} doesn't exist, load model front vector for {cls} failed")
            class_id += 1
        return model_frontv
    
    def get_object_num(self):
        return self.num_obj
    
    def get_sym_list(self):
        return self.symmetry_obj_cls
    
    def get_point_mesh(self):
        if self.refine:
            return self.pt_num_mesh_large
        else:
            return self.pt_num_mesh_small
        
    def get_sample_point_num(self):
        return self.pt_num

        

