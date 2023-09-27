import json
import os
import random
import sys
from audioop import avg
from collections import defaultdict

import numpy as np
import scipy.io as scio
import tqdm

sys.path.append("../")
from dttd.dataset import DTTDDataset
from ycb.dataset import YCBDataset

obj_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19]

def readlines(fpath):
    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            lines = f.read().splitlines()
            return lines
    else:
        print(f"[Error] got a wrong object classes path: {fpath}")
        return []
    
def DTTD_total_counter(root):
    scene_names = [name for name in os.listdir(os.path.join(root,"data")) if os.path.isdir(os.path.join(root,"data",name))]
    total_frame_count = 0
    total_annotation_count = 0
    total_scene_count = 0
    objects_occurance = defaultdict(int)
    for scene in scene_names:
        count_obj_flag = True
        frame_count = 0
        annotation_count = 0
        data_path = os.path.join(root, "data", scene, "data")
        file_names = os.listdir(data_path)
        for file in file_names:
            if file.endswith(".json"):
                frame_count += 1
                with open(os.path.join(data_path, file), "r") as f:
                    meta = json.load(f)
                    if "synthetic" not in scene:
                        annotation_count += len(meta["objects"])
                        if count_obj_flag:
                            for o in meta["objects"]:
                                objects_occurance[o] += 1
                            count_obj_flag = False
                    else:
                        annotation_count += len([obj for obj in meta["objects"] if obj in obj_list])
        print(f"Scene {scene} has {frame_count} frames and {annotation_count} annotations!")
        total_frame_count += frame_count
        total_annotation_count += annotation_count
        total_scene_count += 1
    print(f"Total frame number is {total_frame_count}, total annotation number is {total_annotation_count}")
    print(f"Total scene count is {total_scene_count}")
    print(objects_occurance)
        

def DTTD_analyzer(root, config_dir, sample_num = 500):
    dataset = DTTDDataset(root=root, mode="train", add_noise=False)
    # get objects
    classes = dataset.classes
    dist = {}
    dist[0] = []
    for cls in range(len(classes)):
        dist[cls+1] = []
    # get data list
    train_path = dataset.Trainlist
    test_path = dataset.Testlist
    data = readlines(train_path)+readlines(test_path)
    # analyse scene per frame
    prefix = "data"
    annotations_record = {}
    for dt in tqdm.tqdm(data):
        if dt.startswith('synthetic'):
            continue
        scene_name = dt.split("/")[0]
        with open(os.path.join(root, prefix, f"{dt}_meta.json"), "r") as f:
            meta = json.load(f) # dict_keys(['objects', 'object_poses', 'intrinsic', 'distortion'])
        if scene_name not in annotations_record:
            annotations_record[scene_name] = {}
            annotations_record[scene_name]["unique_object_count"] = [len(meta["objects"])]
        else:
            if len(meta["objects"]) not in annotations_record[scene_name]["unique_object_count"]:
                annotations_record[scene_name]["unique_object_count"].append(len(meta["objects"]))
                print(f'find inconsistent object counts scene: {dt}')
        objs = np.array(meta['objects']).flatten().astype(np.int32) # id of objects that appear in the image
        if len(objs) == 0:
            print(f"find zero object data: {dt}")
        for obj_idx in objs:
            R = np.array(meta['object_poses'][str(obj_idx)])[0:3, 0:3] # (3, 3)
            T = np.array(meta['object_poses'][str(obj_idx)])[0:3, 3:4].T # (1, 3)
            points = dataset.model_points[obj_idx]
            # sample points
            sample_list = list(range(len(points)))
            sample_list = sorted(random.sample(sample_list, min(len(points), sample_num)))
            points = points[sample_list, :]
            # calculate translation
            points = np.add(np.dot(points, R.T), T)
            avg_dist = np.mean(points, axis=0)[2] # get z
            dist[obj_idx].append(avg_dist)
            dist[0].append(avg_dist)
    # print avg distance:
    # print(annotations_record)
    print("Evaluation DTTD average distance")
    print(f"Total object average distance: {sum(dist[0])/max(len(dist[0]), 1)},  max: {max(dist[0])}, min: {min(dist[0])}")
    for cls in range(len(classes)):
        if len(dist[cls+1]) == 0:
            dist[cls+1] = [0]
        print(f"object {classes[cls]}: {sum(dist[cls+1])/len(dist[cls+1])},  max: {max(dist[cls+1])}, min: {min(dist[cls+1])}")
        

def YCB_analyzer(root, config_dir, sample_num = 500):
    dataset = YCBDataset(root=root, mode="train", add_noise=False, config_path=config_dir)
    # get objects
    classes = dataset.classes
    dist = {}
    dist[0] = []
    for cls in range(len(classes)):
        dist[cls+1] = []
    # get data list
    train_path = dataset.Trainlist
    test_path = dataset.Testlist
    data = dataset.real_data_dirs+readlines(test_path)
    
    max_v = 0
    max_dt = None
    max_obj = None
    # analyse scene per frame
    for dt in tqdm.tqdm(data):
        meta = scio.loadmat(os.path.join(root, f"{dt}-meta.mat"))
        objs = meta['cls_indexes'].flatten().astype(np.int32) # id of objects that appear in the image
        for i, obj_idx in enumerate(objs):
            R = np.array(meta['poses'][:, :, i][:, 0:3]) # (3, 3)
            T = np.array([meta['poses'][:, :, i][:, 3:4].flatten()]) # (1, 3)
            points = dataset.model_points[obj_idx]
            # sample points
            sample_list = list(range(len(points)))
            sample_list = sorted(random.sample(sample_list, min(len(points), sample_num)))
            points = points[sample_list, :]
            # calculate translation
            points = np.add(np.dot(points, R.T), T)
            avg_dist = np.mean(points, axis=0)[2] # get z
            if avg_dist > max_v:
                max_v = avg_dist
                max_dt = dt
                max_obj = obj_idx
            dist[obj_idx].append(avg_dist)
            dist[0].append(avg_dist)
    # print avg distance:
    print(max_v, max_dt, max_obj)
    print("Evaluation YCB average distance")
    print(f"Total object average distance: {sum(dist[0])/max(len(dist[0]), 1)},  max: {max(dist[0])}, min: {min(dist[0])}")
    for cls in range(len(classes)):
        if len(dist[cls+1]) == 0:
            dist[cls+1] = [0]
        print(f"object {classes[cls]}: {sum(dist[cls+1])/len(dist[cls+1])},  max: {max(dist[cls+1])}, min: {min(dist[cls+1])}")

if __name__ == "__main__":
    dttd_root = "./dttd_iphone/DTTD_IPhone_Dataset/root"
    dttd_config_dir = "./dttd/DTTD_IPhone_Dataset/root"
    DTTD_total_counter(dttd_root)
    #DTTD_analyzer(dttd_root, dttd_config_dir)
    
    # ycb_root = "./ycb/YCB_Video_Dataset"
    # ycb_config_dir = "./ycb/dataset_config"
    # YCB_analyzer(ycb_root, ycb_config_dir)