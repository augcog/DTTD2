import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

sys.path.append("../../")

from dataset.ycb.dataset import YCBDataset
from model.posenet import PoseNet
from utils.log import Logger
from utils.visualizer import visualize
from utils.transformations import quaternion_matrix
from utils.file import get_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--output', type=str, default = 'eval_results',  help='Directory to save results')
parser.add_argument('--visualize', action='store_true')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 1066.778
cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
bs = 1
dataset_config_dir = '../../dataset/ycb/dataset_config'
ycb_toolbox_dir = './YCB_Video_toolbox'
result_dir = opt.output
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.mkdir(os.path.join(result_dir, "mats"))
if opt.visualize:
    if not os.path.exists(os.path.join(result_dir, "visualize")):
        os.mkdir(os.path.join(result_dir, "visualize"))
    dataset = YCBDataset(opt.dataset_root, "test", add_noise=False, config_path=dataset_config_dir)
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (123, 10, 265), (245, 163, 101), (100, 100, 178)]

def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
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
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
opt.model = get_checkpoint(opt.model)
estimator.load_state_dict(torch.load(opt.model, map_location=torch.device('cuda')))
estimator.eval()

logger = Logger(os.path.join(result_dir, "log.txt"))
logger.log("evaluation of YCBDataset Begins...")
logger.log(f"dataset root: {opt.dataset_root}")
logger.log(f"load model: {opt.model}\n")

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()

logger.log(f"len of evaluation data list: {len(testlist)}")

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

for now in range(min(len(testlist), 2949)):
    img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    label = np.array(posecnn_meta['labels'])
    posecnn_rois = np.array(posecnn_meta['rois'])
    
    lst = posecnn_rois[:, 1:2].flatten()
    results = []
    
    for idx in range(len(lst)):
        itemid = lst[idx]
        try:
            rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            elif len(choose) == 0:
                choose = np.pad(choose, (0, num_points - len(choose)), 'constant')
            else:
                choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            results.append(my_pred.tolist())

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)
        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            results.append([0.0 for i in range(7)])

    scio.savemat(os.path.join(result_dir, "mats", '%04d.mat' % now), {'poses':results})
    
    # visualize
    if opt.visualize:
        img = cv2.imread('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
        meta = scio.loadmat(('{0}/{1}-meta.mat'.format(opt.dataset_root, testlist[now])))
        for idx in range(len(lst)):
            itemid = lst[idx]
            model_pts = np.array(dataset.model_points[itemid])
            R, T = results[idx][0:4], results[idx][4:7]
            R = quaternion_matrix(R)[0:3, 0:3]
            T = np.array(T).reshape(1,3)
            img = visualize(img=img, model_pts=model_pts, R=R, T=T, intrinsics=np.array(meta['intrinsic_matrix']), color=color_list[idx])
        cv2.imwrite(os.path.join(result_dir, "visualize", '%04d.png' % now), img)
    logger.log("Finish No.{0} keyframe".format(now))