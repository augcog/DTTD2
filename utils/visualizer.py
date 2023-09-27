import cv2
import copy
import random
import numpy as np

def visualize(img, model_pts, R, T, intrinsics, color=(0, 255, 0), sample=1.0):
    """Render model points to the input image with (R,T) and camera intrinsics to visualize the results

    Args:
        img (ndarray [H,W,C]): input (original, not cropped) image that will be rendered on. Keep the intrinsics coherent.
        model_pts (ndarray [N, 3]): the points (x, y, z) on the model to be rendered.
        R (ndarray [3, 3]): Rotation of the model.
        T (ndarray [1, 3]): Translation of the model.
        intrinsics (tuple of size 4 | ndarray [3,3]): camera intrinsic parameters. Could be either tuple: (cam_cx, cam_cy, cam_fx, cam_fy) or intrinsic matrix.
        color (tuple of size 3): color of the rendered points.
        sample (float, 0.0~1.0): The sampling proportion of the model points.
        
    Return:
        rendered image
    """
    # valid check
    img, model_pts, R, T, intrinsics = _valid_check(img, model_pts, R, T, intrinsics)
    img = copy.deepcopy(img)
    
    # Transform model to world coordinates
    if sample < 1.0: # sample
        sample_list = list(range(len(model_pts)))
        sample_list = sorted(random.sample(sample_list, round(sample*len(model_pts))))
        model_pts = np.array(model_pts[sample_list, :])
    model_pts = np.add(np.dot(model_pts, R.T), T) # (N, 3)
    
    # Transform model to camera space and render
    cam_cx, cam_cy, cam_fx, cam_fy = intrinsics
    for pts in model_pts:
        x, y, z = pts
        px = round(x*cam_fx/z + cam_cx)
        py = round(y*cam_fy/z + cam_cy)
        if py<0 or py>=img.shape[0]:
            continue
        if px<0 or px>=img.shape[1]:
            continue
        img[py][px] = np.array(color)
    
    # save
    return img


def Draw3DBBox(img, model_pts, R, T, intrinsics, color=[255, 255, 255], thickness=2):
    """Render 3D bounding box of <model_pts> to the input image with (R,T) and camera intrinsics to visualize the results

    Args:
        img (ndarray [H,W,C]): input (original, not cropped) image that will be rendered on. Keep the intrinsics coherent.
        model_pts (ndarray [N, 3]): the points (x, y, z) of the target object CAD model (local coordinate).
        R (ndarray [3, 3]): Rotation of the model.
        T (ndarray [1, 3]): Translation of the model.
        intrinsics (tuple of size 4 | ndarray [3,3]): camera intrinsic parameters. Could be either tuple: (cam_cx, cam_cy, cam_fx, cam_fy) or intrinsic matrix.
        color (tuple of size 3): color of the rendered bbox cube frame.
        thickness (int): The thickness of the rendered bbox cube frame (pixel).
        
    Return:
        rendered image
    """
    # valid check
    _valid_check(img, model_pts, R, T, intrinsics)
    
    # get 3D BBox
    xmax = np.max(model_pts[:, 0])
    ymax = np.max(model_pts[:, 1])
    zmax = np.max(model_pts[:, 2])
    xmin = np.min(model_pts[:, 0])
    ymin = np.min(model_pts[:, 1])
    zmin = np.min(model_pts[:, 2])
    Vertices = np.array([[xmax, ymax, zmax], [xmax, ymax, zmin],\
                        [xmin, ymax, zmax], [xmin, ymax, zmin],\
                        [xmax, ymin, zmax], [xmax, ymin, zmin],\
                        [xmin, ymin, zmax], [xmin, ymin, zmin]])
    
    # Transform
    Vertices = np.add(np.dot(Vertices, R.T), T)
    cam = intrinsics
    cam_cx, cam_cy, cam_fx, cam_fy =  cam[0][2], cam[1][2], cam[0][0], cam[1][1]
    pixels = []
    for v in Vertices:
        x, y, z = v
        px = round(x*cam_fx/z + cam_cx)
        py = round(y*cam_fy/z + cam_cy)
        pixels.append([px, py])
    # Cube edges
    edges = [[0,1], [0,2], [0,4], [1,3], [1,5], [2,3], [2,6], [3,7], [4,5], [4,6], [5,7], [6,7]]
    # Draw Cube
    for edge in edges:
        img = cv2.line(img, pixels[edge[0]], pixels[edge[1]], color, thickness)
    return img


def _valid_check(img, model_pts, R, T, intrinsics):
    img = np.array(img)
    assert img.shape[2] == 3
    model_pts = np.array(model_pts)
    assert model_pts.shape[1] == 3
    R = np.array(R)
    assert R.shape == (3,3)
    T = np.array(T)
    assert T.shape == (1,3)
    if len(intrinsics) == 3 and len(intrinsics[0]) == 3:
        intrinsics = (intrinsics[0][2], intrinsics[1][2], intrinsics[0][0], intrinsics[1][1])
    intrinsics = np.array(intrinsics)
    assert(intrinsics.shape == (4, ))
    return img, model_pts, R, T, intrinsics
        
    
    