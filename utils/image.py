import numpy as np

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