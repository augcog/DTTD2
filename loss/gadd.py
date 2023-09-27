import torch
from torch.nn.modules.loss import _Loss

def loss(pred_r, pred_t, pred_c, tar_r, tar_t, prim_groups, obj_idx, point_cloud, w, device=torch.device("cuda")):
    bs, num_p, _ = pred_c.size() # (bs, 1000, 1)
    pred_r = pred_r / (torch.norm(pred_r, dim=2).view(bs, num_p, 1)) # (bs, 1000, 4)
    pred_r = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1),\
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_p, 1), \
                      (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_p, 1)), dim=2).contiguous().view(bs, num_p, 3, 3) # (bs, 1000, 3, 3)
    pred_t = (pred_t+point_cloud).contiguous().view(bs, num_p, 1, 3)
    
    # get prim group
    sub_dist = torch.zeros(bs).to(device)
    sub_loss = torch.zeros(bs).to(device)
    for i in range(bs):
        obj_groups = prim_groups[obj_idx[i]]
        tar_r_i = tar_r[i].view(1, 3, 3).repeat(num_p, 1, 1).contiguous() # [num_p, 3, 3]
        tar_t_i = tar_t[i].view(1, 3, 1).repeat(num_p, 1, 1).contiguous() # [num_p, 3, 1]
        min_dists = []
        for j, grp in enumerate(obj_groups):
            _, pt_num = grp.size()
            grp = grp.to(device)
            grp = grp.view(1, 3, pt_num).repeat(num_p, 1, 1) # (num_p, 3, pt_num)
            pred = torch.bmm(pred_r[i], grp) + pred_t[i].view(num_p, 3, 1) # (num_p, 3, pt_num)
            tar = torch.bmm(tar_r_i, grp) + tar_t_i.view(num_p, 3, 1) # (num_p, 3, pt_num)
            pred = pred.unsqueeze(dim=1).repeat(1, pt_num, 1, 1).contiguous() # (num_p, pt_num, 3, pt_num)
            tar = tar.permute(0, 2, 1).unsqueeze(dim=3).repeat(1, 1, 1, pt_num).contiguous() # (num_p, pt_num, 3, pt_num)
            min_dist, _ = torch.min(torch.norm(pred-tar, dim=2), dim=2)
            
            if len(obj_groups) == 3 and j == 2:
                min_dist = torch.max(min_dist, dim=1)[0]  # [num_p]
            else:
                min_dist = torch.mean(min_dist, dim=1)  # [num_p]  
            min_dists.append(min_dist.unsqueeze(0))
            
        min_dists = torch.concat(min_dists, dim=0)
        if len(obj_groups) == 3 and obj_groups[2].size()[1] > 1:
            min_dists = torch.max(min_dists, dim=0)[0]  # [num_p]
        else:
            min_dists = torch.mean(min_dists, dim=0)  # [num_p]
        
        sub_dist[i] = torch.mean(min_dists[torch.max(pred_c[i].view(num_p), dim=0)[1]])
        sub_loss[i] = torch.mean(min_dists * pred_c[i].view(num_p) - w * torch.log(pred_c[i].view(num_p)))
        
    gadd = torch.mean(sub_dist)
    gadd_loss = torch.mean(sub_loss)
    
    gadd_loss = torch.where(torch.isinf(gadd_loss), torch.full_like(gadd_loss, 0), gadd_loss)
    gadd_loss = torch.where(torch.isnan(gadd_loss), torch.full_like(gadd_loss, 0), gadd_loss)
    
    return gadd_loss, gadd

class GADD_Loss(_Loss):

    def __init__(self, device):
        super(GADD_Loss, self).__init__(True)
        self.device = device

    def forward(self, pred_r, pred_t, pred_c, tar_r, tar_t, prim_groups, obj_idx, point_cloud, w):
        return loss(pred_r, pred_t, pred_c, tar_r, tar_t, prim_groups, obj_idx, point_cloud, w, self.device)
    
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from dataset.ycb.dataset import YCBDataset
    dataset = YCBDataset("../dataset/ycb/YCB_Video_Dataset/", "train", True, "../dataset/ycb/dataset_config")
    bs = 2
    pred_r = torch.rand((bs, 1000, 4))
    pred_t = torch.rand((bs, 1000, 3))
    pred_c = torch.rand((bs, 1000, 1))
    point_cloud = torch.rand((bs, 1000, 1))
    tar_r = torch.rand((bs, 3, 3))
    tar_t = torch.rand((bs, 1, 3))
    obj_idx = torch.zeros((bs,), dtype=torch.int32)
    l, _ = loss(pred_r, pred_t, pred_c, tar_r, tar_t, dataset.prim_groups, obj_idx, point_cloud, 0.15, torch.device("cpu"))
    print(l)