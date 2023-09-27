import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../")

if torch.cuda.is_available():
    from model.chamfer_distance.chamfer_distance_gpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance
else:
    from model.chamfer_distance.chamfer_distance_cpu import ChamferDistance # https://github.com/chrdiller/pyTorchChamferDistance

class PointCloudEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(PointCloudEncoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(256, 1024, 1)
        self.fc = nn.Linear(1024, emb_dim)
        # TODO: add layer norm, avg pool

    def forward(self, xyz):
        """
        Args:
            xyz: (B, 3, N)
        """
        np = xyz.size()[2]
        x = F.relu(self.conv1(xyz))
        x = F.relu(self.conv2(x))
        global_feat = F.adaptive_max_pool1d(x, 1)
        x0 = torch.cat((x, global_feat.repeat(1, 1, np)), dim=1)
        x = F.relu(self.conv3(x0))
        x = torch.squeeze(F.adaptive_max_pool1d(x, 1), dim=2)
        embedding = self.fc(x)
        return embedding, x0


class PointCloudDecoder(nn.Module):
    def __init__(self, emb_dim, n_pts, out_channels=3):
        super(PointCloudDecoder, self).__init__()
        self.out_channels = out_channels
        self.fc1 = nn.Linear(emb_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, out_channels*n_pts)
        self.norm = nn.LayerNorm(out_channels*n_pts)

    def forward(self, embedding):
        """
        Args:
            embedding: (B, 512)

        """
        bs = embedding.size()[0]
        out = F.relu(self.fc1(embedding))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        out = self.norm(out)
        out_pc = out.view(bs, -1, self.out_channels)
        return out_pc


class PointCloudAE(nn.Module):
    def __init__(self, emb_dim=256, n_pts=1024, latent_dim=None):
        super(PointCloudAE, self).__init__()
        self.num_points = n_pts
        self.encoder = PointCloudEncoder(emb_dim)
        self.decoder = PointCloudDecoder(emb_dim, n_pts)
        if latent_dim is not None:
            self.ptfeat = nn.Conv1d(256+emb_dim, latent_dim, 1)
            # self.ptfeat = nn.Conv1d(256, latent_dim, 1)

    def latent(self, feat, emb):
        return self.ptfeat(torch.cat([feat, emb[:,:,None].repeat(1, 1, self.num_points)], dim=1))
        # return self.ptfeat(feat + emb[:,:,None].repeat(1, 1, self.num_points))

    def forward(self, in_pc, emb=None, recon_ref=None):
        """
        Args:
            in_pc: (B, N, 3)
            emb: (B, 512)

        Returns:
            emb: (B, emb_dim)
            out_pc: (B, n_pts, 3)

        """
        if emb is None:
            xyz = in_pc.permute(0, 2, 1)
            emb, feat = self.encoder(xyz)
        if recon_ref is None:
            recon_ref = in_pc
        out_pc = self.decoder(emb)
        loss = recon_loss(recon_ref, out_pc)
        return feat, emb, out_pc, loss

def recon_loss(points, recon_points):
    chamfer_dist = ChamferDistance()
    points = points.transpose(1,2)
    reconstructed_points = recon_points.transpose(1,2)
    dist1, dist2 = chamfer_dist(points, reconstructed_points)   # calculate loss
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss