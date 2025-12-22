# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
from models.utils import *
from tqdm import tqdm
from torch import nn, einsum
from models.pointnet2_util import PointNetSetAbstractionMsg, PointNetFeaturePropagation
from models.normal_model import ImplicitNet_PINC
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
from models.dpsr import DPSR, PSR2Mesh
psr2mesh = PSR2Mesh.apply

class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    This class implements methods to update the moving average of model parameters over time,
    which can be useful for smoothing the parameters in training.

    Attributes:
        beta (float): The decay rate for the moving average.
        step (int): The current step of the model update.
    """
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """
        Updates the moving average model's parameters.
    
        Args:
            ma_model (torch.nn.Module): The model to update with the moving average.
            current_model (torch.nn.Module): The current model providing new parameters.
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx, norm=False, cond=True):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Conv1d(dim_in, dim_out, 1, 1, 0)
        self.cond = cond
        if cond is True:
            self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
            self._hyper_gate = nn.Linear(dim_ctx, dim_out)
        self.norm = norm
        if norm is True:
            self.bn = nn.GroupNorm(32, dim_out)

    def forward(self, ctx, x):
        if self.cond is True:
            gate = torch.sigmoid(self._hyper_gate(ctx))
            bias = self._hyper_bias(ctx)
            if x.dim() == 3:
                gate = gate.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
            if self.norm is True:
                ret = self.bn(self._layer(x) * gate + bias)
            else:
                ret = self._layer(x) * gate + bias
        else:
            if self.norm is True:
                ret = self.bn(self._layer(x))
            else:
                ret = self._layer(x)
        return ret

class Encoder_Attention(nn.Module):
    def __init__(self, encoder_dim, encoder_bn, k, time_dim, geo_dim=3):
        super().__init__()
        self.k = k
        
        self.q_conv = nn.Conv1d(encoder_dim, encoder_dim, 1)
        self.k_conv = nn.Conv1d(encoder_dim, encoder_dim, 1)
        self.v_conv = nn.Conv1d(encoder_dim, encoder_dim, 1)
        self.geo_mlp = nn.Sequential(
            nn.Conv2d(geo_dim, encoder_dim//2, 1, 1),
            nn.BatchNorm2d(encoder_dim//2) if encoder_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim//2, encoder_dim, 1, 1)
        )
        self.rel_mlp = nn.Sequential(
            nn.Conv2d(encoder_dim, encoder_dim//2, 1, 1),
            nn.BatchNorm2d(encoder_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_dim//2, encoder_dim, 1, 1)
        )
        # self.attn_conv = nn.Conv1d(cfgs.encoder_dim, cfgs.encoder_dim, 1)
        self.out_conv = nn.Sequential(
            nn.Conv1d(encoder_dim, encoder_dim*2, 1, 1),
            nn.BatchNorm1d(encoder_dim*2) if encoder_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv1d(encoder_dim*2, encoder_dim, 1, 1),
            nn.BatchNorm1d(encoder_dim) if encoder_bn else nn.Identity()
        )
        self.time_cond = ConcatSquashLinear(encoder_dim, encoder_dim, time_dim, norm=False)

    def forward(self, pts, feats, time_emb, geos=None):
        if geos == None:
            geos = pts
        
        q = self.q_conv(feats)
        k = self.k_conv(feats)
        v = self.v_conv(feats)

        knn_pts, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)

        knn_geos = index_points(geos, knn_idx)
        geo_embedding = self.geo_mlp(geos.unsqueeze(-1) - knn_geos)


        repeat_q = repeat(q, 'b c n -> b c n k', k=self.k)
        knn_k = index_points(k, knn_idx)
        knn_v = index_points(v, knn_idx)

        attn = torch.softmax(self.rel_mlp(repeat_q - knn_k + geo_embedding), dim=-1)
        agg_feat = torch.einsum('bcnk, bcnk -> bcn', attn, knn_v + geo_embedding) + feats
        out_feat = self.out_conv(agg_feat) + agg_feat
        out_feat = self.time_cond(time_emb, out_feat)

        return out_feat
    


class PUFM(nn.Module):
    def __init__(self, args):
        super(PUFM, self).__init__()
        self.k = 3
        self.time_dim = 3

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3*16, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.conv2 = nn.Conv1d(128, 3, 1)

        self.cond1 = ConcatSquashLinear(96, 96, self.time_dim, norm=False)
        self.cond2 = ConcatSquashLinear(256, 256, self.time_dim, norm=False)
        self.cond3 = ConcatSquashLinear(512, 512, self.time_dim, norm=False)
        self.cond4 = ConcatSquashLinear(1024, 1024, self.time_dim, norm=False)
        self.cond5 = ConcatSquashLinear(256, 256, self.time_dim, norm=False)
        self.cond6 = ConcatSquashLinear(256, 256, self.time_dim, norm=False)
        self.cond7 = ConcatSquashLinear(128, 128, self.time_dim, norm=False)
        self.act = nn.ReLU()
    
    def forward(self, query_pts, mid_pts, t):
        bs = query_pts.shape[0]
        n = query_pts.shape[2]
        t = t.unsqueeze(-1).type(torch.float)
        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)
        l0_points = get_knn_pts(16, query_pts, query_pts, return_idx=False)
        l0_points = l0_points.permute(0, 1, 3, 2).reshape(bs, -1, n)
        l0_xyz = query_pts
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.cond1(time_emb, l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.cond2(time_emb, l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.cond3(time_emb, l3_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.cond4(time_emb, l4_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_points = self.cond5(time_emb, l3_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.cond6(time_emb, l2_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.cond7(time_emb, l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = F.relu(self.conv1(l0_points))
        x = self.conv2(x)

        return x

    
class PUFM_w_attn(nn.Module):
    def __init__(self, args):
        super(PUFM_w_attn, self).__init__()
        self.k = 3
        self.time_dim = 3

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3*16, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512+512+256+256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128+128+256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32+64+256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.conv2 = nn.Conv1d(128, 3, 1)
        self.attn = Encoder_Attention(1024, encoder_bn=True, k=16, time_dim=self.time_dim, geo_dim=3)
        self.cond1 = ConcatSquashLinear(96, 96, self.time_dim, norm=False)
        self.cond2 = ConcatSquashLinear(256, 256, self.time_dim, norm=False)
        self.cond3 = ConcatSquashLinear(512, 512, self.time_dim, norm=False)
        self.cond5 = ConcatSquashLinear(256, 256, self.time_dim, norm=False)
        self.cond6 = ConcatSquashLinear(256, 256, self.time_dim, norm=False)
        self.cond7 = ConcatSquashLinear(128, 128, self.time_dim, norm=False)
        self.act = nn.ReLU()
    
    def forward(self, query_pts, mid_pts, t):
        bs = query_pts.shape[0]
        n = query_pts.shape[2]
        t = t.unsqueeze(-1).type(torch.float)
        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)
        l0_points = get_knn_pts(16, query_pts, query_pts, return_idx=False)
        l0_points = l0_points.permute(0, 1, 3, 2).reshape(bs, -1, n)
        l0_xyz = query_pts 
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.cond1(time_emb, l1_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.cond2(time_emb, l2_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.cond3(time_emb, l3_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
 
        l4_points = self.attn(l4_xyz, l4_points, time_emb, None)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_points = self.cond5(time_emb, l3_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points = self.cond6(time_emb, l2_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points = self.cond7(time_emb, l1_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        x = F.relu(self.conv1(l0_points))
        x = self.conv2(x)

        return x

    
