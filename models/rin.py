# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# RIN: https://arxiv.org/pdf/2212.11972
# --------------------------------------------------------
from models.utils import *
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, DropPath
import math

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.

        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class GlobalNet(nn.Module):
    def __init__(self, input_channels, k):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_channels + 3, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

    def forward(self, x, return_raw_feat=False):
        num_points = x.shape[1]
        _, sample_idx = FPS(x, self.k, return_idx=True)
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        if return_raw_feat is False:
            feat = index_points(feat, sample_idx)
            return feat.permute(0, 2, 1)
        else:
            raw_feat = feat
            feat = index_points(feat, sample_idx)
            return feat.permute(0, 2, 1), raw_feat
    
class GlobalNet_v2(nn.Module):
    def __init__(self, input_channels, k):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_channels + 3, out_channels=32, kernel_size=1),
            Sine(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            Sine(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            Sine(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            Sine(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            Sine(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            Sine(),
        )

    def forward(self, x, return_raw_feat=False):
        num_points = x.shape[1]
        _, sample_idx = FPS(x, self.k, return_idx=True)
        x = x.permute(0, 2, 1)
        feat = self.net(x)
        if return_raw_feat is False:
            feat = index_points(feat, sample_idx)
            return feat.permute(0, 2, 1)
        else:
            raw_feat = feat
            feat = index_points(feat, sample_idx)
            return feat.permute(0, 2, 1), raw_feat
        

class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            num_heads=16,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        kv_dim = dim if not kv_dim else kv_dim
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_q, x_kv):
        B, N_q, C = x_q.shape
        B, N_kv, _ = x_kv.shape
        # [B, N_q, C] -> [B, N_q, H, C/H] -> [B, H, N_q, C/H]
        q = self.wq(x_q).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B, N_kv, C] -> [B, N_kv, H, C/H] -> [B, H, N_kv, C/H]
        k = self.wk(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # [B, N_kv, C] -> [B, N_kv, H, C/H] -> [B, H, N_kv, C/H]
        v = self.wv(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # [B, H, N_q, C/H] @ [B, H, C/H, N_kv] -> [B, H, N_q, N_kv]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # [B, H, N_q, N_kv] @ [B, H, N_kv, C/H] -> [B, H, N_q, C/H]
        x = attn @ v

        # [B, H, N_q, C/H] -> [B, N_q, C]
        x = x.transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Compute_Block(nn.Module):

    def __init__(self, z_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_z1 = norm_layer(z_dim)
        self.attn = CrossAttention(
            z_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_z2 = norm_layer(z_dim)
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z):
        zn = self.norm_z1(z)
        z = z + self.drop_path(self.attn(zn, zn))
        z = z + self.drop_path(self.mlp(self.norm_z2(z)))
        return z

class Read_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_x = norm_layer(x_dim)
        self.norm_z1 = norm_layer(z_dim)
        self.attn = CrossAttention(
            z_dim, x_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_z2 = norm_layer(z_dim)
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        z = z + self.drop_path(self.attn(self.norm_z1(z), self.norm_x(x)))
        z = z + self.drop_path(self.mlp(self.norm_z2(z)))
        return z

class Write_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_z = norm_layer(z_dim)
        self.norm_x1 = norm_layer(x_dim)
        self.attn = CrossAttention(
            x_dim, z_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_x2 = norm_layer(x_dim)
        mlp_hidden_dim = int(x_dim * mlp_ratio)
        self.mlp = Mlp(in_features=x_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, z, x):
        x = x + self.drop_path(self.attn(self.norm_x1(x), self.norm_z(z)))
        x = x + self.drop_path(self.mlp(self.norm_x2(x)))
        return x

class RCW_Block(nn.Module):

    def __init__(self, z_dim, x_dim, num_compute_layers=4, num_heads=16, 
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.read = Read_Block(z_dim, x_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                   attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.write = Write_Block(z_dim, x_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                   attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.compute = nn.ModuleList([
            Compute_Block(z_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, 
                                attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(num_compute_layers)
        ])

    def forward(self, z, x):
        z = self.read(z, x)
        for layer in self.compute:
            z = layer(z)
        x = self.write(z, x)
        return z, x

def unshuffle_points(x_shuffled, perms):
    # x_shuffled: (bs, dim, N)
    x_shuffled = x_shuffled.permute(0, 2, 1)
    bs, N = perms.shape
    inv_perms = torch.zeros_like(perms)
    for b in range(bs):
        inv_perms[b, perms[b]] = torch.arange(N, device=perms.device)
    inv_perms_exp = inv_perms.unsqueeze(1).expand(-1, x_shuffled.shape[1], -1)  # (bs, dim, N)
    x_restored = torch.gather(x_shuffled, dim=2, index=inv_perms_exp)
    x_restored = x_restored.permute(0, 2, 1)
    return x_restored


class Denoiser_backbone(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, k=16,
                 num_z=256, num_x=4096, z_dim=768, x_dim=512, 
                 num_blocks=6, num_compute_layers=4, num_heads=8, 
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.num_z = num_z
        self.num_x = num_x
        self.z_dim = z_dim

        # global blocks
        self.global_cond = GlobalNet(0, k=k)
        
        # input blocks
        self.input_proj = nn.Linear(input_channels, x_dim)
        self.ln_pre = nn.LayerNorm(x_dim)
        self.z_init = nn.Parameter(torch.zeros(1, num_z, z_dim))

        # timestep embedding
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.time_embed = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim)

        # RCW blocks
        self.latent_mlp = Mlp(in_features=z_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ln_latent = nn.LayerNorm(z_dim)
        self.blocks = nn.ModuleList([
            RCW_Block(z_dim, x_dim, num_compute_layers=num_compute_layers, 
                      num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                      drop=drop, attn_drop=attn_drop, drop_path=drop_path, 
                      act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(num_blocks)
        ])
        # for param in self.blocks.parameters():
        #     param.requires_grad = False
        # for param in self.global_cond.parameters():
        #     param.requires_grad = False

        # output blocks
        self.ln_post = nn.LayerNorm(x_dim)
        self.output_proj = nn.Linear(x_dim, output_channels)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.z_init, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        nn.init.constant_(self.ln_latent.weight, 0)
        nn.init.constant_(self.ln_latent.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, t, self_cond, prev_latent, perms=None):
        """
        Forward pass of the model.

        Parameters:
        x: [B, C_in, num_x]
        t: [B]
        self_cond: [B, num_cond, num_x]
        prev_latent: [B, num_z + num_cond + 1, C_latent]

        Returns:
        x_denoised: [B, C_out, num_x]
        """
        x = x.permute(0, 2, 1)
        B, num_x, _ = x.shape
        if self_cond is None:
            self_cond = x
        else:
            self_cond = self_cond.permute(0, 2, 1)
        cond = self.global_cond(self_cond)
        num_cond = cond.shape[1]
        assert num_x == self.num_x
        if prev_latent is not None:
            _, num_z, _ = prev_latent.shape
            assert num_z == self.num_z + num_cond + 1
        else:
            prev_latent = torch.zeros(B, self.num_z + num_cond + 1, self.z_dim).to(x.device)
        
        # timestep embedding, [B, 1, z_dim]
        t_embed = self.time_embed(timestep_embedding(t, self.z_dim)).unsqueeze(1)

        # project x -> [B, num_x, C_x]
        x = self.input_proj(x)
        x = self.ln_pre(x)

        # latent self-conditioning
        z = self.z_init.repeat(B, 1, 1) # [B, num_z, z_dim]
        z = torch.cat([z, cond, t_embed], dim=1) # [B, num_z + num_cond + 1, z_dim]
        prev_latent = prev_latent + self.latent_mlp(prev_latent.detach())
        z = z + self.ln_latent(prev_latent)
        # compute
        for blk in self.blocks:
            z, x = blk(z, x)
        # output 
        if perms is not None:
            x = unshuffle_points(x, perms)
        x = self.ln_post(x)
        x_denoised = self.output_proj(x)
        x_denoised = x_denoised.permute(0, 2, 1)
        return x_denoised, z
    