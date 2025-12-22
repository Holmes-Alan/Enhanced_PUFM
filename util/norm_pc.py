import os
import open3d
import numpy as np
import torch
from models.utils import *
from einops import rearrange


def numpy_to_pc(points):
    pc = open3d.geometry.PointCloud()
    points = open3d.utility.Vector3dVector(points)
    pc.points = points
    return pc

old_pcd = open3d.io.read_point_cloud('/data/point_cloud/real_pc_v2/pudm/002854.xyz')
ref_pcd = open3d.io.read_point_cloud('/data/point_cloud/real_pc_v2/00/pc/002854.xyz')

input_pcd = np.array(old_pcd.points)
input_pcd = torch.from_numpy(input_pcd).float().cuda()

target_pcd = np.array(ref_pcd.points)
target_pcd = torch.from_numpy(target_pcd).float().cuda()

input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
input_pcd = input_pcd.unsqueeze(0)
input_pcd_norm, _, _ = normalize_point_cloud(input_pcd)

target_pcd = rearrange(target_pcd, 'n c -> c n').contiguous()
target_pcd = target_pcd.unsqueeze(0)
target_pcd, centroid, furthest_distance = normalize_point_cloud(target_pcd)

upsampled_pcd = centroid + input_pcd_norm * furthest_distance

upsampled_pcd = rearrange(upsampled_pcd.squeeze(0), 'c n -> n c').contiguous()
upsampled_pcd = upsampled_pcd.detach().cpu().numpy()
# save path
save_path = '/data/point_cloud/real_pc_v2/pudm_new/002854.xyz'
# save_ply(upsampled_pcd, save_path.replace('xyz', 'ply'))
upsampled_pcd = numpy_to_pc(upsampled_pcd)
open3d.io.write_point_cloud(filename=save_path, pointcloud=upsampled_pcd)



# old_pcd = open3d.io.read_point_cloud('/data/point_cloud/real_pc/m333.xyz')

# input_pcd = np.array(old_pcd.points)
# input_pcd = torch.from_numpy(input_pcd).float().cuda()

# input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
# input_pcd = input_pcd.unsqueeze(0)
# input_pcd = input_pcd + 0.02*torch.randn_like(input_pcd)
# # input_pcd_norm, centroid, furthest_distance = normalize_point_cloud(input_pcd)

# upsampled_pcd = input_pcd

# upsampled_pcd = rearrange(upsampled_pcd.squeeze(0), 'c n -> n c').contiguous()
# upsampled_pcd = upsampled_pcd.detach().cpu().numpy()
# # save path
# save_path = '/data/point_cloud/real_pc/m333_noise.xyz'
# # save_ply(upsampled_pcd, save_path.replace('xyz', 'ply'))
# upsampled_pcd = numpy_to_pc(upsampled_pcd)
# open3d.io.write_point_cloud(filename=save_path, pointcloud=upsampled_pcd)