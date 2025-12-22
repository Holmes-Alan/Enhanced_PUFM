import torch
import numpy as np
from glob import glob
import os
import open3d
from models.utils import *
from models.rin import Denoiser_backbone
from einops import rearrange
from time import time
from args.pufm_args import parse_pc_args
from args.utils import str2bool
from tqdm import tqdm
import argparse
import plyfile
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
cd_module = chamfer_3DDist()
from models.utils_ed import Batched_ED
batched_ed = Batched_ED.apply

def numpy_to_pc(points):
    pc = open3d.geometry.PointCloud()
    points = open3d.utility.Vector3dVector(points)
    pc.points = points
    return pc

def save_ply(points, filename, colors=None, normals=None):
    vertex = np.core.records.fromarrays(points.transpose(
        1, 0), names='x, y, z', formats='f4, f4, f4')
    num_vertex = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(
            normals.transpose(1, 0), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == num_vertex
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        assert len(colors) == num_vertex
        if colors.max() <= 1:
            colors = colors * 255
        if colors.shape[1] == 4:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue, alpha', formats='u1, u1, u1, u1')
        else:
            vertex_color = np.core.records.fromarrays(colors.transpose(
                1, 0), names='red, green, blue', formats='u1, u1, u1')
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(num_vertex, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    ply.write(filename)


def estimate_curvature(pcd, k=16):
    # pcd: (B, 3, N)
    B, _, N = pcd.shape
    curvature = torch.zeros(B, N, device=pcd.device)

    for b in range(B):
        points = pcd[b].T  # (N, 3)
        # Compute k-NN distances
        dists = torch.cdist(points, points)  # (N, N)
        knn_idx = dists.topk(k+1, largest=False).indices[:,1:]  # skip self
        for i in range(N):
            neighbors = points[knn_idx[i]]  # (k, 3)
            cov = torch.cov(neighbors.T)
            eigvals, _ = torch.linalg.eigh(cov)
            curvature[b, i] = eigvals[0] / eigvals.sum()  # λ1 / (λ1+λ2+λ3)
    return curvature  # (B, N)


def pcd_update(args, model, interpolated_pcd, original_pcd, t, steps, dt):
    # interpolated_pcd: (b, 3, n)
    pcd_pts_num = interpolated_pcd.shape[-1]
    patch_pts_num = args.num_points * 4
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(interpolated_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, interpolated_pcd, seed)

    patches, centroid, furthest_distance = normalize_point_cloud(patches)

    for param in model.parameters():
        param.requires_grad = False

    updated_patch = patches.clone()
    bs = patches.shape[0]
    alpha = t * torch.ones(bs, device="cuda")
    self_cond = latent_z = None

    # -------------------------------
    # Batched model inference
    # -------------------------------
    batch_size = args.batch_size  # custom batch size
    preds, latents = [], []
    for i in range(0, bs, batch_size):
        end = min(i + batch_size, bs)
        patch_batch = updated_patch[i:end]
        alpha_batch = alpha[i:end]
        sc_batch = None if self_cond is None else self_cond[i:end]
        latent_batch = None if latent_z is None else latent_z[i:end]

        if args.mid_ode is False:
            pred_b, latent_b = model(patch_batch, alpha_batch, sc_batch, latent_batch)
            patch_batch = patch_batch + dt * pred_b
        else:
            pred_b, latent_b = model(patch_batch, alpha_batch, sc_batch, latent_batch)
            patch_batch = patch_batch + (1 / (2 * steps)) * pred_b
            pred_b, latent_b = model(patch_batch, alpha_batch + (1 / (2 * steps)), sc_batch, latent_batch)
            patch_batch = patch_batch + (1 / steps) * pred_b

        preds.append(patch_batch.detach())
        latents.append(latent_b)

    updated_patch = torch.cat(preds, dim=0)
    latent_z = torch.cat([lz for lz in latents if lz is not None], dim=0) if latents[0] is not None else None
    # -------------------------------

    # manifold optimize
    if args.bp is True:
        updated_patch = updated_patch.detach().requires_grad_()
        knn_pts, idx = get_knn_pts(1, patches, updated_patch, return_idx=True)
        knn_pts = knn_pts.squeeze(-1)
        manifold_loss = torch.mean(torch.abs(knn_pts - updated_patch))
        grad = torch.autograd.grad(manifold_loss, updated_patch, retain_graph=True, create_graph=True)[0]
        grad = grad / (grad.norm(dim=2, keepdim=True) + 1e-8)
        updated_patch = updated_patch - 0.01 * grad
        updated_patch = updated_patch.detach()
    else:
        updated_patch = updated_patch.detach()

    updated_patch = updated_patch.clamp(-1, 1)
    updated_patch = centroid + updated_patch * furthest_distance

    updated_pcd = rearrange(updated_patch, 'b c n -> c (b n)').contiguous()
    output_pts_num = interpolated_pcd.shape[-1]
    updated_pcd = updated_pcd.unsqueeze(0)
    updated_pcd = torch.cat((updated_pcd, original_pcd), dim=2)
    updated_pcd = FPS(updated_pcd, output_pts_num)

    return updated_pcd


def pcd_upsample(args, model, input_pcd, t_sched):
    # interpolate: (b, 3, m)
    interpolated_pcd = midpoint_interpolate(args, input_pcd)
    # update: (b, 3, m)
    steps = args.step
    updated_pcd = interpolated_pcd

    for t in tqdm(range(len(t_sched) - 1), "sampling loop"):
        t_input = t_sched[t]
        dt = t_sched[t+1] - t_sched[t]
        updated_pcd = pcd_update(args, model, updated_pcd, input_pcd, t_input, steps, dt)

    return updated_pcd


def test(args):
    model = Denoiser_backbone(num_x=1024, z_dim=512, k=16).cuda()

    model_path = os.path.join(args.ckpt_folder, args.model+'.ckpt')
    ckpt = torch.load(model_path)
    state_dict = ckpt['state_dict']
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # load saved schedule
    saved = torch.load('t_sched_cdf.pt')
    t_sched = saved['t_sched'] 
    cdf = saved['cdf']
    targets = np.linspace(0.0, 1.0, args.step + 1)
    t_np = np.linspace(0.0, 1.0, 50 + 1)
    t_sched = np.interp(targets, cdf, t_np)

    # test input data path list
    test_input_path = glob(os.path.join(args.test_input_path, '*.xyz'))
    test_input_path = sorted(test_input_path)
    # conduct 4X twice to get 16X
    if args.up_rate == 16:
        args.up_rate = 4
        args.double_4X = True

    # log
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = get_logger('test', save_dir)

    # save upsampled point cloud
    pcd_dir = os.path.join(save_dir, 'pcd')
    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)
    # record time
    total_pcd_time = 0.0
    test_loss = 0.0
    count = 0
    # test
    for i, path in tqdm(enumerate(test_input_path), desc='Processing'):
        start = time()
        # each time upsample one point cloud
        pcd = open3d.io.read_point_cloud(path)
        pcd_name = path.split('/')[-1]
        input_pcd = np.array(pcd.points)
        input_pcd = torch.from_numpy(input_pcd).float().cuda()
        # gt point clouds
        gt_path = os.path.join(args.test_gt_path, pcd_name)
        gt = open3d.io.read_point_cloud(gt_path)
        gt_pcd = np.array(gt.points)
        gt_pcd = torch.from_numpy(gt_pcd).float().cuda()
        gt_pcd = gt_pcd.unsqueeze(0)
        # (n, 3) -> (3, n)
        input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
        # (3, n) -> (1, 3, n)
        input_pcd = input_pcd.unsqueeze(0)
        # normalize input
        input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
        # upsample
        upsampled_pcd = pcd_upsample(args, model, input_pcd, t_sched)
        upsampled_pcd = centroid + upsampled_pcd * furthest_distance

        # upsample 16X, conduct 4X twice
        if args.double_4X == True:
            upsampled_pcd, centroid, furthest_distance = normalize_point_cloud(upsampled_pcd)
            upsampled_pcd = pcd_upsample(args, model, upsampled_pcd, t_sched)
            upsampled_pcd = centroid + upsampled_pcd * furthest_distance
            
        # (b, 3, n) -> (n, 3)
        upsampled_pcd = rearrange(upsampled_pcd.squeeze(0), 'c n -> n c').contiguous()
        upsampled_pcd = upsampled_pcd.detach().cpu().numpy()
        # save path
        save_path = os.path.join(pcd_dir, pcd_name)
        upsampled_pcd = numpy_to_pc(upsampled_pcd)
        open3d.io.write_point_cloud(filename=save_path, pointcloud=upsampled_pcd)
        # time
        end = time()
        total_pcd_time += end - start
    logger.info('Average pcd time: {}s'.format(total_pcd_time / len(test_input_path)))

 
def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--test_input_path', default='/data/point_cloud/ScanNet/input_xyz', type=str, help='the test input data path')
    parser.add_argument('--test_gt_path', default='/data/point_cloud/ScanNet/gt_xyz', type=str, help='the test gt data path')
    parser.add_argument('--model', default='rin_self_latent_v2', type=str, help='the pretrained model [pufm, pufm_w_attn]')
    parser.add_argument('--ckpt_folder', default='pretrained_model', type=str, help='the pretrained model folder')
    parser.add_argument('--save_dir', default='/data/point_cloud/ScanNet/pufm_rin_v2/', type=str, help='save upsampled point cloud')
    parser.add_argument('--truncate_distance', default=True, type=str2bool, help='whether truncate distance')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--step', default=6, type=int, help='flow matching sampling steps')
    parser.add_argument('--mid_ode', default=False, type=str2bool, help='whether use midpoint ODE')
    parser.add_argument('--bp', default=False, type=str2bool, help='whether use back projection to improve quality')
    parser.add_argument('--sdf', default=False, type=str2bool, help='whether use INR sdf to improve quality')
    parser.add_argument('--double_4X', default=False, type=str2bool, help='conduct 4X twice to get 16X')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    test_args = parse_test_args()

    model_args = parse_pc_args()

    reset_model_args(test_args, model_args)

    test(model_args)
