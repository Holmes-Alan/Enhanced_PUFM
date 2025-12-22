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
    knn_pts = get_knn_pts(k+1, pcd, pcd, return_idx=False)
    knn = knn_pts[:, :, :, 1:]
    mean = knn.mean(dim=3, keepdim=True)
    diffs = knn - mean
    cov = torch.matmul(diffs.permute(0, 2, 1, 3), diffs.permute(0, 2, 3, 1)) / k # b*N*3*3
    I = 1e-6 * torch.eye(3, device=pcd.device).view(1, 1, 3, 3)
    cov = 0.5 * (cov + cov.transpose(-1, -2)) + I
    eigvals, _ = torch.linalg.eigh(cov)
    curvature = eigvals[..., 0] / eigvals.sum(dim=-1)

    return curvature  # (B, N)



def pcd_update(args, model, interpolated_pcd, original_pcd, t, steps, dt):
    # interpolated_pcd: (b, 3, n)
    pcd_pts_num = interpolated_pcd.shape[-1]
    # 1024
    patch_pts_num = args.num_points * 4
    # extract patch
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    # FPS: (b, 3, fps_pts_num), ensure seeds have a good coverage
    seed = FPS(interpolated_pcd, sample_num)
    # (b*fps_pts_num, 3, patch_pts_num)
    patches = extract_knn_patch(patch_pts_num, interpolated_pcd, seed)

    # normalize each patch
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    # fix the parameters of model while updating the patches
    for param in model.parameters():
        param.requires_grad = False

    updated_patch = patches.clone() 
    bs = patches.shape[0]
    # 1st order sampling
    alpha = t * torch.ones(bs, device="cuda")
    self_cond = latent_z = None

    if args.mid_ode is False:
        pred, latent_z = model(updated_patch, alpha, self_cond, latent_z)
        updated_patch = updated_patch + dt * pred
    else:
        pred, latent_z = model(updated_patch, alpha, self_cond, latent_z)
        updated_patch = updated_patch + (1 / (2*steps)) * pred
        pred, latent_z = model(updated_patch, alpha + (1 / (2*steps)), self_cond, latent_z)
        updated_patch = updated_patch + (1 / steps) * pred
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
    # transform to original scale and merge patches
    updated_patch = updated_patch.clamp(-1, 1)
    updated_patch = centroid + updated_patch * furthest_distance
    # (3, m)
    updated_pcd = rearrange(updated_patch, 'b c n -> c (b n)').contiguous()
    # post process: (1, 3, n)
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
    alpha = 0.01  # adjust
    for t in tqdm(range(len(t_sched) - 1), "sampling loop"):
        t_input = t_sched[t]
        dt = t_sched[t+1] - t_sched[t]
        new_pcd = pcd_update(args, model, updated_pcd, input_pcd, t_input, steps, dt)
        
        if args.bp == 'bp':
            updated_pcd = new_pcd
            updated_pcd = updated_pcd.detach().requires_grad_()
            knn_pts, idx = get_knn_pts(1, interpolated_pcd, updated_pcd, return_idx=True)
            knn_pts = knn_pts.squeeze(-1)
            manifold_loss = torch.mean(torch.abs(knn_pts - updated_pcd))
            grad = torch.autograd.grad(manifold_loss, updated_pcd, retain_graph=True, create_graph=True)[0]
            grad = grad / (grad.norm(dim=2, keepdim=True) + 1e-8)
            updated_pcd = updated_pcd - 0.01 * grad
            updated_pcd = updated_pcd.detach()
        elif args.bp == 'curvature':
            residue_pcd = new_pcd - updated_pcd
            curvature = estimate_curvature(new_pcd).unsqueeze(1)  # (B,1,N)
            updated_pcd = updated_pcd + residue_pcd * (1 + alpha * curvature)
        else:
            updated_pcd = new_pcd

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

        # CD evaluation
        cd_p, dist, _,_ = cd_module(upsampled_pcd.permute(0, 2, 1), gt_pcd)
        dist = (cd_p + dist) / 2.0
        cd = dist.mean().detach().cpu().item()
        test_loss += cd
        count += 1
            
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
    logger.info('Average chamfer dis: {}'.format(test_loss / len(test_input_path)))
    logger.info('Average pcd time: {}s'.format(total_pcd_time / len(test_input_path)))

 
def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    parser.add_argument('--dataset', default='pugan', type=str, help='pu1k or pugan')
    parser.add_argument('--test_input_path', default='/data/point_cloud/PUGAN/pufm_rin_bp/pcd', type=str, help='the test input data path')
    parser.add_argument('--test_gt_path', default='/data/point_cloud/PUGAN_4x/test_pc_v2/input_2048_16X/gt_32768', type=str, help='the test gt data path')
    parser.add_argument('--model', default='rin_self_latent_v2', type=str, help='the pretrained model')
    parser.add_argument('--ckpt_folder', default='pretrained_model', type=str, help='the pretrained model folder')
    parser.add_argument('--save_dir', default='/data/point_cloud/PUGAN_4x/pufm_rin_bp/', type=str, help='save upsampled point cloud')
    parser.add_argument('--truncate_distance', default=True, type=str2bool, help='whether truncate distance')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--step', default=6, type=int, help='flow matching sampling steps')
    parser.add_argument('--mid_ode', default=False, type=str2bool, help='whether use midpoint ODE')
    parser.add_argument('--bp', default='bp', type=str, help='whether use back projection [bp, curvature or none]')
    parser.add_argument('--sdf', default=False, type=str2bool, help='whether use INR sdf to improve quality')
    parser.add_argument('--double_4X', default=False, type=str2bool, help='conduct 4X twice to get 16X')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    test_args = parse_test_args()

    model_args = parse_pc_args()

    reset_model_args(test_args, model_args)

    test(model_args)