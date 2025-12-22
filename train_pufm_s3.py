import os
import torch
import sys
sys.path.append(os.getcwd())
import time
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import PUDataset, PUDataset_self, PUDataset_test
from models.diffusion import *
from models.rin import Denoiser_backbone
# from models.diffusion_v2 import *
from args.pufm_args import parse_pc_args
from args.utils import str2bool
from models.utils import *
from torch.cuda.amp import autocast, GradScaler
import copy
import argparse
from einops import rearrange, reduce
from tqdm import tqdm
import torch.distributions as dist
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
cd_module = chamfer_3DDist()
from emd_assignment import emd_module


def get_alignment_clean(aligner):
    @torch.no_grad()
    def align(noisy, clean):
        noisy = noisy.clone().transpose(1, 2).contiguous()
        clean = clean.clone().transpose(1, 2).contiguous()
        dis, alignment = aligner(noisy, clean, 0.01, 100)
        return alignment.detach()

    return align


def inverse_cdf_sampling(cdf, t_grid, n_steps):
    # linspace of uniform samples in [0, 1]
    linspace = torch.linspace(0, 1, n_steps + 1, device=cdf.device)
    t_sched = torch.zeros_like(linspace)

    for i, v in enumerate(linspace):
        # Find where v would be inserted to maintain order
        idx = torch.searchsorted(cdf, v, right=True).clamp(max=len(t_grid)-1)

        # Linear interpolate between t_grid[idx - 1] and t_grid[idx]
        idx0 = (idx - 1).clamp(min=0)
        idx1 = idx

        cdf0 = cdf[idx0]
        cdf1 = cdf[idx1]
        t0 = t_grid[idx0]
        t1 = t_grid[idx1]

        denom = (cdf1 - cdf0).clamp(min=1e-5)  # avoid division by zero
        t_sched[i] = t0 + (v - cdf0) / denom * (t1 - t0)

    return t_sched

def build_entropic_schedule(loss_curve, t_grid, steps):
    """
    Given loss_curve over t_grid, build an entropic time scheduler with `steps` intervals.
    Returns:
      - t_sched: Tensor of length steps+1 in [0,1]
      - cdf:     The normalized cumulative area of the loss_curve
    """
    # 1) integrate via the trapezoidal rule to get area between t-grid points
    dt       = t_grid[1:] - t_grid[:-1]                        # [N-1]
    avg_loss = 0.5 * (loss_curve[:-1] + loss_curve[1:])       # [N-1]
    area     = avg_loss * dt                                  # [N-1]

    # 2) build CDF by prepending 0, then normalize to [0,1]
    cum = torch.cat([torch.tensor([0.0], device=area.device), torch.cumsum(area, dim=0)])  # [N]
    cum = cum / cum[-1]

    # 3) invert the CDF via numpy.interp
    cum_np     = cum.cpu().numpy()         # [N]
    t_np       = t_grid.cpu().numpy() # [N]
    targets    = np.linspace(0.0, 1.0, steps + 1)    # [steps+1]
    t_sched_np = np.interp(targets, cum_np, t_np)    # linear interpolation
    t_sched = torch.from_numpy(t_sched_np).to(loss_curve.device)
    return t_sched, cum


def train(args):
    set_seed(args.seed)
    start = time.time()

    # load training data
    train_dataset = PUDataset_self(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=False,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)

    # model = PUFM(args)
    model = Denoiser_backbone(num_x=1024, z_dim=512, k=16)
    model = model.cuda()
    if args.pretrained_path is not None:
        # model.load_state_dict(torch.load(args.pretrained_path), strict=False)
        ckpt = torch.load(args.pretrained_path)
        state_dict = ckpt['state_dict']
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print('================ successfully load pretrained model=====================')

    # alignment
    aligner = emd_module.emdModule()
    emd_align = get_alignment_clean(aligner)

    # log loss
    loss_list = []
    model.eval()
    t_grid = torch.linspace(0, 1, 50+1).cuda()
    for t in t_grid:
        epoch_loss = 0
        for i, (input_random, gt_pts) in enumerate(train_loader):
            # (b, n, 3) -> (b, 3, n)
            input_random = rearrange(input_random, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
            bs = gt_pts.shape[0]
            n = gt_pts.shape[2]
            # 4x sampling
            mid_pts = midpoint_interpolate(args, input_random)
            tt = t.reshape(1, ).repeat(bs, )
            alpha = tt[:, None, None]
            #================ generate query points =================
            x_0 = mid_pts + 0.02 * torch.randn_like(mid_pts)
            # EMD align
            align_idxs = emd_align(x_0, gt_pts)
            align_idxs = align_idxs.detach().long()
            align_idxs = align_idxs.unsqueeze(1).expand(-1, 3, -1)
            x_1 = torch.gather(gt_pts, -1, align_idxs)
            # linear interpolation
            x_t = alpha * x_1 + (1 - alpha) * x_0
            #============= model output =================
            self_cond = None
            self_latents = None
            with torch.no_grad():
                model_output, self_latents = model(x_t, tt, self_cond, self_latents)
                self_latents = self_latents.detach()
                pred_delta, _ = model(x_t, tt, self_cond, self_latents)
            #============= MSE loss ================
            velocity = x_1 - x_0
            p2p_loss = F.mse_loss(pred_delta, velocity, reduction='mean')
            loss = p2p_loss 

            epoch_loss += loss.item()
        loss_list.append(epoch_loss)

    loss_tensor = torch.tensor(loss_list, dtype=torch.float32)

    # Entropic schedule: inverse CDF sampling
    t_sched, cdf = build_entropic_schedule(loss_tensor, t_grid.to(loss_tensor.device), steps=4)
    # Save t_sched for inference
    save_dict = {
        'loss': loss_tensor,
        't_sched': t_sched,
        'cdf': cdf
    }
    torch.save(save_dict, 't_sched_cdf.pt')

def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--h5_file_path', default="/data/point_cloud/PU1K/pu1k_pc_all_8192.h5", type=str, help='the path of train dataset')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate, pretrain 1e-4, distill 2e-5')
    parser.add_argument('--global_sigma', default=1.5, type=float, help='global sampling rate')
    parser.add_argument('--grad_lambda', default=0.1, type=float, help='gradient weights')
    parser.add_argument('--epochs', default=500, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--print_rate', default=50, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=1, type=int, help='model save frequency')
    parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')
    parser.add_argument('--pretrained_path', default='pretrained_model/rin_self_latent_v2.ckpt', type=str, help='the pretrained path')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['pu1k', 'pugan']

    model_args = parse_pc_args()

    reset_model_args(train_args, model_args)

    train(model_args)
