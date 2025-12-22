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


def train(args):
    set_seed(args.seed)
    start = time.time()

    # load training data
    train_dataset = PUDataset_self(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)
    # load testing data
    test_dataset = PUDataset_test(args)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   shuffle=False,
                                                   batch_size=1, # args.batch_size
                                                   num_workers=args.num_workers)

    # set up folders for checkpoints and logs
    str_time = datetime.now().isoformat()
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_dir = os.path.join(output_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    logger = get_logger('train', log_dir)
    logger.info('Experiment ID: %s' % (str_time))

    # create model 
    logger.info('========== Build Model ==========')
    model = Denoiser_backbone(num_x=1024, z_dim=512, k=16)
    model = model.cuda()
    if args.pretrained_path is not None:
        ckpt = torch.load(args.pretrained_path)
        state_dict = ckpt['state_dict']
        new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print('================ successfully load pretrained model=====================')
    # get the parameter size
    para_num = sum([p.numel() for p in model.parameters()])
    logger.info("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))
    # log
    logger.info(args)
    logger.info(repr(model))
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    # alignment
    aligner = emd_module.emdModule()
    emd_align = get_alignment_clean(aligner)

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    best_valid_loss = np.Inf
    # train
    logger.info('========== Begin Training ==========')
    for epoch in range(args.epochs):
        logger.info('********* Epoch %d *********' % (epoch + 1))
        # epoch loss
        epoch_p2p_loss = 0.0
        epoch_loss = 0.0
        model.train()

        for i, (input_random, gt_pts) in enumerate(train_loader):
            # (b, n, 3) -> (b, 3, n)
            input_random = rearrange(input_random, 'b n c -> b c n').contiguous().float().cuda()
            gt_pts = rearrange(gt_pts, 'b n c -> b c n').contiguous().float().cuda()
            bs = gt_pts.shape[0]
            n = gt_pts.shape[2]
            # 4x sampling
            mid_pts = midpoint_interpolate(args, input_random)
            # zero sampling
            t = torch.zeros(size=(bs, )).cuda() 
            alpha = t[:, None, None]
            #================ generate query points =================
            noise_pts = mid_pts + 0.02 * torch.randn_like(mid_pts)
            x_0 = noise_pts
            #============= model output =================
            self_cond = None
            self_latents = None
            with torch.no_grad():
                model_output, self_latents = model(x_0, t, self_cond, self_latents)
                self_latents = self_latents.detach()

            pred_delta, _ = model(x_0, t, self_cond, self_latents)
            #============= MSE loss ================
            pred_pts = x_0 + pred_delta
            dist1, dist2, _, _ = cd_module(pred_pts.permute(0, 2, 1), gt_pts.permute(0, 2, 1))
            p2p_loss = (dist1 + dist2) / 2.
            p2p_loss = reduce(p2p_loss, 'b ... -> b (...)', 'mean')
            p2p_loss = torch.sum(p2p_loss)
            loss = p2p_loss 

            epoch_loss += loss.item()
            epoch_p2p_loss += p2p_loss.item()

            # update parameters
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.step_ema(ema_model, model)

            # log
            writer.add_scalar('train/loss', loss, i)
            writer.add_scalar('train/p2p_loss', p2p_loss, i)
            writer.flush()
            if (i+1) % args.print_rate == 0:
                logger.info("epoch: %d/%d, iters: %d/%d, loss: %f, p2p_loss: %f" %
                      (epoch + 1, args.epochs, i + 1, len(train_loader), 
                       epoch_loss / (i+1), epoch_p2p_loss / (i+1)))

        # log
        interval = time.time() - start
        logger.info("epoch: %d/%d, avg loss: %f, time: %d mins %.1f secs" %
          (epoch + 1, args.epochs, 
           epoch_loss / len(train_loader),
           interval / 60, interval % 60))

        # testing
        if (epoch + 1) % args.save_rate == 0:
            count = 0
            test_loss = 0
            model.eval()
            for i, (input_pts, gt_pts) in enumerate(test_loader):
                input_pts = rearrange(input_pts, 'b n c -> b c n').contiguous().float().cuda()
                gt_pts = gt_pts.float().cuda()
                mid_pts = midpoint_interpolate(args, input_pts)
                # #=============================== process as patch ================================
                pcd_pts_num = mid_pts.shape[-1]
                # 1024
                patch_pts_num = args.num_points * 4
                # extract patch
                sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
                # FPS: (b, 3, fps_pts_num), ensure seeds have a good coverage
                seed = FPS(mid_pts, sample_num)
                # (b*fps_pts_num, 3, patch_pts_num)
                patches = extract_knn_patch(patch_pts_num, mid_pts, seed)
                patches, centroid, furthest_distance = normalize_point_cloud(patches)
                updated_patch = patches.clone()
                bs = patches.shape[0]
                n = gt_pts.shape[1]
                steps = 5
                latent_z = None
                self_cond = None
                with torch.no_grad():
                    for t in tqdm(range(steps), "sampling loop"):
                        alpha = t / steps * torch.ones(bs, device="cuda")
                        pred, latent_z = model(updated_patch, alpha, self_cond, latent_z)
                        updated_patch = updated_patch + (1 / steps) * pred
                updated_patch = updated_patch.clamp(-1, 1)
                updated_patch = centroid + updated_patch * furthest_distance
                updated_pcd = rearrange(updated_patch, 'b c n -> c (b n)').contiguous()
                # post process: (1, 3, n)
                output_pts_num = mid_pts.shape[-1]
                updated_pcd = FPS(updated_pcd.unsqueeze(0), output_pts_num)
                # CD evaluation
                dist1, dist2, _,_ = cd_module(updated_pcd.permute(0, 2, 1), gt_pts)
                dist = (dist1 + dist2) / 2.0
                cd = dist.mean().detach().cpu().item()
                test_loss += cd
                count += 1
            test_loss = test_loss / count
            writer.add_scalar('test/loss', test_loss, epoch)
            logger.info("TEST epoch: %d/%d, avg loss: %.5E" %
                        (epoch + 1, args.epochs, 
                         test_loss))
        # save checkpoint
        if (epoch + 1) % args.save_rate == 0:
            if test_loss < best_valid_loss:
                model_name = 'ckpt-epoch-%d.pth' % (epoch+1)
                ema_model_name = 'ema-ckpt-epoch-%d.pth' % (epoch+1)
                model_path = os.path.join(ckpt_dir, model_name)
                ema_model_path = os.path.join(ckpt_dir, ema_model_name)
                torch.save(model.state_dict(), model_path)
                # torch.save(ema_model.state_dict(), ema_model_path)
                best_valid_loss = test_loss


def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--h5_file_path', default="/data/point_cloud/PU1K/pu1k_pc_all_8192.h5", type=str, help='the path of train dataset')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning rate, pretrain 1e-4, distill 2e-5')
    parser.add_argument('--global_sigma', default=1.5, type=float, help='global sampling rate')
    parser.add_argument('--grad_lambda', default=0.1, type=float, help='gradient weights')
    parser.add_argument('--epochs', default=500, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
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
