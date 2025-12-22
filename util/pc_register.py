import imageio
import matplotlib.pyplot
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d
import imageio
from emd_assignment import emd_module
from einops import rearrange
from models.utils import *


def get_alignment_clean(aligner):
    @torch.no_grad()
    def align(noisy, clean):
        noisy = noisy.clone().transpose(1, 2).contiguous()
        clean = clean.clone().transpose(1, 2).contiguous()
        dis, alignment = aligner(noisy, clean, 0.01, 100)
        return alignment.detach()

    return align


def numpy_to_pc(points, x1_centroid, x1_furthest_distance):
    points = points.permute(1, 0).unsqueeze(0)
    points = points * x1_furthest_distance + x1_centroid
    points = points.squeeze(0).permute(1, 0)
    points = points.detach().cpu().numpy()
    pc = o3d.geometry.PointCloud()
    points = o3d.utility.Vector3dVector(points)
    pc.points = points
    return pc

# data loading
p_0 = "/data/point_cloud/PUGAN/test_pc_v2/input_2048_4X/gt_8192/11509_Panda_v4.xyz"
p_1 = "/data/point_cloud/PUGAN/test_pc_v2/input_2048_4X/gt_8192/chair.xyz"
Ndata = 8192
pc = o3d.io.read_point_cloud(p_0)
x0 = np.asarray(pc.points, dtype=np.float32)
pc = o3d.io.read_point_cloud(p_1)
x1 = np.asarray(pc.points, dtype=np.float32)

x0 = torch.from_numpy(x0).float().cuda()
x1 = torch.from_numpy(x1).float().cuda()

x0 = rearrange(x0, 'n c -> c n').contiguous()
x0 = x0.unsqueeze(0)
x0, x0_centroid, x0_furthest_distance = normalize_point_cloud(x0)

x1 = rearrange(x1, 'n c -> c n').contiguous()
x1 = x1.unsqueeze(0)
x1, x1_centroid, x1_furthest_distance = normalize_point_cloud(x1)


# architecture
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3+1,64)  # input = (x_alpha, alpha)
        self.linear2 = torch.nn.Linear(64, 64)
        self.linear3 = torch.nn.Linear(64, 64)
        self.linear4 = torch.nn.Linear(64, 64)
        self.output  = torch.nn.Linear(64, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x, alpha):
        alpha = alpha.type(x.dtype)
        res = torch.cat([x, alpha], dim=1)
        res = self.relu(self.linear1(res))
        res = self.relu(self.linear2(res))
        res = self.relu(self.linear3(res))
        res = self.relu(self.linear4(res))
        res = self.output(res)
        return res
    

# print(x0.dtype)
# print(x1.dtype)
# print(torch.cuda.is_available())
# a = torch.rand(8, 3)
# D = NN()
# b = D(a, torch.ones(8, 1))
# print(b.shape)


# allocating the neural network D
D = NN().to("cuda")
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.001)

# EMD align
# aligner = emd_module.emdModule()
# emd_align = get_alignment_clean(aligner)
# align_idxs = emd_align(x0, x1)
# align_idxs = align_idxs.detach().long()
# align_idxs = align_idxs.unsqueeze(1).expand(-1, 3, -1)
# x1 = torch.gather(x1, -1, align_idxs)
x1 = x1.squeeze(0).permute(1, 0)
x0 = x0.squeeze(0).permute(1, 0)

# training loop
batchsize = 512
for iteration in tqdm(range(65536), "training loop"):
    idx = np.random.randint(0, Ndata, batchsize)
    x0_data = x0[idx, :]
    x1_data = x1[idx, :]
    alpha = torch.rand(batchsize, 1, device="cuda")

    x_alpha = (1-alpha) * x0_data + alpha * x1_data

    loss = torch.sum( (D(x_alpha, alpha) - (x1_data-x0_data))**2 )
    optimizer_D.zero_grad()
    loss.backward()
    optimizer_D.step()


# sampling loop
batchsize = 8192
with torch.no_grad():
    # starting points x_alpha = x_0
    x_alpha = x0[np.random.randint(0, Ndata, batchsize), :]
    # loop
    T = 128
    for t in tqdm(range(T), "sampling loop"):

        # export plot
        save_path = "/data/point_cloud/teaser/register/no_emd/pcd_" + str(t) + ".xyz"
        pcd = numpy_to_pc(x_alpha, x1_centroid, x1_furthest_distance)
        o3d.io.write_point_cloud(filename=save_path, pointcloud=pcd)

        # current alpha value
        alpha = t / T * torch.ones(batchsize, 1, device="cuda")

        # update
        x_alpha = x_alpha + 1/T * D(x_alpha, alpha)

    save_path = "/data/point_cloud/teaser/register/no_emd/pcd_" + str(T) + ".xyz"
    pcd = numpy_to_pc(x_alpha, x1_centroid, x1_furthest_distance)
    o3d.io.write_point_cloud(filename=save_path, pointcloud=pcd)



# filenames = ["pcd_" + str(t) + ".jpg" for t in range(128+1)]
# with imageio.get_writer('/data/point_cloud/teaser/register/emd_vis/p0_p1.gif', mode='I') as writer:
#     for filename in filenames:
#         name = '/data/point_cloud/teaser/register/emd_vis/' + filename
#         image = imageio.imread(name)
#         writer.append_data(image)