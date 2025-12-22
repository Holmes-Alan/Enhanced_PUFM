import open3d as o3d
import numpy as np
from os.path import join
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRasterizer, MeshRenderer,
    PointsRasterizationSettings, PointsRasterizer, PointsRenderer,
    SoftPhongShader, TexturesVertex, AlphaCompositor, look_at_view_transform
)
from os import listdir
from pytorch3d.structures import Pointclouds

def is_pc_file(filename):
    return any(filename.endswith(extension) for extension in [".ply", '.xyz'])

def compute_point_to_surface_distances(points, mesh):
    # Sample points from the mesh surface for distance computation
    mesh_sample_points = np.asarray(mesh.sample_points_uniformly(number_of_points=10000).points)
    
    # Build a KDTree for the sampled mesh points
    tree = cKDTree(mesh_sample_points)
    
    # Query the nearest neighbor distances for the point cloud
    distances, _ = tree.query(points, k=1)
    
    return distances

def simulate_camera_projection_with_pytorch3d(points, distances, camera_position, camera_orientation, image_width, image_height, focal_length):
    """
    Simulate a camera projection of the 3D point cloud using PyTorch3D.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        distances (np.ndarray): Point-to-surface distances for color mapping.
        camera_position (np.ndarray): 3D position of the camera.
        camera_orientation (np.ndarray): 3x3 rotation matrix for camera orientation.
        image_width (int): Width of the output image.
        image_height (int): Height of the output image.
        focal_length (float): Focal length of the camera.

    Returns:
        np.ndarray: Rendered image as a 2D array.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert points and distances to PyTorch tensors
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    distances_tensor = torch.tensor(distances, dtype=torch.float32, device=device).unsqueeze(1)

    # Normalize distances for color mapping
    distances_normalized = (distances_tensor - distances_tensor.min()) / (distances_tensor.max() - distances_tensor.min())
    colors = plt.cm.brg(distances_normalized.cpu().numpy())[:, :3]
    colors_tensor = torch.tensor(colors, dtype=torch.float32, device=device)

    # Define camera
    cameras = FoVPerspectiveCameras(
        R=camera_orientation,
        T=camera_position,
        device=device,
        # znear=0.01,
        fov=6
    )

    # Define rasterization settings
    raster_settings = PointsRasterizationSettings(
        image_size=(image_width, image_height),
        radius=0.008,
        points_per_pixel=60,
    )

    # Create a point cloud object
    colors_tensor = colors_tensor.squeeze(1)
    # colors_tensor = colors_tensor[:, :2]
    point_cloud = Pointclouds(points=[points_tensor], features=[colors_tensor])

    # Create a points rasterizer and renderer
    compositor = AlphaCompositor(background_color=(1.0, 1.0, 1.0))  # Set background color to white
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

    # Render the points
    rendered_image = renderer(point_cloud).clamp(0, 1).cpu().numpy()
    rendered_image = rendered_image[0, ..., :3]

    return rendered_image

def visualize_camera_projection(rendered_image, output_path):
    # Save the rendered image
    plt.imsave(output_path, rendered_image)
    # print(f"Projection image saved to {output_path}")

def main():
    # Load your point cloud (x) and mesh (y)
    point_cloud_file = "/data/point_cloud/PUGAN/mnf_pu_poisson/test/pcd/chair.xyz"  # Replace with your file
    mesh_file = "/data/point_cloud/PUGAN/test_off/chair.off"  # Replace with your file

    # Load point cloud from .xyz file
    points = np.loadtxt(point_cloud_file, delimiter=' ')
    
    # Load mesh from .off file
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    # Compute point-to-surface distances
    distances = compute_point_to_surface_distances(points, mesh)

    # Define camera parameters
    # camera_position = np.array([0, 0, -5])  # Example camera position
    # camera_orientation = np.eye(3)  # Camera facing forward
    camera_orientation, camera_position = look_at_view_transform(20, 0, 40)
    image_width = 500
    image_height = 500
    focal_length = 500

    # Simulate camera projection using PyTorch3D
    rendered_image = simulate_camera_projection_with_pytorch3d(points, distances, camera_position, camera_orientation, image_width, image_height, focal_length)

    # Visualize the camera projection and save the image
    visualize_camera_projection(rendered_image, output_path="camera_projection.png")


def main_batch():
    pre_dir = '/data/point_cloud/teaser/no_emd_5/test/pcd'
    pre_filenames = sorted([x for x in listdir(pre_dir) if is_pc_file(x)])
    mesh_dir = '/data/point_cloud/PUGAN/test_off'
    mesh_filenames = sorted([x for x in listdir(mesh_dir) if is_pc_file(x)])
    out_dir = '/data/point_cloud/teaser/no_emd_vis_5'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in pre_filenames:
        print('pcd name=============', i)
        pre_name = join(pre_dir, i)
        # Load point cloud from .xyz file
        points = np.loadtxt(pre_name, delimiter=' ')
        # Load mesh from .off file
        mesh_name = join(mesh_dir, i)
        mesh_name = mesh_name.replace('xyz', 'off')
        mesh = o3d.io.read_triangle_mesh(mesh_name)
        output_name = join(out_dir, i)
        output_name = output_name.replace('xyz', 'jpg')
        # Compute point-to-surface distances
        distances = compute_point_to_surface_distances(points, mesh)

        # Define camera parameters
        camera_orientation, camera_position = look_at_view_transform(20, 0, 40)
        image_width = 500
        image_height = 500
        focal_length = 500

        # Simulate camera projection using PyTorch3D
        rendered_image = simulate_camera_projection_with_pytorch3d(points, distances, camera_position, camera_orientation, image_width, image_height, focal_length)

        # Visualize the camera projection and save the image
        visualize_camera_projection(rendered_image, output_path=output_name)


if __name__ == "__main__":
    main_batch()
