import numpy as np

def bin_to_xyz(bin_file, xyz_file):
    """
    Convert a .bin point cloud file to a .xyz file.

    Parameters:
    - bin_file: Path to the input .bin file.
    - xyz_file: Path to the output .xyz file.
    """
    # Load binary file into a numpy array
    point_cloud = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)

    # Extract x, y, z columns (ignore intensity if present)
    xyz_points = point_cloud[:, :3]

    # Save as a .xyz file
    np.savetxt(xyz_file, xyz_points, fmt="%.6f", delimiter=" ")

# Example usage
bin_file_path = "/data/point_cloud/real_pc_v2/00/velodyne/001000.bin"
xyz_file_path = "/data/point_cloud/real_pc_v2/00/pc/001000.xyz"
bin_to_xyz(bin_file_path, xyz_file_path)
print(f"Converted {bin_file_path} to {xyz_file_path}")
