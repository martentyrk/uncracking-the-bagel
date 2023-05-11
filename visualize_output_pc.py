"""
Install environment:
    conda create -n pc python=3.8
    conda activate pc
    pip install numpy, tifffile, open3d
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

If you get "ImportError: libgomp.so.1: cannot open shared object file: No such file or directory":
    sudo apt-get install libgomp1

The output in 'test_generation.py' is stored as 'samples.pth' in the output folder.
"""
import numpy as np
import open3d as o3d
import torch
import argparse


def get_points(samples_path, index):
    points = torch.load(samples_path)
    if index is None:
        index = np.random.randint(0, len(points))
    return points[index]


def visualize_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_path", type=str)
    parser.add_argument("--index", type=str, default=None)
    args = parser.parse_args()
    points = get_points(args.samples_path, args.index)
    visualize_pcd(points)
