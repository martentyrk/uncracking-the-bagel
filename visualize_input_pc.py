"""
Install environment:
    conda create -n pc python=3.8
    conda activate pc
    pip install numpy, tifffile, open3d
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

If you get "ImportError: libgomp.so.1: cannot open shared object file: No such file or directory":
    sudo apt-get install libgomp1
"""
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torchvision import transforms
import tifffile as tiff
import argparse

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# Code borrowed from 3D-ADS repo ========================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

rgb_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img


def resize_organized_pc(organized_pc, target_height=224, target_width=224, tensor_out=True):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width),
                                                                 mode='nearest')
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze().permute(1, 2, 0).numpy()


def organized_pc_to_array(organized_pc):
    pc = organized_pc.numpy().reshape(3,-1).transpose()
    return pc


def img_to_color_array(img):
    colors = img.numpy().reshape(3,-1).transpose()
    colors = (colors - colors.min(axis=0)) / (colors.max(axis=0) - colors.min(axis=0))
    return colors


def get_nonzero_points_colors(points, colors):
    p = points
    nonzero = (p[:,0] != 0) & (p[:,1] != 0) & (p[:,2] != 0)
    return p[nonzero], colors[nonzero]


def get_points_colors(rgb_path, tiff_path):
    img = Image.open(rgb_path).convert('RGB')
    img = rgb_transform(img)
    organized_pc = read_tiff_organized_pc(tiff_path)
    resized_organized_pc = resize_organized_pc(organized_pc)
    points = organized_pc_to_array(resized_organized_pc)
    colors = img_to_color_array(img)
    points, colors = get_nonzero_points_colors(points, colors)
    return points, colors


def visualize_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str)
    parser.add_argument("--tiff_path", type=str)
    args = parser.parse_args()
    points, colors = get_points_colors(args.rgb_path, args.tiff_path)
    visualize_pcd(points, colors)
