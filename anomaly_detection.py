"""
src:
    https://github.com/otaheri/chamfer_distance  # TODO install
    https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
    https://pytorch.org/get-started/locally/

setup:
    conda create -n chamfer python=3.8
    conda activate chamfer

    # install pytorch3d
    pip install torch==1.10.0
    pip install pytorch==1.10.0

    sudo apt install g++
    sudo apt install build-essential
    sudo apt-get install manpages-dev

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    conda install -c pytorch3d pytorch3d

    # install chamfer_distance
    pip install git+'https://github.com/otaheri/chamfer_distance'

    #https://github.com/facebookresearch/maskrcnn-benchmark/issues/891
    pip install torch==2.0.0+cu118 torchvision==0.7.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
    pip install -U torch torchvision --no-cache-dir
"""
import numpy as np

# from chamfer_distance import ChamferDistance
# from PVD.metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from PVD.metrics.ChamferDistancePytorch.chamfer_python import distChamfer
import torch
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import open3d as o3d


def chamfer_dist_per_point(gt_pc, pred_pc):
    """
    Compute the chamfer distance between two point clouds, per point.
    Should allow batch dimension.  # TODO allow batch dimension

    :param gt_pc: ground truth point cloud, array of shape (batch_size, num_points, 3) or (num_points, 3)
    :param pred_pc: predicted point cloud, array of shape (batch_size, num_points, 3) or (num_points, 3)
    :return: array of shape (batch_size, num_points, 1) or (num_points, 1)
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(gt_pc)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pred_pc)
    dists = pcd1.compute_point_cloud_distance(pcd2)
    dists = np.asarray(dists)
    dists = (dists - dists.min()) / (dists.max() - dists.min())
    return dists


def normalize_mask(mask, method="tanh", *, threshold=0.5, tanh_scale=1.0):
    """
    Normalize mask values to range [0,1]
    It allows to use different methods to normalize the mask:
    - minmax: normalize to range [0,1]
    - clip: clip values to range [0,1]
    - tanh: apply tanh to values, with scale factor tanh_scale
    - threshold: set values below threshold to 0 and above to 1, with threshold value
    Should allow batch dimension.  # TODO allow batch dimension

    :param mask: input mask array
    :param method: method to use for normalization, one of ['minmax', 'clip', 'tanh', 'threshold']
    :param threshold: threshold value to use for thresholding, only used if method='threshold', defaults to 0.5
    :param tanh_scale: scale factor to use for tanh, only used if method='tanh', defaults to 1.0
    :return: normalized mask
    """
    assert method in ["minmax", "clip", "tanh", "threshold"]
    if method == "minmax":
        mask = mask - mask.min()
        mask = mask / mask.max()
    elif method == "clip":
        mask = torch.clamp(mask, 0, 1)
    elif method == "tanh":
        mask = torch.tanh(mask * tanh_scale)
    elif method == "threshold":
        mask = torch.where(
            mask < threshold, torch.zeros_like(mask), torch.ones_like(mask)
        )
    return mask


def resize_organized_pc(organized_pc, target_height, target_width):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0)
    torch_resized_organized_pc = torch.nn.functional.interpolate(
        torch_organized_pc, size=(target_height, target_width), mode="nearest"
    )
    grid = np.meshgrid(np.arange(0, target_height), np.arange(0, target_width))
    pixel_indices = np.array(grid).transpose(1, 2, 0).reshape(-1, 2)
    pc = torch_resized_organized_pc.squeeze(dim=0).numpy().reshape(3, -1).transpose()
    return pc, pixel_indices


def get_nonzero_points(pc, pixel_indices):
    nonzero = (pc[:, 0] != 0) & (pc[:, 1] != 0) & (pc[:, 2] != 0)
    return pc[nonzero], pixel_indices[nonzero]


def get_pc(tiff_path, target_height, target_width):
    tiff_img = tiff.imread(tiff_path)
    pc, pixel_indices = resize_organized_pc(tiff_img, target_height, target_width)
    pc, pixel_indices = get_nonzero_points(pc, pixel_indices)
    means = np.array([0.02803007, -0.01059183, 0.51991437])
    stds = np.array([0.02647153, 0.02599377, 0.00973472])
    pc = (pc - means) / stds
    return pc, pixel_indices


def apply_dummy_anomaly(pc, pixel_indices, img, target_height, target_width):
    """
    Apply ellipse anomaly to point cloud.
    u,v - ellipse center, random within the point cloud limits
    t - ellipse angle, random
    a,b - ellipse axes, random within the point cloud limits (between 1/32 and 1/16 of the point cloud limits)

    :param pc: array with shape (num_points, 3)
    :return:
    """
    pc = pc.copy()
    xmin, ymin = np.quantile(pc, 0.1, axis=0)[:2]
    xmax, ymax = np.quantile(pc, 0.9, axis=0)[:2]
    width, height = xmax - xmin, ymax - ymin
    u = np.random.uniform(xmin, xmax)
    v = np.random.uniform(ymin, ymax)
    t = np.random.uniform(0, 2 * np.pi)
    a = np.random.uniform(width / 32, width / 16)
    b = np.random.uniform(height / 32, height / 16)
    x, y = pc[:, 0], pc[:, 1]
    where = ((x - u) * np.cos(t) + (y - v) * np.sin(t)) ** 2 / a**2 + (
        (x - u) * np.sin(t) - (y - v) * np.cos(t)
    ) ** 2 / b**2 <= 1
    pc[where, 2] += 10
    gt_mask = np.zeros((target_height, target_width))
    for xy in pixel_indices[where]:
        gt_mask[xy[1], xy[0]] = 1
        img[xy[1], xy[0]] = (255,0,0)
    return pc, gt_mask, img


def predicted_anomaly_mask(pixel_indices, gt_chamfer, target_height, target_width):
    mask = np.zeros((target_height, target_width))
    for xy, ch in zip(pixel_indices, gt_chamfer):
        mask[xy[1], xy[0]] = ch
    return mask


if __name__ == "__main__":
    target_height, target_width = 224, 224
    img = np.array(Image.open("data/example/000.png").resize((target_height, target_width)))
    pc, pixel_indices = get_pc("data/example/000.tiff", target_height, target_width)
    ano_pc, gt_mask, img = apply_dummy_anomaly(pc, pixel_indices, img, target_height, target_width)

    pc1 = torch.tensor(pc)
    pc2 = torch.tensor(ano_pc)
    cd = chamfer_dist_per_point(pc1, pc2)

    mask = predicted_anomaly_mask(pixel_indices, cd, target_height, target_width)

    f, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].scatter(ano_pc[:, 0], -ano_pc[:, 1], s=0.1, c=ano_pc[:, 2])
    axes[1].imshow(img)
    axes[2].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Anomaly Point Cloud")
    axes[1].set_title("Anomaly Image")
    axes[2].set_title("Ground Truth Mask")
    axes[3].set_title("Predicted Mask")
    axes[1].axis("off")
    axes[2].axis("off")
    axes[3].axis("off")
    plt.tight_layout()
    plt.show()
