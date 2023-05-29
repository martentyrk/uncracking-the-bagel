import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import open3d as o3d
from pathlib import Path
import os
import argparse
from tqdm import tqdm

import sys
sys.path.append("./3D-ADS")
from PVD.utils.au_pro_util import calculate_au_pro

def min_dist_per_point(gt_pc, pred_pc):
    """
    Compute the closest distance between two point clouds, per point.
    :param gt_pc: ground truth point cloud, array of shape (num_points, 3)
    :param pred_pc: predicted point cloud, array of shape (num_points, 3)
    :return: array of shape (num_points, 1)
    """
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(gt_pc)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pred_pc)
    dists = pcd1.compute_point_cloud_distance(pcd2)
    dists = np.asarray(dists)
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

def plot_dummy_example(rgb_path, tiff_path, target_height=224, target_width=224):
    img = np.array(Image.open(rgb_path).resize((target_height, target_width)))
    pc, pixel_indices = get_pc(tiff_path, target_height, target_width)
    ano_pc, gt_mask, img = apply_dummy_anomaly(pc, pixel_indices, img, target_height, target_width)
    pc1 = torch.tensor(pc)
    pc2 = torch.tensor(ano_pc)
    cd = min_dist_per_point(pc1, pc2)
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


def predicted_anomaly_mask(pixel_indices, gt_chamfer, target_height, target_width):
    mask = np.zeros((target_height, target_width))
    for xy, ch in zip(pixel_indices, gt_chamfer):
        mask[xy[1], xy[0]] = ch
    return mask

def get_all_files(path, file_type):
    """
    Get all files of a certain type in a folder and its subfolders.
    :param path: path to folder
    :param file_type: file type (e.g. 'png', 'tiff', 'pth')
    :return: list of file paths
    """
    paths = list(Path(path).rglob(f'*.{file_type}'))
    return paths


def path2naming(file_path):
    """
    Get the naming convention for the test files.

    Args:
        file_path (str): file path (e.g. .../test/crack/xyz/0000.tiff)

    Returns:
        root_path (str): root path (e.g. .../test)
        category (str): category (crack, contamination, combined, good, hole)
        data_type (str): data type (gt, xyz, rgb)
        num (str): number (e.g. 0000)
        file_type (str): file type (tiff, png, pth)
    """
    root_path, category, data_type, filename = str(file_path).rsplit('/', maxsplit=3)
    num, file_type = filename.split('.')
    return root_path, category, data_type, num, file_type


def naming_gt(gt_path, category, num):
    return os.path.join(gt_path, category, 'gt', f'{num}.png')

def naming_ano(preds_folder, category, num):
    return os.path.join(preds_folder, category, f'i_{num}_x_hat_0.pth')

def naming_save_npy(save_folder, category, num):
    return os.path.join(save_folder, category, 'pred', f'{num}.npy')

def naming_save_png(save_folder, category, num):
    return os.path.join(save_folder, category, 'pred', f'{num}.png')


def compute_pred_masks(test_folder, preds_folder, save_folder, anomaly_type=None, save_png=True, overwrite=False):
    
    if anomaly_type is not None:
        find_root = os.path.join(test_folder, anomaly_type)
    else:
        find_root = test_folder
    paths = get_all_files(find_root, 'tiff')
    print(f"Found {len(paths)} tiff files in {find_root}")

    for tiff_filename in tqdm(paths, total=len(paths), desc='Computing predicted masks'):
        _, category, data_type, num, file_type = path2naming(str(tiff_filename))
        assert data_type == 'xyz'
        assert file_type == 'tiff'

        gt_filename = naming_gt(test_folder, category, num)
        ano_filename = naming_ano(preds_folder, category, num)
        save_filename = naming_save_npy(save_folder, category, num)

        if os.path.exists(save_filename) and not overwrite:
            # don't overwrite existing files
            continue

        # get gt mask
        gt_mask = np.array(Image.open(gt_filename))
        target_height, target_width = gt_mask.shape[:2]

        # compute predicted mask, unnormalized
        pc, pixel_indices = get_pc(tiff_filename, target_height, target_width)
        ano_pc = torch.load(ano_filename).permute(2,1,0).squeeze().cpu().numpy()
        cd = min_dist_per_point(pc, ano_pc)
        mask = predicted_anomaly_mask(pixel_indices, cd, target_height, target_width)

        # save predicted mask
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        np.save(save_filename, mask)

        if save_png:
            plt.figure()
            plt.imshow(mask)
            plt.colorbar()
            plt.savefig(naming_save_png(save_folder, category, num))
            plt.close()


def compute_au_pro(gt_folder, pred_folder, anomaly_type=None):

    if anomaly_type is not None:
        find_root = os.path.join(gt_folder, anomaly_type)
    else:
        find_root = gt_folder
    paths = get_all_files(find_root, 'png')
    print(f"Found {len(paths)} tiff files in {find_root}")

    gts = []
    predictions = []

    for filename in tqdm(paths, total=len(paths), desc='Computing AU-PRO'):
        _, category, data_type, num, file_type = path2naming(filename)

        if data_type == 'gt' and file_type == 'png':

            # get gt mask
            img = np.array(Image.open(filename))
            img[img > 0] = 1
            gts.append(img)

            pred_filename = naming_save_npy(pred_folder, category, num)
            pred = np.load(pred_filename)
            predictions.append(pred)
            
    integration_limits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    au_pro_dict = {'limits': integration_limits,
                   'au_pro': [],
                   'pro_curve': []
                   }
    for limit in integration_limits:
        au_pro, pro_curve = calculate_au_pro(gts, predictions, integration_limit=limit, num_thresholds=100)
        au_pro_dict['au_pro'].append(au_pro)
        au_pro_dict['pro_curve'].append(pro_curve)
        
    return au_pro_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_folder', type=str, help='Folder containing ground truths.')
    parser.add_argument('--preds_folder', type=str, help='Folder containing reconstructed point clouds.')
    parser.add_argument('--save_folder', type=str, help='Folder to save predicted masks.')
    parser.add_argument('--anomaly_type', type=str, default=None, help='Anomaly type')
    args = parser.parse_args()
    compute_pred_masks(args.test_folder, args.preds_folder, args.save_folder, args.anomaly_type, overwrite=True)
    au_pro = compute_au_pro(args.test_folder, args.save_folder, args.anomaly_type)
    print(f"Area Under PRO curve: {au_pro}")
