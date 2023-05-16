"""
src:
    https://github.com/otaheri/chamfer_distance  # TODO install
    https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
    https://pytorch.org/get-started/locally/

install:
    conda create -n chamfer python=3.9
    conda activate chamfer
    conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath

    conda install -c bottler nvidiacub

    conda install pytorch3d -c pytorch3d

    conda install pytorch-cuda=11.7 -c pytorch -c nvidia

    pip install git+'https://github.com/otaheri/chamfer_distance'

    conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
"""
#from chamfer_distance import ChamferDistance
#from PVD.metrics.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import torch
import matplotlib.pyplot as plt


def get_pc_xy(pc):
    """
    Get the x,y coordinates of a point cloud.
    Allows batch dimension.

    :param pc: array of shape (batch_size, num_points, 3) or (num_points, 3)
    :return: array of shape (batch_size, num_points, 2) or (num_points, 2)
    """
    pc_xy = pc[..., :2]
    return pc_xy


def chamfer_dist_per_point(gt_pc, pred_pc):
    """
    Compute the chamfer distance between two point clouds, per point.
    Should allow batch dimension.  # TODO allow batch dimension

    :param gt_pc: ground truth point cloud, array of shape (batch_size, num_points, 3) or (num_points, 3)
    :param pred_pc: predicted point cloud, array of shape (batch_size, num_points, 3) or (num_points, 3)
    :return: array of shape (batch_size, num_points, 1) or (num_points, 1)
    """
    # TODO use chamfer_distance
    gt_chamfer = torch.rand(*gt_pc.shape[:-1], 1)  # example
    return gt_chamfer

    #chd = ChamferDistance()
    #chd = chamfer_3DDist()
    dist1, dist2, idx1, idx2 = chd(gt_pc, pred_pc)
    print(dist1.shape, dist2.shape, idx1.shape, idx2.shape)


def get_gt_bbox(gt_pc):
    """
    Get the bounding box of the ground truth point cloud.
    Should allow batch dimension.  # TODO allow batch dimension

    :param gt_pc:
    :return:
    """
    # TODO get from original unprocessed data
    if len(gt_pc.shape) == 2:
        gt_pc = gt_pc.unsqueeze(0)
    xmin, ymin = gt_pc[..., 0].min(dim=1).values, gt_pc[..., 1].min(dim=1).values
    xmax, ymax = gt_pc[..., 0].max(dim=1).values, gt_pc[..., 1].max(dim=1).values
    return (xmin, ymin), (xmax, ymax)


def bilinear_spreading(cell_x, cell_y, p_val):
    """
    Given a point value p_val at coordinates cell_x and cell_y inside the cell, spread it to the 4 neighboring cells.
    The spreading is done using bilinear interpolation.

    :param cell_x: x coordinate of the point inside the cell, in range [0,1]
    :param cell_y: y coordinate of the point inside the cell, in range [0,1]
    :param p_val: value of the point inside the cell
    :return: 4 values, one for each neighboring cell
    """
    # TODO check vectorized, perform in a single pass
    p11 = (1 - cell_x) * (1 - cell_y) * p_val
    p12 = (1 - cell_x) * cell_y * p_val
    p21 = cell_x * (1 - cell_y) * p_val
    p22 = cell_x * cell_y * p_val
    return p11, p12, p21, p22


def predicted_anomaly_mask(gt_xy, gt_chamfer, bbox_min, bbox_max, resolution=None):
    """
    Given the ground truth point cloud and its chamfer distance, compute the predicted anomaly mask.
    The mask is a 2D array of the same size as the image, where each pixel is the predicted chamfer distance
    at that pixel.
    Allows batch dimension.

    :param gt_xy: array of shape (num_points, 2) or (batch_size, num_points, 2)
    :param gt_chamfer: array of shape (num_points, 1) or (batch_size, num_points, 1)
    :param bbox_min: tuple (xmin, ymin)
    :param bbox_max: tuple (xmax, ymax)
    :param resolution: tuple (xres, yres), defaults to (xmax-xmin, ymax-ymin)
    :return: array of shape (xres, yres) or (batch_size, xres, yres)
    """
    unsqueeze = False
    if len(gt_xy.shape) == 2:
        unsqueeze = True
        gt_xy = gt_xy.unsqueeze(0)
        gt_chamfer = gt_chamfer.unsqueeze(0)

    xmin, ymin = bbox_min
    xmax, ymax = bbox_max
    if resolution is None:
        resolution = (xmax - xmin, ymax - ymin)
    xres, yres = resolution
    xres, yres = int(xres), int(yres)
    masks = torch.zeros((gt_xy.shape[0], xres, yres))

    for i in range(gt_xy.shape[0]):
        for j in range(gt_xy.shape[1]):

            # normalize coordinates
            x, y = gt_xy[i, j]
            x, y = x - xmin, y - ymin
            x, y = x / (xmax - xmin), y / (ymax - ymin)
            x, y = x * xres, y * yres

            # cell neighbors
            x1, y1 = int(x), int(y)
            x2, y2 = x1 + 1, y1 + 1

            # bilinear spreading
            p11, p12, p21, p22 = bilinear_spreading(x - x1, y - y1, gt_chamfer[i, j, 0])

            # update mask
            for x, y, p in [(x1, y1, p11), (x1, y2, p12), (x2, y1, p21), (x2, y2, p22)]:
                if x < 0 or x >= xres or y < 0 or y >= yres:
                    continue
                masks[i, x, y] = p

    if unsqueeze:
        masks = masks.squeeze(0)
    return masks


def normalize_mask(mask, method='tanh', *, threshold=0.5, tanh_scale=1.0):
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
    assert method in ['minmax', 'clip', 'tanh', 'threshold']
    if method == 'minmax':
        mask = mask - mask.min()
        mask = mask / mask.max()
    elif method == 'clip':
        mask = torch.clamp(mask, 0, 1)
    elif method == 'tanh':
        mask = torch.tanh(mask * tanh_scale)
    elif method == 'threshold':
        mask = torch.where(mask < threshold, torch.zeros_like(mask), torch.ones_like(mask))
    return mask


def plot_mask(mask):
    """
    Plot the mask as a grayscale image.
    Only works for 2D masks, i.e. without batch dimension.

    :param mask: single mask array
    :return: None
    """
    plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    pc1 = torch.rand(100, 3)
    pc2 = torch.rand(200, 3)
    xy = get_pc_xy(pc1)
    cd = chamfer_dist_per_point(pc1, pc2)
    bmin, bmax = get_gt_bbox(pc1)
    masks = predicted_anomaly_mask(xy, cd, bmin, bmax, resolution=(128, 128))
    if len(masks.shape) == 3:
        mask = masks[0]
    else:
        mask = masks
    mask = normalize_mask(mask, 'threshold')
    plot_mask(mask)
