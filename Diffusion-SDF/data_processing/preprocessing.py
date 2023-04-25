"""
Script for preprocessing the acronym meshes and saving them as point clouds and SDFs.
The source meshes are ShapeNet models converted to watertight meshes.

Generate watertight meshes:
    sh watertight.sh path/to/input/shapenet/models path/to/output/shapenet/watertight

Required environment (diffusionsdf-data_processing):
    conda env create -f environment.yml
    conda activate diffusionsdf-data_processing

Troubleshooting - If visualization is not working, some packages might need to be installed manually:
    pip3 install open3d
    conda install -c conda-forge libgcc
    conda install -c conda-forge libstdcxx-ng

Run preprocessing:
    python preprocessing.py [-h] [--test] [--show-progress] [--grasps-path GRASPS_PATH] [--meshes-path MESHES_PATH] [--out-path OUT_PATH]

Expected directory structure:
    grasps/Couch_314e0_1.618.h5  (acronym dataset grasps)
    meshes/314e0.obj  (shapenet processed to watertight meshes)

Test: Runs example with object 'Couch/37cfcafe606611d81246538126da07a8' from the acronym dataset.

Output:
    OUT_PATH/acronym/Couch/314e0/sdf_data.csv - point coordinates and sdf values for point cloud on the surface and near the surface of the mesh
    OUT_PATH/grid_data/acronym/Couch/314e0/grid_gt.csv - point coordinates and sd values for grid points with resolution 128x128x128 in NOCS cube
    OUT_PATH/preprocessed.json - list of preprocessed objects
"""

import json
import os
from argparse import ArgumentParser
from copy import copy

import h5py
import mesh_to_sdf
import numpy as np
import open3d as o3d
import pandas as pd
import trimesh
from tqdm import tqdm

# Required for running mesh_to_sdf without the display
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def load_mesh(filename, meshes_path, scale=None):
    """Load a mesh from a JSON or HDF5 file from the grasp dataset. The mesh will be scaled accordingly.

    Args:
        filename (str): JSON or HDF5 file name.
        scale (float, optional): If specified, use this as scale instead of value from the file. Defaults to None.

    Returns:
        trimesh.Trimesh: Mesh of the loaded object.
    """
    # src: https://github.com/NVlabs/acronym/blob/main/acronym_tools/acronym.py
    if filename.endswith(".json"):
        data = json.load(open(filename, "r"))
        mesh_fname = data["object"].decode('utf-8')
        mesh_scale = data["object_scale"] if scale is None else scale
    elif filename.endswith(".h5"):
        data = h5py.File(filename, "r")
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_scale = data["object/scale"][()] if scale is None else scale
    else:
        raise RuntimeError("Unknown file ending:", filename)

    mesh_fname = mesh_fname.split("/")[-1]  # keep only <id>.obj
    obj_mesh = trimesh.load(os.path.join(meshes_path, mesh_fname))
    obj_mesh = obj_mesh.apply_scale(mesh_scale)

    return obj_mesh


def resize_to_nocs(obj_mesh):
    """
    Target cube: [-1, 1]^3
    Object should be at the center of the cube (centered in [0,]^3) and the diagonal of its own tight box should be 1.
    """
    p1, p2 = obj_mesh.bounds

    # translation
    c = (p1 + p2) / 2  # center
    t = - c

    # scaling
    diagonal_len = np.linalg.norm(p2 - p1)

    # homogeneous transformation matrix
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = t

    obj_mesh_nocs = copy(obj_mesh)
    obj_mesh_nocs.apply_transform(transform)
    obj_mesh_nocs.apply_scale(1.0 / diagonal_len)

    return obj_mesh_nocs


class ProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cat = ''
        self._obj_id = ''
        self._task = ''

    def _update_desc(self):
        self.set_description(f'{self._cat} {self._obj_id} {self._task}')

    def update_obj(self, cat, obj_id):
        self._cat = cat
        self._obj_id = obj_id
        self._task = ''
        self._update_desc()

    def update_task(self, task):
        self._task += '>' + task
        self._update_desc()


def sample_surface(mesh, n, pbar=None):
    if pbar:
        pbar.update_task('surf')
    sampled_points, _ = trimesh.sample.sample_surface(mesh, n)
    return np.array(sampled_points)


def sample_near(points, pbar=None):
    if pbar:
        pbar.update_task('near')
    sampled_points = []
    sampled_points.append(points + np.random.normal(scale=0.005, size=(len(points), 3)))
    sampled_points.append(points + np.random.normal(scale=0.0005, size=(len(points), 3)))
    return np.vstack(sampled_points)


_GRID_QP = None  # reuse, avoid recomputing


def sample_grid(res=128, pbar=None):
    if pbar:
        pbar.update_task('grid')
    global _GRID_QP
    if _GRID_QP is None:
        linsp = np.linspace(-1, 1, res)
        # 128**3 = 2097152
        X, Y, Z = np.meshgrid(linsp, linsp, linsp)
        sampled_points = sorted(list(zip(X.ravel(), Y.ravel(), Z.ravel())))
        _GRID_QP = np.array(sampled_points)
    return _GRID_QP


def signed_distance(mesh, query_points, pbar=None):
    if pbar:
        pbar.update_task('sdv...')
    sds = mesh_to_sdf.mesh_to_sdf(mesh, query_points)
    sds = sds.reshape((sds.shape[0], 1))
    return sds


def save_to_csv(qp_sd, path):
    # save to csv
    df = pd.DataFrame(qp_sd)
    df = df.round(6)  # round floats
    df.to_csv(path, sep=',', header=False, index=False)


def preprocess(mesh, save_path='./surf.csv', n=235000, pbar=None, out_path=None):
    """
    1. sample points
    2. compute sd foreach
    3. save csv
    done for: surface, near surface

    :param mesh:
    :param save_path:
    :param n: number of points to sample on the surface
    :return:
    """
    p_surf = sample_surface(mesh, n, pbar)
    p_near = sample_near(p_surf, pbar)

    # Sanity check visualization
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(p_surf)
    if out_path:
        o3d.io.write_point_cloud(os.path.join(out_path, "test-surf.ply"), pcd1)
        print("Saved to:", os.path.join(out_path, "test-surf.ply"))
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(p_near)
    if out_path:
        o3d.io.write_point_cloud(os.path.join(out_path, "test-near.ply"), pcd2)
        print("Saved to:", os.path.join(out_path, "test-near.ply"))

    psd_surf = np.hstack([p_surf, np.zeros((p_surf.shape[0], 1))])

    sd_near = signed_distance(mesh, p_near, pbar)
    psd_near = np.hstack([p_near, sd_near])
    surf = np.vstack([psd_surf, psd_near])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_to_csv(surf, path=save_path)


def preprocess_grid(mesh, save_path='./grid.csv', res=128, pbar=None, out_path=None):
    """
    1. get grid points
    2. compute sd foreach
    3. save csv

    :param mesh:
    :param save_path:
    :param res: resolution of the grid for nocs cube
    :return:
    """
    p_grid = sample_grid(res, pbar)

    # Sanity check visualization
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(p_grid)
    if out_path:
        o3d.io.write_point_cloud(os.path.join(out_path, "test-grid.ply"), pcd3)
        print("Saved to:", os.path.join(out_path, "test-grid.ply"))

    sd_grid = signed_distance(mesh, p_grid, pbar)
    grid = np.hstack([p_grid, sd_grid])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_to_csv(grid, path=save_path)


def preprocess_all(grasps_path, meshes_path, out_path, show_progress=False):
    info = {}
    grasps = os.listdir(grasps_path)
    if show_progress:
        pbar = ProgressBar(grasps, leave=False)  # tqdm
        grasps = pbar
    else:
        pbar = None
    for filename in grasps:
        cat, obj_id, _ = filename.split('_')
        if pbar:
            pbar.update_obj(cat, obj_id)

        grasp_path = os.path.join(grasps_path, filename)
        mesh = load_mesh(grasp_path, meshes_path=meshes_path)
        mesh = resize_to_nocs(mesh)

        save_path = os.path.join(out_path, 'acronym', cat, obj_id, 'sdf_data.csv')
        preprocess(mesh, save_path=save_path, pbar=pbar)
        save_path = os.path.join(out_path, 'grid_data', 'acronym', cat, obj_id, 'grid_gt.csv')
        preprocess_grid(mesh, save_path=save_path, pbar=pbar)

        info.setdefault(cat, []).append(obj_id)

    info = {'acronym': info}
    with open(os.path.join(out_path, 'preprocessed.json'), 'w') as f:
        json.dump(info, f)
    print(info)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--test', action='store_true', help='run on a single test file (for debugging)')
    argparser.add_argument('--show-progress', action='store_true', help='show progress bar')
    argparser.add_argument('--grasps-path', type=str, default='./grasps', help='path/to/acronym/grasps')
    argparser.add_argument('--meshes-path', type=str, default='./meshes', help='path/to/meshes')
    argparser.add_argument('--out-path', type=str, default='./data', help='path/to/out')
    argparser.add_argument('--start-id', type=str, help='The file you want to start from (optional)')
    args = argparser.parse_args()

    if args.test:
        filename, cat, obj_id = ..., ..., ...
        for filename in os.listdir(args.grasps_path):
            if args.start_id:
                if args.start_id == filename:
                    cat, obj_id, _ = filename.split('_')
            else:
                cat, obj_id, _ = filename.split('_')

        grasp_path = os.path.join(args.grasps_path, filename)
        mesh = load_mesh(grasp_path, meshes_path=args.meshes_path)
        mesh = resize_to_nocs(mesh)
        save_path = os.path.join(args.out_path, 'acronym', cat, obj_id, 'sdf_data.csv')
        print('preprocessing surface...')
        preprocess(mesh, save_path=save_path, out_path=args.out_path)  # provide out_path to save sanity check ply files
        save_path = os.path.join(args.out_path, 'grid_data', 'acronym', cat, obj_id, 'grid_gt.csv')
        print('preprocessing grid...')
        preprocess_grid(mesh, save_path=save_path, out_path=args.out_path)  # provide out_path to save sanity check ply files
        print({'acronym': {cat: [obj_id]}})
    else:
        preprocess_all(args.grasps_path, args.meshes_path, args.out_path, args.show_progress)
