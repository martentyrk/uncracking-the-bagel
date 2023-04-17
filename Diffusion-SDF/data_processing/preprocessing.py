import json
import os
from copy import copy

import h5py
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm


def load_mesh(filename, mesh_root_dir, scale=None):
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

    obj_mesh = trimesh.load(os.path.join(mesh_root_dir, mesh_fname))
    obj_mesh = obj_mesh.apply_scale(mesh_scale)

    return obj_mesh


def show_mesh(obj_mesh):
    trimesh.Scene([obj_mesh]).show()  # blocks execution flow


def resize_to_nocs(obj_mesh):
    p1, p2 = obj_mesh.bounds

    # center
    c = (p1 + p2) / 2

    # translation
    t = - c

    # scaling
    s = np.min(2 / np.abs(p2 - p1))
    S = s * np.eye(3)

    # homogeneous transformation matrix
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = S
    transform[:3, 3] = s * t

    obj_mesh_nocs = copy(obj_mesh)
    obj_mesh_nocs.apply_transform(transform)

    return obj_mesh_nocs


def sample_surface(mesh, n):
    sampled_points, _ = trimesh.sample.sample_surface(mesh, n)
    return np.array(sampled_points)


def sample_near(points):
    sampled_points = []
    for p in tqdm(points, desc='sample near surface'):
        p1 = np.random.multivariate_normal(p, 0.005 * np.eye(3), 1)[0]
        p2 = np.random.multivariate_normal(p, 0.0005 * np.eye(3), 1)[0]
        sampled_points.extend([p1, p2])
    return np.array(sampled_points)


def sample_grid(res=128):
    linsp = np.linspace(-1, 1, res)
    # 128**3 = 2097152
    X, Y, Z = np.meshgrid(linsp, linsp, linsp)
    sampled_points = list(zip(X.ravel(), Y.ravel(), Z.ravel()))
    return np.array(sampled_points)


def signed_distance(mesh, query_points):
    chunks = np.array_split(query_points, len(query_points)//1000)  # lighter chunks
    sds = []
    for chunk in tqdm(chunks, desc='signed distances'):
        chunk_sds = trimesh.proximity.signed_distance(mesh, chunk)
        sds.extend(chunk_sds)
    sds = np.array(sds)
    sds = np.expand_dims(sds, 1)  # unsqueeze
    return sds


def save_to_csv(qp_sd, path):
    # save to csv
    df = pd.DataFrame(qp_sd)
    df = df.round(6)  # round floats
    df.to_csv(path, sep=',', header=False, index=False)


def preprocess(mesh, save_dir='.', n=235000):
    """
    1. sample points
    2. compute sd foreach
    3. save csv
    done for: surface, near surface

    :param mesh:
    :param save_dir:
    :param n: number of points to sample on the surface
    :return:
    """
    p_surf = sample_surface(mesh, n)
    p_near = sample_near(p_surf)
    psd_surf = np.hstack([p_surf, signed_distance(mesh, p_surf)])
    psd_near = np.hstack([p_near, signed_distance(mesh, p_near)])
    surf = np.vstack([psd_surf, psd_near])
    save_to_csv(surf, path=f'{save_dir}/surf.csv')


def preprocess_grid(mesh, save_dir='.', res=128):
    """
    1. get grid points
    2. compute sd foreach
    3. save csv

    :param mesh:
    :param save_dir:
    :param res: resolution of the grid for nocs cube
    :return:
    """
    p_grid = sample_grid(res)
    grid = np.hstack([p_grid, signed_distance(mesh, p_grid)])
    save_to_csv(grid, path=f'{save_dir}/grid.csv')


def preprocess_all(grasps_dir, meshes_root_dir):
    for filename in os.listdir(grasps_dir):
        print(filename)
        cat, obj_id, _ = filename.split('_')
        grasp_path = os.path.join(grasps_dir, filename)

        mesh = load_mesh(grasp_path, mesh_root_dir=meshes_root_dir)
        mesh = resize_to_nocs(mesh)
        print('surface')
        save_dir = os.path.join('.', 'acronym', cat, obj_id)
        preprocess(mesh, save_dir=save_dir)
        print('grid')
        save_dir = os.path.join('.', 'grid_data', 'acronym', cat, obj_id)
        preprocess_grid(mesh, save_dir=save_dir)


if __name__ == '__main__':
    grasp_path = "../data/NVlabs acronym main data-examples/grasps/Mug_10f6e09036350e92b3f21f1137c3c347_0.0002682457830986903.h5"
    mesh_root_dir = "../data/NVlabs acronym main data-examples/"

    mesh = load_mesh(grasp_path, mesh_root_dir)
    mesh = resize_to_nocs(mesh)
    # show_mesh(mesh)

    preprocess_all("../data/NVlabs acronym main data-examples/grasps",
                   "../data/NVlabs acronym main data-examples/")

    print('surface')
    preprocess(mesh)
    print('grid')
    preprocess_grid(mesh)
