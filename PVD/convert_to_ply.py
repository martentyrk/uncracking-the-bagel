import open3d as o3d
import os
import torch
import numpy as np

def get_points(samples_path):
    points = torch.load(samples_path)

    return points


directory = 'output/train_generation/2023-05-15-18-49-43/pcs'

save_to = 'output/train_generation/2023-05-15-18-49-43/ply_files/'

if not os.path.isdir(save_to):
    os.mkdir(save_to)

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    points = torch.load(f)
    for idx, pc in enumerate(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
    
        f_front = filename.split('.')[0]
    
        o3d.io.write_point_cloud(save_to+f_front+'_' +str(idx) +'_'+".ply", pcd)