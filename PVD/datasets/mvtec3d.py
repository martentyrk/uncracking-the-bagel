import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
#DATASETS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets', 'mvtec3d'))



def mvtec3d_classes():
    return [
        "bagel",
        "cable_gland",
        "carrot",
        "cookie",
        "dowel",
        "foam",
        "peach",
        "potato",
        "rope",
        "tire",
    ]


class MVTec3D(Dataset):

    def __init__(self, split, datasets_path, class_name, npoints):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.BAGEL_MEAN = [0.02803007, -0.01059183, 0.51991437]
        self.BAGEL_STD = [0.02647153, 0.02599377, 0.00973472]
        self.cls = class_name
        #self.size = img_size
        self.npoints = npoints
        self.img_path = os.path.join(datasets_path, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((224, 224), interpolation=Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

class MVTec3DTrain(MVTec3D):
    def __init__(self, datasets_path, class_name, npoints=None, normalize=False, grid_downsample=False):
        super().__init__(split="train", datasets_path=datasets_path, class_name=class_name, npoints=npoints)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.normalize = normalize
        self.grid_downsample = grid_downsample

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))
        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')

        img = self.rgb_transform(img)
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        #return (img, resized_organized_pc, resized_depth_map_3channel), label
        
        if self.npoints is not None:
            if self.grid_downsample:
                p = resized_organized_pc
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(p)
                resized_organized_pc = pcd.voxel_down_sample(voxel_size=0.02)
                resized_organized_pc = torch.tensor(resized_organized_pc.points)
            else:
                p = resized_organized_pc
                p = p[(p[:,0] != 0) & (p[:,1] != 0) & (p[:,2] != 0)]
                tr_idxs = np.random.choice(p.shape[0], self.npoints)
                resized_organized_pc = p[tr_idxs, :]
            
            if self.normalize:
                means = torch.tensor(self.BAGEL_MEAN, device=resized_organized_pc.device)
                stds = torch.tensor(self.BAGEL_STD, device=resized_organized_pc.device)
                resized_organized_pc = (resized_organized_pc - means) / stds
        
        out = {
            'idx': idx,
            'train_points': resized_organized_pc,
            'cate_idx': label,
        }
        return out


class MVTec3DTest(MVTec3D):
    def __init__(self, datasets_path, class_name, npoints=None, grid_downsample=False):
        super().__init__(split="test", datasets_path=datasets_path, class_name=class_name, npoints=npoints)
        self.gt_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.grid_downsample = grid_downsample

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        #return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label
        
        if self.npoints is not None:
            p = resized_organized_pc
            p = p[(p[:,0] != 0) & (p[:,1] != 0) & (p[:,2] != 0)]
            tr_idxs = np.random.choice(p.shape[0], self.npoints)
            resized_organized_pc = p[tr_idxs, :]
            means = torch.tensor(self.BAGEL_MEAN, device=resized_organized_pc.device)
            stds = torch.tensor(self.BAGEL_STD, device=resized_organized_pc.device)
            resized_organized_pc = (resized_organized_pc - means) / stds
        
        out = {
            'idx': idx,
            'train_points': resized_organized_pc,
            'cate_idx': label,
        }
        return out
