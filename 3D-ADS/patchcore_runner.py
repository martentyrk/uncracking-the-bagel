from data.mvtec3d import get_data_loader
import torch
from tqdm import tqdm
from feature_extractors.RGB_inet_features import RGBInetFeatures
from feature_extractors.depth_inet_features import DepthInetFeatures
from feature_extractors.raw_features import RawFeatures
from feature_extractors.hog_features import HoGFeatures
from feature_extractors.sift_features import SIFTFeatures
from feature_extractors.fpfh_features import FPFHFeatures
from feature_extractors.rgb_fpfh_features import RGBFPFHFeatures


class PatchCore():
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.methods = {
            # "RGB iNet": RGBInetFeatures(),
            # "Depth iNet": DepthInetFeatures(),
            "Raw": RawFeatures(),
            # "HoG": HoGFeatures(),
            # "SIFT": SIFTFeatures(),
            "FPFH": FPFHFeatures(), #pointcloud based
            # "RGB + FPFH": RGBFPFHFeatures()
        }

    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size)
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            for method in self.methods.values():
                method.add_sample_to_mem_bank(sample)

        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()

    def evaluate(self, class_name):
        au_pros = dict()
        
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size)
        with torch.no_grad():
            for sample, mask, label in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                for method in self.methods.values():
                    method.predict(sample, mask, label)

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            au_pros[method_name] = round(method.au_pro, 3)
            print(
                f'Class: {class_name}, AU-PRO: {method.au_pro:.3f}')
        return au_pros
