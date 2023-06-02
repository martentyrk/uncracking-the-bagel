# Uncracking the Bagel One Crumb at a Time

Implementation of anomaly detection detection on MVTec 3D-AD dataset using point-voxel diffusion.
![](static/overview.png)
*Figure of the anomaly detection pipeline*

## Requirements:

```
python==3.6
pytorch==1.4.0
torchvision==0.5.0
cudatoolkit==10.1
matplotlib==2.2.5
tqdm==4.32.1
open3d==0.9.0
trimesh=3.7.12
scipy==1.5.1
werkzeug==2.0.3
markupsafe==2.0.1
cudatoolkit-dev
scikit-learn
tifffile
Pillow
numpy
wget
timm
kornia
scikit-image
```

## Dataset

To download the dataset locally, go to the [MVTEC website](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad) and fill out your details. (Note: Only download the bagel dataset, since all data will take too much time and is not relevant for reproduction, since we only work on bagels).

To preprocess, run the preprocessing file in the 3D-ADS folder

```
python -u 3D-ADS/utils/preprocessing.py \
    --data_path /path/to/bagels_data \
```

This process can take up to ~4h or more depending on your system.

The preprocessing is done in-place, so the tiff files will remain tiff files. Here is an example of before and after of bagel/train/good/xyz/000.tiff.

Before             |  After
:-------------------------:|:-------------------------:
![](static/000.png)  |  ![](static/000_new.png)


## Running the experiments

Training the generator on normalized healthy samples for 2500 epochs:

```
python -u PVD/train_generation.py \
    --dataroot /path/to/data_folder \
    --normalize \
```

The testing can be run using the pvd_test.py file

```
python -u PVD/pvd_test.py \
    --dataroot path/to/data \
    --anomaly_time 20 \
```

## Citation
[The PVD repository](https://github.com/alexzhou907/PVD)
```
@inproceedings{Zhou_2021_ICCV,
    author    = {Zhou, Linqi and Du, Yilun and Wu, Jiajun},
    title     = {3D Shape Generation and Completion Through Point-Voxel Diffusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {5826-5835}
}
```


[3D-ADS](https://github.com/eliahuhorwitz/3D-ADS)
```
@article{horwitz2022empirical,
  title={An Empirical Investigation of 3D Anomaly Detection and Segmentation},
  author={Horwitz, Eliahu and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2203.05550},
  year={2022}
}
```
