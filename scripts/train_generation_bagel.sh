source activate pvd

python PVD/train_generation.py --dataroot ../PVD/data/ --category bagel --npoints 5000 --niter 2500 --normalize
