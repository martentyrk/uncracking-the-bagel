source activate pvd

python -u PVD/pvd_test.py --dataroot /home/lcur0949/PVD/data --anomaly_time 20 --batch_size 1 --model /home/lcur0949/Computer-vision-2-project/checkpoints/epoch_2499_10k_normalize_true.pth --workers 0 --anomaly --eval_gen --test_folder /home/lcur0949/PVD/data/bagel/test