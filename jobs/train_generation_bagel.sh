#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:0
#SBATCH --job-name=train_bagel_generation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00
#SBATCH --mem=32000M
#SBATCH --output=../job_logs/bagel_generation_train_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source activate pvd

srun python PVD/train_generation.py --dataroot ../PVD/data/bagel/ --category bagel --bs 32 --niter 3