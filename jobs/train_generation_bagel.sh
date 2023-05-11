#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_bagel_generation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:40:00
#SBATCH --mem=32000M
#SBATCH --output=../job_logs/bagel_generation_train_%A.out

module purge

source activate pvd

srun python PVD/train_generation.py --dataroot ../PVD/data/ --category bagel --vizIter 2 --diagiter 2 --niter 8