#!/bin/bash

#SBATCH -A test
#SBATCH -J L1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o result/L1.out
#SBATCH -t 2-00:00:00       
# SBATCH -N 1


module load cuda/10.2
module load gcc/5.5.0
conda activate python36
python3 main.py --batch_size 256 --log_dir ~/logs/L1 --epoch 4000 --lr 0.01 --sigma 100

