#!/bin/bash

#SBATCH -A test
#SBATCH -J sv_stage0V2
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o stage0V2.out
#SBATCH -t 1-00:00:00
#SBATCH -N 1


module load cuda/10.2
module load gcc/5.5.0
conda activate python36
python3 main.py --batch_size 256 --log_dir ~/logs/sv_stage0V2  --epoch 1000 --phi 1000  --lr 0.1 --model1

