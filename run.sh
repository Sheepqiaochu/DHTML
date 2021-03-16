#!/bin/bash

#SBATCH -A test
#SBATCH -J k0_sigma1000
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -o /public/data0/users/yangqiancheng/logs/result/k0_sigma1000.out
#SBATCH -t 1-12:00:00       
#SBATCH -N 1


module load cuda/10.2
module load gcc/5.5.0
conda activate python36
python3 main.py --batch_size 256 --log_dir ~/logs/k0_sigma1000  --epoch 6000 --lr 0.1 --sigma 1000 --phi 0 --model1 /public/data0/users/yangqiancheng/trans/stage0V2.pth.tar

