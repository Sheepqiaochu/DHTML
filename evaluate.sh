#!/bin/bash
epoch=3000
while(($epoch<5010))
do 
    path="/public/data0/users/yangqiancheng/logs/taregt_5000_nodis/models/epoch_"$epoch".pth.tar"
    echo $path
    epoch=`expr 200 + $epoch`
    srun -A test -J test_model -p gpu -N 1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 python main.py --evaluate $path  --batch_size 128
done       

