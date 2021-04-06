#!/bin/bash
epoch=480
while(($epoch<500))
do 
    path="/data/users/yangqiancheng/logs/FGM2_k1m600_sigma10000/models/epoch_"$epoch".pth.tar"
    epoch=`expr 20 + $epoch`
    CUDA_VISIBLE_DEVICES=1  python main.py --evaluate $path  --batch_size 128 --arch ShuffleNetV2 --width_mul 0.5
done       

