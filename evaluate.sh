#!/bin/bash
epoch=20
while(($epoch<640))
do 
    path="/data/users/yangqiancheng/logs/FGM2_k1_sigma400/models/epoch_"$epoch".pth.tar"
    epoch=`expr 20 + $epoch`
    CUDA_VISIBLE_DEVICES=0  python main.py --evaluate $path  --batch_size 128 --arch ShuffleNetV2 --width_mul 0.5
done       

