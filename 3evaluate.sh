#!/bin/bash
epoch=20
while(($epoch<840))
do 
    path="/data/users/yangqiancheng/logs/FGM2_k3_sigma400/models/epoch_"$epoch".pth.tar"
    epoch=`expr 20 + $epoch`
    CUDA_VISIBLE_DEVICES=2  python main.py --evaluate $path  --batch_size 128 --arch ShuffleNetV2 --width_mul 0.5
done       

