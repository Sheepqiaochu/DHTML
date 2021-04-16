#!/bin/bash
epoch=680
while(($epoch<760))
do 
    path="/data/users/yangqiancheng/logs/norm_k4_sigma10/models/epoch_"$epoch".pth.tar"
    epoch=`expr 20 + $epoch`
    CUDA_VISIBLE_DEVICES=3  python main.py --evaluate $path  --batch_size 128 --arch ShuffleNetV2 --width_mul 0.5
done       

