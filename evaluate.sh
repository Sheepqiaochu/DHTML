#!/bin/bash
epoch=20
while(($epoch<480))
do 
    path="/data/users/yangqiancheng/logs/lbp_stage0_sigma1000/models/epoch_"$epoch".pth.tar"
    echo $path
    epoch=`expr 40 + $epoch`
    CUDA_VISIBLE_DEVICES=4  python main.py --evaluate $path  --batch_size 128 --arch ShuffleNetV2 --width_mul 0.5
done       

