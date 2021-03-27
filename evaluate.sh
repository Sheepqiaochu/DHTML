#!/bin/bash
epoch=520
while(($epoch<560))
do 
    path="/data/users/yangqiancheng/logs/lbp_fixedm600_sigma1000k1/models/epoch_"$epoch".pth.tar"
    epoch=`expr 20 + $epoch`
    CUDA_VISIBLE_DEVICES=4  python main.py --evaluate $path  --batch_size 128 --arch ShuffleNetV2 --width_mul 0.5
done       

