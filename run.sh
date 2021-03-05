CUDA_VISIBLE_DEVICES=1 python3 main.py --batch_size 256 --log_dir ~/logs/lbp_sigma1000  --epoch 6000 --lr 0.1 --sigma 1000 --phi 0 --model1 ~/models/epoch_120.pth.tar >> result/lbp_sigma1000 &
