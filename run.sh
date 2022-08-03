#!/bin/bash

 python3 main.py \
     --print_freq 20 \
     --batch_size 128 \
     --aug kaggle \
     --alpha 1 \
     --beta 1 \
     --adp_alpha True \
     --alpha_update cosine\
     --alpha_decay 1000\
     --alpha_min 0.01 \
     --dataset ISIC \
     --data_folder ../ISIC\
     --epochs 400 \
     --learning_rate 0.1 \
     --lr_decay_rate 0.1 \
     --model resnet18 \
     --temp 0.1 \
     --val_size 384 \
     --size 384\
     --cosine \
     --log_dir ./log/ISIC_no_sampling_adpAlpha1to001_R18_kaggle384\
     --trial ISIC_no_sampling_adpAlpha1to001_R18_kaggle384_trail2


## train
#python3 main.py \
#    --dataset ISIC \
#    --data_folder ../ISIC\
#    --model resnet18 \
#    --val_size 384 \
#    --size 384\
#    --resume True\
#    --ckpt ./save/ISIC/aptAlpha_isoc_upsampling.pth
