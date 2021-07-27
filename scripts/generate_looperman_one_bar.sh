#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python generate_audio.py \
    --ckpt "looperman_one_bar_checkpoint.pt" \
    --pics 2000 --data_path "./data/looperman/" \
    --store_path "./generated_looperman_one_bar"
