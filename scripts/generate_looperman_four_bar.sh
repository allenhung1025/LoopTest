#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python generate_looperman_four_bar.py \
    --ckpt "looperman_four_bar_checkpoint.pt" \
    --pics 100 \
    --data_path "./data/looperman_four_bar" \
    --store_path "./generated_audio_looperman_four_bar"
