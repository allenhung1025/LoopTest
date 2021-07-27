#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python generate_audio.py \
    --ckpt "freesound_checkpoint.pt" \
    --pics 2000 --data_path "./data/freesound" \
    --store_path "./generated_freesound_one_bar" \
    --style_mixing
