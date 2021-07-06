#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt "./pretrained_model/looperman/100000.pt" \
    --pics 100 --data_path "./data/looperman/" \
    --store_path "./generated_looperman_one_bar"
