#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt "./pretrained_model/freesound/100000.pt" \
    --pics 100 --data_path "./data/freesound" \
    --store_path "./generated_freesound_one_bar"
