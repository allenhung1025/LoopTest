#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python inception_score.py \
    ./best_model.ckpt \
    --data_dir ../../generated_freesound_one_bar/freesound_checkpoint/100000/mel_80_320  --classes 66 \
    --mean_std_dir ../../data/freesound \
    --store_every_score freesound_styelgan2.pkl

