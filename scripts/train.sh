#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python train_drum.py \
    --size 64 --batch 8 --sample_dir sample_Bandlab_Beats_one_bar \
    --checkpoint_dir checkpoint_Bandlab_Beats_one_bar \
    /home/allenhung/nas189/home/bandlab/BANDLAB_INSTRUMENT/Beats_one_bar/mel_80_320
