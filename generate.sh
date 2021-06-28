#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {100000..100000..10000}
    do
        echo "generate epoch $i model"
        foo=$(printf "%06d" $i)
        echo "$foo"
        CUDA_VISIBLE_DEVICES=2 python generate_audio.py --ckpt "../stylegan2-pytorch/checkpoint_freesound_drum/$foo.pt" --pics 2000 --data_path "/home/allenhung/nas189/home/freesound/drum_audio/mel_80_320" --store_path "./generated_freesound_one_bar"
    done
