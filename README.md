# loop-generation
Open source of ISMIR-21 submission, “A Benchmarking Initiative for Audio-domain Music Generation using the FreeSound Loop Dataset”
## quick start
* Generate loops from looperman pretrained model
``` bash
$ gdown --id 1GQpzWz9ycIm5wzkxLsVr-zN17GWD3_6K -O looperman_checkpoint.pt
$ CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt "looperman_checkpoint.pt" \
    --pics 100 --data_path "./data/looperman" \
    --store_path "./generated_looperman_one_bar"
``` 
* Generate loops from freesound pretrained model
``` bash
$ gdown --id 197DMCOASEMFBVi8GMahHfRwgJ0bhcUND -O freesound_checkpoint.pt 
$ CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt "freesound_checkpoint.pt" \
    --pics 100 --data_path "./data/freesound" \
    --store_path "./generated_freesound_one_bar"
``` 

## Pretrained checkpoint
* [looperman pretrained model link](https://drive.google.com/file/d/1GQpzWz9ycIm5wzkxLsVr-zN17GWD3_6K/view?usp=sharing) 
* [freesound pretrained model link](https://drive.google.com/file/d/197DMCOASEMFBVi8GMahHfRwgJ0bhcUND/view?usp=sharing)