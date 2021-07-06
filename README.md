# loop-generation
Open source of ISMIR-21 submission, “A Benchmarking Initiative for Audio-domain Music Generation using the FreeSound Loop Dataset”
## quick start
* Download model from the link in Prtrained checkpoint
``` bash
$ gdown url
$ CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt checkpoint \
    --pics 100 --data_path "./data/[freesound | looperman]" \
    --store_path "./generated_[freesound | looperman]_one_bar"
``` 
## Pretrained checkpoint
* [looperman pretrained model link](https://drive.google.com/file/d/1GQpzWz9ycIm5wzkxLsVr-zN17GWD3_6K/view?usp=sharing) 
* [freesound pretrained model link](https://drive.google.com/file/d/197DMCOASEMFBVi8GMahHfRwgJ0bhcUND/view?usp=sharing)