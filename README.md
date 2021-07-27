# LoopTest 
[![GitHub](https://img.shields.io/github/license/allenhung1025/loop-generation?label=license)](./LICENSE.md)
![GitHub issues](https://img.shields.io/github/issues/allenhung1025/loop-generation)
![GitHub Repo stars](https://img.shields.io/github/stars/allenhung1025/loop-generation)
* This is the official repository of **A Benchmarking Initiative for Audio-domain Music Generation using the FreeSound Loop Dataset** co-authored with [Paul Chen](https://paulyuchen.com/), [Arthur Yeh](http://yentung.com/) and my supervisor [Yi-Hsuan Yang](http://mac.citi.sinica.edu.tw/~yang/). The paper has been accepted by International Society for Music Information Retrieval Conference 2021.  
* [Demo Page](https://loopgen.github.io/).
* Will put on arxiv link upon release of the paper.
* We not only provided pretrained model to generate loops on your own but also provided scripts for you to evaluate the generated loops.
## Environment
```
$ conda env create -f environment.yml 
```
## Quick Start
``` bash
$ gdown --id 1GQpzWz9ycIm5wzkxLsVr-zN17GWD3_6K -O looperman_one_bar_checkpoint.pt
$ bash scripts/generate_looperman_one_bar.sh
``` 
* Generate loops from four-bar looperman pretrained model
``` bash
$ gdown --id 19rk3vx7XM4dultTF1tN4srCpdya7uxBV -O looperman_four_bar_checkpoint.pt
$ bash scripts/generate_looperman_four_bar.sh
```

* Generate loops from freesound pretrained model
``` bash
$ gdown --id 197DMCOASEMFBVi8GMahHfRwgJ0bhcUND -O freesound_checkpoint.pt 
$ bash scripts/generate_freesound.sh
``` 
## Pretrained Checkpoint
* [Looperman pretrained one-bar model](https://drive.google.com/file/d/1GQpzWz9ycIm5wzkxLsVr-zN17GWD3_6K/view?usp=sharing) 
* [Looperman pretrained four-bar model](https://drive.google.com/file/d/19rk3vx7XM4dultTF1tN4srCpdya7uxBV/view?usp=sharing)
* [Freesound pretrained one-bar model](https://drive.google.com/file/d/197DMCOASEMFBVi8GMahHfRwgJ0bhcUND/view?usp=sharing)

## Benchmarking Freesound Loop Dataset
### Download dataset
``` bash

$ gdown --id 1fQfSZgD9uWbCdID4SzVqNGhsYNXOAbK5
$ unzip freesound_mel_80_320.zip

```
### Training

``` bash
$ CUDA_VISIBLE_DEVICES=2 python train_drum.py \
    --size 64 --batch 8 --sample_dir [sample_dir] \
    --checkpoint_dir [checkpoint_dir] \
    [mel-spectrogram dataset from the proprocessing]
```

### Generate audio
```bash
$ CUDA_VISIBLE_DEVICES=2 python generate_audio.py \
    --ckpt [freesound checkpoint] \
    --pics 2000 --data_path "./data/freesound" \
    --store_path "./generated_freesound_one_bar"
```
### Evaluation
#### NDB_JS
* 2000 looperman melspectrogram [link](https://drive.google.com/file/d/1aFGPYlkkAysVBWp9VacHVk2tf-b4rLIh/view?usp=sharing)
    ``` bash
    $ cd evaluation/NDB_JS
    $ gdown --id 1aFGPYlkkAysVBWp9VacHVk2tf-b4rLIh
    $ unzip looper_2000.zip # contain 2000 looperman mel-sepctrogram
    $ rm looper_2000/.zip
    $ bash compute_ndb_js.sh 
    ```
#### IS
* Short-Chunk CNN [checkpoint](./evaluation/IS/best_model.ckpt)
    ``` bash
    $ cd evaluation/IS
    $ bash compute compute_is_score.sh 
    ```
#### FAD
* FAD looperman ground truth [link](./evaluation/FAD/looperman_2000.stats), follow the official [doc](fad) to download the code and the evaluation.

    ``` bash
    $ ls --color=never generated_freesound_one_bar/*.wav > freesound.csv
    $ python -m frechet_audio_distance.create_embeddings_main --input_files freesound.csv --stats freesound.stats
    $ python -m frechet_audio_distance.compute_fad --background_stats ./looperman_2000.stats --test_stats freesound.stats
    ```



## Train the model with your loop dataset
### Preprocess the Loop Dataset
In the [preprocess](./preprocess) directory and modify some settings (e.g. data path) in the codes and run them with the following orders
``` bash
$ python trim_2_seconds.py # Cut loop into the single bar and stretch them to 2 second.
$ python extract_mel.py # Extract mel-spectrogram from 2-second audio.
$ python make_dataset.py 
$ python compute_mean_std.py 
```

### Train the Model
``` bash
CUDA_VISIBLE_DEVICES=2 python train_drum.py \
    --size 64 --batch 8 --sample_dir [sample_dir] \
    --checkpoint_dir [checkpoint_dir] \
    [mel-spectrogram dataset from the proprocessing]
```
* checkpoint_dir stores model in the designated directory.
* sample_dir stores mel-spectrogram generated from the model.
* You should give the data directory in the end.
* There is an example training [script](./train.sh)

## Vocoder
We use [MelGAN][melgan] as the vocoder. We trained the vocoder with looperman dataset and use the vocoder in generating freesound and looperman models.
The trained vocoder is in [melgan](./melgan) directory.

## References
The code comes heavily from the code below
* [StyleGAN2 from rosinality][stylegan2]
* [Official MelGAN repo][melgan] 
* [Official UNAGAN repo from ciaua][unagan].
* [Official Short Chunk CNN repo][cnn]
* [FAD official document][fad]

[fad]: https://github.com/google-research/google-research/tree/master/frechet_audio_distance
[cnn]: https://github.com/minzwon/sota-music-tagging-models
[stylegan2]: https://github.com/rosinality/stylegan2-pytorch
[unagan]: https://github.com/ciaua/unagan
[melgan]: https://github.com/descriptinc/melgan-neurips

## Citation
If you find this repo useful, please kindly cite with the following information.
```
@inproceedings{ allenloopgen, 
	title={A Benchmarking Initiative for Audio-domain Music Generation using the {FreeSound Loop Dataset}},
	author={Tun-Min Hung and Bo-Yu Chen and Yen-Tung Yeh, and Yi-Hsuan Yang},
	booktitle = {Proc. Int. Society for Music Information Retrieval Conf.},
	year={2021},
}
```
