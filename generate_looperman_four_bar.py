import argparse

import torch
from torchvision import utils
from model_drum_four_bar import Generator
from tqdm import tqdm

import sys
sys.path.append('./melgan')
from modules import Generator_melgan

import yaml
import os

import librosa

import soundfile as sf

import numpy as np

import os
def read_yaml(fp):
    with open(fp) as file:
        # return yaml.load(file)
        return yaml.load(file, Loader=yaml.Loader)

def generate(args, g_ema, device, mean_latent):
    epoch = args.ckpt.split('.')[0]
    
    os.makedirs(f'{args.store_path}/{epoch}', exist_ok=True)
    os.makedirs(f'{args.store_path}/{epoch}/mel_80_320', exist_ok=True)
    feat_dim = 80
    mean_fp = f'{args.data_path}/mean.mel.npy'
    std_fp = f'{args.data_path}/std.mel.npy'
    mean = torch.from_numpy(np.load(mean_fp)).float().view(1, feat_dim, 1).to(device)
    std = torch.from_numpy(np.load(std_fp)).float().view(1, feat_dim, 1).to(device)
    vocoder_config_fp = './melgan/args.yml'
    vocoder_config = read_yaml(vocoder_config_fp)
    
    n_mel_channels = vocoder_config.n_mel_channels
    ngf = vocoder_config.ngf
    n_residual_layers = vocoder_config.n_residual_layers
    sr=44100
    
    vocoder = Generator_melgan(n_mel_channels, ngf, n_residual_layers).to(device)
    vocoder.eval()
    
    vocoder_param_fp = os.path.join('./melgan', 'best_netG.pt')
    vocoder.load_state_dict(torch.load(vocoder_param_fp))
    
    
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )
            np.save(f'{args.store_path}/{epoch}/mel_80_320/{i}.npy', sample.squeeze().data.cpu().numpy())
            
            utils.save_image(
                sample,
                f"{args.store_path}/{epoch}/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
            de_norm = sample.squeeze(0) * std + mean
            audio_output = vocoder(de_norm)
            sf.write(f'{args.store_path}/{epoch}/{i}.wav', audio_output.squeeze().detach().cpu().numpy(), sr)
            print('generate {}th wav file'.format(i))
@torch.no_grad()
def style_mixing(args, generator, step, mean_style, n_source, n_target, device, j):
    index = 5
    # create directory
    os.makedirs(f'./generated_interpolation_{index}/{j}', exist_ok=True)
    
    # load melgan vocoder
    feat_dim = 80
    mean_fp = f'{args.data_path}/mean.mel.npy'
    std_fp = f'{args.data_path}/std.mel.npy'
    mean = torch.from_numpy(np.load(mean_fp)).float().view(1, feat_dim, 1).to(device)
    std = torch.from_numpy(np.load(std_fp)).float().view(1, feat_dim, 1).to(device)
    vocoder_config_fp = './melgan/args.yml'
    vocoder_config = read_yaml(vocoder_config_fp)
    
    n_mel_channels = vocoder_config.n_mel_channels
    ngf = vocoder_config.ngf
    n_residual_layers = vocoder_config.n_residual_layers
    sr=44100
    
    vocoder = Generator_melgan(n_mel_channels, ngf, n_residual_layers).to(device)
    vocoder.eval()
    
    vocoder_param_fp = os.path.join('./melgan', 'best_netG.pt')
    vocoder.load_state_dict(torch.load(vocoder_param_fp))
    
    #generate spectrogram
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.ones(1, 1, 80, 320).to(device) * -1]

    source_image,_ = generator(
        [source_code], truncation=args.truncation, truncation_latent=mean_style
    )
    target_image,_ = generator(
        [target_code], truncation=args.truncation, truncation_latent=mean_style
    )
    
    images.append(source_image)
    
    for i in range(n_source):
        de_norm = source_image[i] * std + mean
        audio_output = vocoder(de_norm)
        sf.write(f'./generated_interpolation_{index}/{j}/source_{i}.wav', audio_output.squeeze().detach().cpu().numpy(), sr)
    
    for i in range(n_target):
        de_norm = target_image[i] * std + mean
        audio_output = vocoder(de_norm)
        sf.write(f'./generated_interpolation_{index}/{j}/target_{i}.wav', audio_output.squeeze().detach().cpu().numpy(), sr)
    
    for i in range(n_target):
        image, _ = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            truncation_latent=mean_style,
            inject_index = index
        )
        
        for k in range(n_source):
            de_norm = image[k] * std + mean
            audio_output = vocoder(de_norm)
            sf.write(f'./generated_interpolation_{index}/{j}/source_{k}_target_{i}.wav', audio_output.squeeze().detach().cpu().numpy(), sr)
        
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    utils.save_image(
            images, f'./generated_interpolation_{index}/{j}/sample_mixing.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
    )
    return images
if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=64, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path store the std and mean of mel",
    )
    parser.add_argument(
        "--store_path",
        type=str,
        help="path store the generated audio",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    parser.add_argument("--style_mixing", action = "store_true")
    parser.add_argument('--n_row', type=int, default=3, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
    # Style mixing
    if args.style_mixing == True:
        step = 0
        for j in range(20):
            img = style_mixing(args,g_ema, step, mean_latent, args.n_col, args.n_row, device, j)
