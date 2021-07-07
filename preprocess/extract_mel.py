# import glob
import os
import librosa
import numpy as np
# from utils.display import *
# from utils.dsp import *
# import hparams as hp
# from multiprocessing import Pool, cpu_count
from multiprocessing import Pool
from librosa.filters import mel as librosa_mel_fn
import torch
from torch import nn
from torch.nn import functional as F

import argparse
'''
Modified from
https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py#L26
'''


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=22050,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec[:, :, :]


def convert_file(path):
    y, _ = librosa.load(path, sr=sr)
    peak = np.abs(y).max()
    if peak_norm or peak > 1.0:
        y /= peak

    y = torch.from_numpy(y)
    y = y[None, None]
    mel = extract_func(y)
    mel = mel.numpy()
    mel = mel[0]
    print(mel.shape)

    return mel.astype(np.float32)


def process_audios(path):
    id = path.split('/')[-1][:-4]

    out_dir = os.path.join(base_out_dir, feat_type)
    os.makedirs(out_dir, exist_ok=True)

    out_fp = os.path.join(out_dir, f'{id}.npy')

    if os.path.exists(out_fp):
        print('Done before')
        return id, 0

    try:
        m = convert_file(path)

        np.save(out_fp, m, allow_pickle=False)
    except Exception:
        return id, 0
    return id, m.shape[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute inception score")

    parser.add_argument("--epoch", type=str, help="path to the model")

    args = parser.parse_args()
    base_out_dir = f'/home/allenhung/nas189/home/bandlab/BANDLAB_INSTRUMENT/Guitar_one_bar'
    os.makedirs(base_out_dir, exist_ok=True)
    clip_dir = f'/home/allenhung/nas189/home/bandlab/BANDLAB_INSTRUMENT/Guitar_one_bar'  #out_dir from step1

    feat_type = 'mel_80_320'
    extension = '.wav'
    peak_norm = True

    n_fft = 1024
    hop_length = 275 #[241, 482, 964, 1928, 3856]
    win_length = 1024
    sampling_rate = 44100
    n_mel_channels = 80 #[80, 40, 20, 10, 5]

    # ### Process ###
    extract_func = Audio2Mel(n_fft, hop_length, win_length, sampling_rate, n_mel_channels)
    sr = sampling_rate

    audio_fns = [fn for fn in os.listdir(clip_dir) if fn.endswith(extension)]

    audio_fns = sorted(list(audio_fns))

    audio_files = [os.path.join(clip_dir, fn) for fn in audio_fns]

    pool = Pool(processes=20)
    dataset = []

    for i, (id, length) in enumerate(pool.imap_unordered(process_audios, audio_files), 1):
        print(id)
        if length == 0:
            continue
        dataset += [(id, length)]
