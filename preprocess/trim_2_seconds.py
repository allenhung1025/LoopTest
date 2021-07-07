import librosa
import pyrubberband as pyrb
import madmom
from madmom.features.downbeats import RNNDownBeatProcessor
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
import os
import matplotlib.pyplot as plot
import numpy as np
import soundfile as sf
from multiprocessing import Pool
loop_dir='/home/allenhung/nas189/home/bandlab/BANDLAB_INSTRUMENT/Guitar'
out_dir='/home/allenhung/nas189/home/bandlab/BANDLAB_INSTRUMENT/Guitar_one_bar'
os.makedirs(out_dir, exist_ok=True)
def one_bar_segment(file):
    file_path = os.path.join(loop_dir, file)
    try:
        y, sr = librosa.core.load(file_path, sr=None) # sr = None will retrieve the original sampling rate = 44100
    except:
        print('load file failed')
        return
    try:
        act = RNNDownBeatProcessor()(file_path)
        down_beat=proc(act) # [..., 2] 2d-shape numpy array
    except:
        print('except happended')
        return
    #print(down_beat)
    #print(len(y) / sr)
    #import pdb; pdb.set_trace()
    #retrieve 1, 2, 3, 4, 1blocks
    count = 0
    bar_list = []
    #print(file)
    name = file.replace('.wav', '')
    print(down_beat)
    for i in range(down_beat.shape[0]):
        if down_beat[i][1] == 1 and i + 4 < down_beat.shape[0] and down_beat[i+4][1] == 1:
            print(down_beat[i: i + 5, :])
            start_time = down_beat[i][0]
            end_time = down_beat[i + 4][0]
            count += 1
            out_path = os.path.join(out_dir, f'{name}_{count}.wav')
            #print(len(y) / sr)
            #print(sr)
            y_one_bar, _ = librosa.core.load(file_path, offset=start_time, duration = end_time - start_time, sr=None)
            y_stretch = pyrb.time_stretch(y_one_bar, sr,  (end_time - start_time) / 2)
            #print((end_time - start_time))
            #print()
            sf.write(out_path, y_stretch, sr)

            print('save file: ',  f'{name}_{count}.wav')
            #y, sr = librosa.core.load(out_path, sr=None)
            #print(librosa.get_duration(y, sr=sr))

if __name__ == '__main__':
    #dur_list = []
    #for file in os.listdir(loop_dir):
    #    file_path = os.path.join(loop_dir, file)
    #    y, sr = librosa.core.load(file_path)
    #    dur = librosa.get_duration(y, sr)
    #    dur_list.append(dur)
    #num_bins = 10
    #plot.hist(dur_list, num_bins, density=True)
    #plot.savefig('./duration.png ')

    proc = DBNDownBeatTrackingProcessor(beats_per_bar=4, fps = 100)
    file_list = list(os.listdir(loop_dir))
    #print(file_list[1])
    #one_bar_segment(file_list[1])
    #print(file_list[:10])
    #for file in os.listdir(loop_dir):
    #    file_list.append(file)
    with Pool(processes=10) as pool:
        pool.map(one_bar_segment, file_list)
