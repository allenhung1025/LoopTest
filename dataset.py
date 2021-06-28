from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data
import numpy as np
import os
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
class MultiResolutionDataset_drum(Dataset):
    def __init__(self, path, transform, resolution=None):
        self.path_list = []
        for file in os.listdir(path):
            if file.endswith('.npy') == True:
                if file.startswith('std') == False and file.startswith('mean') == False:
                    self.path_list.append(os.path.join(path, file))
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        img = np.load(self.path_list[index])
        img = self.transform(img)

        return img

class MultiResolutionDataset_drum_with_filename(Dataset):
    def __init__(self, path, transform, resolution=None):
        self.path_list = []
        for file in os.listdir(path):
            if file.endswith('.npy') == True:
                if file.startswith('std') == False and file.startswith('mean') == False:
                    self.path_list.append(os.path.join(path, file))
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        img = np.load(self.path_list[index])
        img = self.transform(img)

        return img, self.path_list[index].split('/')[-1]

class MultiResolutionDataset_drum_with_label(Dataset):
    def __init__(self, path, transform, label_dictionary, resolution=None):
        self.path_list = []
        for file in os.listdir(path):
            if file.endswith('.npy') == True:
                if file.startswith('std') == False and file.startswith('mean') == False:
                    self.path_list.append(os.path.join(path, file))
        self.resolution = resolution
        self.transform = transform
class MultiResolutionDataset_drum_with_label(Dataset):
    def __init__(self, path, transform, label_dictionary, resolution=None):
        self.path_list = []
        for file in os.listdir(path):
            if file.endswith('.npy') == True:
                if file.startswith('std') == False and file.startswith('mean') == False:
                    self.path_list.append(os.path.join(path, file))
        self.resolution = resolution
        self.transform = transform

        ## read label dictionary
        import pickle as pickle
        with open(label_dictionary, 'rb') as f:
            self.label_dictionary = pickle.load(f)

        ## genre to int dictionary
        self.genre_to_int = {}
        count = 0
        for _, genre in self.label_dictionary.items():
            if self.genre_to_int.get(genre) == None:
                self.genre_to_int[genre] = count
                count += 1
    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        img = np.load(self.path_list[index])
        img = self.transform(img)

        file_name = self.path_list[index].split('/')[-1]
        label = self.genre_to_int[self.label_dictionary[file_name]]
        return img, label
if __name__ == '__main__':
    transform = transforms.Compose(
        [
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    path = '/home/allenhung/nas189/home/style-based-gan-drum/training_data_one_bar_all/mel_80_320_genre_more_than_600'
    label_dictionary = '/home/allenhung/nas189/home/style-based-gan-drum/training_data_one_bar_all/dict_one_bar_more_than_600.pickle'
    dataset = MultiResolutionDataset_drum_with_label(path, transform, label_dictionary)
    loader = data.DataLoader(
        dataset,
        batch_size=2,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )
    for data in loader:
        import pdb; pdb.set_trace()
