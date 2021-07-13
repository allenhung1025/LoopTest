import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
from scipy.stats import entropy
import argparse
import sys
import os
from model import FCN, ShortChunkCNN_one_bar
sys.path.append('../')
sys.path.append('../../')
from dataset import MultiResolutionDataset_drum, data_sampler
from torchvision import transforms
from scipy.stats import ttest_ind
import pickle
#from module import genre_classifier
def inception_score(args, transform):
    #Preprocess
    #100 files under args.data_dir
    #classify into args.class classes
    N = len([name for name in os.listdir(args.data_dir) if os.path.isfile(os.path.join(args.data_dir, name))])
    preds = np.zeros((N, args.classes))

    # load checkpoint
    checkpoint = torch.load(args.path)
    #print(checkpoint['model_state_dict'])
    model =  ShortChunkCNN_one_bar(n_class = args.classes).cuda()
    #print(model)
    #print(checkpoint['model_state_dict'].keys())
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    #Load data
    dataset = MultiResolutionDataset_drum(args.data_dir, transform)
    loader =torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )

    mean_fp = os.path.join(args.mean_std_dir, f'mean.mel.npy')
    std_fp = os.path.join(args.mean_std_dir, f'std.mel.npy')
    feat_dim = 80
    mean = torch.from_numpy(np.load(mean_fp)).float().cuda().view(1, feat_dim, 1)
    std = torch.from_numpy(np.load(std_fp)).float().cuda().view(1, feat_dim, 1)
    #Model inference
    for i, data in enumerate(loader):
        data = data.cuda() # [bs, 1, 80, 320]

        data = data * std + mean
        data = data.squeeze(1)
        output = model(data) # [bs, args.class]

        logit = F.softmax(output, dim = 1).data.cpu().numpy() # [bs, args.class]

        preds[i * 2 : (i + 1) * 2] = logit
    #KL divergence
    scores = []
    py = np.mean(preds, axis=0)

    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))
    #Data PostProcessing
    is_score = np.exp(np.mean(scores))
    std = np.exp(np.std(scores))
    every_score =  np.exp(scores)
    return is_score, std, every_score






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="compute inception score")

    parser.add_argument("path", type=str, help="path to the model")

    parser.add_argument("--data_dir", type=str, help="has 100 npy normalized file under this directory")

    parser.add_argument("--classes", type=int, help="number of classes")

    parser.add_argument("--mean_std_dir", type=str, help="directory which has mean and std npy file")
    parser.add_argument("--store_every_score", type=str)

    args = parser.parse_args()

    transform = transforms.Compose(
        [
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    is_score, std, every_score = inception_score(args, transform)
    with open(f'{args.store_every_score}', 'wb') as f:
        pickle.dump(every_score, f)
    print(is_score, std)
