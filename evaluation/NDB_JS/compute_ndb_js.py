import ndb
import argparse
import os
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="compute ndb and JS divergence")

    parser.add_argument("--real_dir", type=str)

    parser.add_argument("--gen_dir", type=str)

    parser.add_argument("--mean_std_dir", type=str, help="directory which has mean and std npy file")

    args = parser.parse_args()

    dim = 80 * 320
    n_train = 2000 #Don't change this line, this is the amount of looperman dataset
    n_test = 2000 # You can change this line depends on how many generated audio you have in the generation directory

    # load train samples
    mean_fp = os.path.join(args.mean_std_dir, f'mean.mel.npy')
    std_fp = os.path.join(args.mean_std_dir, f'std.mel.npy')
    feat_dim = 80
    mean = np.load(mean_fp).reshape((1, feat_dim, 1))
    std = np.load(std_fp).reshape((1, feat_dim, 1))

    train_samples = np.zeros(shape = [n_train, dim])

    for i, path in enumerate(os.listdir(args.real_dir)):
        train_path = os.path.join(args.real_dir, path)
        train_numpy = np.load(train_path)
        train_numpy = (train_numpy - mean) / std
        train_samples[i, :] = train_numpy.reshape((-1, ))

    # load test samples


    test_samples = np.zeros(shape = [n_test, dim])

    for i, path in enumerate(os.listdir(args.gen_dir)):

        test_path = os.path.join(args.gen_dir, path)
        test_numpy = np.load(test_path)
        test_samples[i, :] = test_numpy.reshape((-1, ))

    # NDB and JSD calculation
    k = 100
    train_ndb = ndb.NDB(training_data=train_samples, number_of_bins=k, whitening=True)

    train_ndb.evaluate(test_samples, model_label='Test')
