
import os, json, glob
from skimage import io
from tqdm import tqdm
import numpy as np
import random
from torchvision import datasets

import h5py


def build_h5(data_dir, save_dir):
    dataset = datasets.MNIST(root=data_dir, download=True, transform=None)

    train_all_file = h5py.File(os.path.join(save_dir, 'train_all.h5'), 'w')
    train_12_file = h5py.File(os.path.join(save_dir, 'train_12.h5'), 'w')
    train_34_file = h5py.File(os.path.join(save_dir, 'train_34.h5'), 'w')
    train_56_file = h5py.File(os.path.join(save_dir, 'train_56.h5'), 'w')
    train_78_file = h5py.File(os.path.join(save_dir, 'train_78.h5'), 'w')
    train_90_file = h5py.File(os.path.join(save_dir, 'train_90.h5'), 'w')

    # train imgs are 250 x 250 resolution extracted from original images
    print('Processing training files')
    for i in range(len(dataset)):
        image, label = dataset[i]
        train_all_file.create_dataset('images/{:d}'.format(i), data=image)
        train_all_file.create_dataset('labels/{:d}'.format(i), data=label)
        if label in [1, 2]:
            train_12_file.create_dataset('images/{:d}'.format(i), data=image)
            train_12_file.create_dataset('labels/{:d}'.format(i), data=label)
        elif label in [3, 4]:
            train_34_file.create_dataset('images/{:d}'.format(i), data=image)
            train_34_file.create_dataset('labels/{:d}'.format(i), data=label)
        elif label in [5, 6]:
            train_56_file.create_dataset('images/{:d}'.format(i), data=image)
            train_56_file.create_dataset('labels/{:d}'.format(i), data=label)
        elif label in [7, 8]:
            train_78_file.create_dataset('images/{:d}'.format(i), data=image)
            train_78_file.create_dataset('labels/{:d}'.format(i), data=label)
        else:
            train_90_file.create_dataset('images/{:d}'.format(i), data=image)
            train_90_file.create_dataset('labels/{:d}'.format(i), data=label)
    print('all: {:d}'.format(len(train_all_file['images'])))
    print('12: {:d}'.format(len(train_12_file['images'])))
    print('34: {:d}'.format(len(train_34_file['images'])))
    print('56: {:d}'.format(len(train_56_file['images'])))
    print('78: {:d}'.format(len(train_78_file['images'])))
    print('90: {:d}'.format(len(train_90_file['images'])))
    train_all_file.close()
    train_12_file.close()
    train_34_file.close()
    train_56_file.close()
    train_78_file.close()
    train_90_file.close()


def extract_imgs(h5_file_path, save_path):
    h5_file = h5py.File(h5_file_path, 'r')
    os.makedirs('{:s}/images'.format(save_path), exist_ok=True)
    for file_key in h5_file['images'].keys():
        # print(file_key)
        img = h5_file['images/{:s}'.format(file_key)][()]
        io.imsave('{:s}/images/{:s}.png'.format(save_path, file_key), img)


build_h5('/share_hd1/db/MNIST/true_images', '/share_hd1/db/MNIST/true_images')
extract_imgs('/share_hd1/db/MNIST/true_images/train_12.h5', '/share_hd1/db/MNIST/true_images/imgs_12')

# build_h5('./true_images', './true_images')
# extract_imgs('./true_images/train_12.h5', './true_images/imgs_12')
