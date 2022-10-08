
import os, json, glob
from skimage import io
from tqdm import tqdm
import numpy as np
import random

import h5py


def build_h5(data_dir, save_dir, modality='MTT'):
    train_file = h5py.File('{:s}/train_acdc.h5'.format(save_dir, modality), "w")
    test_file = h5py.File('{:s}/test_acdc.h5'.format(save_dir, modality), "w")

    ori_train_file = h5py.File('{:s}/ACDC_train.h5'.format(data_dir), 'r')
    ori_test_file = h5py.File('{:s}/ACDC_test.h5'.format(data_dir), 'r')

    print('Processing all training files')
    train_set = ori_train_file['train']
    for pat in train_set.keys():
        for key in train_set[pat].keys():
            if key == 'data_without_label':
                continue

            images = train_set['{:s}/{:s}/data'.format(pat, key)][()]
            labels = train_set['{:s}/{:s}/label'.format(pat, key)][()]
            images = images * ((pow(2, 8) - 1) / images.max())
            images = images.astype("uint8")

            for i in range(images.shape[2]):
                image = images[:, :, i]
                label = labels[:, :, i]
                if np.count_nonzero(label) < 10:
                    continue

                image = image[:, :, None].repeat(3, axis=2)

                train_file.create_dataset('images/{:s}_{:s}_{:d}'.format(pat, key, i), data=image)
                train_file.create_dataset('labels/{:s}_{:s}_{:d}'.format(pat, key, i), data=label)

    print('Processing test files')
    test_set = ori_test_file['test']
    for pat in test_set.keys():
        for key in test_set[pat].keys():
            if key == 'data_without_label':
                continue
            images = test_set['{:s}/{:s}/data'.format(pat, key)][()]
            labels = test_set['{:s}/{:s}/label'.format(pat, key)][()]
            images = images * ((pow(2, 8) - 1) / images.max())
            images = images.astype("uint8")

            for i in range(images.shape[2]):
                image = images[:, :, i]
                label = labels[:, :, i]

                if np.count_nonzero(label) < 10:
                    continue
                image = image[:, :, None].repeat(3, axis=2)

                test_file.create_dataset('images/{:s}_{:s}_{:d}'.format(pat, key, i), data=image)
                test_file.create_dataset('labels/{:s}_{:s}_{:d}'.format(pat, key, i), data=label)

    print('number of images in train set: {:d}'.format(len(train_file['images'].keys())))
    print('number of images in test set: {:d}'.format(len(test_file['images'].keys())))
    train_file.close()
    test_file.close()


def extract_imgs(h5_file_path, save_path):
    h5_file = h5py.File(h5_file_path, 'r')
    for key in h5_file.keys():
        os.makedirs('{:s}/{:s}'.format(save_path, key), exist_ok=True)
        for file_key in list(h5_file[key].keys()):
            img = h5_file['{:s}/{:s}'.format(key, file_key)][()]
            io.imsave('{:s}/{:s}/{:s}.png'.format(save_path, key, file_key), img)


# build_h5('/share_hd1/db/ACDC', '/share_hd1/db/ACDC')

# extract_imgs('/share_hd1/db/ACDC/train_acdc.h5', '/share_hd1/db/ACDC/train_acdc')

# extract_imgs('../../db/brain/train_isles_{:s}.h5'.format(modality), '../../db/brain/isles_train_{:s}'.format(modality))
# extract_imgs('../../db/brain/test_isles.h5', '../../db/brain/isles_test')
# extract_imgs('../../db/Nuclei/256_no_color_norm/train_prostate.h5', '../../db/Nuclei/256_no_color_norm/prostate')
# extract_imgs('../../db/ISLES/ISLES_train.h5', '../../db/ISLES/train')

# extract_imgs('../../results/one_per_task_brain/one_per_task_brain_1/test_latest/Brain_lifelong_resnet_9blocks_epochlatest.h5',
#              '../../results/one_per_task_brain/one_per_task_brain_1/test_latest/images')


h5_file = h5py.File('/share_hd1/db/ACDC/test_acdc.h5', 'r')
print(len(h5_file['images'].keys()))
