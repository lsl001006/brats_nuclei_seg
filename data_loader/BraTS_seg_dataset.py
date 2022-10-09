
import torch.utils.data as data
import os
from PIL import Image
import h5py
import numpy as np


def get_imgs_list(h5_filepath):
    img_list = []

    with h5py.File(h5_filepath, 'r') as h5_file:
        print(h5_filepath)
        img_filenames = list(h5_file['images'].keys())
        label_filenames = list(h5_file['labels'].keys())

        for img_name in img_filenames:
            if img_name in label_filenames:
                item = ('/images/{:s}'.format(img_name),
                        '/labels/{:s}'.format(img_name))
                img_list.append(tuple(item))

    return img_list


class BraTSSegDataset(data.Dataset):
    def __init__(self, h5_filepath, data_transform=None):
        super(BraTSSegDataset, self).__init__()
        self.h5_filepath = h5_filepath
        self.data_transform = data_transform

        self.img_list = get_imgs_list(h5_filepath)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

    def __getitem__(self, index):
        with h5py.File(self.h5_filepath, 'r') as h5_file:
            img_path, label_path = self.img_list[index]
            img, label = h5_file[img_path][()], h5_file[label_path][()]
            if np.max(label) == 1:
                label = (label * 255).astype(np.uint8)
            img = Image.fromarray(img, 'RGB')
            label = Image.fromarray(label)

            if self.data_transform is not None:
                img, label = self.data_transform((img, label))


        return img, label, img_path.split('/')[-1]

    def __len__(self):
        return len(self.img_list)