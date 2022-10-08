from torchvision import datasets, transforms

from base import BaseDataLoader
from data_loader.BraTS_seg_dataset import BraTSSegDataset
from data_loader.Nuclei_seg_dataset import NucleiSegTrainDataset, NucleiSegTestDataset
# from data_loader.ISLES_seg_dataset import ISLESSegDataset
import data_loader.my_transforms as my_transforms


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class Cifar10DataLoader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=32):
        trsfm = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BraTSSegDataLoader(BaseDataLoader):
    def __init__(self, h5_filepath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        if training:
            trsfm = my_transforms.Compose([
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomRotation(10),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelBinarization(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            trsfm = my_transforms.Compose([
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        self.h5_filepath = h5_filepath
        self.dataset = BraTSSegDataset(self.h5_filepath, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ISLESSegDataLoader(BaseDataLoader):
    def __init__(self, h5_filepath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        if training:
            trsfm = my_transforms.Compose([
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomRotation(10),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelBinarization(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            trsfm = my_transforms.Compose([
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        self.h5_filepath = h5_filepath
        self.dataset = ISLESSegDataset(self.h5_filepath, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class NucleiSegDataLoader(BaseDataLoader):
    def __init__(self, h5_filepath, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True,
                 img_size=224):
        self.h5_filepath = h5_filepath
        if training:
            trsfm = my_transforms.Compose([
                my_transforms.RandomResize(0.8, 1.25),
                my_transforms.RandomHorizontalFlip(),
                my_transforms.RandomAffine(0.3),
                my_transforms.RandomRotation(90),
                my_transforms.RandomCrop(img_size),
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
            )
            self.dataset = NucleiSegTrainDataset(self.h5_filepath, data_transform=trsfm)
        else:
            trsfm = my_transforms.Compose([
                my_transforms.LabelEncoding(),
                my_transforms.ToTensor(),
                my_transforms.Normalize((0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504))]
            )
            self.dataset = NucleiSegTestDataset(self.h5_filepath, data_transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)