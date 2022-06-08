import os
import sys
import numpy as np

import imageio
from skimage import transform
from skimage import img_as_float

import torch
from torch.utils import data

from data_utils import create_or_load_statistics, create_distrib, normalize_images, data_augmentation


class DataLoader(data.Dataset):
    def __init__(self, mode, dataset_input_path, images, crop_size, stride_crop, output_path):
        super().__init__()
        assert mode in ['Train', 'Test']

        self.mode = mode
        self.dataset_input_path = dataset_input_path
        self.images = images
        self.crop_size = crop_size
        self.stride_crop = stride_crop

        self.output_path = output_path

        # data and label
        self.data, self.labels = self.load_images()
        self.data[np.where(self.data < -1.0e+38)] = 0  # remove extreme negative values (probably NO_DATA values)
        print(self.data.ndim, self.data.shape, self.data[0].shape, np.min(self.data), np.max(self.data),
              self.labels.shape, np.bincount(self.labels.astype(int).flatten()))

        if self.data.ndim == 4:  # if all images have the same shape
            self.num_channels = self.data.shape[-1]  # get the number of channels
        else:
            self.num_channels = self.data[0].shape[-1]  # get the number of channels

        self.num_classes = 2  # binary - two classes
        # negative classes will be converted into 2 so they can be ignored in the loss
        self.labels[np.where(self.labels < 0)] = 2

        print('num_channels and labels', self.num_channels, self.num_classes, np.bincount(self.labels.flatten()))

        self.distrib, self.gen_classes = self.make_dataset()

        self.mean, self.std = create_or_load_statistics(self.data, self.distrib, self.crop_size,
                                                        self.stride_crop, self.output_path)

        if len(self.distrib) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def load_images(self):
        images = []
        masks = []
        for img in self.images:
            temp_image = img_as_float(imageio.imread(os.path.join(self.dataset_input_path, img + '_stack.tif')))
            temp_mask = imageio.imread(os.path.join(self.dataset_input_path, img + '_mask.tif')).astype(int)
            images.append(temp_image)
            masks.append(temp_mask)

        return np.asarray(images), np.asarray(masks)

    def make_dataset(self):
        return create_distrib(self.labels, self.crop_size, self.stride_crop, self.num_classes, return_all=True)

    def __getitem__(self, index):
        # Reading items from list.
        cur_map, cur_x, cur_y = self.distrib[index][0], self.distrib[index][1], self.distrib[index][2]

        img = np.copy(self.data[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size, :])
        label = np.copy(self.labels[cur_map][cur_x:cur_x + self.crop_size, cur_y:cur_y + self.crop_size])

        # Normalization.
        normalize_images(img, self.mean, self.std)

        if self.mode == 'Train':
            img, label = data_augmentation(img, label)

        img = np.transpose(img, (2, 0, 1))

        # Turning to tensors.
        img = torch.from_numpy(img.copy())
        label = torch.from_numpy(label.copy())

        # Returning to iterator.
        return img.float(), label, cur_map, cur_x, cur_y

    def __len__(self):
        return len(self.distrib)
