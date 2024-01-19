"""
Author: Wouter Van Gansbeke

File with augmentations based on PIL
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numbers
from collections.abc import Sequence
from PIL import Image, ImageFilter
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import random
import math
import sys
from typing import Tuple

# define interpolation modes
INT_MODES = {
    'image': 'bicubic',
    'semseg': 'nearest',
    'class_labels': 'nearest',
    'mask': 'nearest',
    'image_semseg': 'bicubic',
    'image_class_labels': 'bicubic',
}


def resize_operation(img, h, w, mode='bicubic'):
    if mode == 'bicubic':
        img = img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None)
    elif mode == 'bilinear':
        img = img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BILINEAR, reducing_gap=None)
    elif mode == 'nearest':
        img = img.resize((w, h), resample=getattr(Image, 'Resampling', Image).NEAREST, reducing_gap=None)
    else:
        raise NotImplementedError
    return img


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if elem in ['meta', 'text']:
                    continue
                else:
                    sample[elem] = F.hflip(sample[elem])

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip(p=0.5)'


class RandomColorJitter(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __init__(self) -> None:
        self.colorjitter = torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if elem in ['image']:
                    sample[elem] = self.colorjitter(sample[elem])

        return sample

    def __str__(self):
        return f'RandomColorJitter(p={self.p})'


class RandomGaussianBlur(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __init__(self, sigma=[.1, 2.], p=0.2):
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        if random.random() < 0.5:
            for elem in sample.keys():
                if elem in ['image', 'image_semseg']:
                    sigma = random.uniform(self.sigma[0], self.sigma[1])
                    sample[elem] = sample[elem].filter(ImageFilter.GaussianBlur(radius=sigma))
        return sample

    def __str__(self):
        return f'RandomGaussianBlur(p={self.p})'


class CropResize(object):
    def __init__(self, size, crop_mode=None):
        self.size = size
        self.crop_mode = None
        assert self.crop_mode in ['centre', 'random', None]

    def crop_and_resize(self, img, h, w, mode='bicubic'):
        # crop
        if self.crop_mode == 'centre':
            img_w, img_h = img.size
            min_size = min(img_h, img_w)
            if min_size == img_h:
                margin = (img_w - min_size) // 2
                new_img = img.crop((margin, 0, margin+min_size, min_size))
            else:
                margin = (img_h - min_size) // 2
                new_img = img.crop((0, margin, min_size, margin+min_size))

        elif self.crop_mode == 'random':
            img_w, img_h = img.size
            min_size = min(img_h, img_w)
            if min_size == img_h:
                margin = random.randint(0, (img_w - min_size) // 2)
                new_img = img.crop((margin, 0, margin+min_size, min_size))
            else:
                margin = random.randint(0, (img_h - min_size) // 2)
                new_img = img.crop((0, margin, min_size, margin+min_size))
        else:
            new_img = img

        # resize
        if mode == 'bicubic':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BICUBIC, reducing_gap=None)
        elif mode == 'bilinear':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).BILINEAR, reducing_gap=None)
        elif mode == 'nearest':
            new_img = new_img.resize((w, h), resample=getattr(Image, 'Resampling', Image).NEAREST, reducing_gap=None)
        else:
            raise NotImplementedError
        return new_img

    def __call__(self, sample):
        for elem in sample.keys():
            if elem in ['image', 'image_semseg', 'semseg', 'mask', 'class_labels', 'image_class_labels']:
                sample[elem] = self.crop_and_resize(sample[elem], self.size[0], self.size[1], mode=INT_MODES[elem])
        return sample

    def __str__(self) -> str:
        return f"CropResize(size={self.size}, crop_mode={self.crop_mode})"


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem or 'text' in elem:
                continue

            elif elem in ['image', 'image_semseg', 'image_class_labels']:
                sample[elem] = self.to_tensor(sample[elem])  # Regular ToTensor operation

            elif elem in ['semseg', 'mask', 'class_labels']:
                sample[elem] = torch.from_numpy(np.array(sample[elem])).long()  # Torch Long

            else:
                raise NotImplementedError

        return sample

    def __str__(self):
        return 'ToTensor'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem or 'text' in elem:
                continue

            elif elem in ['image', 'image_semseg']:
                sample[elem] = self.normalize(sample[elem])

            else:
                raise NotImplementedError

        return sample

    def __str__(self):
        return f"Normalize(mean={self.normalize.mean}, std={self.normalize.std})"
