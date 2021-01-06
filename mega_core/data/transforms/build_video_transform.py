# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, size=None, target=None):
        for t in self.transforms:
            image, target = t(image, size, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, size=None, target=None):
        image = F.resize(image, size)
        if target is None:
            return image, target
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob
        self.chance = 0.0

    def __call__(self, image, size=None, target=None):
        if target is not None:
            self.chance = random.random()
        if self.chance < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)

        return image, target


class ToTensor(object):
    def __call__(self, image, size=None, target=None):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, size=None, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, target
        return image, target



def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = Compose(
        [
            Resize(min_size, max_size),
            RandomHorizontalFlip(flip_horizontal_prob),
            ToTensor(),
            normalize_transform,
        ]
    )
    return transform
