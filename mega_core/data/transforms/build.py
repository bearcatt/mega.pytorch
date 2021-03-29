# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T

from . import build_image_transform as IT
from . import build_video_transform as VT


def build_transforms(cfg, is_train=True):
    if len(cfg.INPUT.MIN_SIZE_TRAIN) > 1:
        transform = VT.build_transforms(cfg, is_train)
    else:
        transform = IT.build_transforms(cfg, is_train)
    return transform
