# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms

_GLOBAL_SEED = 0
logger = getLogger()


# ---------------------------------------------------------------------------
# I-JEPA: single-view transform (used with mask-based collator)
# ---------------------------------------------------------------------------

def make_transforms(
    crop_size=224,
    crop_scale=(0.3, 1.0),
    color_jitter=1.0,
    horizontal_flip=False,
    color_distortion=False,
    gaussian_blur=False,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
):
    logger.info('making imagenet data transforms')

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    transform_list = []
    transform_list += [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if color_distortion:
        transform_list += [get_color_distortion(s=color_jitter)]
    if gaussian_blur:
        transform_list += [GaussianBlur(p=0.5)]
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize(normalization[0], normalization[1])]

    transform = transforms.Compose(transform_list)
    return transform


# ---------------------------------------------------------------------------
# LeJEPA: multi-view transform
# ---------------------------------------------------------------------------

class MultiViewTransform:
    """Applies a base transform independently V times and stacks the results.

    Returns a tensor of shape (V, C, H, W) per image.
    """

    def __init__(self, base_transform, num_views):
        self.transform = base_transform
        self.num_views = num_views

    def __call__(self, img):
        return torch.stack([self.transform(img) for _ in range(self.num_views)])


def make_lejepa_transforms(
    crop_size=128,
    crop_scale=(0.08, 1.0),
    num_views=4,
    gaussian_blur=True,
    horizontal_flip=True,
    color_distortion=True,
    color_jitter=0.8,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
):
    """Multi-view augmentation pipeline for LeJEPA training."""
    aug_list = [transforms.RandomResizedCrop(crop_size, scale=crop_scale)]
    if horizontal_flip:
        aug_list.append(transforms.RandomHorizontalFlip())
    if color_distortion:
        s = color_jitter
        aug_list += [
            transforms.RandomApply([transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
    if gaussian_blur:
        aug_list.append(GaussianBlur(p=0.5))
    aug_list += [
        transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1]),
    ]
    return MultiViewTransform(transforms.Compose(aug_list), num_views)


def make_lejepa_eval_transform(
    crop_size=128,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
):
    """Single deterministic view for LeJEPA validation.

    Wraps in MultiViewTransform(num_views=1) so the batch shape
    (N, 1, C, H, W) is consistent with the training forward pass.
    """
    eval_transform = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(normalization[0], normalization[1]),
    ])
    return MultiViewTransform(eval_transform, num_views=1)


# ---------------------------------------------------------------------------
# Shared augmentation primitive
# ---------------------------------------------------------------------------

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
