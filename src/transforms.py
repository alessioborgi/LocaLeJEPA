# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from logging import getLogger

from PIL import ImageFilter

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

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
# Recorded transforms — intercept parameters for equivariance metadata
# ---------------------------------------------------------------------------

class RecordedRandomResizedCrop(transforms.RandomResizedCrop):
    """RandomResizedCrop that records the crop box after each call.

    After __call__, ``last_params`` holds normalised coordinates in [0, 1]:
        top, left, height, width  — relative to the PIL image dimensions
        scale                     — fraction of image area that was cropped
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_params = {'top': 0., 'left': 0., 'height': 0., 'width': 0., 'scale': 0.}

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img_w, img_h = img.size  # PIL image: (W, H)
        self.last_params = {
            'top':    i / img_h,
            'left':   j / img_w,
            'height': h / img_h,
            'width':  w / img_w,
            'scale':  float(h * w) / float(img_h * img_w),
        }
        return TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)


class RecordedRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """RandomHorizontalFlip that records whether the flip was applied."""

    def __init__(self, p=0.5):
        super().__init__(p)
        self.last_flipped = False

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            self.last_flipped = True
            return TF.hflip(img)
        self.last_flipped = False
        return img


class _RecordingPipeline:
    """Augmentation pipeline with crop/flip metadata accessible after each call.

    Wraps a Compose while holding references to the RecordedRandomResizedCrop
    and (optionally) RecordedRandomHorizontalFlip so MultiCropTransform can
    read their state immediately after invoking the pipeline.
    """

    def __init__(self, crop, flip, other_transforms):
        self._crop = crop
        self._flip = flip
        self._pipeline = transforms.Compose(
            [crop] + ([flip] if flip is not None else []) + other_transforms
        )

    def __call__(self, img):
        return self._pipeline(img)

    def meta_tensor(self):
        """Return a 6-dim float32 tensor: [top, left, height, width, scale, flipped].

        All values are normalised to [0, 1] so the metadata can be stacked
        directly into a (V, 6) tensor and collated by the default DataLoader.
        """
        p = self._crop.last_params
        f = float(self._flip.last_flipped) if self._flip is not None else 0.
        return torch.tensor(
            [p['top'], p['left'], p['height'], p['width'], p['scale'], f],
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# LeJEPA: DINO-style multi-crop transform (2 global + N local views)
# ---------------------------------------------------------------------------

class MultiCropTransform:
    """DINO-style multi-crop: multiple (pipeline, count) pairs at the same
    output resolution so views stack to (V_total, C, H, W).

    Local views achieve zoom-in semantics via a tighter crop_scale, not a
    smaller output size — keeping the tensor contiguous and satisfying the
    plan's ResNet requirement ("resize local crops to 224").

    When *record_metadata* is True (requires pipelines built via
    ``make_multicrop_transforms(record_metadata=True)``), __call__ returns
    a tuple ``(views, meta)`` instead of just ``views``, where

        views : Tensor[V, C, H, W]
        meta  : Tensor[V, 6]  — [top, left, height, width, scale, flipped]

    Both are normalised to [0, 1].  Datasets pass through the tuple so the
    DataLoader collates items ``(views, meta), label`` into
    ``(batch_views, batch_meta), batch_labels``, leaving future equivariance
    work free of pipeline refactoring.
    """

    def __init__(self, view_specs, record_metadata=False):
        self.view_specs     = view_specs       # list of (pipeline, num_views)
        self.record_metadata = record_metadata

    def __call__(self, img):
        views = []
        meta  = []
        for pipeline, n in self.view_specs:
            for _ in range(n):
                v = pipeline(img)
                views.append(v)
                if self.record_metadata:
                    meta.append(pipeline.meta_tensor())
        stacked = torch.stack(views)
        if self.record_metadata:
            return stacked, torch.stack(meta)  # (V, C, H, W), (V, 6)
        return stacked


def make_multicrop_transforms(
    global_crop_size=224,
    num_global_crops=2,
    num_local_crops=6,
    global_crop_scale=(0.3, 1.0),
    local_crop_scale=(0.05, 0.3),
    use_color_distortion=True,
    color_jitter=0.8,
    use_gaussian_blur=True,
    use_horizontal_flip=True,
    normalization=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
    record_metadata=False,
):
    """DINO-style multi-crop augmentation.

    Both global and local crops are resized to *global_crop_size* so all
    V_total = num_global_crops + num_local_crops views share the same spatial
    size and can be stacked into a single contiguous tensor.

    When *record_metadata* is True, each pipeline is replaced by a
    _RecordingPipeline that captures crop box and flip state after every call.
    MultiCropTransform then returns (views, meta) instead of just views.
    """
    s = color_jitter

    def _other():
        aug = []
        if use_color_distortion:
            aug += [
                transforms.RandomApply([
                    transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        if use_gaussian_blur:
            aug.append(GaussianBlur(p=0.5))
        aug += [
            transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1]),
        ]
        return aug

    def _build_simple(crop_scale):
        aug = [transforms.RandomResizedCrop(global_crop_size, scale=crop_scale)]
        if use_horizontal_flip:
            aug.append(transforms.RandomHorizontalFlip())
        aug += _other()
        return transforms.Compose(aug)

    def _build_recording(crop_scale):
        crop = RecordedRandomResizedCrop(global_crop_size, scale=crop_scale)
        flip = RecordedRandomHorizontalFlip() if use_horizontal_flip else None
        return _RecordingPipeline(crop, flip, _other())

    build = _build_recording if record_metadata else _build_simple

    specs = [(build(global_crop_scale), num_global_crops)]
    if num_local_crops > 0:
        specs.append((build(local_crop_scale), num_local_crops))
    return MultiCropTransform(specs, record_metadata=record_metadata)


# ---------------------------------------------------------------------------
# LeJEPA: legacy uniform multi-view transform (kept for backward compat)
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
