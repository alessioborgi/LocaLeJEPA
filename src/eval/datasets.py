"""11 downstream dataset loaders for linear evaluation.

All datasets use the standard ImageNet center-crop eval transform:
    Resize(256, BICUBIC) → CenterCrop(224) → ToTensor → Normalize(IN-mean, IN-std)

Datasets with no official train/test split (Caltech-101, SUN397) use a seeded
random 80/20 split that is reproducible across runs.
"""

import os
import random

import torch
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def eval_transform(crop_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _rand_split(dataset, frac: float = 0.8, seed: int = 0):
    """Deterministic random 80/20 split for datasets without official splits."""
    n   = len(dataset)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * frac)
    return (
        torch.utils.data.Subset(dataset, idx[:cut]),
        torch.utils.data.Subset(dataset, idx[cut:]),
    )


def get_dataset_splits(name: str, root: str, crop_size: int = 224):
    """Return (train_dataset, test_dataset) for *name*.

    *root* is the parent directory; each dataset lives under root/<name>/.
    Individual paths can be overridden by passing the exact directory.
    """
    t = eval_transform(crop_size)
    r = os.path.join(root, name)

    if name == 'cifar10':
        return (
            datasets.CIFAR10(r, train=True,  download=True, transform=t),
            datasets.CIFAR10(r, train=False, download=True, transform=t),
        )

    if name == 'cifar100':
        return (
            datasets.CIFAR100(r, train=True,  download=True, transform=t),
            datasets.CIFAR100(r, train=False, download=True, transform=t),
        )

    if name == 'food101':
        return (
            datasets.Food101(r, split='train', download=True, transform=t),
            datasets.Food101(r, split='test',  download=True, transform=t),
        )

    if name == 'mit67':
        # MIT Indoor-67: expects <root>/mit67/train/ and <root>/mit67/test/
        # (standard split from the dataset homepage)
        return (
            datasets.ImageFolder(os.path.join(r, 'train'), transform=t),
            datasets.ImageFolder(os.path.join(r, 'test'),  transform=t),
        )

    if name == 'pets':
        return (
            datasets.OxfordIIITPet(r, split='trainval', download=True, transform=t),
            datasets.OxfordIIITPet(r, split='test',     download=True, transform=t),
        )

    if name == 'flowers102':
        return (
            datasets.Flowers102(r, split='train', download=True, transform=t),
            datasets.Flowers102(r, split='test',  download=True, transform=t),
        )

    if name == 'caltech101':
        # No official split — use seeded 80/20
        full = datasets.Caltech101(r, download=True, transform=t)
        return _rand_split(full, frac=0.8, seed=0)

    if name == 'cars':
        # StanfordCars: auto-download was broken in torchvision >= 0.13.
        # Download manually from https://ai.stanford.edu/~jkrause/cars/car_dataset.html
        return (
            datasets.StanfordCars(r, split='train', download=False, transform=t),
            datasets.StanfordCars(r, split='test',  download=False, transform=t),
        )

    if name == 'aircraft':
        return (
            datasets.FGVCAircraft(r, split='trainval', download=True, transform=t),
            datasets.FGVCAircraft(r, split='test',     download=True, transform=t),
        )

    if name == 'dtd':
        return (
            datasets.DTD(r, split='train', download=True, transform=t),
            datasets.DTD(r, split='test',  download=True, transform=t),
        )

    if name == 'sun397':
        # SUN397: manual download required from http://vision.princeton.edu/projects/2010/SUN/
        # No official split — use seeded 80/20
        full = datasets.SUN397(r, download=False, transform=t)
        return _rand_split(full, frac=0.8, seed=0)

    raise ValueError(
        f'Unknown dataset {name!r}. Available: {ALL_DATASETS}'
    )


ALL_DATASETS = [
    'cifar10', 'cifar100', 'food101', 'mit67',
    'pets', 'flowers102', 'caltech101', 'cars',
    'aircraft', 'dtd', 'sun397',
]
