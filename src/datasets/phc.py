import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class PhCDataset(Dataset):
    """Photonic crystal dataset loaded from a .npz archive.

    Expected keys in the archive:
        images : (N, 32, 32) — permittivity maps (float32 or uint8)
        dos    : (N, 400)    — density-of-states vectors (float32)

    Parameters
    ----------
    path       : path to the .npz file
    split      : 'train' | 'test'
    train_frac : fraction of samples used for training
    seed       : random seed for the train/test permutation
    transform  : optional transform applied to the (1, 32, 32) image tensor
    """

    def __init__(
        self,
        path:        str,
        split:       str   = 'train',
        train_frac:  float = 0.6,
        seed:        int   = 42,
        transform           = None,
    ):
        data   = np.load(path)
        images = data['images'].astype(np.float32)
        dos    = data['dos'].astype(np.float32)

        if images.max() > 1.0:
            images = images / 255.0

        N    = len(images)
        rng  = np.random.default_rng(seed)
        perm = rng.permutation(N)
        cut  = int(N * train_frac)
        idx  = perm[:cut] if split == 'train' else perm[cut:]

        self.images    = torch.from_numpy(images[idx]).unsqueeze(1)  # (M, 1, 32, 32)
        self.dos       = torch.from_numpy(dos[idx])                  # (M, 400)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        x = self.images[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.dos[i]


def make_phc_ijepa_loaders(
    path:         str,
    train_transform,
    collator      = None,
    batch_size:   int   = 512,
    num_workers:  int   = 4,
    pin_mem:      bool  = True,
    world_size:   int   = 1,
    rank:         int   = 0,
    train_frac:   float = 0.6,
    seed:         int   = 42,
):
    """Build a training DataLoader for I-JEPA with the mask collator as collate_fn.

    Returns (dataset, train_loader, train_sampler) to match the interface of
    make_imagenette_ijepa and make_imagenet1k.
    """
    train_ds = PhCDataset(path, split='train', train_frac=train_frac, seed=seed,
                          transform=train_transform)
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return train_ds, train_loader, train_sampler


def make_phc_loaders(
    path:         str,
    train_transform,
    test_transform  = None,
    batch_size:   int   = 512,
    num_workers:  int   = 4,
    pin_mem:      bool  = True,
    world_size:   int   = 1,
    rank:         int   = 0,
    train_frac:   float = 0.6,
    seed:         int   = 42,
):
    """Build train and test DataLoaders for a PhC .npz dataset."""
    train_ds = PhCDataset(path, split='train', train_frac=train_frac, seed=seed,
                          transform=train_transform)
    test_ds  = PhCDataset(path, split='test',  train_frac=train_frac, seed=seed,
                          transform=test_transform)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    return train_loader, test_loader, train_sampler
