import torch
import torchvision.datasets as tvdatasets
from logging import getLogger

logger = getLogger()


def make_stl10(
    train_transform,
    val_transform,
    batch_size=256,
    val_batch_size=256,
    pin_mem=True,
    num_workers=4,
    world_size=1,
    rank=0,
    drop_last=True,
    data_root='./data',
):
    """Returns (train_loader, val_loader, train_sampler) for LeJEPA on STL10.

    train_loader uses the unlabeled split (100k images, label=-1).
    val_loader uses the test split (8k labeled images); used only when
    use_probe=True since the unlabeled split carries no class labels.
    """
    train_ds = tvdatasets.STL10(
        root=data_root, split='unlabeled', download=True,
        transform=train_transform,
    )
    val_ds = tvdatasets.STL10(
        root=data_root, split='test', download=True,
        transform=val_transform,
    )
    logger.info('STL10 datasets created (train=unlabeled, val=test)')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_ds,
        num_replicas=world_size,
        rank=rank,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=val_batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    logger.info('STL10 data loaders created')

    return train_loader, val_loader, train_sampler
