import torch
from datasets import load_dataset
from logging import getLogger

logger = getLogger()


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class HFDataset(torch.utils.data.Dataset):
    """ImageNette loaded from HuggingFace (frgfm/imagenette, 160px split).

    Accepts any callable transform so it can be driven by either the
    LeJEPA MultiViewTransform (train) or the eval variant (val).
    """

    def __init__(self, split, transform):
        self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        self.transform = transform

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        return self.transform(img), item["label"]

    def __len__(self):
        return len(self.ds)


# ---------------------------------------------------------------------------
# LeJEPA: multi-view train loader + single-view val loader (with labels)
# ---------------------------------------------------------------------------

def make_imagenette(
    train_transform,
    val_transform,
    batch_size=256,
    val_batch_size=256,
    pin_mem=True,
    num_workers=4,
    world_size=1,
    rank=0,
    drop_last=True,
):
    """Returns (train_loader, val_loader, train_sampler) for LeJEPA.

    train_transform should be a MultiViewTransform returning (V, C, H, W).
    val_transform should be a MultiViewTransform with num_views=1, returning (1, C, H, W).
    """
    train_ds = HFDataset("train",      transform=train_transform)
    val_ds   = HFDataset("validation", transform=val_transform)
    logger.info('Imagenette (LeJEPA) datasets created')

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
    logger.info('Imagenette (LeJEPA) data loaders created')

    return train_loader, val_loader, train_sampler
