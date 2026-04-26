# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np

from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


# ---------------------------------------------------------------------------
# I-JEPA: single-view loader (used with mask collator)
# ---------------------------------------------------------------------------

def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False)
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info('ImageNet dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler


# ---------------------------------------------------------------------------
# LeJEPA: multi-view train loader + single-view val loader (with labels)
# ---------------------------------------------------------------------------

def _apply_class_subset(dataset, class_subset_file):
    """Filter an ImageFolder dataset to the classes listed in *class_subset_file*.

    The file should contain one synset folder name per line (e.g. ``n02869837``).
    Targets are remapped to 0 … len(valid_classes)-1 in sorted order.
    """
    with open(class_subset_file) as f:
        valid = {line.strip() for line in f if line.strip()}
    sorted_valid = sorted(valid)
    new_idx = {dataset.class_to_idx[c]: i for i, c in enumerate(sorted_valid)
               if c in dataset.class_to_idx}
    dataset.samples = [(p, new_idx[t]) for p, t in dataset.samples if t in new_idx]
    dataset.targets = [t for _, t in dataset.samples]
    logger.info('Class subset applied: %d classes, %d samples', len(sorted_valid), len(dataset.samples))
    return dataset


def make_imagenet1k_lejepa(
    train_transform,
    val_transform,
    batch_size,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    copy_data=False,
    drop_last=True,
    val_batch_size=256,
    class_subset_file=None,
):
    """Returns (train_loader, val_loader, train_sampler) for LeJEPA.

    train_transform should be a MultiCropTransform or MultiViewTransform.
    val_transform should produce (1, C, H, W) per image for probe eval.
    Pass *class_subset_file* to restrict to a subset of classes (e.g. ImageNet-100).
    """
    train_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=train_transform,
        train=True,
        copy_data=copy_data,
        index_targets=False,
    )
    val_dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=val_transform,
        train=False,
        copy_data=False,
        index_targets=False,
    )
    if class_subset_file is not None:
        train_dataset = _apply_class_subset(train_dataset, class_subset_file)
        val_dataset   = _apply_class_subset(val_dataset,   class_subset_file)
    logger.info('ImageNet (LeJEPA) datasets created')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=rank,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    logger.info('ImageNet (LeJEPA) data loaders created')

    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_file='imagenet_full_size-061417.tar.gz',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = 'train/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)
        logger.info(f'data-path {data_path}')

        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path
