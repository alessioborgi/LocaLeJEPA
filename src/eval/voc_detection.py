# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VOC07+12 object detection with Faster R-CNN (detectron2).

Requirements
------------
  pip install 'git+https://github.com/facebookresearch/detectron2.git'

Dataset layout
--------------
  <voc_root>/
      VOC2007/
          Annotations/  ImageSets/  JPEGImages/
      VOC2012/
          Annotations/  ImageSets/  JPEGImages/

Training protocol
-----------------
  • Train on VOC2007 trainval + VOC2012 trainval
  • Evaluate on VOC2007 test (AP_all, AP50, AP75)
  • Faster R-CNN with C4 ResNet backbone initialised from the pretrained checkpoint
  • For ViT backbones, use ViTDet (requires additional setup beyond this script)

Key hyper-parameters (from LeJEPA implementation plan §3.4)
------------------------------------------------------------
  base_lr = 0.1   max_iter = 24 000   batch_size = 16
  LR steps at 18 000 and 22 000   linear warmup for 1 000 steps
"""

import logging
import os

import yaml

logger = logging.getLogger()


def _require_detectron2():
    try:
        import detectron2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            'detectron2 is required for VOC detection evaluation.\n'
            'Install with:\n'
            '  pip install git+https://github.com/facebookresearch/detectron2.git'
        ) from e


def register_voc_datasets(voc_root: str) -> None:
    """Register VOC2007 and VOC2012 splits in detectron2's DatasetCatalog."""
    _require_detectron2()
    from detectron2.data.datasets import register_pascal_voc

    splits = [
        ('voc_2007_trainval', 'VOC2007', 'trainval', 2007),
        ('voc_2007_test',     'VOC2007', 'test',     2007),
        ('voc_2012_trainval', 'VOC2012', 'trainval', 2012),
    ]
    for name, folder, split, year in splits:
        register_pascal_voc(name, os.path.join(voc_root, folder), split, year)
    logger.info('Registered VOC2007 and VOC2012 in detectron2 catalog.')


def _map_lejepa_resnet_to_d2(state_dict: dict) -> dict:
    """Remap LeJEPA ResNetEncoder weight keys to detectron2 ResNet keys.

    LeJEPA key prefix          → detectron2 key prefix
    backbone.conv1             → backbone.bottom_up.stem.conv1
    backbone.bn1               → backbone.bottom_up.stem.bn1
    backbone.layer{1-4}        → backbone.bottom_up.res{2-5}
    """
    prefix_map = {
        'backbone.conv1':   'backbone.bottom_up.stem.conv1',
        'backbone.bn1':     'backbone.bottom_up.stem.bn1',
        'backbone.layer1':  'backbone.bottom_up.res2',
        'backbone.layer2':  'backbone.bottom_up.res3',
        'backbone.layer3':  'backbone.bottom_up.res4',
        'backbone.layer4':  'backbone.bottom_up.res5',
    }
    remapped = {}
    for k, v in state_dict.items():
        k_new = k
        for src, dst in prefix_map.items():
            if k.startswith(src):
                k_new = k.replace(src, dst, 1)
                break
        remapped[k_new] = v
    return remapped


def build_voc_cfg(
    voc_root:        str,
    output_dir:      str,
    checkpoint_path: str,
    pretrain_config: str,
    base_lr:         float = 0.1,
    max_iter:        int   = 24_000,
    batch_size:      int   = 16,
    warmup_iters:    int   = 1_000,
):
    """Build a detectron2 CfgNode for Faster R-CNN on VOC07+12.

    Loads backbone weights from *checkpoint_path* into the C4 ResNet backbone.
    Only ResNet-50 is supported; for ViT, use ViTDet separately.
    """
    _require_detectron2()

    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    import torch

    with open(pretrain_config, encoding='utf-8') as f:
        train_cfg = yaml.safe_load(f)

    backbone_name = train_cfg['backbone']['name']
    if backbone_name not in ('resnet50', 'resnet18'):
        raise NotImplementedError(
            f'VOC Faster R-CNN supports ResNet backbones only (got {backbone_name!r}). '
            'For ViT, adapt ViTDet separately.'
        )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file('PascalVOC-Detection/faster_rcnn_R_50_C4.yaml')
    )

    cfg.DATASETS.TRAIN = ('voc_2007_trainval', 'voc_2012_trainval')
    cfg.DATASETS.TEST  = ('voc_2007_test',)

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.MODEL.WEIGHTS             = ''   # loaded manually below
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20

    cfg.SOLVER.BASE_LR            = base_lr
    cfg.SOLVER.MAX_ITER           = max_iter
    cfg.SOLVER.IMS_PER_BATCH      = batch_size
    cfg.SOLVER.STEPS              = (18_000, 22_000)
    cfg.SOLVER.CHECKPOINT_PERIOD  = 5_000
    cfg.SOLVER.WARMUP_ITERS       = warmup_iters
    cfg.SOLVER.WARMUP_METHOD      = 'linear'
    cfg.SOLVER.WARMUP_FACTOR      = 1.0 / warmup_iters

    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MIN_SIZE_TEST  = 800

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    return cfg


def init_model_weights(model, checkpoint_path: str) -> None:
    """Load LeJEPA ResNet weights into a detectron2 Faster R-CNN model in-place."""
    import torch
    ckpt  = torch.load(checkpoint_path, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in ckpt.get('net', {}).items()}
    state = _map_lejepa_resnet_to_d2(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info(
        'Loaded pretrained backbone: %d missing keys, %d unexpected keys.',
        len(missing), len(unexpected),
    )


def train_voc(cfg, checkpoint_path: str) -> None:
    """Run Faster R-CNN training + evaluation on VOC07+12."""
    _require_detectron2()

    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import PascalVOCDetectionEvaluator

    class _VOCTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name):
            return PascalVOCDetectionEvaluator(dataset_name)

    trainer = _VOCTrainer(cfg)
    # Load pretrained backbone before training starts
    init_model_weights(trainer.model, checkpoint_path)
    trainer.train()
