# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VOC07+12 detection evaluation entry point.

Usage
-----
    python eval_voc.py --config configs/eval_voc.yaml

Requires detectron2:
    pip install 'git+https://github.com/facebookresearch/detectron2.git'

Only ResNet-50 backbones are supported for the C4 Faster R-CNN architecture.
For ViT backbones, use ViTDet (separate setup required).
"""

import argparse
import logging
import sys

import yaml

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BundLeJEPA VOC07+12 detection')
    parser.add_argument('--config', type=str, default='configs/eval_voc.yaml')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    from src.eval.voc_detection import register_voc_datasets, build_voc_cfg, train_voc

    register_voc_datasets(cfg['voc_root'])

    d2_cfg = build_voc_cfg(
        voc_root=cfg['voc_root'],
        output_dir=cfg['output_dir'],
        checkpoint_path=cfg['checkpoint'],
        pretrain_config=cfg['pretrain_config'],
        base_lr=cfg.get('base_lr',       0.1),
        max_iter=cfg.get('max_iter',     24_000),
        batch_size=cfg.get('batch_size', 16),
        warmup_iters=cfg.get('warmup_iters', 1_000),
    )

    train_voc(d2_cfg, checkpoint_path=cfg['checkpoint'])
