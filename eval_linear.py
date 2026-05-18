# Copyright (c) 2025 Alessio Borgi, Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Linear evaluation entry point for BundLeJEPA.

Usage
-----
    python eval_linear.py --config configs/eval_linear.yaml

The config specifies the pretrained checkpoint, the 11 downstream datasets,
and evaluation hyper-parameters.  Features are extracted once and optionally
cached to disk so different probe runs don't re-extract.

Output
------
A summary table of top-1 accuracy across all evaluated datasets is printed
at the end.  Per-dataset results are also written to <logging.folder>/eval.csv.
"""

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
import yaml

from src.eval.datasets    import ALL_DATASETS, get_dataset_splits
from src.eval.extract     import load_backbone, extract_features
from src.eval.linear_probe import run_probe

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(cfg: dict) -> None:
    device = torch.device(
        cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    logger.info('Loading backbone from %s', cfg['checkpoint'])
    extract_fn, feat_dim = load_backbone(
        pretrain_config_path=cfg['pretrain_config'],
        checkpoint_path=cfg['checkpoint'],
        device=device,
        crop_size=cfg.get('crop_size', 224),
    )
    logger.info('Feature dimension: %d', feat_dim)

    datasets_root          = cfg['datasets_root']
    batch_size             = cfg.get('batch_size', 256)
    num_workers            = cfg.get('num_workers', 8)
    cache_dir              = cfg.get('cache_dir')
    use_cache              = cfg.get('use_cache', False)
    dataset_names          = cfg.get('datasets', ALL_DATASETS)
    dataset_roots_override = cfg.get('dataset_roots', {})
    output_folder          = cfg.get('output_folder', 'logs/eval/')
    os.makedirs(output_folder, exist_ok=True)

    ckpt_tag = os.path.splitext(os.path.basename(cfg['checkpoint']))[0]

    def cache_path(name, split):
        if not use_cache or cache_dir is None:
            return None
        return os.path.join(cache_dir, ckpt_tag, f'{name}_{split}.pt')

    results = {}

    for name in dataset_names:
        root = dataset_roots_override.get(name, datasets_root)
        logger.info('=== %s ===', name)

        try:
            train_ds, test_ds = get_dataset_splits(
                name, root, crop_size=cfg.get('crop_size', 224)
            )
        except Exception as exc:
            logger.warning('Skipping %s: %s', name, exc)
            continue

        X_train, y_train = extract_features(
            extract_fn, train_ds,
            batch_size=batch_size, num_workers=num_workers, device=device,
            cache_path=cache_path(name, 'train'),
        )
        X_test, y_test = extract_features(
            extract_fn, test_ds,
            batch_size=batch_size, num_workers=num_workers, device=device,
            cache_path=cache_path(name, 'test'),
        )
        logger.info('%s: %d train / %d test samples, feat_dim=%d',
                    name, len(X_train), len(X_test), X_train.shape[1])

        acc, lam = run_probe(
            X_train, y_train, X_test, y_test,
            n_reg=cfg.get('n_reg', 45),
            max_iter=cfg.get('max_iter', 5000),
            val_frac=cfg.get('val_frac', 0.2),
        )
        results[name] = acc
        logger.info('%s  acc=%.4f  (best λ=%.2e)', name, acc, lam)

    # ── summary table ────────────────────────────────────────────────────────
    print('\n' + '=' * 52)
    print(f'{"Dataset":22s}  {"Top-1 Acc":>10s}')
    print('-' * 36)
    for name, acc in results.items():
        print(f'{name:22s}  {acc * 100:9.2f}%')
    if results:
        mean_acc = float(np.mean(list(results.values())))
        print('-' * 36)
        print(f'{"Mean (11 datasets)":22s}  {mean_acc * 100:9.2f}%')
    print('=' * 52)

    # ── write CSV ────────────────────────────────────────────────────────────
    csv_out = os.path.join(output_folder, f'eval_{ckpt_tag}.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'top1_acc'])
        for name, acc in results.items():
            w.writerow([name, f'{acc:.6f}'])
    logger.info('Results written to %s', csv_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BundLeJEPA linear evaluation')
    parser.add_argument('--config', type=str, default='configs/eval_linear.yaml',
                        help='path to eval config YAML')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
