# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Downstream evaluation utilities for LocaLeJEPA.

Currently exposes:
  - :mod:`src.eval.extract`               frozen-backbone loading + feature caching
  - :mod:`src.eval.io`                    crash-safe writes, self-healing cache loads
  - :mod:`src.eval.indomain`              in-domain splits, transform, feature extraction
  - :mod:`src.eval.knn_probe`             similarity-weighted k-NN
  - :mod:`src.eval.linear_probe`          frozen-feature linear probe
  - :mod:`src.eval.transfer`              the seven out-of-domain transfer datasets
  - :mod:`src.eval.rankme`                RankMe effective rank, projector loading
  - :mod:`src.eval.angle_probe`           circular (Yoda) angle recovery
  - :mod:`src.eval.patch_pca`             patch-token PCA visualisation
  - :mod:`src.eval.conformal_autoencoder` Conformal-Autoencoder dimension detection
"""
