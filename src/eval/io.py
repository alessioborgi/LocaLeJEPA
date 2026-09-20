# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Crash-safe writes and cache loads for the evaluation entry points.

Every evaluation job in this project can be killed without warning -- they run on the cluster's
``preemptable`` partition, and a 24-hour wall applies everywhere else. A write interrupted
partway leaves a partial file, and the two consequences are different and both bad:

* **A partial result file** is lost work. The GPU hours that produced it are gone, and the merge
  step that reads ``per_backbone/*.json`` cannot parse what is there.
* **A partial feature cache is worse than lost** -- it is *sticky*. Reproduced 2026-09-19
  (ledger C0-3): ``torch.load`` on a truncated ``.pt`` raises ``PytorchStreamReader failed
  reading zip archive``, and because the writer is only reached *after* the cache-hit branch,
  the bad file is never replaced. Every retry hits the same file and dies the same way, so the
  backbone is wedged until a human deletes it by hand.

  Note this refutes half of what C0-3 predicted. The candidate said a truncated cache would be
  "happily reused", i.e. silent corruption. It is not silent -- it raises. The defect is a
  permanent wedge, not a wrong number. Recorded as observed rather than as claimed.

Both are closed the same way: write to ``<path>.tmp`` and ``os.replace`` onto the destination.
``os.replace`` is atomic within a filesystem, so a reader sees either the old file or the new
one, never a half-written one. ``load_tensor_cache`` then treats an unreadable cache as a miss,
which turns a wedge into a re-extraction -- expensive, but self-healing, and it also covers
caches written by the pre-fix code that still exist on the old cluster.

Why this lives in ``src/eval/`` rather than ``src/utils/``: ``src/utils/__init__.py`` eagerly
re-exports from five modules that pull in hydra, lightning, rich and omegaconf. ``src/eval/`` is
a light package, and every caller of these helpers is an evaluation path, so putting them here
keeps the eval library importable without the training stack.
"""

from __future__ import annotations

import hashlib
import logging
import os
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)

TMP_SUFFIX = ".tmp"


@contextmanager
def atomic_write(path: str | os.PathLike, mode: str = "w", **open_kwargs: Any):
    """Yield a handle whose contents appear at ``path`` only if the block completes.

    Usage mirrors ``open`` exactly, so converting a site is a one-word change::

        with atomic_write(path, "w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2)

    On any exception the temporary file is removed and the original is left untouched. Parent
    directories are created, because several callers write into a per-backbone subdirectory that
    may not exist yet.

    The ``.tmp`` suffix is load-bearing beyond convention: ``_collect_results`` filters the
    per-backbone directory on ``.endswith(".json")``, so a leftover ``foo.json.tmp`` from a
    killed job is skipped rather than parsed.
    """
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp = path + TMP_SUFFIX
    try:
        with open(tmp, mode, **open_kwargs) as handle:
            yield handle
        os.replace(tmp, path)
    except BaseException:
        # BaseException, not Exception: a SIGINT between open and replace must not leave the
        # temporary behind either.
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def atomic_torch_save(obj: Any, path: str | os.PathLike) -> None:
    """``torch.save`` that cannot leave a half-written archive at ``path``."""
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    tmp = path + TMP_SUFFIX
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_tensor_cache(path: str | os.PathLike | None) -> dict | None:
    """Load a feature cache, or return ``None`` if it is absent or unusable.

    Returning ``None`` rather than raising is what makes a damaged cache self-healing: the caller
    treats it as a miss and re-extracts, overwriting the bad file. Without this, a cache written
    by the pre-atomic code -- which exists on the old cluster -- wedges the backbone permanently.

    The ``except Exception`` is deliberate and deliberately broad. ``torch.load`` can raise
    ``RuntimeError`` (truncated zip), ``EOFError`` (empty file), ``UnpicklingError``, or an
    ``OSError`` from the filesystem, and the response to all of them is identical: recompute. A
    cache is an optimisation, never a source of truth, so there is nothing here worth propagating
    an exception for.
    """
    if path is None or not os.path.isfile(path):
        return None

    try:
        blob = torch.load(path, map_location="cpu")
    except Exception as error:  # noqa: BLE001 -- see the docstring
        logger.warning("unreadable feature cache %s (%s: %s); re-extracting",
                       path, type(error).__name__, error)
        return None

    if not isinstance(blob, dict) or "X" not in blob or "y" not in blob:
        logger.warning("feature cache %s has unexpected contents; re-extracting", path)
        return None
    return blob


def checkpoint_digest(checkpoint_path: str | os.PathLike, length: int = 8) -> str:
    """A short identity tag for a checkpoint file.

    Derived from the resolved path, size and mtime -- **not** the file contents. Hashing a
    multi-gigabyte ``.ckpt`` on every evaluation would cost more than the extraction the cache is
    meant to save, and the point here is to separate two *different* checkpoints, not to detect
    deliberate tampering.

    Consequence worth keeping: a checkpoint rebuilt with identical bytes gets a new mtime and so
    a new digest, which is a cache miss. That is the safe direction to fail -- a wasted
    re-extraction rather than features silently borrowed from another run -- and it is why this
    should not be "optimised" down to size alone.
    """
    resolved = os.path.realpath(os.fspath(checkpoint_path))
    stat = os.stat(resolved)
    key = f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.blake2b(key, digest_size=max(1, length // 2)).hexdigest()[:length]


def feature_cache_path(
    cache_dir: str | os.PathLike | None,
    backbone: str,
    dataset: str,
    split: str,
    crop_size: int,
    epoch: Any,
    checkpoint: str | os.PathLike,
) -> str | None:
    """The one place a feature-cache filename is built.

    ``<cache_dir>/<backbone>/<dataset>_<split>_<crop>_ep<epoch>_<digest>.pt``, or ``None`` when
    no ``cache_dir`` is configured (every caller already treats None as "do not cache").

    **Why this exists at all.** The key used to be assembled by hand in five places, and three of
    them *read what the other two wrote* -- ``eval_rankme`` reconstructs the filenames that
    ``eval_indomain`` and ``eval_transfer`` produced. A divergence between a writer and a reader
    does not raise: ``eval_rankme`` logs "no cached features -- skipping this target" and carries
    on, so the failure is a silently dropped target plus an expensive GPU re-extraction. One
    builder means a writer and a reader cannot drift apart.

    **Why ``checkpoint`` is required.** It is the fix for ledger C0-4, reproduced 2026-09-20: the
    key carried the backbone's config *name* and its stored *epoch*, so two genuinely different
    checkpoints sharing both collided, and the second run silently reported features computed by
    the first -- no exception, no warning. Making the argument mandatory means a caller that
    forgets it gets a ``TypeError`` rather than quietly reverting to the unprotected key, which
    is the whole failure mode being closed.

    ``epoch`` is strictly redundant once the digest is present -- a different epoch is a different
    file and so a different digest -- but it stays in the name because operators grep for it when
    working out which run a cache belongs to. ``None`` is accepted and omits that component, for
    callers that never load the epoch.
    """
    if not cache_dir:
        return None

    stem = f"{dataset}_{split}_{crop_size}"
    if epoch is not None:
        stem += f"_ep{epoch}"
    stem += f"_{checkpoint_digest(checkpoint)}"
    return os.path.join(os.fspath(cache_dir), backbone, stem + ".pt")
