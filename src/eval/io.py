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
    feature: str = "encoder",
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
    if feature != "encoder":
        # A pipeline variant is a discriminator (research_rules/05): cls2 features of the same
        # checkpoint must never be served from -- or overwrite -- its encoder-mode cache. The
        # default mode keeps its key unchanged, so no existing cache is orphaned.
        stem += f"_{feature}"
    stem += f"_{checkpoint_digest(checkpoint)}"
    return os.path.join(os.fspath(cache_dir), backbone, stem + ".pt")


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _git_commit() -> str:
    """The commit the generator ran at, suffixed ``-dirty`` when the tree had uncommitted changes.

    The suffix matters more than the hash: a table generated from a dirty tree cannot be tied to
    any commit, so recording the bare hash would overstate what is known about it. Returns
    ``"unknown"`` rather than raising -- a report generator must not die because git is absent,
    and "unknown" is itself accurate provenance.
    """
    import subprocess

    def run(*args):
        return subprocess.run(args, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5)

    try:
        head = run("git", "rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return "unknown"
        dirty = run("git", "status", "--porcelain").stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head.stdout.strip()}-dirty" if dirty else head.stdout.strip()


def provenance_block(script: str, input_dirs, archive_to: str | os.PathLike | None = None) -> str:
    """A markdown block recording where a generated table's numbers came from.

    **Why this exists.** Every ``results/*.md`` in this repo was generated from
    ``logs/.../per_backbone/*.json``, ``/logs`` is gitignored, and no JSON artifact has ever been
    committed on any branch. So none of those tables can be regenerated or audited from a clone,
    and their inputs were produced on a cluster the project has since migrated off. Ledger
    X.fix-P2-2; ``research_rules/05`` is about exactly this.

    The old tables are deliberately **not** being backfilled -- the user's call, and the right
    one: they are all going to be regenerated, so annotating them is work for a reader who does
    not exist. This is the forward-looking half. From the next campaign on, a table says which
    script made it, at which commit, from which files, and each input is copied next to the
    output so the table can actually be re-derived.

    ``archive_to`` is where the inputs are copied. Passing ``None`` records provenance without
    archiving, which is the right choice for a scratch regeneration.

    When no inputs are found this says so plainly rather than emitting an empty table. A report
    generated from nothing is a real state -- it is what a half-finished campaign produces -- and
    a blank provenance block would read as "no provenance recorded" instead of "no inputs".
    """
    import datetime
    import shutil
    import socket

    if isinstance(input_dirs, (str, os.PathLike)):
        input_dirs = [input_dirs]

    found: list[tuple[str, str, int]] = []
    for directory in input_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue
            source = os.path.join(directory, name)
            with open(source, "rb") as handle:
                digest = hashlib.blake2b(handle.read(), digest_size=4).hexdigest()
            found.append((f"{_eval_name(directory)}/{name}", digest, os.path.getsize(source)))
            if archive_to:
                target = os.path.join(os.fspath(archive_to), _eval_name(directory), name)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(source, target)

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "",
        "## Provenance",
        "",
        f"| | |",
        f"|---|---|",
        f"| generated by | `{script}` |",
        f"| at commit | `{_git_commit()}` |",
        f"| on | {stamp}, host `{socket.gethostname()}` |",
        f"| from | {', '.join(f'`{d}`' for d in input_dirs) or '—'} |",
    ]
    if archive_to:
        lines.append(f"| inputs archived to | `{archive_to}` |")
    lines.append("")

    if not found:
        lines += [
            "**No input files were found.** This table was generated from nothing — every cell it",
            "shows is a placeholder. It cannot be regenerated or audited, because there is nothing",
            "to regenerate it from.",
            "",
        ]
        return "\n".join(lines)

    lines += [f"{len(found)} input file(s), blake2b-32 of each:", "", "| file | digest | bytes |",
              "|---|---|---|"]
    lines += [f"| `{name}` | `{digest}` | {size} |" for name, digest, size in found]
    lines.append("")
    return "\n".join(lines)


def refuse_to_replace_with_placeholders(out_path, input_dirs, force: bool = False) -> None:
    """Stop a report generator from overwriting a real table with a table of em dashes.

    FOUND 2026-09-20 by running a generator on a fresh clone, which is the obvious first thing
    anyone does. `/logs` is gitignored, so the per-backbone inputs are absent; every `_load`
    returns None, every cell renders as an em dash, and the script cheerfully writes that over
    the committed table. Verified pre-existing -- it reproduces on the unmodified script.

    Silent destruction of the only copy of a result, triggered by a read-only-looking command.
    The table is in git so it is recoverable, but only if someone notices, and a placeholder
    table is not obviously wrong at a glance.

    Refusing only when inputs are **entirely** absent is deliberate. A partially-finished
    campaign legitimately regenerates a table with some rows still pending -- that is the normal
    working state and must keep working. Zero inputs is the pathological case.
    """
    if force:
        return
    for directory in ([input_dirs] if isinstance(input_dirs, (str, os.PathLike)) else input_dirs):
        if directory and os.path.isdir(directory) and any(
                n.endswith(".json") for n in os.listdir(directory)):
            return
    if not os.path.exists(out_path):
        return
    raise SystemExit(
        f"REFUSING to overwrite {out_path}: no input files were found under "
        f"{', '.join(str(d) for d in ([input_dirs] if isinstance(input_dirs, (str, os.PathLike)) else input_dirs))}. "
        f"Every cell would be a placeholder, replacing a table generated from real data.\n"
        f"If that is genuinely what you want, pass force=True."
    )


def _eval_name(per_backbone_dir) -> str:
    """`logs/eval_indomain_in1k_ep100/per_backbone` -> `eval_indomain_in1k_ep100`."""
    parts = [p for p in str(per_backbone_dir).replace("\\", "/").split("/") if p]
    if parts and parts[-1] == "per_backbone":
        parts = parts[:-1]
    return parts[-1] if parts else "inputs"
