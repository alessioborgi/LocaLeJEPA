#!/usr/bin/env python
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Every decision the AICR launcher makes, as tested Python. The shell scripts stay thin.

Why a helper and not a smarter shell script: the FASRC launcher (``scripts/slurm_submit.sh``)
resumed on the *existence* of ``last.ckpt``, queued its successor before training started, and
let that successor revert to the header's one-GPU request (F32, F33). None of that was testable
and all of it was wrong in a way that looked like success. Here each of those decisions is a
function with a unit test, and the shell only calls them.

Subcommands
-----------
expand      manifest.yaml --out grid.json     cells x seeds -> one run per line, deterministic names
show        grid.json [--index I]             the run a job array task will execute
args        grid.json --index I               shell exports: RUN_NAME, MAX_EPOCHS, OVERRIDES
preflight   grid.json --index I --lock L --python P   tree clean and at the grid's commit, env == lock
geometry    grid.json --index I --world-size W   prints the geometry FIRST, asserts the micro-batch
stage-check --free B --tar-bytes B              /tmp has room for 1.2x the tar, or says why not
resume      --ckpt-dir D --max-epochs E       fresh | resume:<path> | done, decided by global_step
postflight  grid.json --index I --run-dir R --ckpt-dir D [--lock L]   verdict + ``.verified``
successor   --requeue-left N --script S ...   the sbatch argv that re-inherits THIS job's resources
realised-k  metrics.csv                       the realised-K column of the ablation tables
submit      grid.json --partition P --time T  the sbatch command line to paste
eval-preflight  --config C --only NAME --root R   the checkpoint exists and carries .verified
eval-postflight --config C --only NAME --root R   the per-backbone result JSON exists and parses
calibrate-verdict --run-dir R --interrupt-step S --max-steps M   the rehearsed interruption resumed
env-hash    [freeze.txt]                      sha256 of a normalised ``pip freeze``

Exit codes: 0 ok / decided, 1 a check failed (the reason is printed), 3 resumable (postflight only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"

# Training-split sizes, for steps-per-epoch in the geometry printout only.
DATASET_SIZES = {"imagenette": 9469, "imagenet1k": 1281167, "imagenet100": 126689}

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# The keys a postflight compares between the manifest and what the run actually used. `seed`
# is appended by the expansion; the rest are whatever overrides the cell carries.
CHECKPOINT_HPARAMS = ("lamb", "lr_eta_min_ratio", "warmup_epochs")


# ======================================================================================
# manifest -> runs
# ======================================================================================

@dataclass(frozen=True)
class RunSpec:
    """One training run: everything the launcher needs, nothing it must guess."""

    run_name: str
    experiment: str
    overrides: tuple[str, ...]        # "key=value", cell overrides only; seed is separate
    seed: int
    gpus: int
    micro_batch: int                  # the declared per-GPU batch, asserted by `geometry`
    max_epochs: int
    tag: str

    def hydra_overrides(self) -> list[str]:
        return [f"experiment={self.experiment}", *self.overrides, f"seed={self.seed}"]

    def override_map(self) -> dict[str, str]:
        return dict(item.split("=", 1) for item in self.overrides) | {"seed": str(self.seed)}


def format_value(value: Any) -> str:
    """Values as they appear in run names and on the Hydra command line."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        text = repr(value)
        return text[:-2] if text.endswith(".0") else text
    return str(value)


def _sanitise(name: str) -> str:
    name = name.replace("/", "-")
    if not RUN_NAME_RE.match(name):
        raise ValueError(f"run name {name!r} is not filesystem- and Slurm-safe")
    return name


def expand(manifest: Mapping[str, Any]) -> list[RunSpec]:
    """Cartesian product of every block's axes, times the seeds, in manifest order.

    An axis value is a scalar (one override) or a dict (several overrides at once, e.g. the
    batch axis pairs ``data.batch_size`` with ``trainer.accumulate_grad_batches``). Two keys are
    reserved inside a dict value: ``micro_batch`` (declares the per-GPU batch of that cell) and
    ``experiment`` (switches the base config). Run names are deterministic: the block's tag, the
    aliased override values in *canonical* (sorted-key) order, the seed -- so the name is a
    function of the run's overrides and not of how the manifest happened to be written (a YAML
    re-serialised with sorted keys must not rename every run). They double as the checkpoint
    directory, which is what closes F35 (three naming conventions, silent row drop).
    """
    seeds = [int(s) for s in manifest.get("seeds", [0])]
    short = dict(manifest.get("short", {}))
    defaults = {
        "gpus": int(manifest.get("gpus", 1)),
        "micro_batch": int(manifest["micro_batch"]) if "micro_batch" in manifest else None,
        "max_epochs": int(manifest["max_epochs"]),
        "experiment": manifest.get("experiment"),
    }
    runs: list[RunSpec] = []
    seen: set[str] = set()
    for block in manifest["blocks"]:
        tag = str(block["tag"])
        axes: dict[str, list[Any]] = dict(block.get("axes", {}))
        fixed: dict[str, Any] = dict(block.get("overrides", {}))
        names = list(axes)
        for combo in itertools.product(*(axes[name] for name in names)) if names else [()]:
            overrides: dict[str, Any] = dict(fixed)
            parts: list[tuple[str, str]] = []            # (key, aliased text), sorted below
            micro_batch = block.get("micro_batch", defaults["micro_batch"])
            experiment = block.get("experiment", defaults["experiment"])
            for axis, value in zip(names, combo):
                if isinstance(value, Mapping):
                    for key, inner in value.items():
                        if key == "micro_batch":
                            micro_batch = int(inner)
                        elif key == "experiment":
                            experiment = str(inner)
                        else:
                            overrides[key] = inner
                            parts.append((key, f"{short.get(key, key)}{format_value(inner)}"))
                else:
                    overrides[axis] = value
                    parts.append((axis, f"{short.get(axis, axis)}{format_value(value)}"))
            if experiment is None:
                raise ValueError(f"block {tag!r}: no experiment, neither in the block nor the manifest")
            if micro_batch is None:
                raise ValueError(f"block {tag!r}: no micro_batch declared")
            base = "_".join([tag] + [text for _, text in sorted(parts)])
            for seed in seeds:
                run_name = _sanitise(f"{base}_s{seed}")
                if run_name in seen:
                    raise ValueError(f"duplicate run name {run_name!r}: two cells are indistinguishable")
                seen.add(run_name)
                runs.append(RunSpec(
                    run_name=run_name,
                    experiment=str(experiment),
                    overrides=tuple(f"{k}={format_value(v)}" for k, v in sorted(overrides.items())),
                    seed=seed,
                    gpus=int(block.get("gpus", defaults["gpus"])),
                    micro_batch=int(micro_batch),
                    max_epochs=int(block.get("max_epochs", defaults["max_epochs"])),
                    tag=tag,
                ))
    return runs


def load_manifest(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or "blocks" not in manifest:
        raise ValueError(f"{path}: not a campaign manifest (no `blocks`)")
    return manifest


def write_grid(runs: Iterable[RunSpec], out: Path, *, manifest_name: str, commit: str | None) -> None:
    payload = {
        "manifest": manifest_name,
        "commit": commit,
        "runs": [asdict(run) for run in runs],
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out)


def read_grid(path: Path) -> tuple[dict[str, Any], list[RunSpec]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = [RunSpec(**{**entry, "overrides": tuple(entry["overrides"])}) for entry in payload["runs"]]
    return payload, runs


# ======================================================================================
# geometry -- printed before any number that depends on it (05-DECISIONS)
# ======================================================================================

def compose_config(hydra_overrides: list[str]):
    """The experiment as Hydra would compose it, without running anything."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            return compose(config_name="train.yaml", overrides=hydra_overrides)
    finally:
        GlobalHydra.instance().clear()


def _select(cfg, key: str, default=None):
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, key, default=default, throw_on_resolution_failure=False)
    return default if value is None else value


def geometry(spec: RunSpec, world_size: int, cfg=None) -> dict[str, Any]:
    """What the run computes on, derived from the composed config -- and the assertion.

    Every loss is computed on its GPU's shard and both datamodules divide ``data.batch_size`` by
    ``world_size``, so the regularizer's batch is ``batch_size / world_size``: 4 GPUs at
    ``batch_size=256`` is the same experiment as 1 GPU at 64x4, and ``batch_size=64`` on 4 GPUs
    would silently be a 16-sample one. The declared ``micro_batch`` is the contract.
    """
    if cfg is None:
        cfg = compose_config(spec.hydra_overrides())
    batch_size = int(_select(cfg, "data.batch_size"))
    accumulate = int(_select(cfg, "trainer.accumulate_grad_batches", 1))
    if batch_size % world_size != 0:
        raise ValueError(f"data.batch_size={batch_size} is not divisible by world_size={world_size}")
    per_gpu = batch_size // world_size
    if per_gpu != spec.micro_batch:
        raise ValueError(
            f"per-GPU micro-batch is {per_gpu} (batch_size {batch_size} / world_size {world_size}) "
            f"but the manifest declares {spec.micro_batch}: this would be a different experiment"
        )
    transform = _select(cfg, "data.train_transform", {})
    if "num_views" in transform:
        views = int(transform["num_views"])
    else:
        views = int(transform.get("num_global_crops", 0)) + int(transform.get("num_local_crops", 0))
    target = str(_select(cfg, "data._target_", ""))
    dataset = next((name for name in DATASET_SIZES if name in target.lower().replace("_", "")), None)
    if dataset is None and "imagenet1k" in target.lower():
        dataset = "imagenet1k"
    steps_per_epoch = math.ceil(DATASET_SIZES[dataset] / batch_size) if dataset else None
    return {
        "run_name": spec.run_name,
        "experiment": spec.experiment,
        "world_size": world_size,
        "batch_size_global": batch_size,
        "micro_batch_per_gpu": per_gpu,
        "accumulate": accumulate,
        "effective_batch": batch_size * accumulate,
        "views": views,
        "crop_size": int(_select(cfg, "data.crop_size", 0)),
        "precision": str(_select(cfg, "trainer.precision", "32")),
        "num_workers": int(_select(cfg, "data.num_workers", 0)),
        "max_epochs": int(_select(cfg, "trainer.max_epochs", spec.max_epochs)),
        "dataset": dataset,
        "steps_per_epoch": steps_per_epoch,
        "seed": spec.seed,
    }


# ======================================================================================
# resume -- by the step inside the checkpoint, never by existence or mtime
# ======================================================================================

@dataclass(frozen=True)
class Candidate:
    path: Path
    epoch: int | None
    global_step: int | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None


def read_checkpoint_progress(path: Path) -> Candidate:
    """epoch and global_step from a Lightning checkpoint; an unreadable one says why."""
    import torch

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return Candidate(path, int(blob["epoch"]), int(blob["global_step"]))
    except Exception as error:  # noqa: BLE001 - a truncated file raises anything at all
        return Candidate(path, None, None, f"{type(error).__name__}: {error}")


@dataclass(frozen=True)
class Decision:
    kind: str                         # "fresh" | "resume" | "done"
    path: Path | None
    reason: str
    candidates: tuple[Candidate, ...] = field(default_factory=tuple)


def resume_decision(ckpt_dir: Path, max_epochs: int,
                    names: tuple[str, ...] = ("last.ckpt", "on_exception.ckpt")) -> Decision:
    """The three-way decision, from what is *inside* the candidates.

    ``done`` when any readable candidate has finished the last epoch (Lightning's epoch key is
    the 0-based index of the completed epoch, so ``max_epochs - 1``); otherwise ``resume`` from
    the readable candidate with the largest ``global_step`` -- an on-exception checkpoint from
    mid-epoch beats an older ``last.ckpt`` even though its mtime may be older on NFS; ``fresh``
    when nothing readable exists, saying so if something unreadable does.
    """
    candidates = tuple(read_checkpoint_progress(ckpt_dir / name)
                       for name in names if (ckpt_dir / name).is_file())
    readable = [c for c in candidates if c.valid]
    finished = [c for c in readable if c.epoch is not None and c.epoch >= max_epochs - 1]
    if finished:
        best = max(finished, key=lambda c: c.global_step or 0)
        return Decision("done", best.path, f"{best.path.name} is at epoch {best.epoch} "
                        f"(max_epochs {max_epochs})", candidates)
    if readable:
        best = max(readable, key=lambda c: c.global_step or 0)
        return Decision("resume", best.path, f"{best.path.name} at epoch {best.epoch}, "
                        f"global_step {best.global_step}", candidates)
    if candidates:
        broken = "; ".join(f"{c.path.name}: {c.error}" for c in candidates)
        return Decision("fresh", None, f"no readable checkpoint ({broken}); starting over", candidates)
    return Decision("fresh", None, "no checkpoint", candidates)


def verify_artifact(ckpt_path: Path, max_epochs: int) -> tuple[bool, str]:
    candidate = read_checkpoint_progress(ckpt_path)
    if not candidate.valid:
        return False, f"{ckpt_path.name} unreadable: {candidate.error}"
    if candidate.epoch is None or candidate.epoch < max_epochs - 1:
        return False, f"{ckpt_path.name} is at epoch {candidate.epoch}, not {max_epochs - 1}"
    return True, f"{ckpt_path.name} at epoch {candidate.epoch}, global_step {candidate.global_step}"


# ======================================================================================
# postflight -- the artifact, the config it was trained with, the tree, the environment
# ======================================================================================

def _parse_override_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _values_agree(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=0.0)
        except (TypeError, ValueError):
            return False
    return str(expected) == str(actual)


def config_matches_spec(run_dir: Path, spec: RunSpec) -> list[str]:
    """Mismatches between the manifest and what Hydra recorded in ``.hydra/`` -- empty is good.

    ``.hydra/config.yaml`` is the composed config the run actually used and
    ``.hydra/overrides.yaml`` the command line as given. Both are checked: the first catches a
    stale config file, the second a run launched under a different experiment.
    """
    from omegaconf import OmegaConf

    problems: list[str] = []
    config_path = run_dir / ".hydra" / "config.yaml"
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    if not config_path.is_file():
        return [f"{config_path} missing"]
    cfg = OmegaConf.load(config_path)
    given = set(OmegaConf.load(overrides_path)) if overrides_path.is_file() else None
    if given is None:
        problems.append(f"{overrides_path} missing")
    elif f"experiment={spec.experiment}" not in given:
        problems.append(f"experiment={spec.experiment} not among the run's overrides {sorted(given)}")
    for key, text in spec.override_map().items():
        actual = _select(cfg, key, default="<absent>")
        if isinstance(actual, Mapping):
            # A config-group selection (logger=csv, trainer=ddp): the composed value is the
            # whole group, so it is checked against the command line that was given instead.
            if given is not None and f"{key}={text}" not in given:
                problems.append(f"{key}={text} (a config group) not among the run's overrides")
            continue
        if not _values_agree(_parse_override_value(text), actual):
            problems.append(f"{key}: manifest {text!r}, run used {actual!r}")
    return problems


def checkpoint_matches_config(ckpt_path: Path, run_dir: Path) -> list[str]:
    """The checkpoint's own hyper-parameters against the run's config: two artifacts, one story."""
    import torch
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = blob.get("hyper_parameters", {}) or {}
    problems = []
    for name in CHECKPOINT_HPARAMS:
        if name in hparams:
            expected = _select(cfg, f"model.{name}", default="<absent>")
            if not _values_agree(expected, hparams[name]):
                problems.append(f"checkpoint {name}={hparams[name]!r} but config says {expected!r}")
    return problems


def find_metrics_csv(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "csv").glob("version_*/metrics.csv")) if (run_dir / "csv").is_dir() else []
    return candidates[-1] if candidates else None


def metrics_complete(metrics_csv: Path, max_epochs: int, *, local_run: bool) -> list[str]:
    """Every epoch logged, no NaN in the loss, and the realised-K column present on local runs."""
    problems: list[str] = []
    with open(metrics_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        loss_col = "train/loss_epoch"
        if loss_col not in columns:
            return [f"{metrics_csv}: no {loss_col} column"]
        epochs: set[int] = set()
        for row in reader:
            value = row.get(loss_col, "")
            if value == "":
                continue
            if value.lower() in ("nan", "inf", "-inf"):
                problems.append(f"{loss_col} is {value} at epoch {row.get('epoch')}")
            try:
                epochs.add(int(float(row["epoch"])))
            except (KeyError, ValueError):
                problems.append("a loss row has no epoch")
    if len(epochs) != max_epochs:
        problems.append(f"{len(epochs)} epochs logged, expected {max_epochs}")
    if local_run and "train/local/effective_clusters_epoch" not in columns:
        problems.append("local run without train/local/effective_clusters_epoch")
    return problems


def env_lines(text: str) -> set[str]:
    """The pip pins in a lock or a freeze, normalised: ``name==version`` lines only, lower-case,
    ``_`` as ``-``. The cluster's lock holds TWO halves (F24/F38) -- ``pip list --format=freeze``
    and ``conda list --explicit`` -- so a hash of the whole file can never equal a hash of a
    freeze; F44 found the preflight refusing every job for exactly that. Conda's lines are URLs
    and ``@EXPLICIT``/``# …`` headers, none of which contain ``==``."""
    out = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line or line.startswith(("http://", "https://", "-e ")):
            continue
        out.add(line.lower().replace("_", "-"))
    return out


def env_hash(freeze_text: str) -> str:
    """sha256 of the normalised pip pins (see ``env_lines``)."""
    return hashlib.sha256("\n".join(sorted(env_lines(freeze_text))).encode("utf-8")).hexdigest()


def current_env_text(python: str) -> str:
    """``pip list --format=freeze`` -- the same tool the lock was captured with. Plain ``pip freeze``
    writes ``name @ url`` for URL-installed wheels where ``pip list`` writes ``name==version``."""
    out = subprocess.run([python, "-m", "pip", "list", "--format=freeze"], capture_output=True, text=True, check=True)
    return out.stdout


def current_env_hash(python: str) -> str:
    return env_hash(current_env_text(python))


def env_diff(lock_text: str, live_text: str, limit: int = 5) -> str:
    """What differs, for the log: pins only in the lock, pins only in the environment."""
    lock, live = env_lines(lock_text), env_lines(live_text)
    only_lock, only_live = sorted(lock - live), sorted(live - lock)
    parts = []
    if only_lock:
        parts.append("only in lock: " + ", ".join(only_lock[:limit]) + (" …" if len(only_lock) > limit else ""))
    if only_live:
        parts.append("only in env: " + ", ".join(only_live[:limit]) + (" …" if len(only_live) > limit else ""))
    return "; ".join(parts) or "no pin differs"


def tree_state(repo: Path) -> tuple[str | None, bool]:
    """(commit, dirty). ``dirty`` counts modified tracked files only -- the git-ignored
    ``configs/local/default.yaml`` is expected to exist."""
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False
    return commit, bool(status)


@dataclass
class Verdict:
    checks: dict[str, tuple[bool, str]]

    @property
    def ok(self) -> bool:
        return all(passed for passed, _ in self.checks.values())

    def lines(self) -> list[str]:
        return [f"  [{'ok' if passed else 'FAIL'}] {name}: {detail}" for name, (passed, detail) in self.checks.items()]


def postflight(spec: RunSpec, run_dir: Path, ckpt_dir: Path, *, grid_commit: str | None,
               lock_hash: str | None, python: str | None, repo: Path = REPO, lock_text: str | None = None) -> Verdict:
    """Everything G5 asks of a finished pretrain, as one verdict. Writes nothing itself."""
    checks: dict[str, tuple[bool, str]] = {}
    ckpt = ckpt_dir / "last.ckpt"
    ok, detail = verify_artifact(ckpt, spec.max_epochs) if ckpt.is_file() else (False, f"{ckpt} missing")
    checks["artifact"] = (ok, detail)

    problems = config_matches_spec(run_dir, spec)
    checks["config"] = (not problems, "matches the manifest" if not problems else "; ".join(problems))

    if ok and not problems:
        problems = checkpoint_matches_config(ckpt, run_dir)
        checks["checkpoint_hparams"] = (not problems, "agree with the config" if not problems else "; ".join(problems))

    metrics = find_metrics_csv(run_dir)
    local_run = any(item.startswith("model.reg_loss.num_clusters=") for item in spec.overrides) or \
        "local" in spec.experiment
    if metrics is None:
        checks["metrics"] = (False, "no csv/version_*/metrics.csv under the run dir")
    else:
        problems = metrics_complete(metrics, spec.max_epochs, local_run=local_run)
        checks["metrics"] = (not problems, f"{metrics.name} complete" if not problems else "; ".join(problems))

    commit, dirty = tree_state(repo)
    if commit is None:
        checks["tree"] = (False, "not a git checkout")
    elif dirty:
        checks["tree"] = (False, f"{commit[:9]} with uncommitted changes")
    elif grid_commit and not commit.startswith(grid_commit) and not grid_commit.startswith(commit):
        checks["tree"] = (False, f"tree is {commit[:9]}, the grid was expanded at {grid_commit[:9]}")
    else:
        checks["tree"] = (True, commit[:9])

    if lock_hash and python:
        live = current_env_text(python)
        actual = env_hash(live)
        checks["environment"] = (actual == lock_hash, "matches the lock" if actual == lock_hash
                                 else f"pins differ from the lock: {env_diff(lock_text or '', live)}")
    return Verdict(checks)


def preflight(spec: RunSpec, *, grid_commit: str | None, lock_hash: str | None, python: str | None,
              repo: Path = REPO, lock_text: str | None = None) -> Verdict:
    """What a job checks before it stages a byte: the tree and the environment it will train
    with are the ones the grid was expanded from. Refusing here costs seconds; discovering it in
    postflight costs the run."""
    checks: dict[str, tuple[bool, str]] = {}
    commit, dirty = tree_state(repo)
    if commit is None:
        checks["tree"] = (False, "not a git checkout")
    elif dirty:
        checks["tree"] = (False, f"{commit[:9]} with uncommitted changes -- campaign runs come from a committed tree")
    elif grid_commit and grid_commit.endswith("-dirty"):
        checks["tree"] = (False, f"the grid was expanded from a dirty tree ({grid_commit}); re-expand after committing")
    elif grid_commit and not commit.startswith(grid_commit) and not grid_commit.startswith(commit):
        checks["tree"] = (False, f"tree is {commit[:9]}, the grid was expanded at {grid_commit[:9]}")
    else:
        checks["tree"] = (True, commit[:9])
    if lock_hash and python:
        live = current_env_text(python)
        actual = env_hash(live)
        checks["environment"] = (actual == lock_hash, "matches the lock" if actual == lock_hash
                                 else f"pins differ from the lock (F38: env drift): {env_diff(lock_text or '', live)}")
    checks["run"] = (True, f"{spec.run_name}: {spec.experiment} + {len(spec.overrides)} overrides, seed {spec.seed}")
    return Verdict(checks)


def write_marker(ckpt_dir: Path, verdict: Verdict, spec: RunSpec, extra: Mapping[str, Any] | None = None) -> Path:
    """``.verified`` on success, ``.failed`` otherwise -- the eval preflight requires the former."""
    name = ".verified" if verdict.ok else ".failed"
    payload = {"run_name": spec.run_name, "ok": verdict.ok,
               "checks": {k: {"ok": p, "detail": d} for k, (p, d) in verdict.checks.items()},
               **dict(extra or {})}
    path = ckpt_dir / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    other = ckpt_dir / (".failed" if verdict.ok else ".verified")
    if other.exists():
        other.unlink()
    return path


def checkpoint_digest(path: Path) -> str:
    """blake2b of realpath:size:mtime_ns -- the same identity ``src/eval/io.py`` keys feature
    caches on, computed here without importing ``src`` (this helper runs before PYTHONPATH is set)."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return hashlib.blake2b(f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}".encode(), digest_size=8).hexdigest()


# ======================================================================================
# evaluations -- never on a checkpoint the launcher did not verify
# ======================================================================================

def eval_backbone_entry(config_path: Path, only: str) -> dict[str, Any]:
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for entry in cfg.get("backbones", []):
        if entry.get("name") == only:
            return entry
    names = [e.get("name") for e in cfg.get("backbones", [])]
    raise ValueError(f"{config_path} has no backbone named {only!r}; it has {names}")


def eval_preflight(config_path: Path, only: str, root: Path) -> Verdict:
    """The checkpoint the eval config names must exist, be readable, and sit beside a
    ``.verified`` marker written by the launcher's postflight (G5 -> V4). ``best.ckpt`` is
    accepted only when the same directory's ``last.ckpt`` was verified: the marker is per run."""
    checks: dict[str, tuple[bool, str]] = {}
    entry = eval_backbone_entry(config_path, only)
    ckpt = Path(entry["checkpoint"])
    if not ckpt.is_absolute():
        ckpt = root / ckpt
    checks["checkpoint"] = (ckpt.is_file(), str(ckpt))
    if ckpt.is_file():
        candidate = read_checkpoint_progress(ckpt)
        checks["readable"] = (candidate.valid, f"epoch {candidate.epoch}, global_step {candidate.global_step}"
                              if candidate.valid else str(candidate.error))
        marker = ckpt.parent / ".verified"
        if marker.is_file():
            try:
                payload = json.loads(marker.read_text(encoding="utf-8"))
                checks["verified"] = (bool(payload.get("ok")), f"{marker} ok={payload.get('ok')} epoch={payload.get('epoch')}")
            except (OSError, ValueError) as error:
                checks["verified"] = (False, f"{marker} unreadable: {error}")
        else:
            checks["verified"] = (False, f"{marker} missing -- the pretrain's postflight did not pass")
    return Verdict(checks)


def eval_postflight(config_path: Path, only: str, root: Path) -> Verdict:
    """The evaluation's own artifact: the per-backbone JSON exists, parses, and names the backbone."""
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    out_dir = Path(cfg.get("output_folder", "."))
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    result = out_dir / "per_backbone" / f"{only}.json"
    checks: dict[str, tuple[bool, str]] = {"result": (result.is_file(), str(result))}
    if result.is_file():
        try:
            payload = json.loads(result.read_text(encoding="utf-8"))
            checks["parses"] = (isinstance(payload, dict) and bool(payload), f"{len(payload)} top-level keys")
        except ValueError as error:
            checks["parses"] = (False, str(error))
    return Verdict(checks)


# ======================================================================================
# calibration -- the rehearsed interruption actually resumed
# ======================================================================================

def _step_rows(metrics_csv: Path) -> list[int]:
    steps: list[int] = []
    with open(metrics_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("train/loss_step", "") != "":
                steps.append(int(float(row["step"])))
    return steps


def calibrate_verdict(run_dir: Path, *, interrupt_step: int, max_steps: int) -> Verdict:
    """Two launches wrote two CSV versions under one run dir. The first must stop at the
    interruption, the second must continue from the very next step and reach ``max_steps``;
    the learning rate, if logged, must not jump at the seam (a restart from scratch would)."""
    checks: dict[str, tuple[bool, str]] = {}
    versions = sorted((run_dir / "csv").glob("version_*/metrics.csv")) if (run_dir / "csv").is_dir() else []
    checks["launches"] = (len(versions) == 2, f"{len(versions)} metrics.csv under {run_dir}/csv (expected 2)")
    if len(versions) != 2:
        return Verdict(checks)
    first, second = (_step_rows(v) for v in versions)
    if not first or not second:
        checks["steps"] = (False, f"empty step series: first {len(first)} rows, second {len(second)} rows")
        return Verdict(checks)
    checks["interrupted"] = (abs(first[-1] - interrupt_step) <= 1,
                             f"first launch stopped at step {first[-1]} (interrupt at {interrupt_step})")
    spacing = min((b - a for a, b in zip(first, first[1:])), default=1)
    hint = (f"; rows are {spacing} steps apart -- the seam check needs trainer.log_every_n_steps=1 (F46)"
            if spacing > 1 else "")
    checks["resumed"] = (second[0] == first[-1] + 1, f"second launch began at step {second[0]} after {first[-1]}{hint}")
    checks["completed"] = (second[-1] >= max_steps - 1, f"second launch ended at step {second[-1]} (max_steps {max_steps})")
    lr_col = None
    with open(versions[0], newline="", encoding="utf-8") as handle:
        lr_col = next((c for c in (csv.DictReader(handle).fieldnames or []) if c.startswith("lr-")), None)
    if lr_col:
        def series(path):
            with open(path, newline="", encoding="utf-8") as handle:
                return [(int(float(r["step"])), float(r[lr_col])) for r in csv.DictReader(handle) if r.get(lr_col, "") != ""]
        a, b = series(versions[0]), series(versions[1])
        if len(a) >= 2 and b:
            slope = abs(a[-1][1] - a[-2][1])
            jump = abs(b[0][1] - a[-1][1])
            checks["lr_continuous"] = (jump <= max(3 * slope, 1e-12), f"lr jump at the seam {jump:.3e} vs local slope {slope:.3e}")
    return Verdict(checks)


# ======================================================================================
# successor -- inherits what THIS job actually got, and counts down
# ======================================================================================

def successor_argv(env: Mapping[str, str], *, script: str, requeue_left: int, time: str,
                   exports: Mapping[str, str]) -> list[str] | None:
    """The ``sbatch`` argv for a resumable exit, or ``None`` when the budget is spent.

    Resources are read from the running job's ``SLURM_*`` variables, never from a header, so
    the successor cannot revert to a default (the FASRC chain went 4 GPUs -> 1 GPU that way).
    ``--dependency=afterany`` on the current job keeps the two from overlapping on the same
    checkpoint directory (research_rules/05: a live duplicate writing the same files).
    """
    if requeue_left <= 0:
        return None
    required = ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT", "SLURM_JOB_NAME")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError(f"cannot build a successor: {', '.join(missing)} not set")
    gpus = env.get("SLURM_GPUS_ON_NODE") or env.get("SLURM_GPUS_PER_NODE", "1").split(":")[-1]
    tasks = env.get("SLURM_NTASKS_PER_NODE") or gpus
    cpus = env.get("SLURM_CPUS_PER_TASK", "16")
    mem_mb = env.get("SLURM_MEM_PER_NODE")
    argv = [
        "sbatch", "--parsable",
        f"--account={env['SLURM_JOB_ACCOUNT']}",
        f"--partition={env['SLURM_JOB_PARTITION']}",
        "--nodes=1",
        f"--gpus-per-node={gpus}",
        f"--ntasks-per-node={tasks}",
        f"--cpus-per-task={cpus}",
        f"--time={time}",
        "--signal=TERM@120",
        f"--job-name={env['SLURM_JOB_NAME']}",
        f"--dependency=afterany:{env['SLURM_JOB_ID']}",
    ]
    if mem_mb:
        argv.append(f"--mem={mem_mb}M")
    if env.get("SLURM_JOB_GRES"):
        argv.append(f"--gres={env['SLURM_JOB_GRES']}")
    export = ",".join([f"{k}={v}" for k, v in exports.items()] + [f"REQUEUE_LEFT={requeue_left - 1}"])
    argv.append(f"--export=NONE,{export}")
    argv.append(script)
    return argv


# ======================================================================================
# realised K -- the column
# ======================================================================================

def launch_overrides(gpus: int, cpus_per_task: int) -> list[str]:
    """The Hydra overrides every campaign launch adds to the grid's own: the trainer for this GPU
    count, the CSV logger the tables read, one loader worker per allocated core (F46).

    F47: an override the shell composed on its own (``trainer.log_every_n_steps=1`` without the
    ``+`` a missing key needs) killed all nine calibration tasks at launch, and no test could
    have caught it because nothing composed the shell's override set. Every override the launcher
    adds now comes from here, and ``tests/integration/test_campaign_manifests.py`` composes this
    set against every campaign cell.
    """
    trainer = ["trainer=ddp", f"trainer.devices={gpus}", "trainer.num_nodes=1"] if gpus > 1 else ["trainer=gpu"]
    return [*trainer, "logger=csv", f"data.num_workers={cpus_per_task}"]


def calibrate_overrides(max_steps: int, interrupt_step: int | None = None) -> list[str]:
    """Calibrate mode (plan Phase 2 / V2): a short run that measures throughput, SIGTERMs itself
    at ``interrupt_step`` when given (the first launch), and logs every step so the resume seam
    and the LR continuity can be read from ``metrics.csv`` (F46: Lightning's default of 50 hid the
    seam)."""
    out = ["callbacks=calibrate", f"+trainer.max_steps={max_steps}", "+trainer.limit_val_batches=0",
           "+trainer.num_sanity_val_steps=0", "callbacks.learning_rate_monitor.logging_interval=step",
           "trainer.log_every_n_steps=1"]
    if interrupt_step is not None:
        out.append(f"callbacks.self_interrupt.at_step={interrupt_step}")
    return out


def online_probe_curve(metrics_csv: Path) -> dict[str, float | int | None]:
    """The online probe's validation curve from a run's ``metrics.csv``: final epoch and maximum.

    Decision 16: LeJEPA reports the online probe with the epoch chosen on the reported split, so
    gate G4a compares our curve's MAXIMUM to MINIMAL.md's 90.7 and the results tables show
    ``final / max (epoch)`` in the appendix. Reads the per-epoch column only (``val/probe_acc``,
    logged ``on_epoch``); rows without it (the per-step training rows) are skipped. The maximum is
    the FIRST epoch that reaches it. Values are fractions in the CSV and returned in percent.
    """
    candidates = ("val/probe_acc", "val/probe_acc_epoch")
    points: list[tuple[int, float]] = []
    metrics_csv = Path(metrics_csv)
    # a resumed run writes one metrics.csv per launch (version_0, version_1, ...): a csv directory
    # is read version by version, in order, so the curve spans the whole run
    files = sorted(metrics_csv.glob("version_*/metrics.csv"), key=lambda f: int(f.parent.name.split("_")[-1])) \
        if metrics_csv.is_dir() else [metrics_csv]
    for file in files:
        with open(file, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            column = next((c for c in candidates if c in fields), None)
            if column is None:
                continue
            for row in reader:
                if row.get(column, "") == "":
                    continue
                epoch = int(float(row["epoch"])) if row.get("epoch", "") != "" else len(points)
                points.append((epoch, round(100.0 * float(row[column]), 6)))     # percent; 6 dp kills float noise
    if not points:
        return {"epochs": 0, "final": None, "max": None, "argmax_epoch": None}
    best_epoch, best = max(points, key=lambda pt: (pt[1], -pt[0]))
    return {"epochs": len(points), "final": points[-1][1], "max": best, "argmax_epoch": best_epoch}


def realised_k(metrics_csv: Path, last_n: int = 10) -> dict[str, float | int | None]:
    """Epoch means of the local diagnostics from a run's ``metrics.csv``.

    Reads the ``_epoch`` series only: the ``_step`` rows are noisier and Lightning writes both
    into one file with blanks, so averaging all non-empty cells would mix the two.
    """
    column = "train/local/effective_clusters_epoch"
    fallback_col = "train/local/fallback_epoch"
    configured_col = "train/local/configured_clusters_epoch"
    values: list[float] = []
    fallbacks = 0
    configured: float | None = None
    with open(metrics_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            return {"epochs": 0, "mean": None, "last_mean": None, "fallback_epochs": None, "configured": None}
        for row in reader:
            if row.get(column, "") == "":
                continue
            values.append(float(row[column]))
            if row.get(fallback_col, "") not in ("", None) and float(row[fallback_col]) > 0:
                fallbacks += 1
            if row.get(configured_col, "") not in ("", None):
                configured = float(row[configured_col])
    if not values:
        return {"epochs": 0, "mean": None, "last_mean": None, "fallback_epochs": 0, "configured": configured}
    tail = values[-last_n:]
    return {"epochs": len(values), "mean": sum(values) / len(values), "last_mean": sum(tail) / len(tail),
            "fallback_epochs": fallbacks, "configured": configured}


# ======================================================================================
# staging and submission
# ======================================================================================

def stage_check(free_bytes: int, tar_bytes: int, factor: float = 1.2) -> tuple[bool, str]:
    needed = int(tar_bytes * factor)
    if free_bytes < needed:
        return False, f"/tmp has {free_bytes / 1e9:.0f} GB free, staging needs {needed / 1e9:.0f} GB ({factor}x the tar)"
    return True, f"/tmp has {free_bytes / 1e9:.0f} GB free for a {tar_bytes / 1e9:.0f} GB tar"


def submit_command(grid_path: Path, runs: list[RunSpec], *, partition: str, account: str, time: str,
                   script: str, log_dir: str, exports: Mapping[str, str], throttle: int | None = None,
                   cpus_per_gpu: int = 16, mem_per_gpu_gb: int = 60, gpu_type: str | None = None,
                   indices: list[int] | None = None) -> list[str]:
    """One homogeneous job array per grid: every run needs the same GPU count, or the array
    would silently give the wrong resources to some of its tasks."""
    gpus = {run.gpus for run in runs}
    if len(gpus) != 1:
        raise ValueError(f"the grid mixes GPU counts {sorted(gpus)}; split it into one array per count")
    (g,) = gpus
    if indices is None:
        array = f"0-{len(runs) - 1}"
    else:
        array = ",".join(str(i) for i in indices)
    if throttle:
        array += f"%{throttle}"
    argv = [
        "sbatch", "--parsable",
        f"--account={account}", f"--partition={partition}",
        f"--array={array}",
        "--nodes=1",
        f"--ntasks-per-node={g}",
        f"--cpus-per-task={cpus_per_gpu}",
        f"--mem={mem_per_gpu_gb * g}G",
        f"--time={time}",
        "--signal=TERM@120",
        f"--job-name={grid_path.stem}",
        f"--output={log_dir}/%x-%A_%a.out",
    ]
    if partition == "preemptable":
        if not gpu_type:
            raise ValueError("preemptable spans both GPU types: pass --gpu-type rtx_pro_6000 or b200")
        argv.append(f"--gres=gpu:{gpu_type}:{g}")
    else:
        argv.append(f"--gpus-per-node={g}")
    export = ",".join(f"{k}={v}" for k, v in {**exports, "GRID": str(grid_path)}.items())
    argv.append(f"--export=NONE,{export}")
    argv.append(script)
    return argv


# ======================================================================================
# CLI
# ======================================================================================

def _emit_shell(assignments: Mapping[str, Any]) -> None:
    import shlex

    for key, value in assignments.items():
        print(f"export {key}={shlex.quote(str(value))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("expand"); p.add_argument("manifest", type=Path); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("show"); p.add_argument("grid", type=Path); p.add_argument("--index", type=int)
    p = sub.add_parser("args"); p.add_argument("grid", type=Path); p.add_argument("--index", type=int, required=True)
    p = sub.add_parser("preflight"); p.add_argument("grid", type=Path); p.add_argument("--index", type=int, required=True)
    p.add_argument("--lock", type=Path); p.add_argument("--python", default=sys.executable)
    p = sub.add_parser("geometry"); p.add_argument("grid", type=Path); p.add_argument("--index", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True); p.add_argument("--shell", action="store_true")
    p = sub.add_parser("stage-check"); p.add_argument("--free", type=int, required=True)
    p.add_argument("--tar-bytes", type=int, required=True); p.add_argument("--factor", type=float, default=1.2)
    p = sub.add_parser("resume"); p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--max-epochs", type=int, required=True); p.add_argument("--shell", action="store_true")
    p = sub.add_parser("postflight"); p.add_argument("grid", type=Path); p.add_argument("--index", type=int, required=True)
    p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--lock", type=Path); p.add_argument("--python", default=sys.executable)
    p.add_argument("--train-rc", type=int, default=0)
    p = sub.add_parser("successor"); p.add_argument("--requeue-left", type=int, required=True)
    p.add_argument("--time", required=True); p.add_argument("--script", required=True)
    p.add_argument("--export", action="append", default=[], help="KEY=VALUE to carry to the successor")
    p = sub.add_parser("realised-k"); p.add_argument("metrics", type=Path)
    p = sub.add_parser("online-curve"); p.add_argument("metrics", type=Path)
    p = sub.add_parser("launch-overrides"); p.add_argument("--gpus", type=int, required=True); p.add_argument("--cpus", type=int, required=True)
    p = sub.add_parser("calibrate-overrides"); p.add_argument("--max-steps", type=int, required=True); p.add_argument("--interrupt", type=int)
    p = sub.add_parser("submit"); p.add_argument("grid", type=Path); p.add_argument("--partition", required=True)
    p.add_argument("--account", default="dominici_lab_hu"); p.add_argument("--time", required=True)
    p.add_argument("--script", default="scripts/aicr/slurm_train.sh"); p.add_argument("--log-dir", required=True)
    p.add_argument("--export", action="append", default=[]); p.add_argument("--throttle", type=int)
    p.add_argument("--gpu-type"); p.add_argument("--cpus-per-gpu", type=int, default=16)
    p.add_argument("--mem-per-gpu-gb", type=int, default=60); p.add_argument("--indices", help="e.g. 0-9,20")
    p = sub.add_parser("env-hash"); p.add_argument("freeze", type=Path, nargs="?")
    for name in ("eval-preflight", "eval-postflight"):
        p = sub.add_parser(name); p.add_argument("--config", type=Path, required=True)
        p.add_argument("--only", required=True); p.add_argument("--root", type=Path, required=True)
    p = sub.add_parser("calibrate-verdict"); p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--interrupt-step", type=int, required=True); p.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args(argv)

    if args.command == "expand":
        manifest = load_manifest(args.manifest)
        runs = expand(manifest)
        commit, dirty = tree_state(REPO)
        write_grid(runs, args.out, manifest_name=str(manifest.get("name", args.manifest.stem)),
                   commit=(commit + ("-dirty" if dirty else "")) if commit else None)
        tags = sorted({run.tag for run in runs})
        print(f"{len(runs)} runs from {len(manifest['blocks'])} block(s) {tags} x {len(manifest.get('seeds', [0]))} seed(s) -> {args.out}")
        return 0

    if args.command == "show":
        payload, runs = read_grid(args.grid)
        chosen = runs if args.index is None else [runs[args.index]]
        for run in chosen:
            print(json.dumps(asdict(run)))
        return 0

    if args.command == "args":
        _, runs = read_grid(args.grid)
        spec = runs[args.index]
        _emit_shell({"RUN_NAME": spec.run_name, "MAX_EPOCHS": spec.max_epochs,
                     "OVERRIDES": " ".join(spec.hydra_overrides())})
        return 0

    if args.command == "preflight":
        payload, runs = read_grid(args.grid)
        spec = runs[args.index]
        lock_text = args.lock.read_text(encoding="utf-8") if args.lock else None
        lock_hash = env_hash(lock_text) if lock_text is not None else None
        verdict = preflight(spec, grid_commit=payload.get("commit"), lock_hash=lock_hash,
                            python=args.python if args.lock else None, lock_text=lock_text)
        print(f"preflight {spec.run_name}:")
        print("\n".join(verdict.lines()))
        return 0 if verdict.ok else 1

    if args.command == "stage-check":
        ok, detail = stage_check(args.free, args.tar_bytes, args.factor)
        print(detail)
        return 0 if ok else 1

    if args.command == "geometry":
        _, runs = read_grid(args.grid)
        geo = geometry(runs[args.index], args.world_size)
        if args.shell:
            _emit_shell({f"GEO_{k.upper()}": v for k, v in geo.items() if v is not None})
        else:
            for key, value in geo.items():
                print(f"{key:22s} {value}")
        return 0

    if args.command == "resume":
        decision = resume_decision(args.ckpt_dir, args.max_epochs)
        for candidate in decision.candidates:
            print(f"# candidate {candidate.path.name}: epoch={candidate.epoch} global_step={candidate.global_step}"
                  f"{'' if candidate.valid else ' UNREADABLE ' + str(candidate.error)}", file=sys.stderr)
        print(f"# decision: {decision.kind} -- {decision.reason}", file=sys.stderr)
        if args.shell:
            _emit_shell({"DECISION": decision.kind, "RESUME": str(decision.path) if decision.path and decision.kind == "resume" else ""})
        else:
            print(decision.kind if decision.path is None else f"{decision.kind}:{decision.path}")
        return 0

    if args.command == "postflight":
        payload, runs = read_grid(args.grid)
        spec = runs[args.index]
        lock_text = args.lock.read_text(encoding="utf-8") if args.lock else None
        lock_hash = env_hash(lock_text) if lock_text is not None else None
        verdict = postflight(spec, args.run_dir, args.ckpt_dir, grid_commit=payload.get("commit"),
                             lock_hash=lock_hash, python=args.python if args.lock else None, lock_text=lock_text)
        print(f"postflight {spec.run_name} (train.py exit code {args.train_rc} -- informational, not the verdict):")
        print("\n".join(verdict.lines()))
        if verdict.ok:
            progress = read_checkpoint_progress(args.ckpt_dir / "last.ckpt")
            write_marker(args.ckpt_dir, verdict, spec, {
                "train_rc": args.train_rc, "epoch": progress.epoch, "global_step": progress.global_step,
                "digest": checkpoint_digest(args.ckpt_dir / "last.ckpt"), "commit": payload.get("commit")})
            print("VERIFIED")
            return 0
        decision = resume_decision(args.ckpt_dir, spec.max_epochs)
        if decision.kind == "resume":
            print(f"RESUMABLE from {decision.path} ({decision.reason})")
            return 3
        write_marker(args.ckpt_dir, verdict, spec, {"train_rc": args.train_rc})
        print("FAILED")
        return 1

    if args.command == "successor":
        exports = dict(item.split("=", 1) for item in args.export)
        argv_out = successor_argv(os.environ, script=args.script, requeue_left=args.requeue_left,
                                  time=args.time, exports=exports)
        if argv_out is None:
            print("# requeue budget exhausted: no successor", file=sys.stderr)
            return 1
        import shlex
        print(" ".join(shlex.quote(a) for a in argv_out))
        return 0

    if args.command == "realised-k":
        print(json.dumps(realised_k(args.metrics)))
        return 0
    if args.command == "online-curve":
        print(json.dumps(online_probe_curve(args.metrics)))
        return 0
    if args.command == "launch-overrides":
        print(" ".join(launch_overrides(args.gpus, args.cpus)))
        return 0
    if args.command == "calibrate-overrides":
        print(" ".join(calibrate_overrides(args.max_steps, args.interrupt)))
        return 0

    if args.command == "submit":
        _, runs = read_grid(args.grid)
        exports = dict(item.split("=", 1) for item in args.export)
        indices = None
        if args.indices:
            indices = []
            for part in args.indices.split(","):
                if "-" in part:
                    lo, hi = part.split("-"); indices.extend(range(int(lo), int(hi) + 1))
                else:
                    indices.append(int(part))
        argv_out = submit_command(args.grid.resolve(), runs, partition=args.partition, account=args.account,
                                  time=args.time, script=args.script, log_dir=args.log_dir, exports=exports,
                                  throttle=args.throttle, cpus_per_gpu=args.cpus_per_gpu,
                                  mem_per_gpu_gb=args.mem_per_gpu_gb, gpu_type=args.gpu_type, indices=indices)
        import shlex
        print(" ".join(shlex.quote(a) for a in argv_out))
        return 0

    if args.command in ("eval-preflight", "eval-postflight"):
        fn = eval_preflight if args.command == "eval-preflight" else eval_postflight
        verdict = fn(args.config, args.only, args.root)
        print(f"{args.command} {args.only}:")
        print("\n".join(verdict.lines()))
        return 0 if verdict.ok else 1

    if args.command == "calibrate-verdict":
        verdict = calibrate_verdict(args.run_dir, interrupt_step=args.interrupt_step, max_steps=args.max_steps)
        print(f"calibrate-verdict {args.run_dir}:")
        print("\n".join(verdict.lines()))
        print("CALIBRATION VERIFIED" if verdict.ok else "CALIBRATION FAILED")
        return 0 if verdict.ok else 1

    if args.command == "env-hash":
        text = args.freeze.read_text(encoding="utf-8") if args.freeze else sys.stdin.read()
        print(env_hash(text))
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
