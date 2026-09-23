import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import rootutils
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict

ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
CAMPAIGN = "reference_20260922"
METHODS = ("sigreg", "rdmreg")
CONFIG_DIR = ROOT / "configs/experiment/ablations"
CONFIGS = [CONFIG_DIR / f"{CAMPAIGN}.yaml"] + [CONFIG_DIR / f"{CAMPAIGN}_{method}.yaml" for method in METHODS]
FINAL_EPOCH = 99


@dataclass(frozen=True)
class Run:
    method: str
    geometry: str
    seed: int

    @property
    def name(self) -> str:
        """Matches reference.run_name in the config."""
        return f"{CAMPAIGN}_{self.method}_{self.geometry}_seed{self.seed}"

    @property
    def experiment(self) -> str:
        return f"ablations/{CAMPAIGN}_{self.method}"


def queue() -> list[Run]:
    """Seed is the outer loop, so a partial queue still fills every cell with at least one seed."""
    reference = yaml.safe_load(CONFIGS[0].read_text())["reference"]
    runs = [
        Run(method, geometry, seed)
        for seed in reference["seeds"]
        for method in METHODS
        for geometry in reference["geometries"]
    ]
    if len(runs) != 12 or len({run.name for run in runs}) != 12:
        raise RuntimeError("Reference config must define 2 geometries and 3 unique seeds")
    return runs


def command(run: Run, ckpt_path: Path | None) -> list[str]:
    overrides = [f"experiment={run.experiment}", f"reference.geometry={run.geometry}", f"seed={run.seed}"]
    if ckpt_path is not None:
        overrides.append(f"ckpt_path={ckpt_path}")
    return [sys.executable, "src/train.py", "-m", *overrides]


def final_accuracy(run_dir: Path) -> float:
    validation: dict[int, float] = {}
    metrics_files = sorted(
        (run_dir / "csv").glob("version_*/metrics.csv"),
        key=lambda path: int(path.parent.name.removeprefix("version_")),
    )
    for metrics_file in metrics_files:
        with metrics_file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("epoch") and row.get("val/probe_acc"):
                    validation[int(float(row["epoch"]))] = float(row["val/probe_acc"])
    if FINAL_EPOCH not in validation:
        raise RuntimeError(f"Missing val/probe_acc at epoch {FINAL_EPOCH} in {run_dir}")
    return validation[FINAL_EPOCH]


def finished_on_wandb(names: list[str]) -> set[str]:
    """State "finished" is not enough: task_wrapper calls wandb.finish() after a crash too."""
    import wandb

    runs = wandb.Api().runs(
        f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}",
        filters={"group": CAMPAIGN, "display_name": {"$in": names}},
    )
    return {
        run.name
        for run in runs
        if run.state == "finished" and any(row["epoch"] == FINAL_EPOCH for row in run.scan_history(keys=["epoch", "val/probe_acc"]))
    }


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def mark_complete(marker: Path, run: Run, accuracy: float) -> None:
    if not math.isfinite(accuracy):
        raise RuntimeError(f"Non-finite final metric in {marker.parent}")
    write_json(
        marker,
        {
            "run": run.name,
            "final_epoch": FINAL_EPOCH,
            "val_probe_acc": accuracy,
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def compose_run(run: Run, extra: list[str] = ()):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[f"experiment={run.experiment}", f"reference.geometry={run.geometry}", *extra],
        )


def instantiate_data(cfg):
    data_cfg = cfg.data.copy()
    with open_dict(data_cfg):
        data_cfg.pop("has_test_data", None)
    return instantiate(data_cfg)


def probe_memory(runs: list[Run], steps: int) -> None:
    """A few optimizer steps per (method, geometry) in this process, without loggers or checkpoints."""
    import torch

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU memory: {total:.1f} GiB", flush=True)
    failed = []
    cells = sorted({(run.method, run.geometry) for run in runs})
    for method, geometry in cells:
        run = Run(method, geometry, 0)
        with tempfile.TemporaryDirectory() as output_dir:
            cfg = compose_run(run, [f"paths.output_dir={output_dir}"])
            datamodule = instantiate_data(cfg)
            model = instantiate(cfg.model)
            trainer = instantiate(
                cfg.trainer,
                max_steps=steps,
                limit_val_batches=0,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            label = f"{method} {geometry} (grad_checkpointing={cfg.model.encoder.grad_checkpointing})"
            try:
                trainer.fit(model=model, datamodule=datamodule)
            except torch.OutOfMemoryError:
                print(f"probe {label}: out of memory", flush=True)
                failed.append(label)
            else:
                allocated = torch.cuda.max_memory_allocated() / 2**30
                reserved = torch.cuda.max_memory_reserved() / 2**30
                print(f"probe {label}: peak allocated {allocated:.1f} GiB, peak reserved {reserved:.1f} GiB", flush=True)
            del trainer, model, datamodule
            gc.collect()
            torch.cuda.empty_cache()
    if failed:
        raise SystemExit(f"Out of memory: {', '.join(failed)}")


def train(run: Run, output_root: Path, to_file: bool) -> float:
    run_dir = output_root / run.name
    last = run_dir / "checkpoints" / "last.ckpt"
    ckpt_path = last if last.is_file() else None
    print(f"{'resume' if ckpt_path else 'launch'} {run.name}", flush=True)
    if to_file:
        log = output_root / f"{run.name}.log"
        with log.open("a") as stream:
            result = subprocess.run(command(run, ckpt_path), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = "\n".join(log.read_text().splitlines()[-40:])
            raise RuntimeError(f"exit code {result.returncode}, last lines of {log}:\n{tail}")
    else:
        subprocess.run(command(run, ckpt_path), cwd=ROOT, check=True)
    accuracy = final_accuracy(run_dir)
    mark_complete(run_dir / "complete.json", run, accuracy)
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1, help="Jobs trained at the same time on the one GPU")
    parser.add_argument("--max-runs", type=int, default=None, help="Launch at most N unfinished jobs")
    parser.add_argument("--probe-steps", type=int, default=None, help="Only measure peak GPU memory over N optimizer steps per cell")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without writing or using a GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        raise SystemExit("--max-runs must be positive")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    runs = queue()

    if args.dry_run:
        for index, run in enumerate(runs):
            print(f"[{index + 1:02d}/{len(runs)}] {run.name}")
            print("  " + shlex.join(command(run, None)))
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is available")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    if args.probe_steps is not None:
        instantiate_data(compose_run(runs[0])).prepare_data()
        probe_memory(runs, args.probe_steps)
        return

    drive_dir = Path(os.environ["DRIVE_DIR"])
    if not drive_dir.is_dir():
        raise SystemExit(f"DRIVE_DIR={drive_dir} does not exist")
    output_root = drive_dir / CAMPAIGN
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign": CAMPAIGN,
        "config_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in CONFIGS},
        "run_names": [run.name for run in runs],
    }
    manifest_path = output_root / "queue.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit(f"Queue settings differ from the existing manifest: {manifest_path}")
    else:
        write_json(manifest_path, manifest)

    pending = []
    for run in runs:
        run_dir = output_root / run.name
        marker = run_dir / "complete.json"
        if marker.is_file():
            print(f"skip completed {run.name}", flush=True)
            continue
        if run_dir.is_dir():
            try:
                accuracy = final_accuracy(run_dir)
            except RuntimeError:
                pass
            else:
                mark_complete(marker, run, accuracy)
                print(f"recovered completed {run.name}", flush=True)
                continue
        pending.append(run)
    if pending:
        on_wandb = finished_on_wandb([run.name for run in pending])
        for name in sorted(on_wandb):
            print(f"skip finished on W&B {name}", flush=True)
        pending = [run for run in pending if run.name not in on_wandb]
    pending = pending[: args.max_runs]

    if pending:
        instantiate_data(compose_run(pending[0])).prepare_data()
    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(train, run, output_root, args.jobs > 1): run for run in pending}
        for future in as_completed(futures):
            run = futures[future]
            try:
                accuracy = future.result()
            except (RuntimeError, subprocess.CalledProcessError) as error:
                print(f"failed {run.name}: {error}", flush=True)
                failed.append(run.name)
            else:
                print(f"complete {run.name} acc={accuracy:.4f}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
