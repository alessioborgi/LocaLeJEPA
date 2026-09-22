import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
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
CAMPAIGN = "midpoint_20260922"
CONFIG = ROOT / f"configs/experiment/ablations/{CAMPAIGN}.yaml"
LOCAL_ROOT = ROOT / "outputs" / CAMPAIGN
FINAL_EPOCH = 99


@dataclass(frozen=True)
class Run:
    arm: str
    seed: int
    prefix: str

    @property
    def name(self) -> str:
        return f"{self.prefix}_seed{self.seed}"


def queue() -> tuple[list[Run], str]:
    """Run names follow hydra.sweep.subdir in the config."""
    config = yaml.safe_load(CONFIG.read_text())
    params = config["hydra"]["sweeper"]["params"]
    arms = params["midpoint.arm"].split(",")
    seeds = [int(seed) for seed in params["seed"].split(",")]
    lamb = config["model"]["lamb"]
    batch = config["data"]["batch_size"]
    accum = config["trainer"]["accumulate_grad_batches"]
    runs = [Run(arm, seed, f"{CAMPAIGN}_l{lamb}_{arm}_B{batch}x{accum}") for arm in arms for seed in seeds]
    if len(runs) != 24 or len({run.name for run in runs}) != 24:
        raise RuntimeError("Midpoint config must define 8 unique arms and 3 unique seeds")
    return runs, str(lamb)


def command(run: Run) -> list[str]:
    return [
        sys.executable,
        "src/train.py",
        "-m",
        f"experiment=ablations/{CAMPAIGN}",
        f"midpoint.arm={run.arm}",
        f"seed={run.seed}",
    ]


def recorded_metrics(run_dir: Path) -> tuple[float, float]:
    """Read final validation accuracy and realised K over epochs 90-99."""
    validation: dict[int, float] = {}
    realised_k: dict[int, float] = {}
    metrics_files = sorted(
        (run_dir / "csv").glob("version_*/metrics.csv"),
        key=lambda path: int(path.parent.name.removeprefix("version_")),
    )
    for metrics_file in metrics_files:
        with metrics_file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if not row.get("epoch"):
                    continue
                epoch = int(float(row["epoch"]))
                if row.get("val/probe_acc"):
                    validation[epoch] = float(row["val/probe_acc"])
                if row.get("train/local/effective_clusters_epoch"):
                    realised_k[epoch] = float(row["train/local/effective_clusters_epoch"])
    if FINAL_EPOCH not in validation:
        raise RuntimeError(f"Missing val/probe_acc at epoch {FINAL_EPOCH} in {run_dir}")
    missing = [epoch for epoch in range(90, 100) if epoch not in realised_k]
    if missing:
        raise RuntimeError(f"Missing realised K for epochs {missing} in {run_dir}")
    return validation[FINAL_EPOCH], sum(realised_k[epoch] for epoch in range(90, 100)) / 10


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


def mark_complete(marker: Path, run: Run, accuracy: float, realised_k: float) -> None:
    if not math.isfinite(accuracy) or not math.isfinite(realised_k):
        raise RuntimeError(f"Non-finite final metric in {marker.parent}")
    write_json(
        marker,
        {
            "run": run.name,
            "final_epoch": FINAL_EPOCH,
            "val_probe_acc": accuracy,
            "realised_k_last_10_epochs": realised_k,
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def prepare_data() -> None:
    """Download the dataset once, so parallel jobs do not all extract it into the same directory."""
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[f"experiment=ablations/{CAMPAIGN}"])
    data_cfg = cfg.data.copy()
    with open_dict(data_cfg):
        data_cfg.pop("has_test_data", None)
    instantiate(data_cfg).prepare_data()


def train(run: Run, output_root: Path, to_file: bool) -> tuple[float, float]:
    shutil.rmtree(LOCAL_ROOT / run.name, ignore_errors=True)
    print(f"launch {run.name}", flush=True)
    if to_file:
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        log = LOCAL_ROOT / f"{run.name}.log"
        with log.open("w") as stream:
            result = subprocess.run(command(run), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            tail = "\n".join(log.read_text().splitlines()[-40:])
            raise RuntimeError(f"exit code {result.returncode}, last lines of {log}:\n{tail}")
    else:
        subprocess.run(command(run), cwd=ROOT, check=True)
    archive_dir = output_root / run.name
    accuracy, realised_k = recorded_metrics(archive_dir)
    mark_complete(archive_dir / "complete.json", run, accuracy, realised_k)
    return accuracy, realised_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", help="Comma-separated subset of midpoint arms, e.g. k1,k2")
    parser.add_argument("--jobs", type=int, default=1, help="Jobs trained at the same time on the one GPU")
    parser.add_argument("--max-runs", type=int, default=None, help="Launch at most N unfinished jobs")
    parser.add_argument("--dry-run", action="store_true", help="Print the grid without writing or using a GPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_runs is not None and args.max_runs < 1:
        raise SystemExit("--max-runs must be positive")
    if args.jobs < 1:
        raise SystemExit("--jobs must be positive")
    drive_dir = Path(os.environ["DRIVE_DIR"])
    output_root = drive_dir / CAMPAIGN
    runs, lamb = queue()
    if args.arms:
        selected = args.arms.split(",")
        if unknown := set(selected) - {run.arm for run in runs}:
            raise SystemExit(f"Unknown arms: {sorted(unknown)}")
        runs = [run for run in runs if run.arm in selected]
    config_hash = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    manifest = {
        "campaign": CAMPAIGN,
        "config_sha256": config_hash,
        "lambda": lamb,
        "run_names": [run.name for run in queue()[0]],
    }

    if args.dry_run:
        for index, run in enumerate(runs):
            print(f"[{index + 1:02d}/{len(runs)}] {run.name}")
            print("  " + shlex.join(command(run)))
        return

    if not drive_dir.is_dir():
        raise SystemExit(f"DRIVE_DIR={drive_dir} does not exist")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is available")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "queue.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise SystemExit(f"Queue settings differ from the existing manifest: {manifest_path}")
    else:
        write_json(manifest_path, manifest)

    pending = []
    for run in runs:
        archive_dir = output_root / run.name
        marker = archive_dir / "complete.json"
        if marker.is_file():
            print(f"skip completed {run.name}", flush=True)
            continue
        if archive_dir.is_dir():
            try:
                accuracy, realised_k = recorded_metrics(archive_dir)
            except RuntimeError:
                pass
            else:
                mark_complete(marker, run, accuracy, realised_k)
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
        prepare_data()
    failed = []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(train, run, output_root, args.jobs > 1): run for run in pending}
        for future in as_completed(futures):
            run = futures[future]
            try:
                accuracy, realised_k = future.result()
            except (RuntimeError, subprocess.CalledProcessError) as error:
                print(f"failed {run.name}: {error}", flush=True)
                failed.append(run.name)
            else:
                print(f"complete {run.name} acc={accuracy:.4f} Khat={realised_k:.2f}", flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
