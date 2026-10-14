"""The bobblehead angle-probe entry points.

``eval_angle_probe`` extracts and caches features, then probes them once; ``_multiseed`` re-reads
that cache by rebuilding its key, so the round trip between the two is tested directly. The
dataset and the backbone are faked with frames that are already torus coordinates; the probes and
caches are real.
"""

import json
import sys

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import Dataset

import src.eval_angle_probe as probe_main
import src.eval_angle_probe_multiseed as multiseed

pytest.importorskip("matplotlib")


class TorusFrames(Dataset):
    """Each "image" is ``[cos a0, sin a0, cos a1, sin a1]`` plus a little noise."""

    has_labels = True

    def __init__(self, n=60):
        rng = np.random.default_rng(0)
        self.angles = rng.uniform(-np.pi, np.pi, size=(n, 2)).astype(np.float32)
        noise = rng.normal(scale=0.01, size=(n, 4))
        self.points = (np.concatenate([np.cos(self.angles), np.sin(self.angles)], axis=1) + noise).astype(np.float32)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return torch.as_tensor(self.points[i]), torch.as_tensor(self.angles[i])


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(probe_main, "get_bobblehead_dataset", lambda **kwargs: TorusFrames())
    monkeypatch.setattr(
        probe_main, "load_backbone", lambda experiment, checkpoint_path, device, crop_size: ((lambda points: points.float()), 4)
    )


def config(tmp_path, **overrides):
    backbones = []
    for name in ("a", "b"):
        path = tmp_path / "ckpts" / f"{name}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": name}, path)
        backbones.append({"name": name, "experiment": f"exp/{name}", "checkpoint": str(path)})
    cfg = {
        "device": "cpu",
        "crop_size": 16,
        "data_dir": str(tmp_path / "data"),
        "output_folder": str(tmp_path / "out"),
        "cache_dir": str(tmp_path / "cache"),
        "batch_size": 16,
        "num_workers": 0,
        "angle_labels": ["figure 0", "figure 1"],
        "backbones": backbones,
    }
    cfg.update(overrides)
    return cfg


def test_the_probe_recovers_both_angles(tmp_path):
    probe_main.main(config(tmp_path))

    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert list(results) == ["a", "b"]
    for entry in results.values():
        for key in ("angle0", "angle1"):
            assert entry["metrics"][key]["test"]["mae_deg"] < 5.0, "the features are the torus"


def test_fewer_angle_labels_than_angles_falls_back_to_the_angle_key(tmp_path):
    """Regression: ``_plots`` indexed ``angle_labels[j]`` without the ``angle{k}`` fallback the
    logging loop uses, so the run crashed after every backbone had been probed."""
    probe_main.main(config(tmp_path, angle_labels=["figure 0"]))
    assert (tmp_path / "out" / "results.json").is_file()


def test_multiseed_reads_the_cache_the_single_seed_run_wrote(tmp_path, monkeypatch):
    """The two scripts must build the same cache key; a divergence raises FileNotFoundError."""
    cfg = config(tmp_path)
    probe_main.main(cfg)
    path = tmp_path / "multiseed.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["eval_angle_probe_multiseed.py", "--config", str(path), "--n-seeds", "3"])
    multiseed.main()

    summary = json.loads((tmp_path / "out" / "results_multiseed.json").read_text())
    assert list(summary) == ["a", "b"]
    for angle in summary["a"]["angles"].values():
        assert len(set(angle["mae_per_seed"])) == 3, "each seed is a different split"
        assert angle["mae_mean"] < 5.0
