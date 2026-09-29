# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""``scripts/aicr/make_ablation_eval_config.py`` -- the config that re-reads the ablation
checkpoints with the draft's probes (2026-09-22).

The defect each test exists to prevent: an eval that silently runs on a half-trained cell, a
names file that drifts from the config's order (the array index would then evaluate the wrong
checkpoint under the right name), and a protocol that does not match the draft's table.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


def _load():
    path = REPO / "scripts" / "aicr" / "make_ablation_eval_config.py"
    spec = importlib.util.spec_from_file_location("make_ablation_eval_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load()


def grid_file(tmp_path: Path, names: list[str]) -> Path:
    payload = {"manifest": "ablation_grid", "commit": "deadbeef",
               "runs": [{"run_name": n, "experiment": "ablations/base", "tag": "fixk", "seed": 0,
                         "overrides": ["logger=csv"], "gpus": 1, "micro_batch": 64, "max_epochs": 100}
                        for n in names]}
    path = tmp_path / "grid.json"
    path.write_text(json.dumps(payload))
    return path


def cell(root: Path, name: str, *, ckpt: bool = True, verified: bool = True, kind: str = "last") -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    if ckpt:
        (d / f"{kind}.ckpt").write_bytes(b"x")
    if verified:
        (d / ".verified").write_text("{}")


def test_only_verified_cells_with_a_checkpoint_are_evaluated(gen, tmp_path):
    """G5/V4: the launcher's marker is what says a run finished. A cell with a checkpoint but no
    marker is a run whose postflight refused, and its numbers must never reach a table."""
    root = tmp_path / "ckpt"
    cell(root, "a")                                   # complete
    cell(root, "b", verified=False)                   # trained, refused by postflight
    cell(root, "c", ckpt=False)                       # never ran
    cell(root, "d")                                   # complete
    runs = json.loads(grid_file(tmp_path, ["a", "b", "c", "d"]).read_text())["runs"]
    entries, skipped = gen.select_cells(runs, root, "last")
    assert [e["name"] for e in entries] == ["a", "d"]
    assert any("b: not verified" == s for s in skipped) and any("c: no last.ckpt" == s for s in skipped)
    assert entries[0]["checkpoint"] == str(root / "a" / "last.ckpt"), "absolute path, kept verbatim"
    assert "overrides" not in entries[0], "load_backbone instantiates only model.encoder"
    loose, _ = gen.select_cells(runs, root, "last", require_verified=False)
    assert [e["name"] for e in loose] == ["a", "b", "d"], "--allow-unverified widens it, nothing else"


def test_the_order_of_the_names_file_is_the_order_of_the_backbones(gen, tmp_path):
    """The job array reads line i+1 for task i. If the two orders drift, every task evaluates a
    checkpoint under another cell's name and the whole table is silently permuted."""
    root = tmp_path / "ckpt"
    names = ["fixk_b64_lam0.02_k1_x4_s0", "fixk_b256_lam0.3_k32_x1_s2", "fwd_b64_lam0.1_x4_s1"]
    for n in names:
        cell(root, n)
    grid = grid_file(tmp_path, names)
    out, names_file = tmp_path / "cfg.yaml", tmp_path / "names.txt"
    assert gen.main(["--grid", str(grid), "--ckpt-root", str(root), "--data-dir", "/d",
                     "--out", str(out), "--names", str(names_file), "--out-dir", "/o"]) == 0
    cfg = yaml.safe_load(out.read_text())
    listed = names_file.read_text().splitlines()
    assert [b["name"] for b in cfg["backbones"]] == listed == names
    assert all(cfg["backbones"][i]["name"] == listed[i] for i in range(len(names)))


def test_the_generated_config_carries_the_drafts_protocol_and_is_loadable(gen, tmp_path):
    """The point of the exercise is comparability with `tab:k-ablation`: k-NN reported at k=20 and
    a linear head selected on a held-out tenth of the TRAIN split, never on the reported split."""
    root = tmp_path / "ckpt"
    cell(root, "a")
    grid = grid_file(tmp_path, ["a"])
    out, names_file = tmp_path / "cfg.yaml", tmp_path / "names.txt"
    gen.main(["--grid", str(grid), "--ckpt-root", str(root), "--data-dir", "/data/imagenette",
              "--out", str(out), "--names", str(names_file), "--out-dir", "/logs/eval"])
    cfg = yaml.safe_load(out.read_text())
    assert cfg["dataset"] == "imagenette" and cfg["data_dir"] == "/data/imagenette"
    assert cfg["knn"]["report_k"] == 20 and cfg["knn"]["k_values"] == [10, 20, 100, 200]
    assert cfg["linear"]["val_frac"] == 0.1 and "select_on" not in cfg["linear"], "strict selection is the default"
    assert cfg["crop_size"] == 128 and cfg["imagenette_variant"] == "160px"
    assert cfg["cache_dir"] == "/logs/eval/cache" and cfg["output_folder"] == "/logs/eval"
    assert cfg["train_subset_per_class"] is None, "the full train bank, as the draft's protocol says"
    reference = yaml.safe_load((REPO / "configs" / "eval" / "indomain_imagenette.yaml").read_text())
    for key in ("knn", "linear"):
        assert cfg[key] == {k: v for k, v in reference[key].items()}, f"{key} must equal the committed protocol"


def test_nothing_to_evaluate_is_a_non_zero_exit(gen, tmp_path):
    """A silent empty config would submit an array of no-ops and look like success."""
    root = tmp_path / "ckpt"
    cell(root, "a", verified=False)
    grid = grid_file(tmp_path, ["a"])
    assert gen.main(["--grid", str(grid), "--ckpt-root", str(root), "--data-dir", "/d",
                     "--out", str(tmp_path / "c.yaml"), "--names", str(tmp_path / "n.txt"),
                     "--out-dir", "/o"]) == 1
    assert (tmp_path / "n.txt").read_text() == ""


def test_best_checkpoints_can_be_read_too(gen, tmp_path):
    """The draft selects its checkpoint on the online validation accuracy; ours reports `last`.
    Both are reachable, and the kind is never inferred from anything but this flag."""
    root = tmp_path / "ckpt"
    cell(root, "a", kind="best")
    runs = json.loads(grid_file(tmp_path, ["a"]).read_text())["runs"]
    assert gen.select_cells(runs, root, "last")[0] == []
    entries, _ = gen.select_cells(runs, root, "best")
    assert entries[0]["checkpoint"].endswith("best.ckpt")


# ----- shape-bearing overrides (decision 24: the capacity grid varies model.encoder.depth) -----


def shape_grid(tmp_path: Path, overrides: list[str]) -> list[dict]:
    """One run carrying ``overrides``, in the grid-JSON shape `campaign.py expand` writes."""
    payload = {"manifest": "capacity_grid_ep800", "commit": "deadbeef",
               "runs": [{"run_name": "cap_fixk_dep6_lam0.1_k4_s0",
                         "experiment": "ablations/base_depth", "tag": "cap_fixk", "seed": 0,
                         "overrides": overrides, "gpus": 1, "micro_batch": 256, "max_epochs": 800}]}
    path = tmp_path / "shape_grid.json"
    path.write_text(json.dumps(payload))
    return json.loads(path.read_text())["runs"]


def test_a_depth_override_is_forwarded_so_the_checkpoint_meets_its_own_architecture(gen, tmp_path):
    """THE defect this generator used to guarantee, once the capacity grid existed.

    `load_backbone` composes `experiment=<name>` and instantiates `model.encoder` from that alone.
    `ablations/base_depth` declares depth 12, and the capacity grid's depth comes from a MANIFEST
    override -- so without forwarding it, every depth-6 checkpoint would be loaded into a depth-12
    module. `src/eval/extract.py:134-140` raises on the strict key check, so it fails loudly rather
    than silently; but the same mismatch reaches `load_projector`, and 180 evals would fail one by
    one at the end of a queue we are last in.

    INVERTED once: dropping `entry["overrides"] = shape` from `select_cells` leaves the key absent
    and fails.
    """
    root = tmp_path / "ckpt"
    cell(root, "cap_fixk_dep6_lam0.1_k4_s0")
    runs = shape_grid(tmp_path, ["logger=csv", "model.encoder.depth=6", "model.lamb=0.1",
                                "model.reg_loss.num_clusters=4", "trainer.max_epochs=800"])
    entries, _ = gen.select_cells(runs, root, "last")
    assert entries[0]["overrides"] == ["model.encoder.depth=6"], entries[0]


def test_training_only_overrides_are_NOT_forwarded(gen, tmp_path):
    """lambda, K, the batch, the logger and the epoch count cannot reach an instantiated encoder or
    projector, and forwarding them would put training state into an eval config -- where
    `logger=csv` is a config-group selection that would compose a logger the eval never uses, and a
    changed `trainer.max_epochs` would make the emitted config differ from run to run for no reason.
    """
    root = tmp_path / "ckpt"
    cell(root, "cap_fixk_dep6_lam0.1_k4_s0")
    runs = shape_grid(tmp_path, ["logger=csv", "model.lamb=0.1", "model.reg_loss.num_clusters=4",
                                "data.batch_size=256", "trainer.accumulate_grad_batches=1",
                                "trainer.max_epochs=800"])
    entries, _ = gen.select_cells(runs, root, "last")
    assert "overrides" not in entries[0], entries[0]


@pytest.mark.parametrize(
    "override",
    ["model.encoder.depth=6", "model.embedding_dim=512", "model.projection_dim=4",
     "model.projector.hidden_channels=[4]", "model.rectify=true"],
)
def test_every_shape_bearing_key_is_recognised(gen, tmp_path, override: str):
    """Each of these changes the module a checkpoint loads into.

    `model.projection_dim` is the one the d=4 grid moves; its evals happened to survive because
    `load_backbone` builds only the encoder, but `load_projector` -- which the new discrepancy metric
    needs -- rebuilds the projector from the same config and would have been silently wrong.
    """
    root = tmp_path / "ckpt"
    cell(root, "cap_fixk_dep6_lam0.1_k4_s0")
    entries, _ = gen.select_cells(shape_grid(tmp_path, ["logger=csv", override]), root, "last")
    assert entries[0]["overrides"] == [override]


def test_data_crop_size_is_deliberately_not_forwarded(gen, tmp_path):
    """`load_backbone` and `load_projector` take `crop_size` as their own argument and pass it as a
    Hydra override themselves, so emitting it here would send the same key twice."""
    root = tmp_path / "ckpt"
    cell(root, "cap_fixk_dep6_lam0.1_k4_s0")
    entries, _ = gen.select_cells(shape_grid(tmp_path, ["data.crop_size=128"]), root, "last")
    assert "overrides" not in entries[0], entries[0]


def test_a_run_without_an_overrides_key_at_all_is_tolerated(gen, tmp_path):
    """Older grid JSONs on scratch predate this field's use; a missing key must not crash the
    generator that re-reads them."""
    root = tmp_path / "ckpt"
    cell(root, "old")
    runs = [{"run_name": "old", "experiment": "ablations/base"}]
    entries, _ = gen.select_cells(runs, root, "last")
    assert entries[0]["name"] == "old" and "overrides" not in entries[0]
