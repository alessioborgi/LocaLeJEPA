"""Every campaign manifest expands, every cell composes, and the campaign invariants hold.

The manifests are what the cluster runs (scripts/aicr/campaign.py expand -> job arrays). A cell
that does not compose is a job that dies at 03:00; a cell whose per-GPU micro-batch is not the
declared one is a *different experiment* with the right name on it. Both are checked here, on
the workstation, before anything is submitted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.aicr.campaign as C

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = sorted((REPO / "configs" / "campaign").glob("*.yaml"))
EXPECTED_RUNS = {
    "ablation_grid": 252, "ablation_batch128": 24, "in1k_baselines": 6, "in1k_vitl14": 2,
    "in1k_local": 4, "torus": 11, "calibration_1gpu": 2, "calibration_4gpu": 3, "calibration_8gpu": 1,
}


def expand(path: Path) -> list[C.RunSpec]:
    return C.expand(C.load_manifest(path))


def test_every_manifest_is_accounted_for():
    assert {m.stem for m in MANIFESTS} == set(EXPECTED_RUNS), "a new manifest needs its expected run count here"


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[m.stem for m in MANIFESTS])
def test_manifest_expands_to_the_planned_number_of_runs(manifest):
    runs = expand(manifest)
    assert len(runs) == EXPECTED_RUNS[manifest.stem]
    assert len({r.run_name for r in runs}) == len(runs)


def distinct_cells():
    seen = {}
    for manifest in MANIFESTS:
        for run in expand(manifest):
            key = (run.experiment, run.overrides, run.gpus)
            seen.setdefault(key, (manifest.stem, run))
    return sorted(seen.values(), key=lambda x: x[1].run_name)


CELLS = distinct_cells()


@pytest.mark.parametrize("stem, run", CELLS, ids=[f"{s}:{r.run_name}" for s, r in CELLS])
def test_every_distinct_cell_composes_at_its_declared_geometry(stem, run):
    """Composition with the run's overrides, then the geometry assert: the per-GPU micro-batch
    must be the manifest's, every loss is per-shard (no all_gather in src/losses.py)."""
    geo = C.geometry(run, world_size=run.gpus)
    assert geo["micro_batch_per_gpu"] == run.micro_batch
    assert geo["effective_batch"] in (256, 512), geo


@pytest.mark.parametrize("stem, run", CELLS, ids=[f"{s}:{r.run_name}" for s, r in CELLS])
def test_campaign_invariants(stem, run):
    """Decision 6: no flat or rising schedule anywhere in the campaign; logger=csv so the
    realised-K column exists; local arms at cosine-to-peak/1000."""
    cfg = C.compose_config(run.hydra_overrides())
    ratio = float(cfg.model.lr_eta_min_ratio)
    assert ratio < 1.0, f"{run.run_name}: lr_eta_min_ratio {ratio} is flat or rising"
    assert "logger=csv" in run.overrides
    if "local" in run.experiment or "ablations/" in run.experiment:
        assert ratio == 0.001, f"{run.run_name}: the local arm's schedule is cosine to peak/1000"


def test_the_vitl_pair_is_identical_except_the_regularizer():
    """Decision 13, pinned: same views, batch, lr, wd, schedule, lambda, slices, projector."""
    def flat(name):
        cfg = C.compose_config([f"experiment=lejepa/{name}"])
        container = OmegaConf.to_container(cfg, resolve=False)
        container.pop("tags"); container["model"].pop("reg_loss")
        return container
    assert flat("vitl14_in1k_sigreg_paper") == flat("vitl14_in1k_localballreg_paper")
    local = C.compose_config(["experiment=lejepa/vitl14_in1k_localballreg_paper"])
    sig = C.compose_config(["experiment=lejepa/vitl14_in1k_sigreg_paper"])
    assert local.model.reg_loss.loss.num_slices == sig.model.reg_loss.num_slices == 1024
    assert local.model.reg_loss.loss.t_max == sig.model.reg_loss.t_max == 5.0
    assert local.model.lamb == sig.model.lamb == 0.05


def test_the_d4rn50_baselines_are_at_the_published_recipe():
    """Decision 11: lr 2e-3, cosine 2e-3 -> 1e-3, like the ViT and H-ViT baselines."""
    for name in ("lejepa/d4rn50_in1k_sigreg", "rectifiedlpjepa/d4rn50_in1k_rdmreg", "lejepa/vit_in1k_sigreg", "lejepa/hvit_in1k_sigreg"):
        cfg = C.compose_config([f"experiment={name}"])
        assert float(cfg.optimizer.lr) == 2e-3 and float(cfg.model.lr_eta_min_ratio) == 0.5, name


def test_the_control_differs_from_the_local_arm_only_in_the_regularizer_and_lambda():
    """Decision 1: SIGReg under the local recipe -- same lr, warm-up, clip, schedule."""
    control = C.compose_config(["experiment=lejepa/vit_in1k_sigreg_localrecipe"])
    local = C.compose_config(["experiment=lejepa/vit_in1k_localballreg_l01_k4"])
    for key in ("optimizer.lr", "model.warmup_epochs", "trainer.gradient_clip_val", "model.lr_eta_min_ratio"):
        assert OmegaConf.select(control, key) == OmegaConf.select(local, key), key
    assert control.model.reg_loss._target_.endswith("sigreg_loss")
    assert local.model.reg_loss._target_.endswith("local_ballreg_loss")


def test_the_reverse_ramp_counts_down():
    cfg = C.compose_config(["experiment=ablations/dynk_rev"])
    assert cfg.callbacks.dynamic_num_clusters.start == 16 and cfg.callbacks.dynamic_num_clusters.end == 2
