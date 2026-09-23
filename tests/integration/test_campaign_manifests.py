"""Every campaign manifest expands, every cell composes, and the campaign invariants hold.

The manifests are what the cluster runs (scripts/aicr/campaign.py expand -> job arrays). A cell
that does not compose is a job that dies at 03:00; a cell whose per-GPU micro-batch is not the
declared one is a *different experiment* with the right name on it. Both are checked here, on
the workstation, before anything is submitted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import scripts.aicr.campaign as C

REPO = Path(__file__).resolve().parents[2]
MANIFESTS = sorted((REPO / "configs" / "campaign").glob("*.yaml"))
EXPECTED_RUNS = {
    "ablation_grid": 252, "ablation_batch128": 24, "in1k_baselines": 6, "in1k_vitl14": 2,
    "in1k_local": 4, "torus": 11, "calibration_1gpu": 2, "calibration_4gpu": 3, "calibration_8gpu": 1,
    "smoke_1gpu": 6, "smoke_4gpu": 5, "smoke_8gpu": 2, "ablation_grid_ep800": 252,
}


def expand(path: Path) -> list[C.RunSpec]:
    return C.expand(C.load_manifest(path))


def test_every_run_name_is_unique_across_ALL_manifests():
    """F56 (2026-09-23): `expand` refuses duplicates WITHIN a manifest, nothing checked across them. Both
    `torus` and `in1k_baselines` had blocks tagged `vit_sigreg`/`vit_rdmreg` with no axes, so both expanded to
    `vit_sigreg_s0` -- one checkpoint directory and one run directory for an 800-epoch Imagenette run and a
    100-epoch ImageNet-1K one. They resumed from each other's checkpoints. The name IS the identity: it is the
    checkpoint dir, the hydra run dir, and the key every eval config resolves.

    Inverted once by renaming the torus tags back: this test fails with both names listed.
    """
    from collections import defaultdict

    owners = defaultdict(set)
    for manifest in MANIFESTS:
        for run in expand(manifest):
            owners[run.run_name].add((manifest.stem, run.experiment))
    collisions = {name: sorted(v) for name, v in owners.items() if len({exp for _, exp in v}) > 1}
    assert not collisions, "run names shared by different experiments: " + "; ".join(
        f"{name} <- " + ", ".join(f"{stem}:{exp}" for stem, exp in v) for name, v in sorted(collisions.items()))


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


SMOKE = [(s, r) for s, r in CELLS if s.startswith("smoke_")]


def _geo(cfg) -> tuple[int, int]:
    """(per-step batch, accumulation); not every trainer config declares the accumulation."""
    from omegaconf import OmegaConf

    return (int(OmegaConf.select(cfg, "data.batch_size")),
            int(OmegaConf.select(cfg, "trainer.accumulate_grad_batches", default=1) or 1))


def _geometry_pairs_by_experiment():
    """(batch, accumulation) actually run for each experiment, over the REAL cells only."""
    from collections import defaultdict

    pairs = defaultdict(set)
    for stem, run in CELLS:
        if not stem.startswith("smoke_"):
            pairs[run.experiment].add(_geo(C.compose_config(run.hydra_overrides())))
    return pairs


REAL_GEOMETRIES = _geometry_pairs_by_experiment()


@pytest.mark.parametrize("stem, run", SMOKE, ids=[f"{s}:{r.run_name}" for s, r in SMOKE])
def test_a_smoke_cell_rehearses_a_geometry_the_campaign_actually_runs(stem, run):
    """A debug cell exists to rehearse a real one, so it may shorten the run and nothing else. Its
    batch geometry must be one the campaign runs for that experiment -- NOT one the author picked.
    On 2026-09-23 the spherical-ViT smoke cell was given per-step 256, the ablations' geometry,
    where its own lineup runs 64 x 4; it died of CUDA OOM on a 95 GB card in a shape that appears
    nowhere in the campaign. Inverted by restoring that 256: this test fails naming both.
    """
    cfg = C.compose_config(run.hydra_overrides())
    pair = _geo(cfg)
    allowed = REAL_GEOMETRIES.get(run.experiment)
    assert allowed, f"{run.run_name}: {run.experiment} is rehearsed but never run by a real manifest"
    assert pair in allowed, f"{run.run_name}: rehearses {pair}, the campaign runs {sorted(allowed)}"
    assert int(cfg.trainer.max_epochs) == 2, "the only thing a smoke cell shortens is the run"


@pytest.mark.parametrize("stem, run", CELLS, ids=[f"{s}:{r.run_name}" for s, r in CELLS])
def test_every_distinct_cell_composes_at_its_declared_geometry(stem, run):
    """Composition with the run's overrides, then the geometry assert: the per-GPU micro-batch
    must be the manifest's, every loss is per-shard (no all_gather in src/losses.py)."""
    geo = C.geometry(run, world_size=run.gpus)
    assert geo["micro_batch_per_gpu"] == run.micro_batch
    assert geo["effective_batch"] in (256, 512), geo


@pytest.mark.parametrize("stem, run", CELLS, ids=[f"{s}:{r.run_name}" for s, r in CELLS])
def test_the_manifests_max_epochs_is_the_one_the_run_will_train(stem, run):
    """The manifest's `max_epochs` is what `resume`, `verify_artifact` and `metrics_complete` judge the
    run by; `trainer.max_epochs` is what it trains. If they disagree the cell can never verify -- it
    either stops short of the target or overshoots it. Cheap to state, impossible to see in a log."""
    cfg = C.compose_config(run.hydra_overrides())
    assert int(cfg.trainer.max_epochs) == run.max_epochs, \
        f"{run.run_name}: manifest says {run.max_epochs}, the config trains {int(cfg.trainer.max_epochs)}"


@pytest.mark.parametrize("stem, run", CELLS, ids=[f"{s}:{r.run_name}" for s, r in CELLS])
def test_a_k_ramp_arrives_at_the_same_fraction_of_every_run_length(stem, run):
    """`DynamicNumClusters` steps every `step_epochs`, so a ramp written for 100 epochs arrives at 84%
    of it. Copied unchanged into an 800-epoch grid it would arrive at 10% and sit at its endpoint for
    716 epochs -- a different experiment wearing the same name. Every ramp the campaign runs must
    arrive in the last fifth: the ablations' 2->16 at 84%, the torus row's 1->8 at 87.5%. A SMOKE
    cell is exempt, because shortening the run is the one thing it is allowed to do."""
    from omegaconf import OmegaConf

    if stem.startswith("smoke_"):
        return
    cfg = C.compose_config(run.hydra_overrides())
    ramp = OmegaConf.select(cfg, "callbacks.dynamic_num_clusters")
    if ramp is None:
        return
    steps = abs(int(ramp.end) - int(ramp.start)) / int(OmegaConf.select(ramp, "step", default=1) or 1)
    arrival = steps * int(ramp.step_epochs) / int(cfg.trainer.max_epochs)
    assert 0.80 <= arrival <= 0.90, f"{run.run_name}: the ramp arrives at {arrival:.2f} of the run"


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


@pytest.mark.parametrize("stem, run", CELLS, ids=[f"{s}:{r.run_name}" for s, r in CELLS])
def test_the_postflight_local_test_agrees_with_the_module_on_every_cell(stem, run):
    """``config_is_local`` (what the postflight demands ``train/local/*`` from) must equal what the
    module actually logs -- a regularizer that accepts ``stats``. The control is the case that
    differs by name only (F53)."""
    from hydra.utils import instantiate
    from src.models.lejepa_module import _accepts_keyword

    cfg = C.compose_config(run.hydra_overrides())
    reg_loss = instantiate(cfg.model.reg_loss)
    assert C.config_is_local(cfg) == _accepts_keyword(reg_loss, "stats"), run.run_name
    if run.tag == "vit_control":
        assert not C.config_is_local(cfg)


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


# --------------------------------------------------------------------------- targets import (F46)

def _targets(node, found: set[str]) -> None:
    if isinstance(node, dict):
        target = node.get("_target_")
        if isinstance(target, str):
            found.add(target)
        for value in node.values():
            _targets(value, found)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _targets(value, found)


def campaign_targets() -> list[str]:
    """Every ``_target_`` of every distinct cell, as composed -- the strings Hydra would resolve."""
    from omegaconf import OmegaConf

    found: set[str] = set()
    for _, run in CELLS:
        _targets(OmegaConf.to_container(C.compose_config(run.hydra_overrides()), resolve=False), found)
    return sorted(found)


TARGETS = campaign_targets()


def test_the_campaign_has_targets_to_check():
    assert len(TARGETS) > 5, TARGETS


@pytest.mark.parametrize("target", TARGETS)
def test_every_campaign_target_imports_in_this_environment(target):
    """F46: the D4-RN50 cells died on the GPU node with ``Error locating target
    'src.models.components.e2resnet.build_e2resnet'`` -- ``escnn`` was in no environment file, and
    composition never resolves a ``_target_``, so the suite was green with the package absent.
    ``get_method`` is what ``hydra.utils.instantiate`` calls: the launch-time error, without a job
    behind it. An environment that cannot run a campaign cell fails here, on the workstation."""
    from hydra.utils import get_method

    get_method(target)


# --------------------------------------------------------------------------- the launcher's override sets (F47)

@pytest.mark.parametrize("stem, run", CELLS, ids=[f"{s}:{r.run_name}" for s, r in CELLS])
def test_the_launchers_override_sets_compose_for_every_cell(stem, run):
    """F47: the third calibration died at launch on ``trainer.log_every_n_steps=1``, an override
    the shell composed on its own for a key the trainer config did not have (Hydra wants ``+``).
    No test could see it: the shell's overrides were never composed. Now the launcher takes both
    sets from ``campaign.py`` and this test composes them -- plain launch and calibrate mode --
    against every campaign cell at its GPU count, exactly as ``slurm_train.sh`` would."""
    launch = C.launch_overrides(run.gpus, 16)
    cfg = C.compose_config(run.hydra_overrides() + launch)
    assert cfg.data.num_workers == 16 and cfg.trainer.log_every_n_steps == 50
    # the `gpu` trainer config has no sync_batchnorm key (Lightning's default: off); `ddp` sets it False
    assert cfg.model.sync_batchnorm is True and cfg.trainer.get("sync_batchnorm", False) is False, "BatchNorm is synchronized by the module, not by Lightning (F48)"
    # F50: the three checkpoint dirpaths the launcher overrides must exist, last.ckpt must come from a
    # monitor-less callback and best.ckpt from the online probe's, neither versioned
    cfg = C.compose_config(run.hydra_overrides() + launch + ["callbacks.model_checkpoint.dirpath=/x",
                                                             "callbacks.last_checkpoint.dirpath=/x", "callbacks.on_exception.dirpath=/x"])
    assert cfg.callbacks.last_checkpoint.monitor is None and cfg.callbacks.last_checkpoint.save_top_k == 1
    assert cfg.callbacks.model_checkpoint.save_last is False and cfg.callbacks.last_checkpoint.save_last is False
    assert cfg.callbacks.model_checkpoint.enable_version_counter is False and cfg.callbacks.last_checkpoint.enable_version_counter is False
    assert cfg.trainer.devices == run.gpus and (run.gpus == 1 or cfg.trainer.strategy == "ddp")
    assert cfg.logger.csv is not None

    cfg = C.compose_config(run.hydra_overrides() + launch + C.calibrate_overrides(200, 150))
    assert cfg.trainer.log_every_n_steps == 1, "per-step rows, or the seam check reads 149/199 again (F46)"
    assert cfg.trainer.max_steps == 200 and cfg.trainer.limit_val_batches == 0 and cfg.trainer.num_sanity_val_steps == 0
    assert cfg.callbacks.self_interrupt.at_step == 150
    assert cfg.callbacks.learning_rate_monitor.logging_interval == "step"
    assert cfg.callbacks.throughput is not None and cfg.callbacks.on_exception is not None


# --------------------------------------------------------------------------- the D4 encoder as the campaign builds it (F48)

def test_the_campaign_d4_encoder_builds_at_depth_50_and_survives_sync_batchnorm():
    """F48: depth 18 built everywhere while the campaign runs depth 50, and the multi-GPU
    conversion of BatchNorm killed the D4 cells before their first step. This instantiates the
    encoder from the composed D4 experiment config -- the production arguments, nothing tidier --
    forwards at 128 px, and converts it the way the module does under DDP. ~1 min on CPU."""
    pytest.importorskip("escnn", reason="escnn is a post-create install (environment-aicr.yml)")
    from hydra.utils import instantiate
    from src.utils.sync_bn import convert_sync_batchnorm_inplace

    d4 = next(run for _, run in CELLS if run.experiment == "lejepa/d4rn50_in1k_sigreg")   # the deduplicated cell may carry the calibration manifest's tag
    cfg = C.compose_config(d4.hydra_overrides() + C.launch_overrides(d4.gpus, 16))
    assert cfg.model.encoder.depth == 50 and cfg.model.sync_batchnorm is True and cfg.trainer.sync_batchnorm is False
    encoder = instantiate(cfg.model.encoder)
    with torch.no_grad():
        y = encoder.eval()(torch.zeros(1, 3, 128, 128))
    assert tuple(y.shape) == (1, cfg.model.embedding_dim)
    converted = convert_sync_batchnorm_inplace(encoder)
    assert any(isinstance(m, torch.nn.SyncBatchNorm) for m in converted.modules())

