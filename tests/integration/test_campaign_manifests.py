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
    "smoke_1gpu": 8, "smoke_4gpu": 6, "smoke_8gpu": 2, "ablation_grid_ep800": 252,
    "target_and_centering": 24, "in1k_global_ball": 3, "ablation_grid_ep800_d4": 252,
    "capacity_grid_ep800": 180, "capacity_in1k": 4,
    "in1k_attribution": 2, "in1k_vitl14_globalball": 1,
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


def test_the_target_and_centering_grid_is_a_clean_two_by_two_with_locality_OFF():
    """Decision 20d. Two differences between LocaLeJEPA and LeJEPA are uncontrolled in every grid we
    have run: the TARGET (uniform-on-a-ball against the isotropic Gaussian) and CENTRING --
    `localereg` subtracts each neighbourhood's mean and `sigreg` subtracts nothing, so SIGReg pins
    the representation's global mean and ours leaves the location free, at EVERY K including K=1
    (state/10-PAPER-STATE.md, 2026-09-24). This manifest exists to isolate those two, so what it must
    guarantee is that NOTHING ELSE differs between its cells: one recipe, one geometry, one length,
    and `num_clusters=1` throughout so locality never enters. All four combinations also go through
    `local_loss`, because calling `sigreg_loss` directly would make the wrapper a third difference.

    Inverted twice: dropping `model.reg_loss.num_clusters: 1` from one block fails on the K assert,
    and changing one block's `experiment` away from `ablations/base` fails on the single-recipe assert.
    """
    runs = expand(REPO / "configs" / "campaign" / "target_and_centering.yaml")
    assert len(runs) == 24, f"expected 2 targets x 2 centrings x 2 lambdas x 3 seeds, got {len(runs)}"
    cells: dict[tuple[float, str, bool], list[str]] = {}
    recipes: set[tuple] = set()
    for run in runs:
        cfg = C.compose_config(run.hydra_overrides())
        reg = cfg.model.reg_loss
        assert int(reg.num_clusters) == 1, \
            f"{run.run_name}: locality must be OFF in this grid, the config asks for K={reg.num_clusters}"
        assert str(reg._target_).endswith(("local_ballreg_loss", "local_sigreg_loss")), \
            f"{run.run_name}: every cell must go through local_loss, got {reg._target_}"
        key = (float(cfg.model.lamb), str(reg.loss._target_).split(".")[-1], bool(reg.loss.center))
        cells.setdefault(key, []).append(run.run_name)
        recipes.add((float(cfg.optimizer.lr), int(cfg.model.warmup_epochs),
                     float(cfg.model.lr_eta_min_ratio), float(cfg.trainer.gradient_clip_val),
                     _geo(cfg), int(cfg.trainer.max_epochs)))
    assert len(recipes) == 1, \
        f"the cells must differ ONLY in target and centring; {len(recipes)} recipes/geometries seen: {sorted(recipes)}"
    expected = {(lam, target, center)
                for lam in (0.02, 0.1)
                for target in ("ballreg_loss", "sigreg_loss")
                for center in (True, False)}
    assert set(cells) == expected, f"the 2x2x2 is incomplete: missing {sorted(expected - set(cells))}"
    assert all(len(names) == 3 for names in cells.values()), \
        f"every cell needs its three seeds: {{k: len(v) for k, v in cells.items()}}"


def test_the_global_ball_arms_differ_from_the_local_ones_ONLY_by_not_partitioning():
    """Decision 22, run 3. The IN1K evaluations give the local arm +17.5 k-NN over LeJEPA, but that pair
    differs in the regularizer AND lr AND warm-up AND clipping AND lambda AND centring. The Imagenette grid
    says locality contributes ~0, leaving the compact TARGET as the live hypothesis -- and no global-ball run
    exists at any scale. This manifest supplies it, and its whole value depends on being identical to the
    corresponding `in1k_local` arm except for `num_clusters`. If anything else drifts, the comparison is dead.

    Inverted twice, both verified: changing one block's lambda to 0.3 fails the test (lambda is in the
    run name, so it fails on the pair lookup), and changing a non-name override such as data.batch_size
    fails on the 'differ ONLY in num_clusters' assert itself, printing both differing keys.
    """
    ball = {r.run_name: r for r in expand(REPO / "configs" / "campaign" / "in1k_global_ball.yaml")}
    local = {r.run_name: r for r in expand(REPO / "configs" / "campaign" / "in1k_local.yaml")}
    assert len(ball) == 3
    pairs = [("vit_globalball_lam0.1_k1_s0", "vit_local_lam0.1_k4_s0"),
             ("hvit_globalball_lam0.1_k1_s0", "hvit_local_lam0.1_k4_s0"),
             ("d4rn50_globalball_lam0.1_k1_s0", "d4rn50_local_lam0.1_k4_s0")]
    for bname, lname in pairs:
        b, l = ball[bname], local[lname]
        assert b.experiment == l.experiment, f"{bname}: experiment drifted from {lname}"
        assert b.gpus == l.gpus and b.micro_batch == l.micro_batch and b.max_epochs == l.max_epochs
        bo, lo = b.override_map(), l.override_map()
        assert int(bo["model.reg_loss.num_clusters"]) == 1, f"{bname}: the global arm must be K=1"
        assert int(lo["model.reg_loss.num_clusters"]) > 1, f"{lname}: the local arm must be K>1"
        differing = {k for k in set(bo) | set(lo) if bo.get(k) != lo.get(k)}
        assert differing == {"model.reg_loss.num_clusters"}, \
            f"{bname} vs {lname} must differ ONLY in num_clusters; also differ in {sorted(differing)}"


def test_the_d4_grid_narrows_the_PROJECTOR_and_nothing_else():
    """Decision 22, run 4. With d=16 and a per-step batch of 64, K=4 already leaves exactly 16 samples for a
    16-dimensional target, so every larger K estimates a 16-D distribution from fewer points than dimensions
    -- the high-K cells were never a test of locality. At d=4 a 16-sample neighbourhood is 4x the target
    dimension, so K up to 8 becomes estimable, and the ball-vs-Gaussian signal is ~10x larger.

    What must hold: the projector output narrows to 4, `embedding_dim` stays 512 so the probe reads the same
    width and the numbers stay comparable to the d=16 grid, and the ball radius is still derived (radius
    null -> sqrt(d+2)) rather than pinned to the old d.

    Inverted by also overriding `model.embedding_dim`: this test fails on the 512 assert.
    """
    runs = expand(REPO / "configs" / "campaign" / "ablation_grid_ep800_d4.yaml")
    assert len(runs) == 252
    seen = set()
    for run in runs:
        key = (run.experiment, run.overrides)
        if key in seen:
            continue
        seen.add(key)
        cfg = C.compose_config(run.hydra_overrides())
        assert int(cfg.model.projection_dim) == 4, f"{run.run_name}: projection_dim is {cfg.model.projection_dim}"
        assert int(cfg.model.embedding_dim) == 512, \
            f"{run.run_name}: embedding_dim must stay 512 or the probe is not comparable to the d=16 grid"
        assert int(list(cfg.model.projector.hidden_channels)[-1]) == 4, \
            f"{run.run_name}: the projector must actually emit 4 dimensions"
        reg = cfg.model.reg_loss
        if "loss" in reg and "radius" in reg.loss:
            assert reg.loss.radius is None, \
                f"{run.run_name}: radius must stay null so it resolves to sqrt(d+2) at the NEW d"


def test_no_d4_run_name_can_collide_with_the_d16_grid():
    """F56: two runs sharing a name share a checkpoint directory and resume from each other. The d=4 grid is
    the same 252-cell design at a different projector width, so its names are one prefix away from the
    originals."""
    d16 = {r.run_name for r in expand(REPO / "configs" / "campaign" / "ablation_grid_ep800.yaml")}
    d4 = {r.run_name for r in expand(REPO / "configs" / "campaign" / "ablation_grid_ep800_d4.yaml")}
    assert not (d16 & d4), f"colliding names: {sorted(d16 & d4)[:5]}"
    assert all(n.startswith("d4ep800_") for n in d4)


def test_the_capacity_grid_varies_DEPTH_and_nothing_else():
    """Decision 24. The whole experiment rests on depth being the only axis that moves.

    `embedding_dim` must stay 512 so the probe reads the same width as every other grid, and
    `projection_dim` must stay 16 so the collapse cliff -- which sits at roughly `d` samples per
    neighbourhood -- stays where the reference grid put it. A capacity change that also moved the
    projected dimension would shift the cliff underneath the axis being measured, which is exactly
    the confound the d=4 grid was built to create deliberately and this grid must avoid.

    Batch geometry is checked here too: b=256 with accumulate 1 gives N=256 per shard, so K=16
    leaves 16 samples per neighbourhood -- at the cliff, not past it. At b=64 only K<=4 would
    survive, which is why this grid has one batch and the reference grid has two.

    Inverted by adding `model.projection_dim: 4` to the manifest's overrides: fails the 16 assert.
    """
    runs = expand(REPO / "configs" / "campaign" / "capacity_grid_ep800.yaml")
    assert len(runs) == 180
    depths = set()
    seen = set()
    for run in runs:
        key = (run.experiment, run.overrides)
        if key in seen:
            continue
        seen.add(key)
        cfg = C.compose_config(run.hydra_overrides())
        depths.add(int(cfg.model.encoder.depth))
        assert int(cfg.model.embedding_dim) == 512, \
            f"{run.run_name}: embedding_dim must stay 512 or the probe is not comparable"
        assert int(cfg.model.projection_dim) == 16, \
            f"{run.run_name}: projection_dim sets the collapse cliff; depth must not move it"
        assert int(cfg.data.batch_size) == 256 and int(cfg.trainer.accumulate_grad_batches) == 1, \
            f"{run.run_name}: one batch geometry only, N=256 per shard"
    assert depths == {6, 3}, f"the capacity axis must be depth 6 and 3, got {sorted(depths)}"


def test_the_capacity_grid_omits_depth_12_because_it_is_already_trained():
    """Depth 12 is the existing `ablation_grid_ep800` at b=256 -- 75 fixed-K cells plus 15
    forward-ramp cells, verified. Re-running it would cost ~150 GPU-h to reproduce numbers we hold,
    and would introduce a second set of depth-12 results that could disagree with the first.

    Also pins the REVERSE ramp out: it collapsed in the reference grid, so a capacity counterpart
    would buy a second question rather than evidence (decision 24).
    """
    runs = expand(REPO / "configs" / "campaign" / "capacity_grid_ep800.yaml")
    assert not any("depth=12" in o for r in runs for o in r.overrides), \
        "depth 12 is the reference grid and must not be re-trained"
    assert {r.tag for r in runs} == {"cap_fixk", "cap_fwd"}, \
        "the reverse ramp has no capacity counterpart"


CAPACITY_MANIFESTS = ("capacity_grid_ep800.yaml", "capacity_in1k.yaml")
SMOKE_MANIFESTS = ("smoke_1gpu.yaml", "smoke_4gpu.yaml", "smoke_8gpu.yaml")


def _depths(manifest: str) -> set[int]:
    found = set()
    for run in expand(REPO / "configs" / "campaign" / manifest):
        for override in run.overrides:
            if override.startswith("model.encoder.depth="):
                found.add(int(override.split("=", 1)[1]))
    return found


def _experiments(manifest: str) -> set[str]:
    return {r.experiment for r in expand(REPO / "configs" / "campaign" / manifest)}


@pytest.mark.parametrize("manifest", CAPACITY_MANIFESTS)
def test_the_smoke_grids_rehearse_EVERY_depth_a_capacity_grid_runs(manifest: str):
    """The debug cycle (V2) exists to run a new shape through the whole lifecycle -- preflight,
    staging, training to max_epochs, the callbacks, postflight, the .verified marker -- before a wave
    commits GPU time to it. Depth is the capacity grids' new axis, and nothing had ever trained at
    anything but depth 12.

    The geometry test above structurally cannot catch an unrehearsed depth: it compares
    (batch, accumulation), which is identical at every depth, so a smoke cell that silently rehearsed
    depth 12 would pass every other assertion in this file while rehearsing nothing new.

    Inverted once by dropping `model.encoder.depth: 6` from `smoke_capacity_dep6`: fails with 6
    listed as unrehearsed.
    """
    wanted = _depths(manifest)
    assert wanted, f"{manifest} declares no depth axis -- has the capacity grid changed shape?"
    rehearsed = set().union(*(_depths(s) for s in SMOKE_MANIFESTS))
    assert not (wanted - rehearsed), \
        f"{manifest}: depth(s) {sorted(wanted - rehearsed)} are run but rehearsed by no smoke cell"


@pytest.mark.parametrize("manifest", CAPACITY_MANIFESTS)
def test_the_smoke_grids_rehearse_EVERY_capacity_experiment_config(manifest: str):
    """Each capacity grid uses experiment configs that exist only for it -- `ablations/base_depth`
    and `ablations/dynk_depth` on Imagenette, `lejepa/vit_in1k_localballreg_depth` on ImageNet. Every
    one needs a lifecycle rehearsal.

    This is what forces an IN1K smoke cell to exist at all: the depths themselves are already
    rehearsed on Imagenette at 1 GPU, so a depth-only check would pass with no ImageNet cell, and the
    combination of a reduced-depth encoder with the ImageNet data path under 4-GPU DDP would reach
    the cluster unexercised.

    Inverted once by removing `smoke_vit_in1k_capdep` from smoke_4gpu: fails naming
    `lejepa/vit_in1k_localballreg_depth`.
    """
    used = _experiments(manifest)
    rehearsed = set().union(*(_experiments(s) for s in SMOKE_MANIFESTS))
    assert not (used - rehearsed), f"{manifest}: unrehearsed config(s) {sorted(used - rehearsed)}"


@pytest.mark.parametrize("manifest", MANIFESTS, ids=[m.stem for m in MANIFESTS])
def test_every_manifest_is_submittable_as_ONE_array(manifest):
    """`submit_command` builds one homogeneous job array per grid and RAISES on a mixed GPU count,
    because an array gives every task the same resources -- an 8-GPU run in a 4-GPU array would get
    half of what it needs and fail at Lightning's device check.

    This was written after decision 26 nearly shipped one manifest holding the two 4-GPU controls and
    the 8-GPU ViT-L global ball. Nothing in the suite would have caught it; the first symptom would
    have been `campaign.py submit` refusing, after the configs and tests were written.

    Inverted by putting the ViT-L block back into `in1k_attribution`: this fails on that manifest.
    """
    runs = expand(manifest)
    counts = {r.gpus for r in runs}
    assert len(counts) == 1, f"{manifest.stem} mixes GPU counts {sorted(counts)}; split it per count"
    C.submit_command(manifest, runs, partition="b200-batch", account="a", time="24:00:00",
                     script="s.sh", log_dir="/tmp", exports={})


def test_the_two_controls_differ_from_their_own_local_arm_ONLY_in_the_regularizer():
    """Decision 26. A control is only a control if the recipe is held exactly.

    The hyperbolic and equivariant rows each have a Gaussian arm already, but it runs LeJEPA's
    published recipe and differs from the local arm in FIVE settings (lr 2e-3 -> 1e-3, lambda
    0.02 -> 0.1, warm-up 1 -> 10, lr_eta_min_ratio 0.5 -> 0.001, clipping none -> 1.0). That is why
    their recipe and target contributions are still bundled. These two runs hold all five at the local
    arm's values, so the only thing left moving is the regularizer.

    The regularizer must also be `sigreg_loss` with NO `num_clusters` key -- the absence is what makes
    it a control rather than a K=1 local run: there is no clustering code path at all.

    Inverted by setting `model.lamb: 0.02` on either block: fails naming lamb.
    """
    pairs = {"hvit_control": "lejepa/hvit_in1k_localballreg_l01_k4",
             "d4rn50_control": "lejepa/d4rn50_in1k_localballreg_l01_k4"}
    keys = ["optimizer.lr", "optimizer.weight_decay", "model.lamb", "model.warmup_epochs",
            "model.lr_eta_min_ratio", "trainer.gradient_clip_val", "data.crop_size",
            "model.embedding_dim", "model.projection_dim"]
    runs = expand(REPO / "configs" / "campaign" / "in1k_attribution.yaml")
    assert {r.tag for r in runs} == set(pairs)
    for run in runs:
        cfg = C.compose_config(run.hydra_overrides())
        ref = C.compose_config([f"experiment={pairs[run.tag]}"])
        diffs = [(k, OmegaConf.select(ref, k), OmegaConf.select(cfg, k))
                 for k in keys if OmegaConf.select(ref, k) != OmegaConf.select(cfg, k)]
        assert not diffs, f"{run.run_name}: recipe differs from its local arm: {diffs}"
        assert OmegaConf.select(cfg, "model.reg_loss._target_") == "src.losses.sigreg_loss"
        assert OmegaConf.select(cfg, "model.reg_loss.num_clusters") is None, \
            f"{run.run_name}: a control must have NO num_clusters, not a K of 1"


def test_the_vitl_trio_shares_one_recipe_so_its_two_steps_are_clean():
    """Decision 26. ViT-L needs no control because its pair is already recipe-matched (decision 13),
    so the global ball turns one number into two clean steps: sigreg -> ball K=1 is the TARGET, and
    ball K=1 -> ball K=4 is LOCALITY. That only holds if all three share every optimiser setting.

    lambda is 0.05 here, not the 0.1 the 128 px rows use -- LeJEPA's own value, carried by decision 13.
    Asserting it stops someone 'fixing' the global ball to 0.1 and silently breaking the comparison.

    Inverted by overriding `model.lamb: 0.1` on the global-ball block: fails naming lamb.
    """
    gb = expand(REPO / "configs" / "campaign" / "in1k_vitl14_globalball.yaml")
    assert len(gb) == 1 and gb[0].gpus == 8
    cfgs = [C.compose_config(["experiment=lejepa/vitl14_in1k_sigreg_paper"]),
            C.compose_config(gb[0].hydra_overrides()),
            C.compose_config(["experiment=lejepa/vitl14_in1k_localballreg_paper",
                              "model.reg_loss.num_clusters=4"])]
    for key in ("optimizer.lr", "optimizer.weight_decay", "model.lamb", "model.warmup_epochs",
                "model.lr_eta_min_ratio", "data.crop_size", "model.projection_dim"):
        vals = {OmegaConf.select(c, key) for c in cfgs}
        assert len(vals) == 1, f"the ViT-L trio disagrees on {key}: {vals}"
    assert float(OmegaConf.select(cfgs[1], "model.lamb")) == 0.05, "ViT-L keeps LeJEPA's lambda"
    assert int(OmegaConf.select(cfgs[1], "model.reg_loss.num_clusters")) == 1


def test_the_in1k_capacity_grid_matches_the_geometry_of_the_cells_it_compares_against():
    """Decision 25. The four new cells are read AGAINST two we already own --
    `vit_globalball_lam0.1_k1_s0` (depth 12, K=1) and `vit_local_lam0.1_k4_s0` (depth 12, K=4) -- so
    any difference in geometry would make the comparison meaningless rather than merely noisy.

    Both reference arms run `data.batch_size: 256` with `accumulate_grad_batches: 1` on 4 GPUs. The
    regularizer is computed PER SHARD (there is no all_gather in src/losses.py), so it sees N=64.
    That N is what sets the collapse cliff relative to the 16-d projector, and it must be identical.
    lambda is 0.1 on both references, so depth and K are the only things allowed to move.

    Inverted by setting `gpus: 2` in the manifest: the per-shard N becomes 128 and this fails.
    """
    runs = expand(REPO / "configs" / "campaign" / "capacity_in1k.yaml")
    assert len(runs) == 4
    refs = [r for m in ("in1k_local.yaml", "in1k_global_ball.yaml")
            for r in expand(REPO / "configs" / "campaign" / m)]
    ref_geo = {(r.gpus, r.micro_batch) for r in refs}
    assert ref_geo == {(4, 64)}, f"the reference arms' geometry moved: {ref_geo}"

    depths, ks = set(), set()
    for run in runs:
        geo = C.geometry(run, world_size=run.gpus)
        assert (run.gpus, geo["micro_batch_per_gpu"]) == (4, 64), f"{run.run_name}: {geo}"
        assert geo["effective_batch"] == 256, f"{run.run_name}: {geo}"
        cfg = C.compose_config(run.hydra_overrides())
        assert float(cfg.model.lamb) == 0.1, f"{run.run_name}: lambda must match both references"
        assert int(cfg.model.embedding_dim) == 512, f"{run.run_name}: the probe width must not move"
        assert int(cfg.model.projection_dim) == 16, \
            f"{run.run_name}: the projected dim sets the collapse cliff; depth must not move it"
        assert int(cfg.trainer.max_epochs) == 100, f"{run.run_name}: the references are 100 epochs"
        depths.add(int(cfg.model.encoder.depth))
        ks.add(int(cfg.model.reg_loss.num_clusters))
    assert depths == {6, 3}, f"depths {sorted(depths)}"
    assert ks == {1, 4}, f"K values {sorted(ks)} -- must match the two reference arms"


def test_the_in1k_capacity_grid_omits_depth_12_because_it_is_already_trained():
    """depth 12 at K=1 and K=4 are `vit_globalball_lam0.1_k1_s0` and `vit_local_lam0.1_k4_s0`, both
    trained and evaluated. Re-running them would cost ~73 GPU-h to reproduce numbers we hold and
    would create a second depth-12 result that could disagree with the first."""
    runs = expand(REPO / "configs" / "campaign" / "capacity_in1k.yaml")
    assert not any("depth=12" in o for r in runs for o in r.overrides)
    assert {r.tag for r in runs} == {"vit_capdep"}


def test_the_in1k_capacity_tag_begins_with_a_backbone_family_prefix():
    """`sync_eval_configs.py:family()` matches `tag == prefix or tag.startswith(prefix + '_')` over
    ("vitl14","d4rn50","hvit","vit","svit","nvit") and RAISES otherwise -- at MODULE IMPORT, where
    `RENDERED = SYNC.render_all()` runs, so an unmatched tag errors collection of the whole
    eval-config test file rather than failing one assertion. A tag like `cap1k_vit` would do exactly
    that; `vit_capdep` resolves to family `vit`.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_sync_for_family", REPO / "scripts/aicr/sync_eval_configs.py")
    sync = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sync
    spec.loader.exec_module(sync)
    for run in expand(REPO / "configs" / "campaign" / "capacity_in1k.yaml"):
        assert sync.family(run) == "vit", run.tag


def test_no_capacity_run_name_can_collide_with_any_other_grid():
    """F56: two runs sharing a name share a checkpoint directory and resume from each other, which
    cost four runs on 2026-09-23. Depth must therefore appear in the name -- without it, the depth-6
    and depth-3 cells of one (lambda, K, seed) would be the same string.
    """
    capacity = {r.run_name for r in expand(REPO / "configs" / "campaign" / "capacity_grid_ep800.yaml")}
    assert len(capacity) == 180
    for other in ("ablation_grid_ep800.yaml", "ablation_grid_ep800_d4.yaml", "ablation_grid.yaml"):
        names = {r.run_name for r in expand(REPO / "configs" / "campaign" / other)}
        assert not (capacity & names), f"collides with {other}: {sorted(capacity & names)[:5]}"
    assert all(n.startswith("cap_") for n in capacity), \
        "the cap_ prefix is what make_ablation_tables.py --tag-prefix reaches"
    assert sum("dep6" in n for n in capacity) == 90
    assert sum("dep3" in n for n in capacity) == 90


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

