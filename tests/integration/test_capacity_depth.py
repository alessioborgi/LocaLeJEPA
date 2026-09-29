"""The capacity axis: transformer depth must move capacity and nothing else.

Decision 24 tests whether locality pays once the encoder can no longer satisfy the global
constraint, by varying the number of transformer blocks. The whole comparison rests on two
properties that are cheap to assert here and expensive to discover after 310 GPU-hours:

1. **depth 12 explicit == no depth key at all.** The capacity grid's depth-6 and depth-3 cells are
   compared against the existing 800-epoch grid's depth-12 cells, which were trained through
   `ablations/base` and never carried a `depth` key. If passing `depth=12` to timm changed the
   architecture even slightly, that comparison would be between two different experiments.
2. **depth must not move the embedding width.** The collapse cliff sits at roughly `d` samples per
   neighbourhood, so a depth change that also moved the projected dimension would shift the cliff
   underneath the experiment and confound the very axis being measured.

It also pins the F58 trap: `model.encoder.depth` is reachable by an ORDINARY override. If the key
were not declared in a config, the manifest would need `+model.encoder.depth`, whose leading `+` is
not part of the key path -- so `campaign.py`'s `override_map()` would have to strip it and the
run-name builder would carry it into the `short:` alias.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"

REFERENCE = "ablations/base"
CAPACITY = "ablations/base_depth"
CAPACITY_RAMP = "ablations/dynk_depth"

# `vit_small_patch8_224` is depth 12; this is the value `base_depth.yaml` declares so that the key
# exists without changing the model.
REFERENCE_DEPTH = 12


def compose_experiment(name: str, overrides: tuple[str, ...] = ()):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            return compose(config_name="train", overrides=[f"experiment={name}", *overrides])
    finally:
        GlobalHydra.instance().clear()


def test_the_capacity_configs_exist():
    """A checker that matches nothing is the most common way a test suite lies
    (research_rules/00)."""
    for name in (CAPACITY, CAPACITY_RAMP):
        assert (CONFIG_DIR / "experiment" / f"{name}.yaml").is_file(), f"missing {name}.yaml"


def test_depth_twelve_builds_the_same_encoder_as_the_reference_ablation_base():
    """THE load-bearing test of the capacity axis, and the reason `base_depth.yaml` declares 12.

    The depth-12 reference cells of the capacity grid are NOT re-run: they are the existing
    800-epoch grid's cells, trained through `ablations/base`, which has no `depth` key. Comparing
    depth-6 and depth-3 cells against them is only legitimate if handing timm an explicit
    `depth=12` reproduces the architecture it builds by default.

    Asserted on the state_dict's keys, shapes and total parameter count rather than on a forward
    pass, because a structural difference is what would invalidate the comparison and a numerical
    tolerance would be beside the point. No tolerance is needed for an equality of shapes, which
    makes this a stronger assertion than any threshold.

    INVERTED once: changing `base_depth.yaml` to `depth: 6` fails on the parameter count
    (11,021,312 against 21,664,640) and on the key list.
    """
    import hydra

    reference = hydra.utils.instantiate(compose_experiment(REFERENCE).model.encoder)
    explicit = hydra.utils.instantiate(compose_experiment(CAPACITY).model.encoder)

    ref_sd, exp_sd = reference.state_dict(), explicit.state_dict()
    assert list(ref_sd) == list(exp_sd), "explicit depth=12 changed the parameter NAMES"
    assert all(ref_sd[k].shape == exp_sd[k].shape for k in ref_sd), \
        "explicit depth=12 changed a parameter SHAPE"
    assert sum(p.numel() for p in reference.parameters()) \
        == sum(p.numel() for p in explicit.parameters())


@pytest.mark.parametrize("depth", [12, 6, 3])
def test_depth_moves_the_block_count_and_leaves_the_embedding_at_512(depth: int):
    """Depth must be the ONLY thing that moves.

    The embedding width is pinned by `timm_vit.yaml`'s `num_classes: ${model.embedding_dim}`, so
    timm's own head keeps the encoder output at 512 whatever the block count. That is what lets the
    projector, the online probe and every eval path stay untouched -- and it keeps the collapse
    cliff (~`d` samples per neighbourhood, with d the PROJECTED dimension) where the reference grid
    put it, instead of moving it underneath the capacity axis.

    INVERTED once: dropping `num_classes` from `timm_vit.yaml` makes the output 384-d (the backbone
    width) at every depth and fails the width assertion.
    """
    import hydra
    import torch

    cfg = compose_experiment(CAPACITY, (f"model.encoder.depth={depth}",))
    encoder = hydra.utils.instantiate(cfg.model.encoder)

    assert len(encoder.blocks) == depth, f"asked for {depth} blocks, built {len(encoder.blocks)}"
    with torch.no_grad():
        out = encoder(torch.zeros(1, 3, int(cfg.data.crop_size), int(cfg.data.crop_size)))
    assert out.shape == (1, int(cfg.model.embedding_dim)), \
        f"depth {depth} changed the embedding width to {tuple(out.shape)}"
    assert int(cfg.model.projection_dim) == 16, \
        "the projected dimension sets the collapse cliff; depth must not move it"


def test_depth_is_reachable_by_an_ordinary_override_and_needs_no_hydra_plus():
    """F58's trap, closed by construction.

    `depth` is absent from `configs/model/encoder/timm_vit.yaml`, so a bare CLI override of it
    would be rejected by Hydra's struct mode and would need `+model.encoder.depth=6`. A leading `+`
    is not part of the key path: `campaign.py:86-95 override_map()` strips it so postflight can look
    the key up, and the run-name builder resolves its `short:` alias on the RAW key, so the alias
    would have to contain the `+` too. Declaring the key in the experiment config avoids all of it.

    INVERTED once: removing `model.encoder.depth` from `base_depth.yaml` raises
    ConfigAttributeError / ConfigCompositionException on the plain override.
    """
    cfg = compose_experiment(CAPACITY, ("model.encoder.depth=6",))
    assert int(cfg.model.encoder.depth) == 6


def test_the_reference_experiment_still_has_no_depth_key():
    """`timm_vit.yaml` must stay untouched -- it is shared by 22 experiment configs, and adding a
    key there would change the composed config of every one of them, including the ImageNet arms
    already trained. The capacity axis is additive: a new experiment config, nothing else.

    INVERTED once: adding `depth: 12` to `timm_vit.yaml` makes the key present here and fails.
    """
    cfg = compose_experiment(REFERENCE)
    assert "depth" not in cfg.model.encoder, \
        "the shared encoder config grew a `depth` key; 22 experiments compose through it"


def test_the_capacity_ramp_config_carries_both_the_depth_key_and_the_forward_ramp():
    """The `cap_fwd` block needs both, and `ablations/dynk` derives from `base`, not `base_depth` --
    so a depth-aware ramp config has to exist or the forward-ramp cells would silently train at
    depth 12.

    The ramp direction matters: `DynamicNumClusters` counts DOWN when `end < start`, so a swapped
    pair would run the reverse ramp, which collapsed in the reference grid.

    INVERTED once: pointing `dynk_depth.yaml`'s defaults at `/experiment/ablations/base` drops the
    depth key and fails the first assertion.
    """
    cfg = compose_experiment(CAPACITY_RAMP)
    assert int(cfg.model.encoder.depth) == REFERENCE_DEPTH
    ramp = cfg.callbacks.dynamic_num_clusters
    assert (int(ramp.start), int(ramp.end)) == (2, 16), "cap_fwd must ramp UP, 2 -> 16"
    assert int(ramp.step) == 2
