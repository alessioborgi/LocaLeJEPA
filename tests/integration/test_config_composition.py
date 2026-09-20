"""Every experiment config must compose and instantiate.

A broken config is the cheapest possible failure to catch and the most expensive to discover
late: a typo in a YAML key surfaces only when that experiment is launched, which on a saturated
cluster can be hours after submission. There are 56 experiment configs on this branch and
nothing has ever checked that they all still build.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"

EXPERIMENTS = sorted(
    str(p.relative_to(CONFIG_DIR / "experiment").with_suffix(""))
    for p in (CONFIG_DIR / "experiment").rglob("*.yaml")
)

# escnn is deliberately not installed (see the coverage omit list), so the D4-equivariant
# encoder cannot be instantiated here. Composition is still checked; only instantiation is not.
NEEDS_ESCNN = ("d4rn50", "e2resnet", "d4resnet")


def compose_experiment(name: str):
    os.environ.setdefault("PROJECT_ROOT", str(REPO))
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            return compose(config_name="train", overrides=[f"experiment={name}"])
    finally:
        GlobalHydra.instance().clear()


def test_experiment_configs_were_discovered():
    """Guard against the glob silently matching nothing -- a checker that matches nothing is the
    most common way a test suite lies (research_rules/00)."""
    assert len(EXPERIMENTS) >= 10, f"expected many experiment configs, found {EXPERIMENTS}"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_config_composes(name):
    """Resolution included: ``${model.embedding_dim}`` style interpolations must resolve, which
    is where a renamed key actually bites."""
    cfg = compose_experiment(name)
    assert cfg.model is not None and cfg.data is not None and cfg.trainer is not None


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_declares_a_regularizer_and_an_encoder(name):
    cfg = compose_experiment(name)
    assert "_target_" in cfg.model.reg_loss
    assert "_target_" in cfg.model.encoder


@pytest.mark.parametrize("name", [n for n in EXPERIMENTS if not any(s in n for s in NEEDS_ESCNN)])
def test_every_regularizer_instantiates(name):
    """The regularizer is a partial, so building it catches a bad keyword that composition alone
    would let through."""
    import hydra

    cfg = compose_experiment(name)
    loss_fn = hydra.utils.instantiate(cfg.model.reg_loss)
    assert callable(loss_fn)


def test_the_local_configs_bind_a_cluster_count():
    """``local_*`` regularizers need ``num_clusters``; without it they would fall back to the
    default of 8 rather than the k the experiment's name claims."""
    for name in EXPERIMENTS:
        if "localballreg" not in name and "localsigreg" not in name:
            continue
        cfg = compose_experiment(name)
        assert "num_clusters" in cfg.model.reg_loss, f"{name} does not pin k"


def test_batch_size_is_never_raised_above_64_for_local_regularizers():
    """For every ``local_*`` config the k-means sees the **per-step** batch, so batch size is a
    method parameter, not a memory knob. Pinned so a well-meaning speed-up on a larger GPU shows
    up as a failing test rather than as a silently different method.
    """
    for name in EXPERIMENTS:
        if "local" not in name:
            continue
        cfg = compose_experiment(name)
        assert cfg.data.batch_size <= 256, (
            f"{name} uses batch_size={cfg.data.batch_size}; raising it changes what the k-means "
            "clusters and therefore changes the method"
        )


def test_headline_pair_differs_only_where_documented():
    """The baseline/ours pair for the plain ViT on ImageNet-1K. Pins the seven differences that
    `state/02-METHOD-AND-CLAIMS.md` records, so that a change to any of them is visible."""
    base = compose_experiment("lejepa/vit_in1k_sigreg")
    ours = compose_experiment("lejepa/vit_in1k_localballreg_l01_k4")

    assert base.optimizer.lr == 2e-3 and ours.optimizer.lr == 1e-3
    assert ours.model.warmup_epochs == 10
    assert ours.trainer.get("gradient_clip_val") == 1.0
    assert base.trainer.get("gradient_clip_val") is None
    assert base.model.lamb == 0.02 and ours.model.lamb == 0.1
    assert base.model.lr_eta_min_ratio == 0.5 and ours.model.lr_eta_min_ratio == 1.0
    assert base.data.batch_size == ours.data.batch_size, "the batch is the same"
    assert base.model.embedding_dim == ours.model.embedding_dim


def test_both_arms_resolve_to_the_same_absolute_learning_rate_floor():
    """0.5 x 2e-3 and 1.0 x 1e-3 are both 1e-3 -- the historical value the ratios were pinned to
    reproduce. Stated here because it explains why the ratios differ at all."""
    base = compose_experiment("lejepa/vit_in1k_sigreg")
    ours = compose_experiment("lejepa/vit_in1k_localballreg_l01_k4")
    assert base.model.lr_eta_min_ratio * base.optimizer.lr == pytest.approx(1e-3)
    assert ours.model.lr_eta_min_ratio * ours.optimizer.lr == pytest.approx(1e-3)
