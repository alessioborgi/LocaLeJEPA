"""Realised K reaches the logger: ``local_loss`` reports through ``stats`` and the module logs it
under ``train/local/*`` -- the column every ablation table's K axis is read from (plan C1).

Without it the K axis of an ablation is the *requested* K. At batch 64 with ``min_cluster_size``
4, K=16 runs with about ten clusters and K=32 with none, and the tables would say otherwise.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch import nn

from conftest import TagEncoder, tagged_views
from src.losses import local_ballreg_loss, local_loss, sigreg_loss
from src.models.lejepa_module import LeJEPAModule, _accepts_keyword

LOCAL_KEYS = {"local/configured_clusters", "local/effective_clusters",
              "local/kept_fraction", "local/fallback"}


def _inner(proj: torch.Tensor) -> torch.Tensor:
    return proj.float().square().mean()


# Euclidean k-means: the tag encoder's embeddings are collinear, and cosine clustering would put
# every sample in one cluster. Two clusters of at least two samples out of eight.
LOCAL = partial(local_loss, loss=_inner, num_clusters=2, min_cluster_size=2,
                normalize_for_kmeans=False, kmeans_backend="torch")


def build(reg_loss) -> LeJEPAModule:
    module = LeJEPAModule(
        encoder=TagEncoder(4), optimizer=torch.optim.AdamW, projector=nn.Identity(),
        online_probe=nn.Linear(4, 3), embedding_dim=4, projection_dim=4, num_classes=3,
        reg_loss=reg_loss,
    )
    module.log = lambda *a, **k: None
    return module


def test_a_local_regularizer_reports_under_the_local_namespace():
    _, metrics = build(LOCAL).model_step(tagged_views(8, 2))

    assert LOCAL_KEYS <= set(metrics), sorted(metrics)
    assert metrics["local/configured_clusters"] == 2.0
    assert metrics["local/effective_clusters"] in (1.0, 2.0)
    assert 0.0 < metrics["local/kept_fraction"] <= 1.0
    assert metrics["local/fallback"] == 0.0


def test_a_global_regularizer_reports_nothing_local():
    """An empty column must mean "not a local run", never "K was as configured"."""
    _, metrics = build(sigreg_loss).model_step(tagged_views(8, 2))
    assert not [key for key in metrics if key.startswith("local/")]


def test_the_fallback_is_visible_in_the_metrics():
    """``min_cluster_size`` above the batch: every cluster is dropped and the global test runs.
    Before C1 this was indistinguishable from a healthy local step (ledger C0-1)."""
    _, metrics = build(partial(LOCAL, min_cluster_size=100)).model_step(tagged_views(8, 2))
    assert metrics["local/fallback"] == 1.0
    assert metrics["local/effective_clusters"] == 0.0
    assert metrics["local/kept_fraction"] == 0.0


def test_training_step_logs_the_local_metrics_with_the_train_prefix():
    """What ``metrics.csv`` will hold: the keys are ``train/local/<name>``, the same series the
    campaign's realised-K reader looks for."""
    module = build(LOCAL)
    logged: dict[str, object] = {}
    module.log = lambda name, value, **kwargs: logged.__setitem__(name, value)

    module.training_step((tagged_views(8, 2), torch.zeros(8, dtype=torch.long)), 0)

    assert {f"train/{key}" for key in LOCAL_KEYS} <= set(logged), sorted(logged)


def test_dynamic_k_rewrapping_keeps_the_report():
    """``DynamicNumClusters`` replaces ``pl_module.reg_loss`` with a partial of itself every
    epoch. The detection is made once at construction; it must survive that re-wrapping, or the
    dynamic-K rows would lose exactly the column that explains them."""
    module = build(LOCAL)
    module.reg_loss = partial(module.reg_loss, num_clusters=2)   # what the callback does

    _, metrics = module.model_step(tagged_views(8, 2))
    assert LOCAL_KEYS <= set(metrics)


@pytest.mark.parametrize("fn, expected", [
    (sigreg_loss, False),
    (local_ballreg_loss, True),
    (partial(local_ballreg_loss, num_clusters=2), True),
    (partial(partial(local_ballreg_loss, num_clusters=2), num_clusters=4), True),
    (lambda proj, **kwargs: proj, False),          # swallows it; would report nothing
    (lambda proj, stats=None: proj, True),
    (42, False),                                   # not callable at all
])
def test_accepts_keyword_reads_the_declared_signature(fn, expected):
    assert _accepts_keyword(fn, "stats") is expected
