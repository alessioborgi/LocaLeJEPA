"""The two k-means backends and their shared input validation.

Clustering is the part of ``local_loss`` with no closed form to check against, so these tests
pin structural invariants instead: valid label ranges, a genuine partition, determinism under a
seed, and the empty-cluster path that a naive implementation gets wrong.
"""

from __future__ import annotations

import pytest
import torch

from src.losses import (
    _fast_pytorch_kmeans_labels,
    _torch_kmeans_labels,
    _validate_kmeans_input,
)

BACKENDS = [_torch_kmeans_labels, _fast_pytorch_kmeans_labels]
BACKEND_IDS = ["torch", "fast_pytorch_kmeans"]


# --------------------------------------------------------------------------------------
# shared validation
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(16,), (4, 16, 8), (2, 3, 4, 5)])
def test_validation_rejects_anything_but_a_2d_point_cloud(shape):
    with pytest.raises(ValueError, match=r"2D point cloud \(N, D\)"):
        _validate_kmeans_input(torch.randn(*shape), 4)


def test_validation_rejects_an_empty_cloud():
    with pytest.raises(ValueError, match="at least one point"):
        _validate_kmeans_input(torch.randn(0, 8), 4)


def test_k_is_clipped_to_the_sample_count():
    """Asking for more clusters than points is not an error -- k is clipped to N, which is what
    keeps ``local_loss`` working on a short final batch."""
    _, _, k = _validate_kmeans_input(torch.randn(3, 8), num_clusters=10)
    assert k == 3


def test_k_is_clipped_up_to_at_least_one():
    for requested in (0, -5):
        _, _, k = _validate_kmeans_input(torch.randn(16, 8), num_clusters=requested)
        assert k == 1


def test_validation_returns_the_observed_shape():
    n, d, k = _validate_kmeans_input(torch.randn(32, 7), num_clusters=4)
    assert (n, d, k) == (32, 7, 4)


# --------------------------------------------------------------------------------------
# both backends: structural guarantees
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_labels_are_in_range_and_cover_every_point(backend):
    x = torch.randn(64, 16)
    labels = backend(x, num_clusters=4)
    assert labels.shape == (64,)
    assert labels.dtype == torch.long
    assert int(labels.min()) >= 0 and int(labels.max()) < 4


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_labels_partition_the_batch(backend):
    """Every point lands in exactly one cluster, so the bincount must sum to N. A backend that
    dropped or duplicated points would break ``local_loss``'s disjointness silently."""
    labels = backend(torch.randn(64, 16), num_clusters=4)
    assert int(torch.bincount(labels, minlength=4).sum()) == 64


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_more_clusters_requested_than_points(backend):
    """k is clipped to N, so each of the three points gets its own cluster."""
    labels = backend(torch.randn(3, 8), num_clusters=10)
    assert labels.shape == (3,) and int(labels.max()) < 3


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_single_cluster_puts_everything_in_label_zero(backend):
    labels = backend(torch.randn(32, 8), num_clusters=1)
    assert int(labels.min()) == 0 and int(labels.max()) == 0


def _three_blobs(seed: int, d: int = 8) -> torch.Tensor:
    """Three blobs 50 units apart along orthogonal axes -- about as separable as data gets."""
    torch.manual_seed(seed)
    centres = torch.eye(3, d) * 50.0
    return torch.cat([centres[i] + 0.1 * torch.randn(20, d) for i in range(3)])


def _partition_quality(backend, seeds: int = 100) -> dict[str, int]:
    """Count, over many seeds, how often the clustering recovers the obvious answer."""
    stats = {"ideal": 0, "split": 0, "collapsed": 0}
    for seed in range(seeds):
        x = _three_blobs(seed)
        torch.manual_seed(seed)
        labels = backend(x, num_clusters=3)
        pure = all(len(labels[b * 20:(b + 1) * 20].unique()) == 1 for b in range(3))
        n_clusters = len(labels.unique())
        stats["ideal"] += int(pure and n_clusters == 3)
        stats["split"] += int(not pure)
        stats["collapsed"] += int(n_clusters < 3)
    return stats


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_clustering_is_initialisation_sensitive_even_on_trivially_separable_data(backend):
    """FINDING, pinned deliberately.

    Both backends seed centroids from a random subset of points and run a fixed ten Lloyd
    iterations. On three blobs **50 units apart** -- unambiguous by any measure -- they recover
    the obvious partition only about **80%** of the time. Measured over 100 seeds: roughly 15%
    of runs produce fewer clusters than requested (two centroids land in one blob, leaving
    another unclaimed) and roughly 5% split a single blob across two labels.

    This matters because the neighbourhoods are the method: ``local_loss`` applies its test
    *within* clusters, so a partition driven by initialisation rather than structure changes
    what the regularizer measures. It also compounds the ``min_cluster_size`` finding in
    test_losses_local.py -- the configured ``num_clusters`` is an upper bound for two
    independent reasons, and neither is reported at runtime.

    The band below is deliberately wide. **If this test fails because the rate went up, the
    initialisation has been improved** (k-means++ or more iterations) -- that is a welcome change
    and a deliberate one, and results will have moved with it, so the test should be updated as
    part of that decision rather than loosened in advance.
    """
    stats = _partition_quality(backend)
    assert 60 <= stats["ideal"] <= 95, (
        f"expected the known ~80% recovery rate, got {stats}; a rate near 100 means the "
        f"initialisation changed, a rate below 60 means it regressed"
    )
    assert stats["collapsed"] > 0, "at least some seeds should fail to form all three clusters"


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
def test_a_good_initialisation_recovers_the_blobs_exactly(backend):
    """The positive control: when the initialisation does happen to place one centroid per blob,
    the assignment step is correct -- three pure clusters, no splits. Seed 0 is such a case for
    both backends, so this isolates assignment quality from initialisation luck.
    """
    x = _three_blobs(0)
    torch.manual_seed(0)
    labels = backend(x, num_clusters=3)
    assert len(labels.unique()) == 3
    for block in range(3):
        assert len(labels[block * 20:(block + 1) * 20].unique()) == 1


# --------------------------------------------------------------------------------------
# the in-repo torch fallback specifically
# --------------------------------------------------------------------------------------

def test_torch_backend_is_deterministic_under_a_seed():
    """Centroids are initialised from a random permutation, so the result is seed-dependent --
    but it must be reproducible, or a re-run of the same job would cluster differently."""
    x = torch.randn(64, 16)
    torch.manual_seed(42)
    first = _torch_kmeans_labels(x, num_clusters=4)
    torch.manual_seed(42)
    second = _torch_kmeans_labels(x, num_clusters=4)
    assert torch.equal(first, second)


def test_torch_backend_survives_an_empty_cluster():
    """Duplicated points make it easy for a centroid to end up owning nothing. The update must
    leave such a centroid where it was rather than dividing by a zero count and producing NaNs
    that would silently poison every subsequent assignment.
    """
    x = torch.cat([torch.zeros(8, 4), torch.ones(8, 4)])
    labels = _torch_kmeans_labels(x, num_clusters=8, iters=5)
    assert torch.isfinite(x).all()
    assert int(labels.min()) >= 0 and int(labels.max()) < 8


def test_torch_backend_iters_is_floored_at_one():
    """``max(int(iters), 1)`` -- zero or negative iterations must still produce an assignment
    rather than returning the zero-initialised label buffer."""
    x = torch.cat([torch.zeros(16, 4) - 5.0, torch.zeros(16, 4) + 5.0])
    labels = _torch_kmeans_labels(x, num_clusters=2, iters=0)
    assert len(labels.unique()) == 2, "one Lloyd step must still have happened"


def test_torch_backend_normalize_changes_the_geometry():
    """normalize=True clusters by direction, so two points along the same ray belong together
    however different their magnitudes. With normalize=False they need not."""
    x = torch.tensor([[1.0, 0.0], [100.0, 0.0], [0.0, 1.0], [0.0, 100.0]])
    torch.manual_seed(0)
    cosine = _torch_kmeans_labels(x, num_clusters=2, normalize=True)
    assert cosine[0] == cosine[1] and cosine[2] == cosine[3], (
        "cosine clustering must group by direction, not magnitude"
    )
    assert cosine[0] != cosine[2]


def test_fast_backend_reports_a_useful_error_when_missing(monkeypatch):
    """The ImportError is caught and re-raised with the config switch that works around it. If
    the package were absent on a cluster, this message is what the user would see."""
    import builtins

    real_import = builtins.__import__

    def fail(name, *args, **kwargs):
        if name == "fast_pytorch_kmeans":
            raise ImportError("simulated absence")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    with pytest.raises(ImportError, match="model.reg_loss.kmeans_backend=torch"):
        _fast_pytorch_kmeans_labels(torch.randn(16, 8), num_clusters=2)
