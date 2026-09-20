"""``local_loss`` -- the proposed method, and the file with the most silent-wrongness surface.

Two behaviours here are pinned as **findings, not endorsements**: the fallback to the global
test, and the gap between the configured and effective cluster count. Both are real and both
would change results if altered. Until 2026-09-20 both were also *unreported*; the campaign plan's
C1 made them observable through an opt-in ``stats`` out-param, and the tests that pinned the
silence were inverted in the same commit (the standing rule: a fix that pins a finding inverts
its test). The behaviours themselves are unchanged and still pinned.
"""

from __future__ import annotations

import inspect
from functools import partial

import pytest
import torch

from src.losses import ballreg_loss, local_ballreg_loss, local_loss, local_sigreg_loss, sigreg_loss


class Spy:
    """Records the subset size of every call, so the *partition* can be asserted, not just the
    returned scalar. Returns a differentiable zero so gradient tests still work."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, proj: torch.Tensor) -> torch.Tensor:
        self.sizes.append(int(proj.shape[1]))
        return proj.float().sum() * 0.0


TORCH_BACKEND = dict(kmeans_backend="torch")


# --------------------------------------------------------------------------------------
# the silent fallback -- ledger finding C0-1
# --------------------------------------------------------------------------------------

def test_falls_back_to_the_global_test_when_every_cluster_is_too_small():
    """FINDING, pinned deliberately.

    With N=8, k=4 and min_cluster_size=4 the clusters average two samples, so every one is
    dropped and ``local_loss`` calls the inner test **once on all N samples** -- i.e. the global
    test. It returns a bare scalar: no flag, no log, no warning, indistinguishable from a genuine
    local computation.

    This is why a cheap smoke test at data.batch_size=8 verifies none of the local code path.
    """
    spy = Spy()
    local_loss(torch.randn(4, 8, 16), loss=spy, num_clusters=4, min_cluster_size=4,
               **TORCH_BACKEND)
    assert spy.sizes == [8], (
        "expected a single call covering the whole batch, i.e. the global fallback; "
        f"got calls on subsets {spy.sizes}"
    )


def test_does_not_fall_back_at_the_real_training_batch_size():
    """The counterpart: at the batch the ImageNet-1K configs actually use (64) with k=4, all
    four clusters clear the minimum and the local path is genuinely taken."""
    spy = Spy()
    local_loss(torch.randn(4, 64, 16), loss=spy, num_clusters=4, min_cluster_size=4,
               **TORCH_BACKEND)
    assert len(spy.sizes) == 4
    assert sum(spy.sizes) == 64, "the accepted clusters must partition the batch"
    assert min(spy.sizes) >= 4


def test_the_fallback_is_reported_through_stats_and_the_return_is_unchanged():
    """INVERTED 2026-09-20 (C1). This test pinned the *absence* of a signal: the fallback
    returned a bare scalar and nothing else. The signal now exists, opt-in -- pass ``stats={}``
    and the fallback is written there, with the realised count (zero surviving clusters) and the
    kept fraction (nothing kept locally). The return type is unchanged on purpose: every caller
    and every saved config keeps working, and a caller that does not ask still gets nothing."""
    stats: dict[str, float] = {}
    out = local_loss(torch.randn(4, 8, 16), loss=Spy(), num_clusters=4, min_cluster_size=4,
                     stats=stats, **TORCH_BACKEND)
    assert isinstance(out, torch.Tensor) and out.shape == ()
    assert stats == {"configured_clusters": 4.0, "effective_clusters": 0.0,
                     "kept_fraction": 0.0, "fallback": 1.0}


# --------------------------------------------------------------------------------------
# effective k vs configured k -- the second finding
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("k, min_expected", [(2, 2), (4, 4)])
def test_small_k_keeps_every_cluster_at_batch_64(k, min_expected):
    """k=2 and k=4 -- the settings the reported runs use -- lose no clusters at N=64."""
    worst = min(_clusters_used(k, n=64, seed=s) for s in range(20))
    assert worst == min_expected


def test_large_k_loses_clusters_at_batch_64_and_stats_say_so():
    """FINDING kept, its SILENCE inverted 2026-09-20 (C1).

    ``num_clusters`` is an upper bound, not the effective count: clusters below
    ``min_cluster_size`` are skipped. At N=64 a configured k=16 runs with roughly ten, so an
    ablation sweeping k past ~8 at this batch size is not sweeping what its label says -- which
    is why the campaign's tables carry a realised-K column. The report must agree with an
    independent count of the inner-loss calls (the Spy), seed by seed.
    """
    used, reported = [], []
    for seed in range(20):
        torch.manual_seed(seed)
        spy, stats = Spy(), {}
        local_loss(torch.randn(4, 64, 16), loss=spy, num_clusters=16, min_cluster_size=4,
                   stats=stats, **TORCH_BACKEND)
        used.append(0 if spy.sizes == [64] else len(spy.sizes))
        reported.append(int(stats["effective_clusters"]))
    assert max(used) < 16, "expected fewer effective clusters than configured"
    assert min(used) >= 2, "but not so few that the statistic degenerates"
    assert reported == used, "stats must agree with an independent count of the inner calls"


def _clusters_used(k: int, n: int, seed: int) -> int:
    torch.manual_seed(seed)
    spy = Spy()
    local_loss(torch.randn(4, n, 16), loss=spy, num_clusters=k, min_cluster_size=4,
               **TORCH_BACKEND)
    # a single call spanning the whole batch is the global fallback, not one accepted cluster
    if spy.sizes == [n]:
        return 0
    return len(spy.sizes)


# --------------------------------------------------------------------------------------
# weighting
# --------------------------------------------------------------------------------------

def test_unweighted_gives_every_surviving_cluster_equal_say():
    """Default weighted=False means a 30-sample cluster counts the same as a 5-sample one."""
    sizes = {}

    def loss(proj: torch.Tensor) -> torch.Tensor:
        n = int(proj.shape[1])
        sizes[n] = sizes.get(n, 0) + 1
        return torch.tensor(float(n))          # value equals the cluster size

    out = local_loss(torch.randn(2, 64, 16), loss=loss, num_clusters=4, weighted=False,
                     **TORCH_BACKEND)
    observed = [n for n, count in sizes.items() for _ in range(count)]
    assert float(out) == pytest.approx(sum(observed) / len(observed))


def test_weighted_averages_by_cluster_size():
    """weighted=True weights each cluster by its share of the accepted samples, so the result is
    sum(n_i^2)/sum(n_i) when the inner loss returns the cluster size."""
    observed: list[int] = []

    def loss(proj: torch.Tensor) -> torch.Tensor:
        observed.append(int(proj.shape[1]))
        return torch.tensor(float(proj.shape[1]))

    out = local_loss(torch.randn(2, 64, 16), loss=loss, num_clusters=4, weighted=True,
                     **TORCH_BACKEND)
    total = sum(observed)
    assert float(out) == pytest.approx(sum(n * n for n in observed) / total, rel=1e-5)


# --------------------------------------------------------------------------------------
# gradients and detachment
# --------------------------------------------------------------------------------------

def test_gradients_reach_the_projections():
    proj = torch.randn(4, 64, 16, requires_grad=True)
    local_loss(proj, loss=lambda p: ballreg_loss(p, center=True), num_clusters=4,
               **TORCH_BACKEND).backward()
    assert proj.grad is not None and float(proj.grad.abs().sum()) > 0.0


def test_the_clustering_input_is_detached(monkeypatch):
    """No gradient may flow through the k-means assignment. Capture what the backend receives
    and assert it carries no graph -- a dropped ``.detach()`` would make the labels
    differentiable and silently change the optimisation."""
    seen = {}
    import src.losses as losses_module
    original = losses_module._torch_kmeans_labels

    def capture(x, num_clusters, **kwargs):
        seen["requires_grad"] = x.requires_grad
        seen["grad_fn"] = x.grad_fn
        return original(x, num_clusters, **kwargs)

    monkeypatch.setattr(losses_module, "_torch_kmeans_labels", capture)
    proj = torch.randn(4, 64, 16, requires_grad=True)
    local_loss(proj, loss=lambda p: ballreg_loss(p), num_clusters=4, **TORCH_BACKEND)

    assert seen["requires_grad"] is False and seen["grad_fn"] is None


# --------------------------------------------------------------------------------------
# validation and dispatch
# --------------------------------------------------------------------------------------

def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expects proj with shape \(V, N, D\)"):
        local_loss(torch.randn(8, 16), loss=sigreg_loss)


def test_rejects_empty_batch():
    with pytest.raises(ValueError, match="at least one sample"):
        local_loss(torch.randn(4, 0, 16), loss=sigreg_loss)


@pytest.mark.parametrize("num_global", [0, 5, -1])
def test_rejects_num_global_outside_the_view_range(num_global):
    with pytest.raises(ValueError, match=r"num_global must be in \[1, 4\]"):
        local_loss(torch.randn(4, 16, 8), loss=sigreg_loss, num_global=num_global)


def test_accepts_num_global_at_both_ends():
    for num_global in (1, 4):
        out = local_loss(torch.randn(4, 64, 16), loss=sigreg_loss, num_global=num_global,
                         num_clusters=2, **TORCH_BACKEND)
        assert torch.isfinite(out)


def test_rejects_an_unknown_backend():
    with pytest.raises(ValueError, match="Unknown kmeans_backend"):
        local_loss(torch.randn(4, 16, 8), loss=sigreg_loss, kmeans_backend="nope")


def test_both_backends_partition_the_batch_the_same_way():
    """fast_pytorch_kmeans and the in-repo torch fallback are offered as interchangeable. They
    will not agree on labels, but both must produce a valid partition of the accepted samples --
    otherwise ``kmeans_backend`` is not the free switch the config claims."""
    for backend in ("torch", "fast_pytorch_kmeans"):
        spy = Spy()
        torch.manual_seed(0)
        local_loss(torch.randn(4, 64, 16), loss=spy, num_clusters=4, min_cluster_size=4,
                   kmeans_backend=backend)
        assert sum(spy.sizes) <= 64 and len(spy.sizes) >= 1


# --------------------------------------------------------------------------------------
# the two presets
# --------------------------------------------------------------------------------------

def test_local_sigreg_defaults_to_the_uncentred_gaussian_test():
    torch.manual_seed(0)
    preset = local_sigreg_loss(torch.randn(4, 64, 16), num_clusters=2, **TORCH_BACKEND)
    torch.manual_seed(0)
    explicit = local_loss(torch.randn(4, 64, 16), loss=sigreg_loss, num_clusters=2,
                          **TORCH_BACKEND)
    torch.testing.assert_close(preset, explicit)


def test_local_ballreg_binds_centring_into_its_default_inner_test():
    """The preset is ``partial(ballreg_loss, center=True)``. Pin that by comparing against both
    explicit forms: it must equal the centred one and differ from the uncentred one."""
    from functools import partial

    proj = torch.randn(4, 64, 16)

    def run(fn, **kwargs):
        torch.manual_seed(0)
        return fn(proj, num_clusters=2, **TORCH_BACKEND, **kwargs)

    preset = run(local_ballreg_loss)
    centred = run(local_loss, loss=partial(ballreg_loss, center=True))
    uncentred = run(local_loss, loss=partial(ballreg_loss, center=False))

    torch.testing.assert_close(preset, centred)
    assert abs(float(preset) - float(uncentred)) > 1e-3, (
        "if the preset matched the uncentred test too, center=True would not be bound"
    )


def test_per_cluster_centring_makes_translation_free_under_euclidean_clustering():
    """With ``normalize_for_kmeans=False`` the k-means is Euclidean and therefore
    translation-equivariant: shifting every point moves the centroids identically and leaves the
    partition unchanged. Per-cluster centring then removes the shift entirely, so the statistic
    is **exactly** unchanged -- asserted at zero tolerance, which a weaker claim would not earn.
    """
    proj = torch.randn(4, 64, 16)

    def run(x):
        torch.manual_seed(0)
        return local_ballreg_loss(x, num_clusters=2, normalize_for_kmeans=False, **TORCH_BACKEND)

    torch.testing.assert_close(run(proj), run(proj + 5.0), rtol=0, atol=0)


def test_default_cosine_clustering_is_not_translation_invariant():
    """FINDING, pinned deliberately.

    Under the default ``normalize_for_kmeans=True`` the clustering is cosine-based, so a global
    translation changes every point's *direction* and the partition moves with it. The result is
    that ``local_ballreg_loss`` is **not** translation-invariant even though its inner test is
    centred per cluster -- measured here at ~5% for a shift of 5.0.

    This is a property of the method as configured, not a bug, but it is undocumented and it
    means the regularizer's value depends on where the projector happens to place the cloud. A
    change that made the clustering translation-invariant would alter results, so it should be a
    deliberate decision rather than a silent one.
    """
    proj = torch.randn(4, 64, 16)

    def run(x):
        torch.manual_seed(0)
        return float(local_ballreg_loss(x, num_clusters=2, **TORCH_BACKEND))

    plain, shifted = run(proj), run(proj + 5.0)
    assert abs(plain - shifted) / plain > 0.01, (
        "expected the cosine partition to move under translation; if this now holds, the "
        "clustering has been made translation-invariant and results will have changed"
    )


# --------------------------------------------------------------------------------------
# the clustering input is detached
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("backend", ["torch", "fast_pytorch_kmeans"])
@pytest.mark.parametrize("num_global", [None, 2])
def test_the_clustering_input_is_detached_from_the_graph(monkeypatch, backend, num_global):
    """``cluster_points = proj[...].mean(dim=0).detach()`` -- the k-means sees a constant.

    ADDED AFTER A SURVIVING MUTANT, and it took some thought to find an observable at all.
    Deleting the ``.detach()`` changes **neither the loss value nor any gradient**: the backend
    returns integer labels, and no gradient flows through an argmin. Every value- and
    gradient-based test in this file passes on the defect, which is why it survived.

    What it does change is what the graph holds. Ten Lloyd iterations of ``(N, D) x (D, k)``
    matmuls would be retained from the forward pass until backward, for a tensor whose gradient
    is then discarded -- pure memory, on the path that runs every step of a 100-epoch ImageNet
    run at batch 64. That is the kind of regression that surfaces as an OOM at hour six on a
    partition where the job is not requeued, and is attributed to anything but this.

    So the contract is asserted where it is visible: at the boundary. The tensor handed to the
    backend must not require grad. White-box, deliberately -- a black-box test for this defect
    does not exist.

    Parametrised over ``num_global`` because the detach appears on **two separate lines**, one per
    branch, and the first version of this test only ever took the ``None`` one -- so the mutant on
    the other line survived a second time. Two copies of a guard need two tests.
    """
    import src.losses as losses_module

    name = f"_{'torch' if backend == 'torch' else 'fast_pytorch'}_kmeans_labels"
    original = getattr(losses_module, name)
    seen = {}

    def spy(points, *args, **kwargs):
        seen["requires_grad"] = points.requires_grad
        seen["grad_fn"] = points.grad_fn
        return original(points, *args, **kwargs)

    monkeypatch.setattr(losses_module, name, spy)

    proj = torch.randn(4, 64, 16, requires_grad=True)
    out = losses_module.local_loss(proj, loss=sigreg_loss, num_clusters=4,
                                   num_global=num_global, kmeans_backend=backend)

    assert seen, "the backend was never called -- the test is not exercising the local path"
    assert seen["requires_grad"] is False, "the clustering input is still attached to the graph"
    assert seen["grad_fn"] is None
    assert out.requires_grad, "the loss itself must still be differentiable w.r.t. proj"


def test_detaching_the_clustering_does_not_detach_the_loss():
    """The complement, and the reason the detach is narrow rather than global: gradients must
    still reach ``proj`` through the per-cluster statistic. Detaching one line too many would
    make the regularizer a constant, and training would proceed on the invariance term alone --
    the exact silent failure LeJEPA exists to avoid."""
    proj = torch.randn(4, 64, 16, requires_grad=True)
    local_loss(proj, loss=sigreg_loss, num_clusters=4).backward()

    assert proj.grad is not None
    assert float(proj.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------------------
# realised K -- the ``stats`` out-param (campaign plan C1, 2026-09-20)
# --------------------------------------------------------------------------------------

STATS_KEYS = {"configured_clusters", "effective_clusters", "kept_fraction", "fallback"}


def test_stats_report_the_partition_exactly(monkeypatch):
    """Pin the *values*, on a partition chosen by hand: the k-means backend is replaced by fixed
    labels 30/30/2/2 at ``min_cluster_size=4``, so exactly two clusters survive and 60 of the 64
    samples are kept. Random data cannot pin this -- its partition is whatever k-means finds."""
    import src.losses as losses_module

    labels = torch.tensor([0] * 30 + [1] * 30 + [2] * 2 + [3] * 2)
    monkeypatch.setattr(losses_module, "_torch_kmeans_labels",
                        lambda x, num_clusters, **kwargs: labels)
    spy, stats = Spy(), {}
    local_loss(torch.randn(4, 64, 16), loss=spy, num_clusters=4, min_cluster_size=4,
               stats=stats, **TORCH_BACKEND)

    assert sorted(spy.sizes) == [30, 30]
    assert set(stats) == STATS_KEYS
    assert stats == {"configured_clusters": 4.0, "effective_clusters": 2.0,
                     "kept_fraction": 60 / 64, "fallback": 0.0}


def _deterministic(proj: torch.Tensor) -> torch.Tensor:
    return proj.float().square().mean()


def test_k_equals_one_is_the_global_statistic():
    """K=1 is the "locality off" control inside every ablation table: one cluster holds the whole
    batch, so the local loss must *be* the global one.

    Not a per-call identity, though. The sketched tests draw fresh random slices on every call
    (``_sketch_test``: ``A = torch.randn(D, num_slices)``), so two calls on the same input differ
    by sampling noise -- measured 0.609 vs 0.632 at 256 slices when this test was first written
    with the naive assertion. Two things are pinned instead:

    1. with a deterministic inner test the identity is exact: the one cluster IS the batch;
    2. with the real centred ball test the *only* difference is the RNG stream: replaying the
       backend's single ``randperm`` draw (its centroid initialisation) and then calling the
       inner test reproduces the local value bitwise. Nothing else happens between the batch
       and the statistic.
    """
    torch.manual_seed(0)
    proj = torch.randn(4, 64, 16)

    stats: dict[str, float] = {}
    local = local_loss(proj, loss=_deterministic, num_clusters=1, stats=stats, **TORCH_BACKEND)
    torch.testing.assert_close(local, _deterministic(proj), rtol=1e-6, atol=1e-6)
    assert stats == {"configured_clusters": 1.0, "effective_clusters": 1.0,
                     "kept_fraction": 1.0, "fallback": 0.0}

    inner = partial(ballreg_loss, center=True)
    torch.manual_seed(1)
    local = local_loss(proj, loss=inner, num_clusters=1, **TORCH_BACKEND)
    torch.manual_seed(1)
    torch.randperm(64)                       # the backend's one draw, replayed
    torch.testing.assert_close(local, inner(proj), rtol=0, atol=0)


def test_stats_none_leaves_the_loss_bit_identical():
    """The out-param is opt-in and free: the same seed with and without ``stats`` gives the same
    tensor, bitwise -- same ops on the same values in the same process, so exact is earned."""
    def run(**extra):
        torch.manual_seed(0)
        return local_loss(torch.randn(4, 64, 16), loss=sigreg_loss, num_clusters=4,
                          **TORCH_BACKEND, **extra)

    torch.testing.assert_close(run(), run(stats={}), rtol=0, atol=0)


@pytest.mark.parametrize("preset", [local_sigreg_loss, local_ballreg_loss])
def test_the_presets_declare_and_pass_stats_through(preset):
    """The Hydra targets are the presets, not ``local_loss``. ``stats`` must be a *declared*
    parameter on each: the module detects it by signature, and a bare ``**kwargs`` would not
    do -- it would forward the dict and yet look, from outside, like a regularizer that cannot
    report."""
    parameter = inspect.signature(preset).parameters.get("stats")
    assert parameter is not None and parameter.kind is inspect.Parameter.KEYWORD_ONLY

    stats: dict[str, float] = {}
    preset(torch.randn(4, 64, 16), num_clusters=2, stats=stats, **TORCH_BACKEND)
    assert set(stats) == STATS_KEYS
    assert stats["effective_clusters"] == 2.0 and stats["fallback"] == 0.0
