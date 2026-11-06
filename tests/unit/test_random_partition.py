"""The random-partition control (decision 36): ``local_loss(..., random_partition=True)``.

Would random partitions with matched cluster sizes give similar gains? The control runs k-means as usual and then
shuffles its labels across the samples of that step. What the read-out relies on, and what was only read from the code
before these tests:

* off by default, and then byte-identical to today, including the random stream;
* on, each step keeps k-means' exact sizes, so the size filter and the realised-K statistics mean the same thing;
* on, membership no longer depends on the data;
* a fresh shuffle per call, reaching the loss through both public wrappers and the config.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from src.losses import ballreg_loss, local_ballreg_loss, local_loss, local_sigreg_loss, sigreg_loss

TORCH_BACKEND = dict(kmeans_backend="torch")


class MemberSpy:
    """Records which samples each call received, by the sample id planted in the first coordinate."""

    def __init__(self) -> None:
        self.groups: list[list[int]] = []

    def __call__(self, proj: torch.Tensor) -> torch.Tensor:
        self.groups.append(sorted(int(x) for x in proj[0, :, 0].round().tolist()))
        return proj.float().sum() * 0.0


def blobs(n_per_blob: int = 16, n_blobs: int = 4, views: int = 2, dim: int = 8, seed: int = 0) -> torch.Tensor:
    """Tight, far-apart blobs that k-means separates perfectly, so its labels are the blob ids. The FIRST coordinate is
    the sample's id. It is scaled down in the points k-means sees through the blob centres dominating everything else."""
    g = torch.Generator().manual_seed(seed)
    n = n_per_blob * n_blobs
    centres = 1000.0 * torch.eye(n_blobs, dim - 1)
    labels = torch.arange(n_blobs).repeat_interleave(n_per_blob)
    points = centres[labels] + 0.01 * torch.randn(n, dim - 1, generator=g)
    ids = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    single = torch.cat([ids, points], dim=1)
    return single.unsqueeze(0).repeat(views, 1, 1)


def blob_of(sample: int, n_per_blob: int = 16) -> int:
    return sample // n_per_blob


def run(proj, seed=0, **kwargs):
    spy = MemberSpy()
    torch.manual_seed(seed)
    local_loss(proj, loss=spy, num_clusters=4, normalize_for_kmeans=False, **TORCH_BACKEND, **kwargs)
    return spy.groups


def test_kmeans_alone_recovers_the_blobs_so_the_fixture_can_see_a_shuffle():
    """If k-means did not recover the blobs, the purity test below would compare two shuffles."""
    for group in run(blobs()):
        assert len({blob_of(s) for s in group}) == 1, group


def test_off_by_default_and_off_is_the_old_behaviour_with_the_same_random_stream():
    """The arms already trained must not move: no shuffle and not one extra draw from the generator.

    The last assertion shows the check can see a difference: with the key on, the stream does move."""
    proj = blobs()
    torch.manual_seed(7)
    default = MemberSpy()
    local_loss(proj, loss=default, num_clusters=4, normalize_for_kmeans=False, **TORCH_BACKEND)
    after_default = torch.get_rng_state()

    torch.manual_seed(7)
    off = MemberSpy()
    local_loss(proj, loss=off, num_clusters=4, normalize_for_kmeans=False, random_partition=False, **TORCH_BACKEND)
    after_off = torch.get_rng_state()
    assert default.groups == off.groups
    assert torch.equal(after_default, after_off)

    torch.manual_seed(7)
    on = MemberSpy()
    local_loss(proj, loss=on, num_clusters=4, normalize_for_kmeans=False, random_partition=True, **TORCH_BACKEND)
    assert not torch.equal(after_default, torch.get_rng_state())


def test_on_keeps_kmeans_exact_sizes_in_the_same_step():
    """Matched sizes are the control's definition, and they must hold step by step, also when k-means is unbalanced."""
    unbalanced = torch.cat([blobs(n_per_blob=16, n_blobs=1, seed=1) + 0.0, blobs(n_per_blob=16, n_blobs=4, seed=2)[:, 16:40, :]], dim=1)
    unbalanced[:, :, 0] = torch.arange(unbalanced.shape[1], dtype=torch.float32)
    for seed in range(5):
        off = run(unbalanced, seed=seed)
        on = run(unbalanced, seed=seed, random_partition=True)
        assert sorted(map(len, off)) == sorted(map(len, on)), (seed, off, on)


def test_on_is_a_partition_no_sample_is_lost_or_repeated():
    groups = run(blobs(), random_partition=True)
    assert sorted(s for g in groups for s in g) == list(range(64))


def test_on_removes_the_data_dependence():
    """Every k-means group is pure in blob id. A random group of 16 out of four blobs of 16 almost never is, and over
    many steps its largest blob share sits near chance (about 0.44 for this size). Inverted by deleting the shuffle."""
    shares = []
    for seed in range(40):
        for group in run(blobs(), seed=seed, random_partition=True):
            counts = torch.bincount(torch.tensor([blob_of(s) for s in group]), minlength=4)
            shares.append(counts.max().item() / len(group))
    assert sum(shares) / len(shares) < 0.6, f"mean largest blob share {sum(shares) / len(shares):.3f}"


def test_each_call_draws_a_fresh_shuffle():
    """A permutation fixed once would be a frozen, data-independent partition of batch POSITIONS, which is a different
    control. Two consecutive steps from the same stream must differ."""
    proj = blobs()
    torch.manual_seed(3)
    first, second = MemberSpy(), MemberSpy()
    local_loss(proj, loss=first, num_clusters=4, normalize_for_kmeans=False, random_partition=True, **TORCH_BACKEND)
    local_loss(proj, loss=second, num_clusters=4, normalize_for_kmeans=False, random_partition=True, **TORCH_BACKEND)
    assert sorted(first.groups) != sorted(second.groups)


def test_the_size_filter_and_the_statistics_read_the_same_with_the_shuffle():
    """A run's realised K comes from these statistics, and the read-out compares it with the K=4 arm's."""
    proj = blobs()
    off_stats, on_stats = {}, {}
    run(proj, seed=5, stats=off_stats)
    run(proj, seed=5, stats=on_stats, random_partition=True)
    assert off_stats == on_stats


def test_every_cluster_too_small_still_falls_back_to_the_global_test():
    """The fallback is pinned as a finding in test_losses_local.py. The shuffle must not move it."""
    proj = blobs(n_per_blob=2, n_blobs=4)
    stats = {}
    groups = run(proj, stats=stats, random_partition=True)
    assert len(groups) == 1 and len(groups[0]) == 8
    assert stats["fallback"] == 1.0


@pytest.mark.parametrize("wrapper", [local_ballreg_loss, local_sigreg_loss], ids=["ball", "gaussian"])
def test_the_key_reaches_local_loss_through_both_wrappers(wrapper):
    """The runs instantiate the wrappers from config, not local_loss. A wrapper that dropped the keyword would train
    plain k-means under the control's name."""
    proj = blobs()
    pure, shuffled = MemberSpy(), MemberSpy()
    torch.manual_seed(0)
    wrapper(proj, loss=pure, num_clusters=4, normalize_for_kmeans=False, **TORCH_BACKEND)
    torch.manual_seed(0)
    wrapper(proj, loss=shuffled, num_clusters=4, normalize_for_kmeans=False, random_partition=True, **TORCH_BACKEND)
    assert all(len({blob_of(s) for s in g}) == 1 for g in pure.groups)
    assert any(len({blob_of(s) for s in g}) > 1 for g in shuffled.groups)


@pytest.mark.parametrize("inner", [partial(ballreg_loss, center=True), partial(sigreg_loss, center=True)], ids=["ball", "gaussian"])
def test_a_real_test_statistic_runs_and_keeps_its_gradient(inner):
    proj = torch.randn(2, 64, 16, requires_grad=True)
    loss = local_loss(proj, loss=inner, num_clusters=4, random_partition=True, **TORCH_BACKEND)
    loss.backward()
    assert torch.isfinite(loss) and proj.grad is not None and proj.grad.abs().sum() > 0
