"""RankMe -- the label-free effective rank used as a downstream-performance proxy.

The formula has an exact closed form on constructed spectra, so most of this is checked against
arithmetic rather than against the implementation. The subsampling pair is checked for the
contract nothing else states: the indices must select exactly the rows.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.eval.rankme import (
    classical_rank_from_singular_values,
    rankme,
    rankme_from_singular_values,
    singular_values,
    subsample_indices,
    subsample_rows,
)


# --------------------------------------------------------------------------------------
# the metric, against its definition
# --------------------------------------------------------------------------------------

def test_rankme_of_a_flat_spectrum_is_the_dimension():
    """RankMe is exp(Shannon entropy of the normalised spectrum). k equal singular values give
    entropy log(k), so RankMe is exactly k -- the maximum attainable."""
    for k in (2, 8, 64):
        sigma = torch.ones(k, dtype=torch.float64)
        assert rankme_from_singular_values(sigma) == pytest.approx(k, rel=1e-4)


def test_rankme_of_a_rank_one_spectrum_is_one():
    """All mass on one direction: entropy 0, RankMe 1. This is the collapse signature."""
    sigma = torch.zeros(32, dtype=torch.float64)
    sigma[0] = 5.0
    assert rankme_from_singular_values(sigma) == pytest.approx(1.0, abs=1e-3)


def test_rankme_is_scale_invariant():
    """The spectrum is normalised to a probability vector first, so a global rescale cannot
    change the rank -- otherwise a backbone producing larger features would look higher-rank."""
    sigma = torch.tensor([4.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    assert rankme_from_singular_values(sigma) == pytest.approx(
        rankme_from_singular_values(sigma * 1000.0), rel=1e-6
    )


def test_rankme_is_bounded_by_the_spectrum_length():
    sigma = torch.rand(16, dtype=torch.float64) + 0.1
    assert 1.0 <= rankme_from_singular_values(sigma) <= 16.0 + 1e-6


def test_rankme_orders_a_decaying_spectrum_below_a_flat_one():
    flat = torch.ones(16, dtype=torch.float64)
    decaying = torch.tensor([0.9 ** i for i in range(16)], dtype=torch.float64)
    assert rankme_from_singular_values(decaying) < rankme_from_singular_values(flat)


# --------------------------------------------------------------------------------------
# singular values and the classical rank
# --------------------------------------------------------------------------------------

def test_singular_values_rejects_anything_but_a_matrix():
    with pytest.raises(ValueError, match="Expected a 2-D feature matrix"):
        singular_values(torch.randn(4, 4, 4))


def test_singular_values_are_computed_in_float64():
    """Features are cached as fp16; computing the spectrum in that precision would lose the tail
    that the entropy is most sensitive to."""
    assert singular_values(torch.randn(32, 8, dtype=torch.float16)).dtype == torch.float64


def test_singular_values_of_a_known_matrix():
    """A diagonal matrix's singular values are the absolute diagonal entries, sorted."""
    x = torch.diag(torch.tensor([3.0, 1.0, 2.0]))
    torch.testing.assert_close(
        singular_values(x), torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64),
        rtol=1e-5, atol=1e-6,
    )


def test_classical_rank_counts_significant_directions():
    """Threshold rank: sigma > sigma_max * max(N, K) * eps. A genuinely rank-2 matrix in a
    3-dimensional embedding must come back as 2."""
    sigma = torch.tensor([5.0, 3.0, 0.0], dtype=torch.float64)
    assert classical_rank_from_singular_values(sigma, shape=(100, 3)) == 2


def test_rankme_on_a_low_rank_matrix_is_near_its_true_rank():
    """Construct an exactly rank-3 matrix; both the entropic and the classical rank should
    report 3, which is the check that ties the two together."""
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(16, 3))[0]
    x = torch.randn(500, 3) @ basis.T
    out = rankme(x)
    assert out["classical_rank"] == 3
    assert out["rankme"] == pytest.approx(3.0, rel=0.2)


def test_rankme_reports_its_inputs_shape_and_spectrum():
    out = rankme(torch.randn(50, 7))
    assert out["n"] == 50 and out["dim"] == 7
    assert len(out["spectrum"]) == 7


def test_centring_changes_the_rank_when_there_is_a_mean_direction():
    """A large shared offset is one dominant direction. Uncentred it inflates the spectrum's
    first component and depresses RankMe; centring removes it. The paper convention is
    uncentred, and the report carries both -- so the two must actually differ."""
    torch.manual_seed(0)
    x = torch.randn(500, 8) + 50.0
    assert rankme(x, center=False)["rankme"] < rankme(x, center=True)["rankme"]


# --------------------------------------------------------------------------------------
# subsampling -- the contractual pair
# --------------------------------------------------------------------------------------

def test_subsample_indices_select_exactly_the_rows_subsample_rows_returns():
    """THE contract. ``subsample_rows`` and ``subsample_indices`` are used on different objects
    -- features and their labels -- and are only correct together. If they diverged, every
    RankMe row would be computed on features paired with the wrong labels, silently.
    """
    torch.manual_seed(0)
    x = torch.randn(200, 6)
    rows = subsample_rows(x, 50, seed=3)
    indices = subsample_indices(200, 50, seed=3)
    torch.testing.assert_close(rows, x[indices], rtol=0, atol=0)


def test_subsampling_is_deterministic_for_a_seed():
    """Note ``subsample_indices`` returns a plain ``list[int]`` while ``subsample_rows`` returns
    a tensor -- an asymmetry worth pinning, since callers index with the list."""
    a = subsample_indices(1000, 64, seed=7)
    b = subsample_indices(1000, 64, seed=7)
    c = subsample_indices(1000, 64, seed=8)
    assert isinstance(a, list) and all(isinstance(i, int) for i in a)
    assert a == b and a != c


def test_subsample_indices_are_sorted_and_unique():
    """Sorted keeps the subset in dataset order; unique means no row is counted twice, which
    would bias the spectrum."""
    idx = subsample_indices(500, 100, seed=0)
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx)


def test_subsampling_returns_everything_when_the_request_exceeds_the_data():
    x = torch.randn(20, 4)
    torch.testing.assert_close(subsample_rows(x, 1000, seed=0), x, rtol=0, atol=0)
    assert len(subsample_indices(20, 1000, seed=0)) == 20
