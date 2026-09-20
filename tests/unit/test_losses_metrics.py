"""Invariance, sparsity, contrastive loss, and the generalized-Gaussian sampler.

Most of these have exact closed forms, so they are asserted exactly rather than approximately.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.losses import (
    invariance_loss,
    l0_sparsity_metric,
    l1_sparsity_metric,
    nt_xent_loss,
    sample_generalized_gaussian,
    sigma_gn,
)


# --------------------------------------------------------------------------------------
# invariance_loss
# --------------------------------------------------------------------------------------

def test_invariance_is_exactly_zero_for_identical_views(identical_views):
    """Not "small" -- exactly zero. Every view equals the mean of the views, so every residual
    is the zero tensor and no floating-point slack is involved."""
    assert float(invariance_loss(identical_views)) == 0.0


def test_invariance_is_positive_when_views_differ(gaussian_proj):
    assert float(invariance_loss(gaussian_proj)) > 0.0


def test_num_global_restricts_the_reference_to_the_first_k_views():
    """With num_global=2 the reference is the mean of views 0 and 1 only. Construct a case where
    that changes the answer: views 0,1 identical and views 2,3 far away. Against all four views
    the reference sits between the groups; against the first two it sits on top of them."""
    v, n, d = 4, 8, 4
    proj = torch.zeros(v, n, d)
    proj[2:] = 10.0

    ref_all = invariance_loss(proj, num_global=None)
    ref_global = invariance_loss(proj, num_global=2)
    assert float(ref_global) > float(ref_all), (
        "a reference pinned to the two global views must leave the distant local views "
        "further from it than a reference averaged over all four"
    )


def test_num_global_of_v_equals_no_num_global(gaussian_proj):
    """Passing num_global = V is the same reference as None; a slice off-by-one would break it."""
    torch.testing.assert_close(
        invariance_loss(gaussian_proj, num_global=gaussian_proj.shape[0]),
        invariance_loss(gaussian_proj, num_global=None),
    )


def test_normalize_projects_onto_the_unit_sphere_first():
    """With normalize=True only direction matters, so scaling a view must not change the loss."""
    proj = torch.randn(3, 16, 8)
    scaled = proj.clone()
    scaled[0] *= 7.0
    torch.testing.assert_close(
        invariance_loss(proj, normalize=True),
        invariance_loss(scaled, normalize=True),
        rtol=1e-5, atol=1e-6,
    )
    assert not torch.isclose(invariance_loss(proj), invariance_loss(scaled))


# --------------------------------------------------------------------------------------
# sparsity metrics -- exact by construction
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [4, 16, 128])
def test_l1_sparsity_is_one_over_d_for_a_one_hot_vector(d):
    """(1/D)(||z||_1 / ||z||_2)^2 attains its minimum 1/D exactly at a one-hot vector."""
    z = torch.zeros(1, d)
    z[0, 0] = 3.0                       # magnitude must not matter: the ratio is scale-free
    assert float(l1_sparsity_metric(z)) == pytest.approx(1.0 / d, rel=1e-6)


@pytest.mark.parametrize("d", [4, 16, 128])
def test_l1_sparsity_is_one_for_a_uniform_vector(d):
    """...and its maximum 1 exactly at a constant vector."""
    assert float(l1_sparsity_metric(torch.full((1, d), 2.0))) == pytest.approx(1.0, rel=1e-6)


def test_l0_sparsity_is_the_exact_nonzero_fraction():
    z = torch.zeros(2, 10)
    z[0, :3] = 1.0          # 3/10
    z[1, :7] = -1.0         # 7/10
    assert float(l0_sparsity_metric(z)) == pytest.approx(0.5)   # mean of 0.3 and 0.7


def test_l1_sparsity_survives_an_all_zero_row():
    """The eps guard exists so a dead row yields 0 rather than NaN -- a NaN here would propagate
    silently through a logged mean."""
    assert torch.isfinite(l1_sparsity_metric(torch.zeros(1, 8)))


# --------------------------------------------------------------------------------------
# sigma_gn -- exact known values
# --------------------------------------------------------------------------------------

def test_sigma_gn_at_p1_is_one_over_sqrt_two():
    """p=1 is the Laplace case, whose unit-variance scale is 1/sqrt(2)."""
    assert sigma_gn(1.0) == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-12)


def test_sigma_gn_at_p2_is_one():
    """p=2 is the Gaussian case, already unit-variance."""
    assert sigma_gn(2.0) == pytest.approx(1.0, rel=1e-12)


def test_sigma_gn_gives_unit_variance_for_a_generic_p():
    """The point of sigma_gn is that GN_p(0, sigma_gn(p)) has variance 1. Check it for p=3 by
    sampling, which tests the formula through its consequence rather than restating it.
    Tolerance 5% on 200k draws; the defect it must catch (a wrong gamma ratio) moves the
    variance by tens of percent."""
    p = 3.0
    x = sample_generalized_gaussian((200_000,), p, scale=sigma_gn(p))
    assert float(x.var()) == pytest.approx(1.0, rel=0.05)


# --------------------------------------------------------------------------------------
# sample_generalized_gaussian
# --------------------------------------------------------------------------------------

def test_p_of_two_is_gaussian():
    x = sample_generalized_gaussian((100_000,), 2.0, loc=0.0, scale=1.0)
    assert float(x.mean()) == pytest.approx(0.0, abs=0.02)
    assert float(x.var()) == pytest.approx(1.0, rel=0.05)
    assert float((x.abs() ** 4).mean()) == pytest.approx(3.0, rel=0.1)   # Gaussian kurtosis


def test_p_of_one_is_laplace():
    """Laplace(0, b) has variance 2b^2 -- distinct from the Gaussian's b^2, so this
    discriminates the fast path rather than merely checking it runs."""
    x = sample_generalized_gaussian((200_000,), 1.0, loc=0.0, scale=1.0)
    assert float(x.var()) == pytest.approx(2.0, rel=0.05)


def test_generic_p_path_is_centred_on_loc():
    x = sample_generalized_gaussian((100_000,), 4.0, loc=2.5, scale=1.0)
    assert float(x.mean()) == pytest.approx(2.5, abs=0.05)


def test_generic_p_is_symmetric_about_loc():
    """The sign/Gamma reparameterization must give a symmetric law; a dropped sign draw would
    make it one-sided."""
    x = sample_generalized_gaussian((100_000,), 3.0, loc=0.0)
    assert float((x > 0).float().mean()) == pytest.approx(0.5, abs=0.01)


def test_requested_shape_is_honoured():
    assert sample_generalized_gaussian((3, 4, 5), 1.0).shape == (3, 4, 5)
    assert sample_generalized_gaussian((3, 4, 5), 2.0).shape == (3, 4, 5)
    assert sample_generalized_gaussian((3, 4, 5), 3.0).shape == (3, 4, 5)


@pytest.mark.parametrize("p", [0.0, -1.0])
def test_non_positive_p_raises(p):
    with pytest.raises(ValueError, match="must be > 0"):
        sample_generalized_gaussian((4,), p)


# --------------------------------------------------------------------------------------
# nt_xent_loss
# --------------------------------------------------------------------------------------

def test_nt_xent_requires_exactly_two_views():
    with pytest.raises(ValueError, match="exactly 2 views"):
        nt_xent_loss(torch.randn(3, 8, 4))


def test_nt_xent_is_lower_when_paired_views_agree():
    """Matched pairs must score below shuffled pairs. Direction-only, so no tolerance needed."""
    n, d = 32, 8
    z = torch.randn(n, d)
    aligned = nt_xent_loss(torch.stack([z, z + 0.01 * torch.randn(n, d)]))
    shuffled = nt_xent_loss(torch.stack([z, z[torch.randperm(n)]]))
    assert float(aligned) < float(shuffled)


def test_nt_xent_excludes_self_similarity():
    """``sim.fill_diagonal_(-inf)``: a sample's similarity to *itself* must not enter its own
    softmax denominator, where it would compete with the positive pair it is supposed to select.

    REWRITTEN AFTER A SURVIVING MUTANT. The previous version asserted only ``loss > 0``, which the
    defect also satisfies -- deleting the mask gives ~0.693, comfortably positive. A one-sided
    bound on a quantity the defect *raises* can never catch it.

    The closed form. With orthonormal rows and identical views, row i's logits are: -inf at i
    (masked), 1/T = 14.29 at the positive i+n, and 0 at the 2n-2 others. The positive dominates
    by e^14.29, so the loss is log(1 + (2n-2)e^-14.29) ~ 1.9e-5 -- measured exactly that.

    Delete the mask and sim[i,i] = 1/T as well: two identical maxima, the softmax splits evenly,
    and the loss becomes log(2) = 0.6931 -- measured 0.69316. The bound of 0.01 sits between
    them with four orders of margin on one side and two on the other.
    """
    z = torch.eye(16)                       # orthonormal: cos(i, j) = 0 for i != j, 1 for i == j
    loss = float(nt_xent_loss(torch.stack([z, z])))

    assert loss < 0.01, (
        f"loss {loss:.4f}; log(2) = 0.693 is what leaving self-similarity in the denominator gives"
    )
