"""The distribution-matching regularizers.

The central property of every one of them is *discrimination*: input drawn from the target
distribution must score below input drawn from something else. That is direction-only and needs
no tolerance, which makes it a far stronger assertion than any threshold on an absolute value.
"""

from __future__ import annotations

import math

import pytest
import torch

from conftest import uniform_ball
from src.losses import (
    SubReg,
    ballreg_loss,
    rdmreg_loss,
    sigreg_loss,
    subreg_loss,
    susreg_loss,
)

BIG = dict(knots=17, num_slices=256, t_max=3.0)   # enough slices that the ranking is stable


# --------------------------------------------------------------------------------------
# sigreg -- isotropic Gaussian target
# --------------------------------------------------------------------------------------

def test_sigreg_prefers_gaussian_input_over_uniform_ball(gaussian_proj, ball_proj):
    assert float(sigreg_loss(gaussian_proj, **BIG)) < float(sigreg_loss(ball_proj, **BIG))


def test_sigreg_penalises_collapse(gaussian_proj, collapsed_proj):
    """Collapse is the failure mode the regularizer exists to prevent."""
    assert float(sigreg_loss(collapsed_proj, **BIG)) > float(sigreg_loss(gaussian_proj, **BIG))


def test_sigreg_centring_makes_the_test_blind_to_location(gaussian_proj):
    """center=True subtracts the per-view sample mean, so the statistic constrains shape only.
    A large shift must then cost nothing; without centring it must cost a great deal."""
    shifted = gaussian_proj + 5.0
    torch.manual_seed(0)
    centred_plain = sigreg_loss(gaussian_proj, center=True, **BIG)
    torch.manual_seed(0)
    centred_shifted = sigreg_loss(shifted, center=True, **BIG)
    torch.testing.assert_close(centred_plain, centred_shifted, rtol=1e-4, atol=1e-6)

    torch.manual_seed(0)
    uncentred_shifted = sigreg_loss(shifted, center=False, **BIG)
    assert float(uncentred_shifted) > float(centred_shifted) * 2.0


def test_sigreg_accepts_non_float32_input(gaussian_proj):
    """The docstring promises "any dtype"; the implementation upcasts internally."""
    out = sigreg_loss(gaussian_proj.to(torch.float64), **BIG)
    assert out.dtype == torch.float32 and torch.isfinite(out)


# --------------------------------------------------------------------------------------
# ballreg -- uniform-ball target
# --------------------------------------------------------------------------------------

def test_ballreg_prefers_ball_input_over_gaussian(gaussian_proj, ball_proj):
    assert float(ballreg_loss(ball_proj, **BIG)) < float(ballreg_loss(gaussian_proj, **BIG))


def test_ballreg_penalises_collapse(ball_proj, collapsed_proj):
    assert float(ballreg_loss(collapsed_proj, **BIG)) > float(ballreg_loss(ball_proj, **BIG))


def test_default_radius_gives_identity_covariance():
    """The *consequence* of radius = sqrt(d+2), checked by sampling rather than by restating the
    formula. Cov[Unif(ball_R)] = R^2/(d+2) I, so the default is exactly the whitened target.

    Tolerance 3% on 200k draws; using sqrt(d) instead would put the diagonal at d/(d+2), i.e.
    0.8 for d=8 -- twenty percent off, far outside the band.
    """
    d = 8
    x = uniform_ball(1, 200_000, d).squeeze(0)
    cov = x.T @ x / x.shape[0]
    assert float(cov.diag().mean()) == pytest.approx(1.0, rel=0.03)
    off_diagonal = cov - torch.diag(cov.diag())
    assert float(off_diagonal.abs().mean()) < 0.02


def test_scale_grid_is_a_no_op_at_the_default_radius(ball_proj):
    """t_scale = sqrt(d+2)/R, which is exactly 1 when R is the default. The two calls must agree
    bit-for-bit under the same seed -- if they differ, the scaling is being applied where the
    comment says it is not."""
    torch.manual_seed(0)
    without = ballreg_loss(ball_proj, scale_grid=False, **BIG)
    torch.manual_seed(0)
    with_scaling = ballreg_loss(ball_proj, scale_grid=True, **BIG)
    torch.testing.assert_close(without, with_scaling, rtol=0, atol=0)


def test_scale_grid_restores_the_collapse_penalty_at_a_small_radius(collapsed_proj):
    """The documented reason scale_grid exists: on a grid fixed in absolute units a small target
    ball is nearly indistinguishable from a collapsed distribution, so the statistic barely
    penalises collapse. Scaling the grid equalises that power across radii."""
    torch.manual_seed(0)
    unscaled = ballreg_loss(collapsed_proj, radius=0.05, scale_grid=False, **BIG)
    torch.manual_seed(0)
    scaled = ballreg_loss(collapsed_proj, radius=0.05, scale_grid=True, **BIG)
    assert float(scaled) > float(unscaled), (
        "with the grid scaled to the target's own width, a collapsed batch must be penalised "
        "more than it is on the absolute grid"
    )


def test_ballreg_centring_makes_the_test_blind_to_location(ball_proj):
    shifted = ball_proj + 5.0
    torch.manual_seed(0)
    a = ballreg_loss(ball_proj, center=True, **BIG)
    torch.manual_seed(0)
    b = ballreg_loss(shifted, center=True, **BIG)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_ballreg_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expects proj with shape \(V, N, D\)"):
        ballreg_loss(torch.randn(16, 8))


# --------------------------------------------------------------------------------------
# susreg -- uniform on the sphere
# --------------------------------------------------------------------------------------

def test_susreg_prefers_spherical_input(collapsed_proj):
    """susreg normalises its input, so *any* non-degenerate cloud becomes spherical directions.
    The discriminating case is therefore a degenerate one: directions concentrated in a cap."""
    v, n, d = 2, 512, 8
    spread = torch.randn(v, n, d)
    concentrated = torch.randn(v, n, d) * 0.01
    concentrated[..., 0] += 1.0               # nearly all mass on one pole
    assert float(susreg_loss(spread, **BIG)) < float(susreg_loss(concentrated, **BIG))


def test_susreg_is_scale_invariant():
    """L2-normalisation happens first, so a global rescale cannot change the statistic."""
    proj = torch.randn(2, 256, 8)
    torch.manual_seed(0)
    a = susreg_loss(proj, **BIG)
    torch.manual_seed(0)
    b = susreg_loss(proj * 37.0, **BIG)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_susreg_switches_to_the_clt_gaussian_above_d_256():
    """For d > 256 the projection of Unif(S^{d-1}) is approximated by N(0, 1/d). Both branches
    must produce a finite statistic; the boundary is the thing worth exercising."""
    below = susreg_loss(torch.randn(1, 64, 256), knots=9, num_slices=16, t_max=3.0)
    above = susreg_loss(torch.randn(1, 64, 257), knots=9, num_slices=16, t_max=3.0)
    assert torch.isfinite(below) and torch.isfinite(above)


# --------------------------------------------------------------------------------------
# subreg -- Gaussian in frozen subspaces
# --------------------------------------------------------------------------------------

def test_subreg_prefers_gaussian_input():
    v, n, d, k, ds = 2, 256, 16, 4, 4
    mats = torch.stack([torch.linalg.qr(torch.randn(d, ds))[0].T for _ in range(k)])
    gauss = torch.randn(v, n, d)
    ball = uniform_ball(v, n, d)
    assert float(subreg_loss(gauss, mats, **BIG)) < float(subreg_loss(ball, mats, **BIG))


def test_subreg_module_registers_its_matrices_as_a_buffer():
    """A buffer, not a parameter: the projections are frozen, and they must follow the module
    across devices and into checkpoints."""
    module = SubReg(projection_dim=16, num_subspaces=4)
    assert "proj_matrices" in dict(module.named_buffers())
    assert not list(module.parameters()), "the projections must not be trainable"
    assert module.proj_matrices.shape == (4, 4, 16)


def test_subreg_module_reports_the_function_name_as_its_metric_key():
    """LeJEPAModule derives the logged metric name from the regularizer; the module sets
    __name__ so it keys identically to the plain function."""
    assert SubReg(projection_dim=16).__name__ == "subreg_loss"


def test_subreg_module_rejects_too_many_subspaces():
    with pytest.raises(ValueError, match="too large for projection_dim"):
        SubReg(projection_dim=4, num_subspaces=8)


def test_subreg_matrices_are_row_orthonormal():
    """QR gives orthonormal columns; the code transposes, so each subspace's rows must be
    orthonormal -- P P^T = I. A dropped transpose would silently change the projection."""
    module = SubReg(projection_dim=16, num_subspaces=4)
    for matrix in module.proj_matrices:
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(matrix.shape[0]),
                                   rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------------------
# rdmreg -- sliced Wasserstein to a (rectified) generalized Gaussian
# --------------------------------------------------------------------------------------

def test_rdmreg_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"expects proj with shape \(V, N, D\)"):
        rdmreg_loss(torch.randn(16, 8))


def test_rdmreg_prefers_input_matching_its_target():
    """Rectified target: ReLU'd features should score below unrectified ones, since half the
    target's mass sits at zero."""
    v, n, d = 2, 512, 16
    raw = torch.randn(v, n, d)
    rectified = torch.relu(raw)
    torch.manual_seed(0)
    matched = rdmreg_loss(rectified, num_projections=64, rectify_target=True)
    torch.manual_seed(0)
    mismatched = rdmreg_loss(raw, num_projections=64, rectify_target=True)
    assert float(matched) < float(mismatched)


def test_rdmreg_unrectified_target_prefers_unrectified_input():
    """The rectify_target=False branch matches a plain generalized Gaussian, reversing the
    preference above -- which is what makes the flag meaningful rather than cosmetic."""
    v, n, d = 2, 512, 16
    raw = torch.randn(v, n, d)
    torch.manual_seed(0)
    on_raw = rdmreg_loss(raw, num_projections=64, rectify_target=False)
    torch.manual_seed(0)
    on_rectified = rdmreg_loss(torch.relu(raw), num_projections=64, rectify_target=False)
    assert float(on_raw) < float(on_rectified)


def test_rdmreg_is_non_negative():
    """A sorted-sample squared distance cannot be negative."""
    assert float(rdmreg_loss(torch.randn(2, 64, 8), num_projections=16)) >= 0.0
