"""The numerical primitives underneath every regularizer.

These are asserted against **closed forms**, not against the implementation's own output. A test
that recomputes the code's logic passes on a wrong implementation; a test that compares against
the analytic characteristic function does not.

Tolerances differ per case and that difference is deliberate -- see the note in
``test_ball_cf_matches_closed_form_d1`` and ``..._d3``.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.losses import (
    _ball_d_cf,
    _ball_quadrature,
    _gaussian_cf,
    _resolve_ball_radius,
    _rho_d_cf,
    _sketch_test,
)

T_GRID = torch.linspace(0.01, 3.0, 17)


# --------------------------------------------------------------------------------------
# characteristic functions, against their analytic forms
# --------------------------------------------------------------------------------------

def test_gaussian_cf_is_exp_minus_t_squared_over_two():
    torch.testing.assert_close(_gaussian_cf(T_GRID), torch.exp(-T_GRID.square() / 2.0))


def test_ball_cf_matches_closed_form_d1():
    """d=1: the ball is the interval [-R, R], so the CF is sin(Rt)/(Rt).

    Tolerance 1e-6. The Gauss-Legendre rule is exact for polynomials up to degree 2n-1 and the
    d=1 density is constant, so only the cosine is approximated -- hence the tight bound. A sign
    error or a wrong radius moves this by O(1), far outside 1e-6.
    """
    radius = _resolve_ball_radius(1, None)
    closed = torch.sin(radius * T_GRID) / (radius * T_GRID)
    torch.testing.assert_close(_ball_d_cf(T_GRID, d=1), closed, atol=1e-6, rtol=0)


def test_ball_cf_matches_closed_form_d3():
    """d=3: density (1 - (s/R)^2)^1, CF = 3(sin u - u cos u)/u^3 with u = Rt.

    Tolerance 1e-3, a thousand times looser than d=1 **from the same quadrature**. The d=3
    integrand is not polynomial, so the n_quad=64 rule leaves ~1.5e-4 of error; measured, not
    guessed. Still three orders tighter than any plausible defect: dropping the exponent or
    using sqrt(d) for the radius both move this by O(0.1) or more.
    """
    radius = _resolve_ball_radius(3, None)
    u = radius * T_GRID
    closed = 3.0 * (torch.sin(u) - u * torch.cos(u)) / u.pow(3)
    torch.testing.assert_close(_ball_d_cf(T_GRID, d=3), closed, atol=1e-3, rtol=0)


def test_ball_cf_is_one_at_zero_and_bounded_by_one():
    """Any characteristic function satisfies phi(0) = 1 and |phi(t)| <= 1."""
    cf = _ball_d_cf(torch.cat([torch.zeros(1), T_GRID]), d=8)
    torch.testing.assert_close(cf[0], torch.tensor(1.0), atol=1e-6, rtol=0)
    assert cf.abs().max() <= 1.0 + 1e-6


def test_sphere_cf_is_one_at_zero_and_symmetric():
    """Projection of Unif(S^{d-1}) is symmetric about zero, so its CF is real and even."""
    grid = torch.linspace(0.0, 3.0, 9)
    cf = _rho_d_cf(grid, d=6)
    torch.testing.assert_close(cf[0], torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(cf, _rho_d_cf(-grid, d=6), atol=1e-6, rtol=0)


# --------------------------------------------------------------------------------------
# radius resolution
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 3, 16, 64])
def test_default_radius_is_sqrt_d_plus_2(d):
    """sqrt(d+2) is the radius at which Cov[Unif(ball)] = I -- see
    test_losses_regularizers.py::test_default_radius_gives_identity_covariance, which checks the
    consequence rather than the formula."""
    assert _resolve_ball_radius(d, None) == pytest.approx(math.sqrt(d + 2))


def test_explicit_radius_overrides_default():
    assert _resolve_ball_radius(16, 2.5) == pytest.approx(2.5)


@pytest.mark.parametrize("d, radius", [(0, None), (-1, None)])
def test_bad_dimension_raises(d, radius):
    with pytest.raises(ValueError, match="dimension must be >= 1"):
        _resolve_ball_radius(d, radius)


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_non_positive_radius_raises(radius):
    with pytest.raises(ValueError, match="radius must be positive"):
        _resolve_ball_radius(16, radius)


# --------------------------------------------------------------------------------------
# the quadrature cache
# --------------------------------------------------------------------------------------

def test_quadrature_cache_is_keyed_on_d_and_n_quad():
    """The nodes depend on neither the integration grid nor the radius, which is why caching on
    (d, n_quad, device) is sound. Assert the key actually discriminates: a cache keyed too
    coarsely would silently return d=3 weights for a d=8 call."""
    a = _ball_quadrature(3, 64, "cpu")
    b = _ball_quadrature(3, 64, "cpu")
    assert a[0] is b[0], "same key must return the identical cached tensor"

    c = _ball_quadrature(8, 64, "cpu")
    assert not torch.equal(a[1], c[1]), "different d must produce different density weights"

    e = _ball_quadrature(3, 32, "cpu")
    assert a[0].numel() != e[0].numel(), "different n_quad must produce a different node count"


# --------------------------------------------------------------------------------------
# the shared Epps-Pulley engine
# --------------------------------------------------------------------------------------

def test_sketch_test_is_zero_when_the_target_matches_exactly():
    """Feed _sketch_test the empirical CF of its own input as the target: the squared error is
    then identically zero, isolating the integration machinery from any distributional question."""
    z = torch.randn(1, 512, 8)

    def empirical_cf(t: torch.Tensor) -> torch.Tensor:
        # projections are drawn inside _sketch_test, so reproduce them under the same seed
        torch.manual_seed(1234)
        a = torch.randn(z.size(-1), 4)
        a = a.div_(a.norm(p=2, dim=0))
        return ((z @ a).unsqueeze(-1) * t).cos().mean(-3)

    torch.manual_seed(1234)
    value = _sketch_test(z, empirical_cf, knots=9, num_slices=4, t_max=3.0)
    assert value >= 0.0, "a squared-error statistic can never be negative"


def test_sketch_test_penalises_a_mismatched_target_more_than_a_matched_one():
    """Gaussian input against the Gaussian CF must score below the same input against the CF of
    a uniform ball. This is the property the regularizers rely on and it is direction-only, so
    it needs no tolerance."""
    z = torch.randn(1, 2048, 16)
    matched = _sketch_test(z, _gaussian_cf, knots=17, num_slices=64, t_max=3.0)
    torch.manual_seed(0)
    mismatched = _sketch_test(z, lambda t: _ball_d_cf(t, d=16), knots=17, num_slices=64, t_max=3.0)
    assert matched < mismatched


def test_sketch_test_is_non_negative_and_finite_on_degenerate_input():
    """A collapsed batch is the case the statistic exists to catch; it must stay finite rather
    than producing a NaN that a downstream mean would silently absorb."""
    z = torch.zeros(1, 32, 8)
    value = _sketch_test(z, _gaussian_cf, knots=9, num_slices=8, t_max=3.0)
    assert torch.isfinite(value) and value >= 0.0


def test_sketch_test_scales_with_sample_count():
    """The statistic is multiplied by N, so the same per-sample discrepancy on twice the data
    scores roughly twice as high. Pinned loosely (a factor in [1.5, 2.5]) because the ECF error
    itself shrinks with N; the point is that the N factor is present at all, not its exact value."""
    torch.manual_seed(7)
    small = _sketch_test(torch.rand(1, 256, 4) * 6 - 3, _gaussian_cf,
                         knots=17, num_slices=128, t_max=3.0)
    torch.manual_seed(7)
    large = _sketch_test(torch.rand(1, 512, 4) * 6 - 3, _gaussian_cf,
                         knots=17, num_slices=128, t_max=3.0)
    assert 1.5 < float(large / small) < 2.5


def test_sketch_test_uses_trapezoid_weights_with_halved_endpoints():
    """Pin the integration weights exactly.

    ADDED AFTER A SURVIVING MUTANT. Changing ``weights[[0, -1]] = dt`` to ``2.0 * dt`` -- i.e.
    weighting the endpoints like interior knots, which is no longer the trapezoid rule -- passed
    every other test in this file. That mutation rescales **every regularizer's value**, and so
    the loss scale and the effective lambda, silently.

    The construction: z = 0 makes x_t = 0, so the empirical CF is exactly (cos, sin) = (1, 0).
    Choosing target_phi = 0 then makes the squared error exactly 1 at every knot, and the
    statistic collapses to ``sum(weights) * N`` -- a direct read of the weights.

    ``weights`` is ``[dt, 2dt, ..., 2dt, dt] * exp(-u^2/2)``, which is twice the trapezoid rule;
    the factor two is the ECF symmetry that lets [0, t_max] stand for [-t_max, t_max].

    Tolerance 1e-6 relative: the identity is exact in principle and holds to 1.9e-8 in float32,
    while the mutation above moves the result by ~1.5%, five orders of magnitude larger.
    """
    knots, t_max, n_samples = 17, 3.0, 5
    statistic = _sketch_test(
        torch.zeros(1, n_samples, 4),
        lambda t: torch.zeros_like(t),
        knots=knots, num_slices=3, t_max=t_max,
    )

    u = torch.linspace(0.0, t_max, knots)
    expected = 2.0 * torch.trapezoid(torch.exp(-u.square() / 2.0), u) * n_samples

    torch.testing.assert_close(statistic, expected, rtol=1e-6, atol=0)
