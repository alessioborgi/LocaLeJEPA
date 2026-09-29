"""The residual-discrepancy metric must measure distance from a target, comparably across arms.

This metric is the premise test for the capacity campaign: if a frozen ViT-S/8 already reaches its
target under the global statistic, the global constraint was satisfiable and locality was never
load-bearing. A metric that was silently wrong would not fail loudly -- it would return plausible
numbers and send 310 GPU-hours after a refuted hypothesis. So the tests below pin the three things
that turn `src/losses.py`'s training statistic into a comparable measurement, and they assert
against closed forms and direction-only inequalities rather than against the code's own output.

The strongest assertions here need no tolerance at all: a sample drawn FROM the ball must score
closer to the ball than to the Gaussian, and vice versa. That is a property of the measurement, not
of a threshold, and it fails for any implementation that has the two targets confused.
"""

from __future__ import annotations

import pytest
import torch

from conftest import uniform_ball
from src.eval.discrepancy import (
    TARGETS,
    covariance_metrics,
    discrepancy,
    discrepancy_both_targets,
    local_discrepancy,
)

# Enough samples that the empirical characteristic function is not dominated by sampling noise;
# 2048 keeps every test in this file under a second on CPU.
N = 2048
D = 16


def ball(n: int = N, d: int = D, radius: float | None = None) -> torch.Tensor:
    """(n, d) drawn uniformly from the ball, via conftest's inverse-CDF helper."""
    return uniform_ball(1, n, d, radius=radius)[0]


def gaussian(n: int = N, d: int = D) -> torch.Tensor:
    return torch.randn(n, d)


# ----- the metric measures what it claims (direction-only, no tolerance) -----


def test_a_ball_sample_is_closer_to_the_ball_than_to_the_gaussian():
    """The load-bearing assertion, and it needs no tolerance.

    Samples drawn from the uniform ball must score lower against the ball target than against the
    Gaussian. This is direction-only, which makes it a far stronger claim than any threshold on an
    absolute value (the convention stated in test_losses_regularizers.py:4-5).

    INVERTED once: swapping the two branches of `_target_phi` fails this and the next test
    simultaneously, which is exactly the defect a self-scored metric would hide.
    """
    scores = discrepancy_both_targets(ball())
    assert scores["ball"]["distance"] < scores["gaussian"]["distance"], scores


def test_a_gaussian_sample_is_closer_to_the_gaussian_than_to_the_ball():
    """The converse. Both directions are needed: a metric that always returned the ball as nearer
    would pass the previous test alone."""
    scores = discrepancy_both_targets(gaussian())
    assert scores["gaussian"]["distance"] < scores["ball"]["distance"], scores


def test_a_collapsed_sample_is_far_from_both_targets():
    """Collapse is the failure the regularizer exists to prevent, so the metric must see it.

    Direction-only again: a nearly-constant embedding must score worse against BOTH targets than a
    correctly-distributed sample does against either. Tiny noise is added because a perfectly
    constant matrix makes the characteristic function degenerate rather than merely bad.
    """
    collapsed = torch.zeros(N, D) + 1e-4 * torch.randn(N, D)
    collapsed_scores = discrepancy_both_targets(collapsed)
    healthy = discrepancy_both_targets(ball())
    worst_healthy = max(v["distance"] for v in healthy.values())
    best_collapsed = min(v["distance"] for v in collapsed_scores.values())
    assert best_collapsed > worst_healthy, (collapsed_scores, healthy)


# ----- the three corrections that make it a metric rather than a loss -----


def test_the_statistic_and_the_distance_scale_differently_and_both_are_reported():
    """WRITTEN AFTER THE FIRST VERSION OF THIS MODULE WAS WRONG.

    I first reported `statistic / n_samples` alone, documented as "normalised by N so it is
    comparable across sample counts". Measured, that is false. The `* z.size(-2)` inside
    `_sketch_test` is the Epps-Pulley normalisation, so the RAW statistic is the quantity that holds
    still under a good fit -- at d=16 it stays near 1.0 from N=512 to N=8192 for samples drawn from
    the ball, while `statistic / N` falls as 1/N. There is no single N-independent number; the two
    answer different questions, so both are reported.

    Bands are wide because these are sampling estimates over a 4x change in N; the defect each one
    catches is a factor of 4, far outside either band.
    """
    small = discrepancy(ball(n=1024), "ball")
    large = discrepancy(ball(n=4096), "ball")

    # The statistic holds still under a good fit -- that is what the N factor is for.
    assert large["statistic"] == pytest.approx(small["statistic"], rel=0.5), (small, large)
    # The distance falls toward zero, because the fit really is good.
    assert large["distance"] < 0.5 * small["distance"], (small, large)
    assert large["n_samples"] == 4096.0


def test_a_real_mismatch_makes_the_statistic_GROW_with_the_sample_size():
    """The other half of the scaling story, and the reason the statistic is worth reporting.

    Under a genuine mismatch the statistic diverges rather than holding still -- that divergence is
    the test's power. Measured: a Gaussian sample scored against the ball goes 1.09 -> 3.98 from
    N=512 to N=8192, while a matched sample stays near 1.0. So a statistic that grew with N under a
    GOOD fit would indicate a broken target, not a distant one.

    INVERTED once: scoring the ball sample against the ball here fails, which is the point.
    """
    small = discrepancy(gaussian(n=1024), "ball")
    large = discrepancy(gaussian(n=4096), "ball")
    assert large["statistic"] > small["statistic"], (small, large)


def test_the_metric_separates_a_mismatch_from_a_good_fit_at_EQUAL_sample_size():
    """The property the metric is actually used for.

    Every arm is evaluated on the same split, so what matters is not behaviour across N but
    discrimination at fixed N. Both numbers must rank a mismatched sample above a matched one, on
    the same sample count, with no tolerance needed.
    """
    matched = discrepancy(ball(n=4096), "ball")
    mismatched = discrepancy(gaussian(n=4096), "ball")
    assert mismatched["distance"] > matched["distance"], (matched, mismatched)
    assert mismatched["statistic"] > matched["statistic"], (matched, mismatched)


def test_the_same_seed_reproduces_and_a_different_seed_does_not():
    """`_sketch_test` draws fresh slice directions on every call, so an unseeded metric disagrees
    with itself. Reproducibility is asserted exactly (rtol=0, atol=0) because a seeded draw is
    deterministic, not approximately so.

    The second half matters too: a metric that ignored `seed` entirely would pass the first half.

    INVERTED once: deleting the `torch.manual_seed(seed)` line fails the exact-equality half.
    """
    Z = ball()
    assert discrepancy(Z, "ball", seed=7)["distance"] == discrepancy(Z, "ball", seed=7)["distance"]
    assert discrepancy(Z, "ball", seed=7)["distance"] != discrepancy(Z, "ball", seed=8)["distance"]


def test_centering_is_an_explicit_choice_and_it_changes_the_score():
    """Centring lives in the loss WRAPPERS, not the statistic: `ballreg_loss` defaults to
    `center=False`, `local_ballreg_loss` bakes in `center=True`, and `local_sigreg_loss` does not
    centre at all. A metric that inherited whichever convention its arm used would not be comparable
    across arms, so it is a parameter here -- and it must actually do something.
    """
    shifted = ball() + 3.0
    centred = discrepancy(shifted, "ball", center=True)["distance"]
    uncentred = discrepancy(shifted, "ball", center=False)["distance"]
    assert centred < uncentred, (centred, uncentred)


# ----- the radius convention, which is what makes the two targets comparable -----


def test_covariance_metrics_read_identity_covariance_on_a_ball_at_the_identity_radius():
    """WAS VACUOUS, AND THE INVERSION CAUGHT IT.

    This test was originally titled as pinning the module's radius convention. It does not:
    `conftest.uniform_ball` computes `math.sqrt(d + 2)` itself (conftest.py:108) and never calls
    `_resolve_ball_radius`, so breaking the module's radius to `sqrt(d)` left this PASSING. What it
    really pins is that `covariance_metrics` reads unit diagonal and near-zero off-diagonal on a
    sample that genuinely has identity covariance -- worth keeping, but under an honest name. The
    radius convention is pinned by the next test, through the module's own code path.

    Tolerance `rel=0.05` on a 2048-sample estimate of a quantity whose true value is exactly 1.
    """
    metrics = covariance_metrics(ball())
    assert metrics["cov_diag_mean"] == pytest.approx(1.0, rel=0.05), metrics
    assert abs(metrics["cov_offdiag_mean"]) < 0.1, metrics


def test_the_default_radius_is_sqrt_d_plus_two_through_the_modules_own_path():
    """The real pin on `_resolve_ball_radius`, and it has to route through `discrepancy` to be one.

    sqrt(d + 2) is the radius at which Cov[Unif(ball)] = I, which is what normalises the ball
    exactly as the isotropic Gaussian is normalised and puts the two discrepancies on one scale.

    Equality with the explicit value is asserted EXACTLY: the seed is pinned, so the same radius
    draws the same slices and must give bit-identical output. Inequality with sqrt(d) is what makes
    the test non-vacuous -- a module that ignored `radius` entirely would pass the first assertion.

    INVERTED once: changing `_resolve_ball_radius` to `d ** 0.5` makes the default agree with the
    sqrt(d) call and disagree with the sqrt(d + 2) one, failing both assertions.
    """
    Z = ball(n=512)
    default = discrepancy(Z, "ball", radius=None)["statistic"]
    identity_radius = discrepancy(Z, "ball", radius=(D + 2) ** 0.5)["statistic"]
    wrong_radius = discrepancy(Z, "ball", radius=D**0.5)["statistic"]

    assert default == identity_radius, (default, identity_radius)
    assert default != wrong_radius, "the radius argument is being ignored"


def test_covariance_metrics_see_collapse_and_isotropy_differently():
    """The diagnostics ported from the training module must separate the two regimes they exist to
    separate. An isotropic sample has unit mean diagonal; a collapsed one has essentially zero."""
    healthy = covariance_metrics(gaussian())
    collapsed = covariance_metrics(torch.zeros(N, D) + 1e-4 * torch.randn(N, D))
    assert healthy["cov_diag_mean"] == pytest.approx(1.0, rel=0.05)
    assert collapsed["cov_diag_mean"] < 1e-6
    assert collapsed["std_mean"] < healthy["std_mean"]


# ----- the local variant must surface a degenerate partition -----


def test_local_discrepancy_reports_the_partition_it_actually_used():
    """`local_loss`'s `stats` out-param is the only way to tell a real local score from a fallback.

    With K=4 on 256 samples every cluster clears `min_cluster_size`, so `fallback` must be 0 and the
    effective count must not exceed the configured one.
    """
    value, stats = local_discrepancy(ball(n=256), "ball", num_clusters=4)
    assert value > 0
    assert stats["configured_clusters"] == 4.0
    assert stats["fallback"] == 0.0
    assert 0 < stats["effective_clusters"] <= 4.0
    assert 0 < stats["kept_fraction"] <= 1.0


def test_a_degenerate_partition_sets_the_fallback_flag_rather_than_scoring_silently():
    """THE trap this metric could hide. When every cluster falls below `min_cluster_size`,
    `local_loss` scores the whole batch globally instead (`src/losses.py:309-310`) and returns a
    number that looks like a good local score. Asking for 8 clusters from 8 samples forces it.

    A caller that reads the value without reading `fallback` would record a global score in a local
    column, for every collapsed cell in the grid.

    INVERTED once: dropping `stats` from the `local_loss` call leaves the dict empty and fails.
    """
    _, stats = local_discrepancy(ball(n=8, d=4), "ball", num_clusters=8)
    assert stats["fallback"] == 1.0, stats


# ----- shape and argument guards -----


@pytest.mark.parametrize("bad", [torch.zeros(4, 8, 16), torch.zeros(16)])
def test_only_two_dimensional_features_are_accepted(bad: torch.Tensor):
    """The feature cache holds (N, D). `ballreg_loss` itself refuses anything but (V, N, D), which is
    why this module calls `_sketch_test` directly -- so it has to do its own rank check, or a (V, N, D)
    batch would be silently averaged over views and reported as one number."""
    with pytest.raises(ValueError, match="N, D"):
        discrepancy(bad, "ball")


def test_an_unknown_target_is_refused_by_name():
    """Silently defaulting to one of the two targets would produce a whole column of wrong numbers
    with no error anywhere."""
    with pytest.raises(ValueError, match="target must be one of"):
        discrepancy(ball(), "uniform-on-the-sphere")  # type: ignore[arg-type]


def test_a_single_sample_cannot_estimate_a_characteristic_function():
    """An empirical CF over one point is that point's phase, not a distribution. Returning a number
    there would put a meaningless value in a table cell rather than raising."""
    with pytest.raises(ValueError, match="at least 2 samples"):
        discrepancy(ball(n=1), "ball")


def test_local_discrepancy_also_refuses_anything_but_two_dimensional_features():
    """It wraps the input in a view axis itself, so a (V, N, D) input would silently become
    (1, V, N, D) and `local_loss`'s own rank check would report a confusing shape."""
    with pytest.raises(ValueError, match="N, D"):
        local_discrepancy(torch.zeros(2, 64, 16), "ball", num_clusters=2)


def test_local_discrepancy_can_leave_the_neighbourhoods_uncentred():
    """`center` defaults to True here because that is what `local_ballreg_loss` bakes in, but
    `local_sigreg_loss` deliberately does not centre -- so the uncentred path has to work and has to
    give a different answer, or the flag is decorative."""
    Z = ball(n=256) + 2.0
    centred, _ = local_discrepancy(Z, "ball", num_clusters=4, center=True)
    uncentred, _ = local_discrepancy(Z, "ball", num_clusters=4, center=False)
    assert centred != uncentred
    assert centred < uncentred, (centred, uncentred)


@pytest.mark.parametrize(
    "bad, match",
    [(torch.zeros(4, 8, 16), "N, D"), (torch.zeros(1, 16), "at least 2 samples")],
)
def test_covariance_metrics_guard_their_input(bad: torch.Tensor, match: str):
    """A covariance from one sample is a division by zero; a (V, N, D) batch would be read as an
    (N, D) matrix with the wrong N."""
    with pytest.raises(ValueError, match=match):
        covariance_metrics(bad)


def test_both_targets_are_scored_and_named():
    scores = discrepancy_both_targets(ball(n=256))
    assert set(scores) == set(TARGETS)
    for target, values in scores.items():
        assert set(values) == {"statistic", "distance", "n_samples"}, target
        assert values["statistic"] > 0 and values["distance"] > 0, target
        assert values["n_samples"] == 256.0
