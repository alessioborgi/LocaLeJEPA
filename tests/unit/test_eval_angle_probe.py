"""The circular ridge probe behind the Yoda rotation-angle results.

Angles are regressed as (cos, sin) pairs, so the metrics have to wrap correctly at the branch cut
-- an error of 359 degrees is an error of 1 degree. That wrapping is the thing most likely to be
silently wrong, and it is checked against known angles rather than against the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.angle_probe import (
    _circular_metrics,
    _ridge_solve,
    circular_linear_probe,
    low_data_curve,
    pca_2d,
)


def circular_data(n: int = 400, dim: int = 12, noise: float = 0.01, seed: int = 0):
    """Features that are a linear image of (cos t, sin t) plus noise, so a linear probe can
    recover the angle exactly in the noiseless limit."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, n)
    basis = rng.normal(size=(2, dim))
    feats = np.stack([np.cos(angles), np.sin(angles)], 1) @ basis
    return feats + noise * rng.normal(size=feats.shape), angles


# --------------------------------------------------------------------------------------
# the ridge solve
# --------------------------------------------------------------------------------------

def test_ridge_recovers_an_exact_linear_map_at_zero_regularisation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    weights = rng.normal(size=(5, 2))
    x_aug = np.concatenate([x, np.ones((200, 1))], 1)
    solved = _ridge_solve(x_aug, x @ weights, lam=1e-10)
    np.testing.assert_allclose(solved[:-1], weights, rtol=1e-4, atol=1e-5)


def test_ridge_shrinks_towards_zero_as_lambda_grows():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(100, 4)), np.ones((100, 1))], 1)
    y = rng.normal(size=(100, 2))
    weak = np.abs(_ridge_solve(x, y, 1e-6)).sum()
    strong = np.abs(_ridge_solve(x, y, 1e6)).sum()
    assert strong < weak


def test_ridge_does_not_penalise_the_bias_column():
    """``reg[-1, -1] = 0`` -- the bias must stay free, or a strongly regularised fit could not
    represent a constant offset. Assumes the bias is the LAST column, which is a shape contract
    the caller must honour."""
    x = np.concatenate([np.zeros((50, 3)), np.ones((50, 1))], 1)
    y = np.full((50, 2), 7.0)
    solved = _ridge_solve(x, y, lam=1e6)
    np.testing.assert_allclose(solved[-1], [7.0, 7.0], rtol=1e-3)


# --------------------------------------------------------------------------------------
# circular metrics
# --------------------------------------------------------------------------------------

def test_perfect_prediction_gives_zero_error_and_unit_r2():
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pred = np.stack([np.cos(angles), np.sin(angles)], 1)
    out = _circular_metrics(angles, pred)
    assert out["mae_deg"] == pytest.approx(0.0, abs=1e-6)
    assert out["r2"] == pytest.approx(1.0, abs=1e-6)


def test_error_wraps_across_the_branch_cut():
    """THE property. True angle just below 2*pi, prediction just above 0: the separation is
    2 degrees, not 358. A metric that subtracted raw angles would report the latter and make a
    near-perfect probe look useless."""
    true = np.array([2 * np.pi - np.deg2rad(1.0)])
    pred_angle = np.deg2rad(1.0)
    pred = np.stack([np.cos([pred_angle]), np.sin([pred_angle])], 1)
    assert _circular_metrics(true, pred)["mae_deg"] == pytest.approx(2.0, abs=1e-3)


def test_error_never_exceeds_180_degrees():
    """Wrapped to [0, pi], so the worst possible error is antipodal."""
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 2 * np.pi, 500)
    pred_angle = rng.uniform(0, 2 * np.pi, 500)
    pred = np.stack([np.cos(pred_angle), np.sin(pred_angle)], 1)
    assert _circular_metrics(true, pred)["mae_deg"] <= 180.0 + 1e-6


def test_median_and_mean_errors_are_both_reported():
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 2 * np.pi, 200)
    pred = np.stack([np.cos(true), np.sin(true)], 1)
    out = _circular_metrics(true, pred)
    assert {"r2", "mae_deg", "median_deg"} <= set(out)


def test_degenerate_targets_do_not_produce_a_nan_r2():
    """A constant target has zero total sum of squares. The ``max(ss_tot, 1e-12)`` guard exists
    so the row reports a finite number rather than a NaN a downstream mean would absorb."""
    true = np.zeros(32)
    pred = np.stack([np.cos(true), np.sin(true)], 1)
    assert np.isfinite(_circular_metrics(true, pred)["r2"])


# --------------------------------------------------------------------------------------
# the probe end to end
# --------------------------------------------------------------------------------------

def test_probe_recovers_a_linearly_encoded_angle():
    """Results are keyed ``angle{k}`` -- one block per regressed angle -- each with train/val/test
    metric dicts and the selected ridge."""
    feats, angles = circular_data(noise=0.01)
    metrics, _ = circular_linear_probe(feats, angles, seed=0)
    assert metrics["angle0"]["test"]["r2"] > 0.95
    assert metrics["angle0"]["test"]["mae_deg"] < 10.0


def test_probe_fails_on_features_that_carry_no_angle():
    """The negative control: random features must not predict the angle. Without it, a bug that
    leaked the target across the split would look like success."""
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(400, 12))
    angles = rng.uniform(0, 2 * np.pi, 400)
    metrics, _ = circular_linear_probe(feats, angles, seed=0)
    assert metrics["angle0"]["test"]["r2"] < 0.5


def test_probe_is_deterministic_for_a_seed():
    """The split is drawn from ``default_rng(seed)``, so the reported number is reproducible --
    unlike the linear probe in src/eval/linear_probe.py, which leaks to the global RNG."""
    feats, angles = circular_data()
    first, _ = circular_linear_probe(feats, angles, seed=3)
    second, _ = circular_linear_probe(feats, angles, seed=3)
    assert first["angle0"]["test"]["r2"] == second["angle0"]["test"]["r2"]


def test_a_different_seed_draws_a_different_split():
    feats, angles = circular_data(noise=0.3)
    a, _ = circular_linear_probe(feats, angles, seed=0)
    b, _ = circular_linear_probe(feats, angles, seed=1)
    assert a["angle0"]["test"]["r2"] != b["angle0"]["test"]["r2"]


def test_probe_reports_the_ridge_it_selected_from_the_grid():
    feats, angles = circular_data()
    grid = (1e-2, 1.0, 1e2)
    metrics, _ = circular_linear_probe(feats, angles, seed=0, ridge_grid=grid)
    assert metrics["angle0"]["ridge"] in grid


def test_probe_accepts_a_one_dimensional_angle_array():
    """``angles.ndim == 1`` is promoted to a single column, which lets the one-figure case share
    a code path with the two-figure one."""
    feats, angles = circular_data()
    metrics, _ = circular_linear_probe(feats, angles.reshape(-1), seed=0)
    assert set(metrics) == {"angle0"}


def test_probe_handles_two_angles_at_once():
    """The Yoda dataset regresses two figure rotations; each gets its own block."""
    feats, angles = circular_data()
    two = np.stack([angles, (angles * 2) % (2 * np.pi)], 1)
    metrics, viz = circular_linear_probe(feats, two, seed=0)
    assert set(metrics) == {"angle0", "angle1"}
    assert set(viz) == {"angle0", "angle1"}


def test_probe_returns_held_out_predictions_for_plotting():
    feats, angles = circular_data()
    _, viz = circular_linear_probe(feats, angles, seed=0)
    assert set(viz["angle0"]) == {"true", "pred", "pred_cos", "pred_sin"}


# --------------------------------------------------------------------------------------
# the low-data curve
# --------------------------------------------------------------------------------------

def test_low_data_curve_improves_with_more_training_data():
    """Keyed ``angle{k}`` with parallel lists, plus the ``train_sizes`` that produced them."""
    feats, angles = circular_data(n=600, noise=0.05)
    out = low_data_curve(feats, angles, train_sizes=(25, 200), n_seeds=2, base_seed=0)
    assert out["train_sizes"] == [25, 200]
    r2 = out["angle0"]["r2_mean"]
    assert r2[-1] > r2[0], "more labelled data must help"


def test_low_data_curve_reports_a_spread_across_seeds():
    """``mae_std`` over n_seeds draws -- the spread is what makes a difference between two
    curves interpretable rather than anecdotal."""
    feats, angles = circular_data(n=300, noise=0.2)
    out = low_data_curve(feats, angles, train_sizes=(30,), n_seeds=4, base_seed=0)
    assert out["angle0"]["mae_std"][0] > 0.0


def test_low_data_curve_error_falls_as_data_grows():
    feats, angles = circular_data(n=600, noise=0.05)
    out = low_data_curve(feats, angles, train_sizes=(25, 300), n_seeds=2, base_seed=0)
    mae = out["angle0"]["mae_mean"]
    assert mae[-1] < mae[0]


def test_low_data_curve_silently_caps_a_request_larger_than_the_pool():
    """FINDING, minor but pinned. ``min(ts, len(pool_idx))`` caps the requested train size
    without reporting it, so a curve can contain two points that used the same data and look
    like a plateau. ``train_sizes`` still echoes the *requested* value, not the used one.
    """
    feats, angles = circular_data(n=120)
    out = low_data_curve(feats, angles, train_sizes=(25, 100_000), n_seeds=1, base_seed=0)
    assert out["train_sizes"] == [25, 100_000], "the requested size is echoed back unchanged"
    assert np.isfinite(out["angle0"]["mae_mean"][-1]), "and the oversized request still runs"


def test_pca_2d_returns_two_components():
    rng = np.random.default_rng(0)
    assert pca_2d(rng.normal(size=(100, 9))).shape == (100, 2)


def test_pca_2d_recovers_a_planar_structure():
    """Points on a circle embedded in 9 dimensions must come back as a ring: the radii of the
    2-D projection should be nearly constant."""
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(2, 9))
    coords = pca_2d(np.stack([np.cos(angles), np.sin(angles)], 1) @ basis)
    radii = np.linalg.norm(coords - coords.mean(0), axis=1)
    assert radii.std() / radii.mean() < 0.2
