"""The conformal autoencoder: intrinsic-dimension detection on the bobblehead torus.

This is the one evaluation in the repo whose *output is a claim about the data* rather than an
accuracy -- "the manifold has dimension d". A defect here does not lower a number, it produces a
different scientific conclusion, and nothing downstream would contradict it.

Two properties carry the whole method and both have closed forms, so both are checked against
mathematics rather than against the implementation:

* the per-sample Jacobian really is per-sample (the ``.sum()`` trick in
  ``latent_input_jacobian`` is correct only because samples are independent), and
* the orthogonality penalty touches only *off-diagonal* Gram entries -- including the diagonal
  would penalise gradient magnitude itself and drive every latent to zero, which reads exactly
  like a successful dimension detection reporting dimension 0.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.eval.conformal_autoencoder import (
    CAEConfig,
    ConformalAutoencoder,
    FeedForward,
    circular_recovery,
    latent_input_jacobian,
    orthogonality_loss,
    train_conformal_autoencoder,
)

CPU = torch.device("cpu")


# ======================================================================================
# architecture
# ======================================================================================

def test_the_layer_stack_is_n_tanh_blocks_then_n_linear_layers():
    """The paper's Definition B.1: "5 Tanh-activated layers followed by two linear layers, all of
    width w". Counted explicitly, because an off-by-one in either loop changes the model class
    while leaving every shape assertion satisfied."""
    net = FeedForward(in_dim=6, out_dim=3, width=8, n_tanh=5, n_linear=2).net

    assert sum(isinstance(m, nn.Tanh) for m in net) == 5
    assert sum(isinstance(m, nn.Linear) for m in net) == 7, "5 tanh-preceding + 2 linear tail"
    assert isinstance(net[-1], nn.Linear), "the last layer is linear, not an activation"


def test_only_the_final_layer_maps_to_the_output_dimension():
    """The linear tail is width -> width -> out_dim, not width -> out_dim -> out_dim. With
    latent_dim=8 and width=256 the second form would squeeze through an 8-wide bottleneck twice
    and impose a dimension the experiment is supposed to *measure*."""
    linears = [m for m in FeedForward(6, 3, 8, n_tanh=2, n_linear=3).net if isinstance(m, nn.Linear)]
    assert [m.out_features for m in linears] == [8, 8, 8, 8, 3]


def test_a_purely_linear_network_is_allowed():
    """n_tanh=0 gives an affine map. Used by the Jacobian tests below, where it has a closed
    form, so it must actually be constructible."""
    net = FeedForward(4, 2, 8, n_tanh=0, n_linear=1)
    assert net(torch.randn(3, 4)).shape == (3, 2)


def test_at_least_one_linear_layer_is_required():
    """n_linear=0 leaves the network ending in a tanh of the wrong width -- it would not even
    map to ``out_dim``. Refused with the reason stated."""
    with pytest.raises(ValueError, match=r"n_linear must be >= 1"):
        FeedForward(4, 2, 8, n_linear=0)


def test_the_autoencoder_round_trips_through_the_latent_layer():
    model = ConformalAutoencoder(input_dim=6, latent_dim=3, width=8, n_tanh=1)
    x = torch.randn(5, 6)
    reconstruction, latent = model(x)

    assert latent.shape == (5, 3) and reconstruction.shape == (5, 6)
    torch.testing.assert_close(latent, model.encode(x))
    torch.testing.assert_close(reconstruction, model.decode(latent))


def test_the_latent_layer_is_over_specified_on_purpose():
    """Definition 2.2. The latent width is an upper bound the method prunes down to; if the
    encoder were sized to the true dimension there would be nothing to detect."""
    model = ConformalAutoencoder(input_dim=60, latent_dim=8, width=16, n_tanh=1)
    assert model.latent_dim == 8 and model.encode(torch.randn(2, 60)).shape[1] == 8


# ======================================================================================
# the per-sample Jacobian
# ======================================================================================

def linear_model(weight: torch.Tensor) -> ConformalAutoencoder:
    """An autoencoder whose encoder is exactly ``x -> W x``, so dnu/dx = W for every sample."""
    out_dim, in_dim = weight.shape
    model = ConformalAutoencoder(input_dim=in_dim, latent_dim=out_dim, width=4,
                                 n_tanh=0, n_linear=1)
    with torch.no_grad():
        model.encoder.net[0].weight.copy_(weight)
        model.encoder.net[0].bias.zero_()
    return model


def test_the_jacobian_of_a_linear_encoder_is_its_weight_matrix():
    """The closed form. For ``nu(x) = Wx`` the Jacobian is W at every input, identically -- which
    pins the row/column convention as well as the values. A transposed result would have the
    right shape whenever L == n and would silently transpose the Gram matrix below."""
    weight = torch.randn(3, 5)
    x = torch.randn(7, 5, requires_grad=True)

    jac = latent_input_jacobian(linear_model(weight), x)

    assert jac.shape == (7, 3, 5)
    for sample in range(7):
        torch.testing.assert_close(jac[sample], weight, rtol=1e-5, atol=1e-6)


def test_each_sample_gets_its_own_gradient_not_the_batch_sum():
    """THE CORRECTNESS ARGUMENT FOR THE ``.sum()`` TRICK.

    ``torch.autograd.grad(z[:, j].sum(), x)`` computes one backward pass for the whole batch. It
    yields the correct *per-sample* gradients only because d(sum_b z[b,j])/dx[b'] vanishes for
    b' != b -- samples are independent. The code says so in a comment; this checks it.

    A nonlinear encoder is required: with a linear one every sample has the same Jacobian, so a
    batch-summed result would be indistinguishable from a per-sample one.
    """
    model = ConformalAutoencoder(input_dim=4, latent_dim=3, width=8, n_tanh=2)
    x = torch.randn(6, 4, requires_grad=True)
    batched = latent_input_jacobian(model, x)

    for sample in range(6):
        alone = x[sample:sample + 1].detach().clone().requires_grad_(True)
        torch.testing.assert_close(latent_input_jacobian(model, alone)[0], batched[sample],
                                   rtol=1e-5, atol=1e-6)


def test_one_samples_jacobian_does_not_move_when_another_sample_changes():
    """The same independence property from the other direction, and the one that would actually
    break: gradient leakage between samples would make the penalty depend on how the batch was
    shuffled."""
    model = ConformalAutoencoder(input_dim=4, latent_dim=2, width=8, n_tanh=2)
    x = torch.randn(4, 4)

    first = latent_input_jacobian(model, x.clone().requires_grad_(True))
    perturbed = x.clone()
    perturbed[3] += 10.0
    second = latent_input_jacobian(model, perturbed.requires_grad_(True))

    torch.testing.assert_close(first[:3], second[:3], rtol=1e-5, atol=1e-6)


def test_the_jacobian_stays_differentiable():
    """``create_graph=True``. The orthogonality penalty is a function *of the gradient*, so
    training requires a second derivative; without it the penalty would contribute no gradient
    at all and the method would quietly reduce to a plain autoencoder."""
    model = ConformalAutoencoder(input_dim=4, latent_dim=3, width=8, n_tanh=1)
    x = torch.randn(5, 4, requires_grad=True)

    orthogonality_loss(latent_input_jacobian(model, x)).backward()

    weight_grad = model.encoder.net[0].weight.grad
    assert weight_grad is not None and float(weight_grad.abs().sum()) > 0.0


# ======================================================================================
# the orthogonality penalty
# ======================================================================================

def test_orthogonal_latent_gradients_incur_no_penalty():
    """The objective's fixed point. Rows of the identity are mutually orthogonal, so the penalty
    is exactly zero -- not approximately."""
    jac = torch.eye(4).unsqueeze(0).expand(3, 4, 4).contiguous()
    assert float(orthogonality_loss(jac)) == 0.0
    assert float(orthogonality_loss(jac, norm="l1")) == 0.0


def test_the_penalty_ignores_the_diagonal_of_the_gram_matrix():
    """THE DEFECT THIS TEST EXISTS FOR.

    ``triu_indices(offset=1)`` excludes the diagonal. The diagonal entries are ||grad nu_j||^2 --
    the very quantities whose *non-vanishing* signals an active coordinate. Including them would
    penalise every latent's gradient magnitude, drive them all to zero, and report an inferred
    dimension of 0 or 1 for any dataset whatsoever, looking like a confident answer.

    Scaling the identity by 100 multiplies every diagonal entry by 10,000 and leaves every
    off-diagonal entry at zero. The penalty must not move.
    """
    identity = torch.eye(4).unsqueeze(0)
    assert float(orthogonality_loss(identity * 100.0)) == 0.0


def test_the_l2_penalty_is_the_sum_of_squared_inner_products():
    """Eq. (6), second term, against the closed form. Two latent gradients at a known angle:
    (1,0) and (cos t, sin t) have inner product cos t, so the penalty is cos^2 t."""
    for angle in (0.0, 0.3, 1.0):
        jac = torch.tensor([[[1.0, 0.0], [np.cos(angle), np.sin(angle)]]])
        assert float(orthogonality_loss(jac)) == pytest.approx(np.cos(angle) ** 2, abs=1e-6)


def test_the_l1_penalty_is_the_sum_of_absolute_inner_products():
    """The paper's Algorithm 1 note / Example C.1 variant, which shrinks redundant coordinates
    harder: |cos t| rather than cos^2 t, which is larger for every inner product below 1."""
    jac = torch.tensor([[[1.0, 0.0], [0.5, 0.866]]])
    assert float(orthogonality_loss(jac, norm="l1")) == pytest.approx(0.5, abs=1e-6)
    assert float(orthogonality_loss(jac, norm="l2")) == pytest.approx(0.25, abs=1e-6)


@pytest.mark.parametrize("bad", ["L1", "l_1", "", "ll", None])
def test_an_unrecognised_norm_is_refused(bad):
    """REPLACES ``test_an_unrecognised_norm_falls_back_to_l2``, which pinned the defect.

    There was no validation: anything that was not the exact string ``"l1"`` fell through to the
    l2 branch. So ``ortho_norm: L1`` in a config silently ran Eq. (6) instead of the variant that
    pushes harder toward lower dimension -- and the two give **different inferred dimensions**,
    which is this experiment's entire output. A wrong answer, reported confidently, from a
    capitalisation.

    ``configs/eval/conformal.yaml`` is splatted straight into ``CAEConfig(**kwargs)``, so an
    unknown *key* already raised -- but a bad *value* passed silently. Now both fail loudly.
    """
    jac = torch.tensor([[[1.0, 0.0], [0.5, 0.866]]])
    with pytest.raises(ValueError, match="ortho_norm must be one of"):
        orthogonality_loss(jac, norm=bad)


def test_both_valid_norms_are_still_accepted():
    """The guard must not be so tight that it breaks the documented configuration."""
    jac = torch.tensor([[[1.0, 0.0], [0.5, 0.866]]])
    assert float(orthogonality_loss(jac, norm="l1")) == pytest.approx(0.5, abs=1e-6)
    assert float(orthogonality_loss(jac, norm="l2")) == pytest.approx(0.25, abs=1e-6)


def test_a_bad_norm_is_caught_before_a_full_training_run():
    """It reaches ``orthogonality_loss`` on the first batch, so a typo costs one forward pass
    rather than the 2000 epochs the default protocol asks for."""
    X, _, _ = torus_features()
    with pytest.raises(ValueError, match="ortho_norm must be one of"):
        train_conformal_autoencoder(X, short_config(ortho_norm="L2"), CPU)


def test_every_distinct_pair_is_counted_exactly_once():
    """L(L-1)/2 pairs. Counting each pair twice would double the effective alpha -- invisible in
    any single run, and enough to make a reported alpha sweep incomparable across latent widths.
    """
    jac = torch.ones(1, 4, 3)          # all six pairs have inner product 3
    assert float(orthogonality_loss(jac)) == pytest.approx(6 * 9.0)


def test_the_penalty_is_averaged_over_the_batch_not_summed():
    """A summed penalty would scale with batch size, so the same alpha would mean something
    different at batch 64 and batch 256."""
    single = torch.tensor([[[1.0, 0.0], [0.6, 0.8]]])
    assert float(orthogonality_loss(single.expand(8, 2, 2))) == pytest.approx(
        float(orthogonality_loss(single)), abs=1e-6)


# ======================================================================================
# training
# ======================================================================================

def torus_features(n=192, dim=6, seed=0):
    """Points on a 2-torus linearly embedded in R^6: intrinsic dimension 2, ambient 6.

    Exactly the bobblehead structure in miniature -- two independent angles -- so the training
    loop is exercised on data with a known answer, even though 20 epochs is far too few to
    actually recover it.
    """
    rng = np.random.default_rng(seed)
    a, b = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    coords = np.stack([np.cos(a), np.sin(a), np.cos(b), np.sin(b)], axis=1)
    return torch.tensor(coords @ rng.normal(size=(4, dim)), dtype=torch.float32), a, b


def short_config(**kwargs):
    return CAEConfig(**{"latent_dim": 4, "width": 16, "n_tanh": 2, "n_linear": 1,
                        "epochs": 20, "batch_size": 64, "log_every": 5,
                        "grad_norm_eval_batch": 64, **kwargs})


def test_a_short_run_produces_a_complete_and_self_consistent_result():
    """Shapes and internal consistency of ``CAEResult``. The histories are what the paper's
    figures are drawn from, so a mismatched length silently misaligns a plot's x-axis."""
    X, _, _ = torus_features()
    model, result = train_conformal_autoencoder(X, short_config(), CPU)

    assert result.grad_norms.shape == (4,)
    assert result.latents.shape == (X.shape[0], 4)
    assert result.latent_variances.shape == (4,)

    n_logs = len(result.epochs_logged)
    for history in (result.loss_history, result.recon_history, result.ortho_history):
        assert history.shape == (n_logs,)
    assert result.grad_norm_history.shape == (n_logs, 4)
    assert result.inferred_dim == len(result.active_components)


def test_the_final_epoch_is_always_logged():
    """``epoch % log_every == 0 or epoch == epochs - 1``. Without the second clause the reported
    final loss would come from whichever earlier epoch happened to be a multiple -- up to
    ``log_every`` epochs of training reported as if it had not happened."""
    X, _, _ = torus_features()
    _, result = train_conformal_autoencoder(X, short_config(epochs=12, log_every=5), CPU)
    assert result.epochs_logged.tolist() == [0, 5, 10, 11]


def test_training_reduces_the_reconstruction_error():
    """The weakest possible claim about learning, and the right one for a 20-epoch test: not that
    it converges, only that the loop is wired up. Zero alpha isolates reconstruction."""
    X, _, _ = torus_features()
    _, result = train_conformal_autoencoder(X, short_config(alpha=0.0, epochs=40), CPU)
    assert result.recon_history[-1] < result.recon_history[0]


def test_the_same_seed_gives_the_same_result():
    """``research_rules/00``: a run that cannot be repeated cannot be debugged. The seed covers
    both torch (init, shuffling) and numpy."""
    X, _, _ = torus_features()
    _, first = train_conformal_autoencoder(X, short_config(seed=3), CPU)
    _, second = train_conformal_autoencoder(X, short_config(seed=3), CPU)

    np.testing.assert_allclose(first.grad_norms, second.grad_norms, rtol=0, atol=0)
    np.testing.assert_allclose(first.latents, second.latents, rtol=0, atol=0)


def test_a_different_seed_gives_a_different_result():
    """Otherwise the test above would pass on a model that ignores its seed entirely -- and the
    multi-seed spread the results are quoted with would be identically zero."""
    X, _, _ = torus_features()
    _, first = train_conformal_autoencoder(X, short_config(seed=0), CPU)
    _, second = train_conformal_autoencoder(X, short_config(seed=1), CPU)
    assert not np.allclose(first.grad_norms, second.grad_norms)


def test_standardisation_is_recorded_so_it_can_be_undone():
    """The latents are computed on standardised features. Anything mapping back to input space
    needs the mean and std, so they travel with the result rather than being recomputed by a
    caller that might standardise differently."""
    X, _, _ = torus_features()
    _, result = train_conformal_autoencoder(X, short_config(standardize=True), CPU)

    np.testing.assert_allclose(result.extras["feature_mean"][0], X.mean(0).numpy(),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.extras["feature_std"][0], X.std(0).numpy(),
                               rtol=1e-5, atol=1e-6)


def test_standardisation_can_be_switched_off_and_records_an_identity():
    """The no-op must still be *recorded* as a no-op, so a consumer can apply the same formula
    either way rather than branching."""
    X, _, _ = torus_features()
    _, result = train_conformal_autoencoder(X, short_config(standardize=False), CPU)

    np.testing.assert_allclose(result.extras["feature_mean"], np.zeros((1, X.shape[1])))
    np.testing.assert_allclose(result.extras["feature_std"], np.ones((1, X.shape[1])))


def test_a_constant_feature_does_not_divide_by_zero():
    """``clamp_min(1e-6)`` on the standard deviation. A dead feature dimension is normal in a
    frozen backbone's output, and without the clamp every latent becomes NaN at step one."""
    X, _, _ = torus_features()
    X[:, 0] = 2.5
    _, result = train_conformal_autoencoder(X, short_config(epochs=3), CPU)
    assert np.isfinite(result.latents).all()


def test_the_active_set_is_thresholded_relative_to_the_strongest_component():
    """``active_frac * max``, not an absolute cut. Gradient norms have no natural scale -- they
    depend on the feature normalisation and the network width -- so an absolute threshold would
    have to be retuned for every dataset, and silently reports the wrong dimension when it is
    not."""
    X, _, _ = torus_features()

    _, permissive = train_conformal_autoencoder(X, short_config(active_frac=0.0, seed=5), CPU)
    _, strict = train_conformal_autoencoder(X, short_config(active_frac=0.99, seed=5), CPU)

    assert permissive.inferred_dim == 4, "a zero threshold keeps every latent"
    assert strict.inferred_dim <= permissive.inferred_dim
    assert strict.inferred_dim >= 1, "the maximum component is never below the threshold"


def test_the_active_components_are_the_ones_above_the_threshold():
    """Stated as the definition rather than trusting the returned list, since ``inferred_dim`` is
    the reported scientific quantity and it is just this list's length."""
    X, _, _ = torus_features()
    _, result = train_conformal_autoencoder(X, short_config(active_frac=0.5, seed=2), CPU)

    threshold = 0.5 * result.grad_norms.max()
    assert result.active_components == [i for i, g in enumerate(result.grad_norms) if g > threshold]


def test_early_stopping_shortens_a_run_that_stops_improving():
    """``patience`` counts *logged checkpoints*, not epochs -- worth knowing, since with the
    default ``log_every=100`` a patience of 3 means 300 epochs, not 3.

    The exact count is ``patience + 1``: ``best_loss`` starts at infinity, so the first
    checkpoint always improves on it and resets the counter, and only the following ``patience``
    checkpoints can be stale. ``min_delta=1e9`` makes every subsequent one stale by construction.
    """
    X, _, _ = torus_features()
    _, result = train_conformal_autoencoder(
        X, short_config(epochs=100, log_every=1, patience=3, min_delta=1e9), CPU)

    assert result.epochs_logged.tolist() == [0, 1, 2, 3], "one warm-up checkpoint, then patience"
    assert result.epochs_logged[-1] < 99, "the run stopped early"


def test_patience_is_disabled_by_default():
    """patience=0 means "run every epoch". A default of 1 would silently truncate the paper's
    2000-epoch protocol at the first noisy checkpoint."""
    assert CAEConfig().patience == 0

    X, _, _ = torus_features()
    _, result = train_conformal_autoencoder(X, short_config(epochs=11, log_every=5), CPU)
    assert result.epochs_logged[-1] == 10


def test_the_defaults_match_the_papers_protocol():
    """Width 256, 5 tanh layers, 2 linear, 2000 epochs, alpha 1.0 -- Definition B.1 and Table 2.
    Any reported dimension was produced under these; a drifted default breaks the comparison to
    the published result without breaking anything visible."""
    cfg = CAEConfig()
    assert (cfg.width, cfg.n_tanh, cfg.n_linear) == (256, 5, 2)
    assert (cfg.epochs, cfg.alpha, cfg.ortho_norm) == (2000, 1.0, "l2")


# ======================================================================================
# circular recovery
# ======================================================================================

def test_angles_are_recovered_exactly_from_their_own_cosine_and_sine():
    """The best case, and the calibration point for every other number this function reports:
    when the latents literally contain (cos t, sin t), the regression is exact, so R^2 must be 1
    and the angular error 0. Anything less means the regression or the atan2 decode is wrong.
    """
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    latents = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    out = circular_recovery(latents, theta, active=[0, 1])["angle0"]
    assert out["cos_sin_r2"] == pytest.approx(1.0, abs=1e-8)
    assert out["mean_abs_angular_error_rad"] == pytest.approx(0.0, abs=1e-8)


def test_an_unrelated_latent_recovers_nothing():
    """The control. Without it, a test suite can only show the metric is high when it should be,
    never that it is low when it should be -- and a metric that is always high is the failure
    mode that matters."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 300)
    noise = rng.normal(size=(300, 2))

    out = circular_recovery(noise, theta, active=[0, 1])["angle0"]
    assert out["cos_sin_r2"] < 0.1
    assert out["mean_abs_angular_error_rad"] > 1.0, "near the ~pi/2 expected of a random guess"


def test_the_angular_error_wraps_around_the_circle():
    """``np.angle(np.exp(1j * d))`` maps the difference into (-pi, pi]. Predicting 0.01 rad when
    the truth is 2*pi - 0.01 is an error of 0.02, not of 6.26 -- and the naive subtraction would
    report the latter, making a near-perfect recovery look like a total failure."""
    theta = np.array([0.01, 0.02, 0.03])
    shifted = theta + 2 * np.pi - 0.02
    latents = np.stack([np.cos(shifted), np.sin(shifted)], axis=1)

    out = circular_recovery(latents, theta, active=[0, 1])["angle0"]
    assert out["mean_abs_angular_error_rad"] < 0.1


def test_each_angle_column_is_reported_separately():
    """The bobblehead manifold is a 2-torus: two independent angles. Reporting one pooled number
    would hide the case the experiment is designed to detect -- one figure's rotation recovered
    and the other's not."""
    theta_a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    rng = np.random.default_rng(1)
    theta_b = rng.uniform(0, 2 * np.pi, 200)
    latents = np.stack([np.cos(theta_a), np.sin(theta_a)], axis=1)

    out = circular_recovery(latents, np.stack([theta_a, theta_b], axis=1), active=[0, 1])
    assert set(out) == {"angle0", "angle1"}
    assert out["angle0"]["cos_sin_r2"] > 0.99, "recovered"
    assert out["angle1"]["cos_sin_r2"] < 0.1, "independent, so not recovered"


def test_a_one_dimensional_angle_array_is_promoted():
    """Callers pass (N,) for a single angle and (N, k) for several. Both must work."""
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    latents = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    assert set(circular_recovery(latents, theta, active=[0, 1])) == {"angle0"}


def test_an_empty_active_set_falls_back_to_all_latents():
    """``Z = latents[:, active] if active else latents``. An empty active set would otherwise
    index to a zero-width matrix and the lstsq would fail -- and an empty active set is a
    *possible outcome* of the detection, not a programming error."""
    theta = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    latents = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    out = circular_recovery(latents, theta, active=[])["angle0"]
    assert out["cos_sin_r2"] == pytest.approx(1.0, abs=1e-8)


def test_the_regression_carries_an_intercept():
    """A column of ones is appended. Without it, latents offset from the origin -- which is the
    normal case for an unconstrained encoder -- cannot represent (cos, sin) and the recovery
    would look poor for a perfectly good embedding."""
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    latents = np.stack([np.cos(theta) + 100.0, np.sin(theta) - 50.0], axis=1)

    assert circular_recovery(latents, theta, active=[0, 1])["angle0"]["cos_sin_r2"] == \
        pytest.approx(1.0, abs=1e-6)


def test_degrees_and_radians_agree():
    """Both are reported and the markdown tables quote degrees. A mismatch would misstate every
    published error by a factor of 57."""
    rng = np.random.default_rng(2)
    theta = rng.uniform(0, 2 * np.pi, 100)
    out = circular_recovery(rng.normal(size=(100, 2)), theta, active=[0, 1])["angle0"]

    assert out["mean_abs_angular_error_deg"] == pytest.approx(
        np.degrees(out["mean_abs_angular_error_rad"]))


def test_a_constant_angle_does_not_divide_by_zero():
    """``max(ss_tot, 1e-12)``. A degenerate label column -- every frame at the same angle -- has
    zero total variance; R^2 is undefined and must not be NaN, because a NaN would propagate
    into an aggregate and take an otherwise valid table with it."""
    theta = np.full(50, 1.234)
    rng = np.random.default_rng(3)
    out = circular_recovery(rng.normal(size=(50, 2)), theta, active=[0, 1])["angle0"]
    assert np.isfinite(out["cos_sin_r2"])
