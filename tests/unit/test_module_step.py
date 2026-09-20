"""The training step: loss arithmetic, covariance diagnostics, and the online probe.

The diagnostics matter as much as the loss here. ``_covariance_metrics`` is what the
BallRegRadius write-up reads to tell a collapsed run from a healthy one, so a wrong statistic
would not crash anything -- it would quietly mislabel a collapse.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from conftest import TagEncoder, tagged_views
from src.losses import invariance_loss, sigreg_loss
from src.models.lejepa_module import LeJEPAModule


def build(lamb: float = 0.02, probe: bool = True, **kwargs) -> LeJEPAModule:
    module = LeJEPAModule(
        encoder=TagEncoder(4),
        optimizer=torch.optim.AdamW,
        projector=nn.Identity(),
        online_probe=nn.Linear(4, 3) if probe else None,
        embedding_dim=4,
        projection_dim=4,
        num_classes=3,
        lamb=lamb,
        use_online_probe=probe,
        **kwargs,
    )
    module.log = lambda *a, **k: None          # Lightning's log needs a Trainer; we only need
    return module                              # the arithmetic, so make it a no-op


# --------------------------------------------------------------------------------------
# the loss is a convex combination
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("lamb", [0.0, 0.02, 0.1, 1.0])
def test_total_loss_is_lamb_reg_plus_one_minus_lamb_inv(lamb):
    """loss = lamb * reg + (1 - lamb) * inv, recomputed from the reported components. A swapped
    weighting would still train and still look plausible; only the arithmetic catches it."""
    module = build(lamb=lamb)
    torch.manual_seed(0)
    _, metrics = module.model_step(tagged_views(4, 3))

    reg = metrics["sigreg_loss"]
    inv = metrics["invariance_loss"]
    torch.testing.assert_close(metrics["loss"], lamb * reg + (1.0 - lamb) * inv,
                               rtol=1e-6, atol=1e-7)


def test_reported_contributions_sum_to_the_loss():
    module = build(lamb=0.1)
    torch.manual_seed(0)
    _, m = module.model_step(tagged_views(4, 3))
    torch.testing.assert_close(m["inv_contrib"] + m["reg_contrib"], m["loss"],
                               rtol=1e-6, atol=1e-7)


def test_lamb_of_zero_is_pure_invariance():
    module = build(lamb=0.0)
    torch.manual_seed(0)
    _, m = module.model_step(tagged_views(4, 3))
    torch.testing.assert_close(m["loss"], m["invariance_loss"], rtol=1e-6, atol=1e-7)


def test_the_regularizer_is_logged_under_its_own_name():
    """``reg_name`` keys the metric, so a config change from sigreg to localereg renames the
    logged series. Runs are compared across those series, so the key must track the function."""
    module = build()
    assert module.reg_name == "sigreg_loss"

    from functools import partial
    module_ball = LeJEPAModule(
        encoder=TagEncoder(4), optimizer=torch.optim.AdamW, projector=nn.Identity(),
        online_probe=None, embedding_dim=4, projection_dim=4, use_online_probe=False,
        reg_loss=partial(sigreg_loss, center=True),
    )
    assert module_ball.reg_name == "sigreg_loss", "partials report the wrapped function's name"


# --------------------------------------------------------------------------------------
# covariance diagnostics
# --------------------------------------------------------------------------------------

def test_covariance_metrics_on_a_known_input():
    """Whitened input: unit diagonal, zero off-diagonal, zero mean. Checked against the values
    the statistics must take rather than against the code's own output.

    Tolerances are sampling tolerances on 8192 draws (~1% standard error), not slack.
    """
    module = build()
    torch.manual_seed(0)
    proj = torch.randn(2, 4096, 8)
    m = module._covariance_metrics(proj)

    assert float(m["cov_diag_mean"]) == pytest.approx(1.0, rel=0.05)
    assert float(m["cov_offdiag_mean"]) < 0.05
    assert float(m["mean_abs"]) < 0.05
    assert float(m["std_mean"]) == pytest.approx(1.0, rel=0.05)


def test_covariance_metrics_detect_collapse(collapsed_proj):
    """The signal the radius write-up relies on: a collapsed batch has a near-zero covariance
    diagonal. Measured at 1e-5 there against 0.60 for a healthy run."""
    m = build()._covariance_metrics(collapsed_proj)
    assert float(m["cov_diag_mean"]) < 1e-3
    assert float(m["std_mean"]) < 1e-2


def test_norm_mean_matches_the_expected_chi_norm():
    """E||x|| for x ~ N(0, I_d) is approximately sqrt(d) for moderate d; d=16 gives ~3.94."""
    torch.manual_seed(0)
    m = build()._covariance_metrics(torch.randn(2, 4096, 16))
    assert float(m["norm_mean"]) == pytest.approx(3.94, rel=0.03)


def test_covariance_metrics_are_computed_without_gradients():
    """Decorated ``@torch.no_grad``: diagnostics must never contribute to the graph."""
    proj = torch.randn(2, 64, 8, requires_grad=True)
    for value in build()._covariance_metrics(proj).values():
        assert not value.requires_grad


# --------------------------------------------------------------------------------------
# the online probe
# --------------------------------------------------------------------------------------

def test_probe_sees_the_first_view_of_every_sample():
    """``emb.reshape(N, V, D)[:, 0]`` -- which is only "the first view of every sample" because
    ``encode`` is sample-major. Assert it by value, using the tagged encoder: the probe's input
    must be n*10 + 0 for each sample."""
    module = build()
    seen = {}

    class Recorder(nn.Module):
        def forward(self, x):
            seen["input"] = x.clone()
            return torch.zeros(x.shape[0], 3)

    module.online_probe = Recorder()
    n, v = 4, 3
    module.training_step((tagged_views(n, v), torch.zeros(n, dtype=torch.long)), 0)

    assert seen["input"].shape == (n, 4)
    for sample in range(n):
        assert float(seen["input"][sample, 0]) == sample * 10 + 0


def test_probe_input_is_detached_from_the_encoder():
    """The probe is a diagnostic, not an objective: its gradient must not reach the encoder, or
    the "frozen-feature" probe would be training the features it is meant to measure -- and the
    reported online-probe accuracy would be measuring a representation the probe helped shape.

    REWRITTEN AFTER A SURVIVING MUTANT. The previous version was **vacuous**: it used the default
    ``TagEncoder``, which has no parameters, so ``emb.requires_grad`` was False whether or not
    ``.detach()`` was there. Deleting the detach passed. A ``requires_grad`` assertion is only
    evidence if something upstream actually requires grad -- hence the parameterised encoder here,
    and the explicit check that the setup is non-degenerate.
    """
    module = build()
    module.encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    captured = {}

    class Recorder(nn.Module):
        def forward(self, x):
            captured["requires_grad"] = x.requires_grad
            return torch.zeros(x.shape[0], 3, requires_grad=True)

    module.online_probe = Recorder()
    views = tagged_views(3, 2)
    emb, _ = module.forward(views)
    assert emb.requires_grad, "the encoder must be trainable, or this test proves nothing"

    module.training_step((views, torch.zeros(3, dtype=torch.long)), 0)
    assert captured["requires_grad"] is False


def test_the_probe_contributes_no_gradient_to_the_encoder():
    """The same guarantee stated as a consequence rather than as a flag, so it holds even if the
    detach is moved or reimplemented: the encoder's gradient from the full training loss must
    equal its gradient from the LeJEPA loss alone.

    Reseeded around both calls, since the regularizer draws fresh sketch directions per call.
    """
    module = build()
    module.encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    weight = module.encoder[1].weight
    views, labels = tagged_views(4, 2), torch.zeros(4, dtype=torch.long)

    torch.manual_seed(0)
    with_probe = torch.autograd.grad(module.training_step((views, labels), 0), weight)[0]
    torch.manual_seed(0)
    _, metrics = module.model_step(views)
    without_probe = torch.autograd.grad(metrics["loss"], weight)[0]

    torch.testing.assert_close(with_probe, without_probe, rtol=1e-6, atol=1e-7)


def test_probe_loss_is_added_to_the_total():
    module = build()
    views, labels = tagged_views(4, 2), torch.zeros(4, dtype=torch.long)
    torch.manual_seed(0)
    with_probe = module.training_step((views, labels), 0)

    module_without = build(probe=False)
    torch.manual_seed(0)
    _, metrics = module_without.model_step(views)
    assert float(with_probe.detach()) > float(metrics["loss"].detach()), "the probe's cross-entropy is added on"


def test_no_probe_means_no_probe_metric():
    module = build(probe=False)
    loss = module.training_step((tagged_views(3, 2), torch.zeros(3, dtype=torch.long)), 0)
    assert torch.isfinite(loss)


def test_validation_step_is_a_no_op_without_a_probe():
    module = build(probe=False)
    assert module.validation_step((tagged_views(2, 2), torch.zeros(2, dtype=torch.long)), 0) is None


# --------------------------------------------------------------------------------------
# the nGPT hypersphere hook
# --------------------------------------------------------------------------------------

def test_optimizer_step_renormalises_encoders_that_ask_for_it():
    """``optimizer_step`` calls ``encoder.norm_weights_()`` when present -- the nGPT convention
    that keeps weight rows on the unit hypersphere. Encoders without it must be untouched."""
    module = build()
    calls = []

    class NormalisingEncoder(TagEncoder):
        def norm_weights_(self):
            calls.append(1)

    module.encoder = NormalisingEncoder(4)
    module.optimizer_step = LeJEPAModule.optimizer_step.__get__(module)

    import lightning.pytorch as pl
    original = pl.LightningModule.optimizer_step
    pl.LightningModule.optimizer_step = lambda self, *a, **k: None
    try:
        module.optimizer_step()
    finally:
        pl.LightningModule.optimizer_step = original
    assert calls == [1]
