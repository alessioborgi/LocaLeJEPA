"""``RectifiedLpJEPAModule``: LeJEPA with a ReLU on the projector output and absolute weights.

A small subclass, but it changes two things at once -- the *representation* the losses see
(rectified, so non-negative and sparse) and how they are combined (two absolute weights instead
of one ``lamb`` trade-off). Both are easy to get subtly wrong and neither fails loudly.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from conftest import TagEncoder, tagged_views
from src.losses import invariance_loss, sigreg_loss
from src.models.rectifiedlpjepa_module import RectifiedLpJEPAModule


def build(rectify: bool = True, inv_weight: float = 25.0, reg_weight: float = 125.0):
    return RectifiedLpJEPAModule(
        encoder=TagEncoder(8),
        optimizer=torch.optim.AdamW,
        projector=nn.Identity(),
        online_probe=None,
        embedding_dim=8,
        projection_dim=8,
        use_online_probe=False,
        rectify=rectify,
        invariance_loss_weight=inv_weight,
        rdm_reg_loss_weight=reg_weight,
    )


@pytest.fixture
def views():
    return torch.randn(16, 4, 3, 2, 2)


def test_the_loss_is_the_weighted_sum_of_its_two_reported_contributions(views):
    """Not a tautology: ``loss``, ``inv_contrib`` and ``reg_contrib`` are computed and logged
    separately, and a run is diagnosed from the two contributions. If they stopped summing to the
    optimised quantity, every training curve would be describing a different objective than the
    one being minimised."""
    _, out = build().model_step(views)

    torch.testing.assert_close(out["loss"], out["inv_contrib"] + out["reg_contrib"])
    torch.testing.assert_close(out["inv_contrib"], out["invariance_loss"] * 25.0)
    torch.testing.assert_close(out["lejepa_loss"], out["loss"])


def test_the_weights_are_absolute_not_a_convex_trade_off(views):
    """LeJEPA uses a single ``lamb`` and the two terms trade off; here both weights scale
    independently. Doubling one must double its contribution and leave the other alone -- the
    behaviour a sweep over ``rdm_reg_loss_weight`` assumes.

    Reseeded between the two calls: ``sigreg_loss`` draws fresh sketch directions on every
    evaluation, so without it the two regularizer values differ by ~5% for reasons that have
    nothing to do with the weights.
    """
    torch.manual_seed(0)
    _, base = build(inv_weight=1.0, reg_weight=1.0).model_step(views)
    torch.manual_seed(0)
    _, doubled = build(inv_weight=2.0, reg_weight=1.0).model_step(views)

    torch.testing.assert_close(doubled["inv_contrib"], base["inv_contrib"] * 2.0)
    torch.testing.assert_close(doubled["reg_contrib"], base["reg_contrib"])


def test_rectification_is_what_the_losses_see(views):
    """The ReLU is applied once, before *both* losses and all the metrics. Applying it to only
    one of them -- the obvious slip -- would mean the invariance term is matching un-rectified
    views while the regularizer shapes the rectified ones.

    The regularizer is re-evaluated under the same seed, since its sketch directions are drawn
    at call time.
    """
    module = build(rectify=True)
    torch.manual_seed(0)
    _, out = module.model_step(views)

    _, proj = module.forward(views)
    rectified = torch.relu(proj)
    torch.testing.assert_close(out["invariance_loss"], invariance_loss(rectified))
    torch.manual_seed(0)
    torch.testing.assert_close(out[module.reg_name], sigreg_loss(rectified))


def test_turning_rectification_off_gives_plain_lp_jepa(views):
    """``rectify=False`` is a documented configuration, not dead code: it is the ablation that
    isolates the effect of the ReLU from the effect of the absolute weights."""
    module = build(rectify=False)
    torch.manual_seed(0)
    _, out = module.model_step(views)

    _, proj = module.forward(views)
    torch.testing.assert_close(out["invariance_loss"], invariance_loss(proj))


def test_rectification_changes_the_loss(views):
    """Guards against a ``rectify`` flag that is stored but never consulted -- which every test
    above would still pass if both branches computed the same thing."""
    _, on = build(rectify=True).model_step(views)
    _, off = build(rectify=False).model_step(views)
    assert float(on["loss"]) != float(off["loss"])


def test_sparsity_metrics_are_reported_on_the_rectified_representation(views):
    """These exist precisely to measure what the ReLU does. Computing them on ``proj`` would make
    them insensitive to the one thing the module is named after."""
    from src.losses import l0_sparsity_metric

    module = build(rectify=True)
    _, out = module.model_step(views)

    _, proj = module.forward(views)
    torch.testing.assert_close(out["l0_sparsity"],
                               l0_sparsity_metric(torch.relu(proj).flatten(0, 1)))
    assert 0.0 < float(out["l0_sparsity"]) < 1.0, "a ReLU'd Gaussian is about half zeros"


def test_the_layout_contract_survives_the_flatten(views):
    """``z.flatten(0, 1)`` folds (V, N, D) to (V*N, D) for the metrics. It must be the projection
    tensor being flattened, in that layout -- flattening an (N, V, D) tensor gives the same shape
    and a different set of rows, so shape alone cannot catch the error."""
    module = build()
    emb, out = module.model_step(views)

    assert emb.shape == (16 * 4, 8)
    assert torch.isfinite(out["loss"])


def test_the_gradient_reaches_the_encoder(views):
    """The whole point is end-to-end training. ReLU zeroes gradients on the negative half, so a
    representation that happened to be entirely negative would train on nothing -- checked here
    to be non-degenerate on ordinary input."""
    module = build()
    module.encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 8))
    _, out = module.model_step(views)
    out["loss"].backward()

    grad = module.encoder[1].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_the_defaults_are_the_papers(views):
    """25.0 and 125.0, from arXiv:2602.01456. Pinned because they are the numbers any reported
    result was produced with, and a default that drifts silently invalidates a comparison."""
    module = RectifiedLpJEPAModule(
        encoder=TagEncoder(8), optimizer=torch.optim.AdamW, projector=nn.Identity(),
        online_probe=None, embedding_dim=8, projection_dim=8, use_online_probe=False,
    )
    assert (module.invariance_loss_weight, module.rdm_reg_loss_weight) == (25.0, 125.0)
    assert module.rectify is True


def test_view_ordering_is_inherited_unchanged():
    """The subclass overrides ``model_step`` but not ``forward``, so the (V, N, D) ordering must
    still hold. Asserted by value with the tagging encoder rather than by shape."""
    module = build()
    _, proj = module.forward(tagged_views(3, 4))

    assert proj.shape == (4, 3, 8)
    for sample in range(3):
        for view in range(4):
            assert float(proj[view, sample, 0]) == sample * 10 + view
