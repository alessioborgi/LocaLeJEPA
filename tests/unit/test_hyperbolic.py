"""The Poincare-ball head of the hyperbolic ViT.

Adapted third-party code (Ermolov et al., CVPR 2022), but it is *in the campaign* -- H-ViT is one
of the three backbone families in the results tables -- so it is Tier A, not vendored-and-omitted.

The whole construction rests on one invariant: every output lies strictly inside the ball of
radius 1/sqrt(c). A point at or outside the boundary has infinite hyperbolic distance to the
origin, so the downstream metric silently becomes meaningless rather than raising. That invariant
is checked here under conditions far harsher than training produces -- norms of 1e6, exact zeros,
float32 extremes -- because the cost of discovering it fails after an 800-epoch run is the run.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.components.hyperbolic import ToPoincare, expmap0, project


CURVATURES = [0.01, 0.1, 1.0, 10.0]


# --------------------------------------------------------------------------------------
# project: the containment guarantee
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("c", CURVATURES)
def test_projection_puts_every_point_strictly_inside_the_ball(c):
    """Including points 1e6 times the ball radius. The clamp is the only thing between an
    exploding gradient step and an infinite distance."""
    radius = 1.0 / math.sqrt(c)
    x = torch.randn(256, 16) * radius * 1e6
    norms = project(x, c=c).norm(dim=-1)

    assert (norms < radius).all(), "a projected point must be inside the open ball"
    assert torch.isfinite(norms).all()


@pytest.mark.parametrize("c", CURVATURES)
def test_the_boundary_is_approached_but_never_reached(c):
    """``eps=1e-3`` buys a margin of exactly ``1e-3 / sqrt(c)``. Pinned, because the margin is
    what keeps ``artanh`` finite downstream, and a 'harmless' eps change would remove it."""
    radius = 1.0 / math.sqrt(c)
    far = torch.full((1, 8), 1e9)
    achieved = float(project(far, c=c).norm())

    assert achieved == pytest.approx((1.0 - 1e-3) * radius, rel=1e-5)
    assert achieved < radius


def test_a_point_already_inside_is_left_untouched():
    """``scale`` clamps at 1.0, so the projection is the identity on the interior. Without the
    clamp it would *push points outward* onto the boundary sphere -- collapsing the ball's
    interior, which is where all the geometry lives."""
    inside = torch.randn(64, 8) * 0.01
    torch.testing.assert_close(project(inside, c=1.0), inside, rtol=0, atol=0)


def test_the_origin_survives_projection():
    """Norm zero divides by zero unless clamped to ``finfo.tiny``. An all-zero feature vector is
    not hypothetical -- a dead ReLU block produces one."""
    origin = torch.zeros(4, 8)
    out = project(origin, c=1.0)
    assert torch.isfinite(out).all() and float(out.abs().max()) == 0.0


# --------------------------------------------------------------------------------------
# expmap0: the exponential map at the origin
# --------------------------------------------------------------------------------------

def test_the_exponential_map_fixes_the_origin():
    """exp_0(0) = 0 in any Riemannian manifold. Here it is also the division-by-zero case."""
    out = expmap0(torch.zeros(4, 8), c=1.0)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros(4, 8), atol=1e-7, rtol=0)


@pytest.mark.parametrize("c", CURVATURES)
def test_the_exponential_map_scales_the_norm_by_tanh_and_keeps_the_direction(c):
    """The closed form: ``exp_0(u) = tanh(sqrt(c)|u|) * u / (sqrt(c)|u|)``, so the image norm is
    ``tanh(sqrt(c)|u|)/sqrt(c)`` and the direction is unchanged.

    Asserted against the formula, not against the code -- and the direction half is the one that
    matters, since a defect that perturbed direction would leave every norm-based test happy
    while scrambling the metric.

    The tangent norm is scaled to ``0.5/sqrt(c)`` so the map stays well clear of saturation; the
    closed form does NOT hold once ``project`` clamps, which the next test covers deliberately.
    """
    u = torch.nn.functional.normalize(torch.randn(128, 16), dim=-1) * (0.5 / math.sqrt(c))
    mapped = expmap0(u, c=c)

    sqrt_c = math.sqrt(c)
    expected_norm = torch.tanh(sqrt_c * u.norm(dim=-1)) / sqrt_c
    torch.testing.assert_close(mapped.norm(dim=-1), expected_norm, rtol=1e-5, atol=1e-6)

    cosine = torch.nn.functional.cosine_similarity(mapped, u, dim=-1)
    torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=0, atol=1e-5)


@pytest.mark.parametrize("c", CURVATURES)
def test_the_map_saturates_onto_a_single_shell_for_large_tangent_norms(c):
    """WHY THE CAMPAIGN USES c=0.1 AND clip_r=2.3, stated as a test.

    Two things compose here. ``tanh`` saturates at 1, so the image norm approaches the ball
    radius; then ``project`` clamps anything within ``eps`` of the boundary onto the shell at
    ``(1 - 1e-3)/sqrt(c)``. Past that point the map is no longer injective in the radial
    direction: tangent vectors of norm 10, 100 and 1000 all land on the *same shell*, and every
    difference in feature magnitude is destroyed.

    This is not a defect -- it is what keeps the representation finite -- but it is the reason
    the hyperparameters are not free. At ``c=1.0``, 16-dimensional features of ordinary scale
    (norm ~ 4) are already saturated; at ``c=0.1`` with the tangent norm clipped to 2.3 they are
    not, which is the configuration H-ViT actually trains in.

    Discovered by writing the closed-form test above without the saturation guard and watching
    it fail at exactly 1e-3 relative error for c >= 1.
    """
    direction = torch.nn.functional.normalize(torch.randn(1, 16), dim=-1)
    shell = (1.0 - 1e-3) / math.sqrt(c)

    norms = torch.stack([
        # lengths in units of the ball radius, so "saturated" means the same thing at every c
        expmap0(direction * length / math.sqrt(c), c=c).norm()
        for length in (10.0, 100.0, 1000.0)
    ])
    torch.testing.assert_close(norms, torch.full((3,), shell), rtol=1e-5, atol=0)


def test_the_configured_h_vit_operating_point_is_not_saturated():
    """The complement of the test above, on the values the campaign actually uses: c=0.1 with the
    tangent norm clipped to 2.3 gives ``tanh(0.316 * 2.3) = 0.62`` of the ball radius, comfortably
    short of the 0.999 shell. If either hyperparameter moves, this fails and says so."""
    at_the_clip = float(expmap0(torch.ones(1, 1) * 2.3, c=0.1).norm()) * math.sqrt(0.1)
    assert at_the_clip == pytest.approx(math.tanh(math.sqrt(0.1) * 2.3), rel=1e-5)
    assert at_the_clip < 0.9, "the configured operating point must leave gradient headroom"


def test_the_map_is_radially_monotone():
    """A longer tangent vector must land further from the origin. tanh is monotone, so this holds
    exactly; it is the property that makes the head order-preserving, which is what a metric
    learning objective needs from it."""
    directions = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
    lengths = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0, 20.0]).unsqueeze(-1)
    norms = expmap0(directions * lengths, c=0.1).norm(dim=-1)

    assert (norms[1:] > norms[:-1]).all()


def test_near_the_origin_the_map_is_close_to_the_identity():
    """tanh(x) ~ x for small x, so the ball is locally Euclidean. This is why a randomly
    initialised head does not immediately saturate."""
    small = torch.randn(64, 8) * 1e-3
    torch.testing.assert_close(expmap0(small, c=1.0), small, rtol=1e-3, atol=1e-7)


# --------------------------------------------------------------------------------------
# ToPoincare
# --------------------------------------------------------------------------------------

def test_the_module_output_is_always_a_valid_ball_point():
    layer = ToPoincare(c=0.1, clip_radius=2.3)
    out = layer(torch.randn(256, 32) * 1e4)

    assert (out.norm(dim=-1) < layer.ball_radius).all()
    assert torch.isfinite(out).all()


def test_the_ball_radius_is_one_over_root_c():
    assert ToPoincare(c=0.25).ball_radius == pytest.approx(2.0)
    assert ToPoincare(c=0.1).ball_radius == pytest.approx(1.0 / math.sqrt(0.1))


def test_clipping_bounds_the_tangent_norm_before_the_map():
    """clip_r=2.3 is the official implementation's value. It bounds the tangent vector, which
    bounds how far into the saturated region of tanh the map can go -- and the saturated region
    is where gradients vanish. Setting it aside is a training-stability decision, so the
    behaviour is pinned rather than left implicit."""
    layer = ToPoincare(c=0.1, clip_radius=2.3)
    huge = torch.randn(32, 8) * 1000.0

    expected = expmap0(torch.nn.functional.normalize(huge, dim=-1) * 2.3, c=0.1)
    torch.testing.assert_close(layer(huge), expected, rtol=1e-5, atol=1e-6)


def test_clipping_leaves_a_short_vector_alone():
    """Only the norms above clip_r are touched -- the clamp is one-sided. Scaling short vectors
    *up* to clip_r would erase all magnitude information below the threshold."""
    layer = ToPoincare(c=0.1, clip_radius=2.3)
    short = torch.randn(32, 8) * 0.01
    torch.testing.assert_close(layer(short), expmap0(short, c=0.1), rtol=0, atol=0)


def test_clipping_can_be_switched_off():
    layer = ToPoincare(c=0.1, clip_radius=None)
    x = torch.randn(16, 8) * 50.0
    torch.testing.assert_close(layer(x), expmap0(x, c=0.1), rtol=0, atol=0)
    assert (layer(x).norm(dim=-1) < layer.ball_radius).all(), "still inside the ball"


@pytest.mark.parametrize("c", [0.0, -1.0])
def test_a_non_positive_curvature_is_refused(c):
    """c <= 0 is not a hyperbolic ball at all: ``1/sqrt(c)`` is infinite or complex. Refused at
    construction, where a config typo is cheap, rather than producing NaNs at step one."""
    with pytest.raises(ValueError, match="curvature parameter c must be positive"):
        ToPoincare(c=c)


@pytest.mark.parametrize("clip_radius", [0.0, -2.3])
def test_a_non_positive_clip_radius_is_refused(clip_radius):
    """clip_r=0 would map every input to the origin -- a model that trains, converges, and has
    learned nothing at all."""
    with pytest.raises(ValueError, match="clip_radius must be positive or None"):
        ToPoincare(c=0.1, clip_radius=clip_radius)


def test_the_repr_states_both_parameters():
    """These end up in the printed model summary, which is the only place the curvature is
    visible in a run log."""
    assert "c=0.1" in repr(ToPoincare(c=0.1, clip_radius=2.3))
    assert "clip_radius=2.3" in repr(ToPoincare(c=0.1, clip_radius=2.3))


def test_gradients_flow_through_the_whole_head():
    """Both clamps use ``clamp(max=1.0)`` on a scale factor rather than clamping the tensor, so
    the gradient survives even for a point that was clipped. A ``detach`` or a hard clamp here
    would silently stop learning for exactly the large-norm features that most need to move."""
    x = torch.randn(8, 16, requires_grad=True) * 1.0
    x.retain_grad()
    ToPoincare(c=0.1, clip_radius=2.3)(x).square().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------------------
# HyperbolicViT
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hvit():
    from src.models.components.hyperbolic_vit import HyperbolicViT

    return HyperbolicViT(model_name="vit_tiny_patch16_224", embedding_dim=16,
                         img_size=64, hyp_c=0.1, clip_r=2.3)


def test_the_encoder_maps_images_into_the_ball(hvit):
    out = hvit(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 16)
    assert (out.norm(dim=-1) < hvit.ball_radius).all()


def test_the_encoders_ball_radius_is_the_heads(hvit):
    """Exposed so ``src/eval`` can normalise distances without reaching into the head."""
    assert hvit.ball_radius == hvit.to_poincare.ball_radius


def test_the_tangent_head_starts_orthogonal_with_zero_bias(hvit):
    """The official initialisation. Orthogonal rows mean the initial embedding preserves the
    encoder's feature geometry instead of collapsing it onto a random subspace; a zero bias means
    the map starts centred at the ball's origin, where tanh is linear."""
    weight = hvit.tangent_head.weight
    gram = weight @ weight.T
    torch.testing.assert_close(gram, torch.eye(16), rtol=0, atol=1e-5)
    assert float(hvit.tangent_head.bias.detach().abs().max()) == 0.0


def test_a_non_positive_embedding_dimension_is_refused():
    from src.models.components.hyperbolic_vit import HyperbolicViT

    with pytest.raises(ValueError, match="embedding_dim must be positive"):
        HyperbolicViT(model_name="vit_tiny_patch16_224", embedding_dim=0, img_size=64)


def test_the_body_is_headless_so_the_tangent_head_sees_features(hvit):
    """``num_classes=0`` in the timm call. If the body kept a classifier, the tangent head would
    be fitted on logits and ``num_features`` would not be the width it is sized against."""
    assert hvit.body.num_classes == 0
    assert hvit.tangent_head.in_features == hvit.body.num_features


# --------------------------------------------------------------------------------------
# the saturation guard
# --------------------------------------------------------------------------------------

def test_a_saturating_curvature_and_clip_radius_warn():
    """Ledger T.suite-P2-9 made actionable.

    ``hyp_c`` and ``clip_r`` are not independently sweepable: once ``tanh(sqrt(c) * clip_r)``
    approaches 1, every tangent vector lands on the same shell and all magnitude information is
    destroyed -- with no error, and a loss curve that looks entirely healthy. Someone sweeping
    ``hyp_c`` upward would find the representation quietly collapsing and have nothing to point
    at.

    A warning rather than a raise: it is a bad operating point, not an invalid one, and the
    ablation that demonstrates the collapse needs to be runnable.
    """
    from src.models.components.hyperbolic_vit import HyperbolicViT

    with pytest.warns(RuntimeWarning, match="exponential map is saturated"):
        HyperbolicViT(model_name="vit_tiny_patch16_224", embedding_dim=16, img_size=64,
                      hyp_c=4.0, clip_r=2.3)


def test_the_configured_operating_point_does_not_warn(recwarn):
    """c=0.1 / clip_r=2.3 reaches 0.62 of the ball radius. If this ever starts warning, the
    campaign's H-ViT runs have moved into the saturated regime and the guard is telling you
    before the results do."""
    from src.models.components.hyperbolic_vit import HyperbolicViT

    HyperbolicViT(model_name="vit_tiny_patch16_224", embedding_dim=16, img_size=64,
                  hyp_c=0.1, clip_r=2.3)

    assert not [w for w in recwarn if "saturated" in str(w.message)]


def test_no_warning_when_clipping_is_disabled():
    """With ``clip_r=None`` there is no bound on the tangent norm to compare against, so the
    check has nothing to say -- and must not crash trying."""
    from src.models.components.hyperbolic_vit import HyperbolicViT

    HyperbolicViT(model_name="vit_tiny_patch16_224", embedding_dim=16, img_size=64,
                  hyp_c=4.0, clip_r=None)
