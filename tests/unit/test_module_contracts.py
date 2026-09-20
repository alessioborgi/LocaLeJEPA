"""The layout contracts every loss in this repository depends on.

``encode`` returns a flat ``(N*V, D)`` tensor and ``forward`` reshapes it to ``(V, N, D)``. If
that reshape were transposed the wrong way, every regularizer would still run, every loss would
still be finite, and the statistics would be computed across the wrong axis. Nothing currently
checks it.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from conftest import TagEncoder, tagged_views
from src.models.lejepa_module import LeJEPAModule

D = 4


def build(embedding_dim: int = D, projection_dim: int = D) -> LeJEPAModule:
    return LeJEPAModule(
        encoder=TagEncoder(embedding_dim),
        optimizer=torch.optim.AdamW,
        projector=nn.Identity(),
        online_probe=nn.Identity(),
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
    )


# --------------------------------------------------------------------------------------
# _view_groups
# --------------------------------------------------------------------------------------

def test_single_tensor_is_treated_as_one_resolution_group():
    groups, batch, views = LeJEPAModule._view_groups(torch.zeros(3, 4, 3, 2, 2))
    assert len(groups) == 1 and batch == 3 and views == 4


def test_a_list_is_treated_as_several_resolution_groups():
    groups, batch, views = LeJEPAModule._view_groups(
        [torch.zeros(3, 2, 3, 4, 4), torch.zeros(3, 6, 3, 2, 2)]
    )
    assert len(groups) == 2 and batch == 3 and views == 8, "view count must sum across groups"


def test_a_tuple_is_accepted_like_a_list():
    """MultiCropTransform returns a tuple for mixed resolutions; the default collate preserves
    that, so both container types must work."""
    _, batch, views = LeJEPAModule._view_groups(
        (torch.zeros(2, 2, 3, 4, 4), torch.zeros(2, 3, 3, 2, 2))
    )
    assert batch == 2 and views == 5


def test_rejects_an_empty_group_list():
    with pytest.raises(RuntimeError, match="at least one group"):
        LeJEPAModule._view_groups([])


@pytest.mark.parametrize("shape", [(3, 4, 2, 2), (3, 4, 3, 2, 2, 1), (8, 4)])
def test_rejects_anything_but_five_dimensions(shape):
    with pytest.raises(RuntimeError, match=r"shape \(N, V, C, H, W\)"):
        LeJEPAModule._view_groups(torch.zeros(*shape))


def test_rejects_groups_with_different_batch_sizes():
    """A collate bug that desynchronised the groups would otherwise reshape into nonsense."""
    with pytest.raises(RuntimeError, match="same batch size"):
        LeJEPAModule._view_groups([torch.zeros(3, 2, 3, 2, 2), torch.zeros(4, 2, 3, 2, 2)])


# --------------------------------------------------------------------------------------
# encode -- row ordering
# --------------------------------------------------------------------------------------

def test_encode_is_sample_major():
    """Row ``n*V + v`` must be view ``v`` of sample ``n``.

    The docstring states this and the online probe relies on it: ``training_step`` reshapes to
    ``(N, V, D)`` and takes ``[:, 0]`` as "the first view of every sample". If the ordering were
    view-major, the probe would silently train on view 0 of the first N/V samples instead.
    """
    n, v = 3, 4
    emb = build().encode(tagged_views(n, v))

    assert emb.shape == (n * v, D)
    for sample in range(n):
        for view in range(v):
            assert float(emb[sample * v + view, 0]) == sample * 10 + view


def test_encode_puts_global_views_first_across_groups():
    """The grouped layout's contract: globals occupy the low view indices, locals follow. The
    invariance loss's ``num_global`` slice depends on it."""
    n = 3
    globals_ = tagged_views(n, 2, size=3)               # 3x3 crops, values n*10 + {0,1}
    locals_ = torch.zeros(n, 2, 3, 2, 2)                # 2x2 crops -- a genuinely mixed resolution
    for sample in range(n):
        for view in range(2):
            locals_[sample, view] = sample * 10 + 2 + view   # values n*10 + {2,3}

    emb = build().encode([globals_, locals_])
    assert emb.shape == (n * 4, D)
    for sample in range(n):
        for view in range(4):
            assert float(emb[sample * 4 + view, 0]) == sample * 10 + view


def test_encode_rejects_an_encoder_with_the_wrong_output_width():
    """A mismatch between the encoder's width and ``embedding_dim`` must fail loudly here rather
    than reshaping into a silently wrong (V, N, D)."""
    module = build(embedding_dim=D)
    module.encoder = TagEncoder(D + 1)
    with pytest.raises(RuntimeError, match=r"Expected embeddings with shape"):
        module.encode(tagged_views(2, 2))


# --------------------------------------------------------------------------------------
# forward -- the (V, N, D) transpose
# --------------------------------------------------------------------------------------

def test_forward_returns_view_major_projections():
    """``proj[v, n]`` is view ``v`` of sample ``n`` -- the transpose of ``encode``'s layout.

    This is the single most load-bearing line in the module: every loss indexes ``proj[:k]`` as
    "the first k views". Transposed, ``num_global`` would slice samples instead of views.
    """
    n, v = 3, 4
    _, proj = build().forward(tagged_views(n, v))

    assert proj.shape == (v, n, D)
    for sample in range(n):
        for view in range(v):
            assert float(proj[view, sample, 0]) == sample * 10 + view


def test_forward_rejects_a_projector_with_the_wrong_output_width():
    module = build(embedding_dim=D, projection_dim=D)
    module.projector = nn.Linear(D, D + 3)
    with pytest.raises(RuntimeError, match=r"Projector must return"):
        module.forward(tagged_views(2, 2))


def test_forward_and_encode_agree_on_the_same_values():
    """The transpose must be a pure reshape: the multiset of values is unchanged."""
    emb, proj = build().forward(tagged_views(4, 3))
    torch.testing.assert_close(
        proj.transpose(0, 1).reshape(-1, D), emb, rtol=0, atol=0
    )
