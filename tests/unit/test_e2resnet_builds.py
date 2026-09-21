"""The vendored D4-RN50 encoder builds and is D4-invariant -- in this environment (F46).

Before 2026-09-21 no test touched ``e2resnet.py``: ``pyproject.toml`` said escnn was installed on
neither machine, and the D4 cells died at instantiation on the GPU node. Two things are pinned:
escnn's null-space solver runs on its deterministic SVD path (the randomized one is refused by
scikit-learn >= 1.4 and is unseeded), and the built encoder is invariant to the group it claims.
"""

import numpy as np
import pytest
import torch

escnn = pytest.importorskip("escnn", reason="escnn is a post-create install (environment-aicr.yml)")

from src.models.components import e2resnet  # noqa: E402


def test_the_randomized_svd_path_is_switched_off():
    import escnn.group._numerical as numerical

    assert numerical.randomized_svd is None


@pytest.fixture(scope="module")
def d4_encoder():
    torch.manual_seed(0)
    return e2resnet.build_e2resnet(group="d4", depth=18, num_classes=16, initialize=True).eval()


def test_d4_resnet18_builds_and_maps_images_to_embeddings(d4_encoder):
    y = d4_encoder(torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(1)))
    assert tuple(y.shape) == (2, 16) and torch.isfinite(y).all()


def test_d4_resnet18_is_invariant_to_the_dihedral_group(d4_encoder):
    """A rotation by 90 degrees or a horizontal flip of the input must not move the embedding;
    an arbitrary 40-degree rotation is NOT in the group and must move it, or the check is vacuous.

    Odd input size on purpose: every stride-2 stage (33 -> 17 -> 9 -> 5 -> 3) then samples a grid
    that is symmetric under the group, which is what exact invariance needs. At an EVEN size the
    strided grid is off-centre by half a pixel and the same model moves by ~0.4 under a quarter
    turn (measured at 32 px) -- so at the campaign's 128 px crops the D4 encoder is invariant only
    approximately. That is the architecture's property (equivision's too), recorded, not fixed."""
    x = torch.randn(1, 3, 33, 33, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        ref = d4_encoder(x)
        for g in (torch.rot90(x, 1, dims=(2, 3)), torch.rot90(x, 2, dims=(2, 3)), torch.flip(x, dims=(3,))):
            assert torch.allclose(d4_encoder(g), ref, atol=1e-4, rtol=1e-4), (d4_encoder(g) - ref).abs().max()
        other = torch.nn.functional.grid_sample(
            x, torch.nn.functional.affine_grid(
                torch.tensor([[[np.cos(0.7), -np.sin(0.7), 0.0], [np.sin(0.7), np.cos(0.7), 0.0]]], dtype=torch.float32),  # 40 deg
                x.shape, align_corners=False), align_corners=False)
        assert not torch.allclose(d4_encoder(other), ref, atol=1e-2)
