"""Shared fixtures.

Everything here is synthetic and offline: no dataset download, no checkpoint on disk, no GPU.
The core suite must run on a workstation, on a cluster login node, and in a reviewer's fresh
clone, so anything that reaches the network or a filesystem outside ``tmp_path`` belongs in the
``gpu`` or ``slow`` tiers instead.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch


def pytest_sessionstart(session):
    """Refuse to run while ``tests/mutate.py`` has a defect planted in ``src/``.

    Without this, a test run started in another shell during a mutation run reports failures from
    a deliberately corrupted tree -- which looks exactly like a real regression, and is gone by
    the time you look again because the harness reverts after each mutant. Worse, a ``git add -A``
    in that window can stage the planted defect and commit it under a message saying the opposite.
    Both happened here on 2026-09-19; the second one only missed by a few seconds.
    """
    if os.environ.get("LOCALEJEPA_MUTATION_RUNNER"):
        return
    lock = Path(__file__).resolve().parents[1] / ".mutation-in-progress"
    if lock.exists():
        pytest.exit(
            "A mutation run is in progress -- src/ currently holds a PLANTED DEFECT.\n"
            "Any failure you see now is fabricated, and committing now would capture it.\n"
            "Wait for tests/mutate.py to finish, or delete " + str(lock) + " if it crashed.",
            returncode=3,
        )


# --------------------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _seeded():
    """Seed every test.

    ``autouse`` because several code paths under test draw randomly at call time --
    ``_sketch_test`` samples sketch directions, ``_torch_kmeans_labels`` samples initial
    centroids -- so an unseeded test is flaky rather than merely non-reproducible.
    """
    torch.manual_seed(0)
    yield


# --------------------------------------------------------------------------------------
# projections in the (V, N, D) layout every loss expects
# --------------------------------------------------------------------------------------

V_DEFAULT, N_DEFAULT, D_DEFAULT = 4, 64, 16


@pytest.fixture
def gaussian_proj() -> torch.Tensor:
    """Standard-normal projections: the distribution ``sigreg_loss`` targets."""
    return torch.randn(V_DEFAULT, N_DEFAULT, D_DEFAULT)


@pytest.fixture
def ball_proj() -> torch.Tensor:
    """Uniform on the centred d-ball of radius sqrt(d+2): the distribution ``ballreg_loss``
    targets, and the radius at which its covariance is the identity."""
    return uniform_ball(V_DEFAULT, N_DEFAULT, D_DEFAULT)


@pytest.fixture
def collapsed_proj() -> torch.Tensor:
    """Every sample of every view at (nearly) the same point -- the failure mode the regularizers
    exist to penalise.

    The collapse must span views as well as samples: ``_covariance_metrics`` flattens (V, N, D)
    to (V*N, D), so a cloud that is constant within each view but differs across views still has
    full variance and is not collapsed at all. Tiny noise keeps the statistics finite.
    """
    base = torch.randn(1, 1, D_DEFAULT)
    return base.expand(V_DEFAULT, N_DEFAULT, D_DEFAULT) \
        + torch.randn(V_DEFAULT, N_DEFAULT, D_DEFAULT) * 1e-4


@pytest.fixture
def identical_views() -> torch.Tensor:
    """All V views of each sample identical: ``invariance_loss`` must be exactly zero."""
    single = torch.randn(1, N_DEFAULT, D_DEFAULT)
    return single.expand(V_DEFAULT, N_DEFAULT, D_DEFAULT).contiguous()


# --------------------------------------------------------------------------------------
# helpers importable by tests
# --------------------------------------------------------------------------------------

def uniform_ball(v: int, n: int, d: int, radius: float | None = None) -> torch.Tensor:
    """Draw uniformly from the centred d-ball.

    Direction on the sphere times a radius scaled by ``u**(1/d)``, which is the inverse-CDF of
    the radial density ``d * r**(d-1)`` -- a uniform draw in ``r`` would concentrate near the
    centre and would not be the distribution the target CF describes.
    """
    r_max = math.sqrt(d + 2) if radius is None else radius
    direction = torch.randn(v, n, d)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    radius_draw = torch.rand(v, n, 1) ** (1.0 / d)
    return direction * radius_draw * r_max


# --------------------------------------------------------------------------------------
# LeJEPAModule construction without lightning's Trainer
# --------------------------------------------------------------------------------------

class TagEncoder(torch.nn.Module):
    """Returns the constant baked into each view, so row ordering can be asserted by value.

    Every pixel of view ``v`` of sample ``n`` is set to ``n*10 + v`` by the fixtures below; this
    encoder simply surfaces that number, making the (V, N, D) layout directly readable.
    """

    def __init__(self, embedding_dim: int = 4) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        return flat[:, : self.embedding_dim].contiguous()


class FakeTrainer:
    """The two attributes ``_build_scheduler`` reads, and nothing else.

    A real ``Trainer`` needs a datamodule, an accelerator and a fit loop to populate
    ``estimated_stepping_batches``; the scheduler only needs the number, so faking it keeps the
    schedule tests at microseconds and free of lightning's setup.
    """

    def __init__(self, estimated_stepping_batches: int, max_epochs: int) -> None:
        self.estimated_stepping_batches = estimated_stepping_batches
        self.max_epochs = max_epochs
        self.world_size = 1


def tagged_views(n: int, v: int, size: int = 2) -> torch.Tensor:
    """(N, V, C, H, W) where every pixel of view ``v`` of sample ``n`` equals ``n*10 + v``."""
    views = torch.zeros(n, v, 3, size, size)
    for sample in range(n):
        for view in range(v):
            views[sample, view] = sample * 10 + view
    return views


# --------------------------------------------------------------------------------------
# on-disk fixtures: synthetic, tiny, and confined to tmp_path
# --------------------------------------------------------------------------------------

def write_imagefolder(root, classes: int = 3, per_class: int = 4, size: int = 8) -> None:
    """A minimal ImageFolder tree of real JPEGs.

    Real files rather than mocks, because the code under test goes through
    ``torchvision.datasets.ImageFolder``, whose class ordering, ``class_to_idx`` mapping and
    sample ordering are precisely what several tests assert. Colour varies per class so a
    transform pipeline has something to preserve.
    """
    from PIL import Image

    for c in range(classes):
        class_dir = root / f"n{c:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            shade = 20 + 70 * c
            Image.new("RGB", (size, size), (shade, 255 - shade, (shade + 40) % 256)).save(
                class_dir / f"{i:04d}.JPEG"
            )


@pytest.fixture
def imagefolder(tmp_path):
    """3 classes x 4 images. Returns the root."""
    root = tmp_path / "data"
    write_imagefolder(root)
    return root


@pytest.fixture
def imagenette_tree(tmp_path):
    """The layout ImagenetteDataModule expects: <root>/imagenette2-160/{train,val}/<class>/*."""
    base = tmp_path / "imagenette" / "imagenette2-160"
    for split in ("train", "val"):
        write_imagefolder(base / split, classes=3, per_class=4)
    return tmp_path / "imagenette"


@pytest.fixture
def imagenet_tree(tmp_path):
    """The layout ImageNet1KDataModule expects, including the 061417 sub-path."""
    base = tmp_path / "imagenet" / "imagenet_full_size" / "061417"
    for split in ("train", "val"):
        write_imagefolder(base / split, classes=3, per_class=4)
    return tmp_path / "imagenet"


def lightning_checkpoint(encoder_state: dict, prefix: str = "encoder.", epoch: int = 7) -> dict:
    """A checkpoint shaped like one ``src/train.py`` writes, without running Lightning.

    ``load_backbone`` only reads ``state_dict`` and ``epoch``, so faking those exactly exercises
    the prefix-stripping and strictness logic that a real checkpoint would.
    """
    return {
        "state_dict": {f"{prefix}{k}": v for k, v in encoder_state.items()},
        "epoch": epoch,
        "global_step": epoch * 100,
    }
