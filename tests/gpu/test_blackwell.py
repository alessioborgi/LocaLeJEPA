"""GPU-only checks. Run inside an allocation: ``pytest -m gpu``.

These are excluded from the default run so the core suite stays portable -- it must pass on a
workstation, on a login node and in a reviewer's clone. What lives here is everything that cannot
be established without real hardware, chiefly the silent-failure mode that dominated the cluster
bring-up: a Hopper-era build installs, imports, allocates and runs while emitting no Blackwell
kernels.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@CUDA
def test_the_device_is_blackwell():
    """sm_120 on RTX PRO 6000, sm_100 on B200 -- the two AICR GPU types."""
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) in {(12, 0), (10, 0)}, (
        f"expected a Blackwell capability, got {(major, minor)} on "
        f"{torch.cuda.get_device_name(0)}"
    )


@CUDA
def test_the_build_carries_kernels_for_this_device():
    """THE check. A build compiled only for older architectures runs without error and emits
    nothing for this GPU, so every kernel falls back to JIT or to a slower path. It cost the
    bring-up a day of suspicion before it was ruled out, and it is invisible to every other test
    in this suite because they all run on CPU.
    """
    major, minor = torch.cuda.get_device_capability(0)
    assert f"sm_{major}{minor}" in torch.cuda.get_arch_list(), (
        f"torch {torch.__version__} (cuda {torch.version.cuda}) has no sm_{major}{minor} "
        f"kernels; arch_list is {torch.cuda.get_arch_list()}"
    )


@CUDA
def test_matmul_is_finite_and_accurate_in_fp32_and_bf16():
    """bf16-mixed is what ``trainer=gpu`` and ``trainer=ddp`` use, so a broken bf16 path would
    corrupt every campaign run while leaving the fp32 tests green.

    REWRITTEN AFTER FAILING ON THE CLUSTER, and the hardware was right. The first version
    compared ``(a @ a).sum()`` between the two precisions with a 5% relative bound. That sum is a
    **near-total cancellation**: for standard-normal ``a`` the signed sum is ~0.06% of the
    unsigned mass (measured at n=1024), because 16.7M entries of random sign almost entirely
    annihilate. Relative error on a quantity that has cancelled away is amplified by exactly the
    factor it cancelled by, so the test was measuring the cancellation, not the arithmetic. On
    sm_120 it read 14.7% and failed; the same bf16 result is accurate to 0.29% in Frobenius norm.

    The tolerances, both stated as `research_rules/00` requires. **Must tolerate:** ~0.3%
    Frobenius-relative error, which is what bf16's ~8 mantissa bits give over 4096 accumulations
    -- measured 0.29% on CPU at n=1024. **Must catch:** a wrong or absent kernel, which moves
    this to O(1), two orders of magnitude clear of the 2% bound.

    Compared on the full matrix rather than any reduction of it, so no cancellation can hide an
    error or manufacture one.
    """
    a = torch.randn(4096, 4096, device="cuda")
    fp32 = a @ a
    bf16 = (a.bfloat16() @ a.bfloat16()).float()

    assert torch.isfinite(fp32).all(), "fp32 matmul produced non-finite values"
    assert torch.isfinite(bf16).all(), "bf16 matmul produced non-finite values"

    relative = float((bf16 - fp32).norm() / fp32.norm())
    assert relative < 0.02, (
        f"bf16 matmul differs from fp32 by {relative:.1%} in Frobenius norm; bf16 should give "
        f"~0.3% over this many accumulations"
    )


@CUDA
def test_the_regularizers_run_on_device_and_match_cpu():
    """Same seed, same input, both devices. The statistic is a reduction over random sketches,
    so it will not match bit-for-bit -- a 2% band catches a genuinely wrong kernel while
    tolerating a different reduction order (research_rules/00 on exact equality)."""
    from src.losses import ballreg_loss, sigreg_loss

    proj = torch.randn(4, 64, 16)
    for loss_fn in (sigreg_loss, ballreg_loss):
        torch.manual_seed(0)
        on_cpu = float(loss_fn(proj))
        torch.manual_seed(0)
        on_gpu = float(loss_fn(proj.cuda()))
        assert abs(on_cpu - on_gpu) / on_cpu < 0.02, f"{loss_fn.__name__}: {on_cpu} vs {on_gpu}"


@CUDA
def test_fast_pytorch_kmeans_works_on_this_device():
    """The dependency with the most Blackwell risk: a 9.7 KB pure-Python wheel over torch that
    may route through torch.compile/triton, and triton on a new architecture is a known source
    of friction. If this fails, ``model.reg_loss.kmeans_backend=torch`` is the documented
    fallback -- and the next test proves it is a real one."""
    from src.losses import _fast_pytorch_kmeans_labels

    labels = _fast_pytorch_kmeans_labels(torch.randn(256, 32, device="cuda"), num_clusters=4)
    assert labels.shape == (256,)
    assert int(labels.min()) >= 0 and int(labels.max()) < 4


@CUDA
def test_both_kmeans_backends_produce_a_valid_partition_on_device():
    from src.losses import local_loss, sigreg_loss

    proj = torch.randn(4, 64, 16, device="cuda")
    for backend in ("torch", "fast_pytorch_kmeans"):
        torch.manual_seed(0)
        out = local_loss(proj, loss=sigreg_loss, num_clusters=4, kmeans_backend=backend)
        assert torch.isfinite(out), f"{backend} produced {out}"


@CUDA
def test_the_local_regularizer_backpropagates_on_device():
    from src.losses import ballreg_loss, local_loss

    proj = torch.randn(4, 64, 16, device="cuda", requires_grad=True)
    local_loss(proj, loss=lambda p: ballreg_loss(p, center=True), num_clusters=4).backward()
    assert proj.grad is not None and torch.isfinite(proj.grad).all()


@CUDA
def test_a_module_moved_to_device_still_honours_the_layout_contract():
    """The (V, N, D) reshape must survive the device move -- a contiguity assumption that holds
    on CPU could in principle differ."""
    from torch import nn

    from conftest import TagEncoder, tagged_views
    from src.models.lejepa_module import LeJEPAModule

    module = LeJEPAModule(
        encoder=TagEncoder(4), optimizer=torch.optim.AdamW, projector=nn.Identity(),
        online_probe=None, embedding_dim=4, projection_dim=4, use_online_probe=False,
    ).cuda()

    _, proj = module.forward(tagged_views(3, 4).cuda())
    assert proj.shape == (4, 3, 4)
    for sample in range(3):
        for view in range(4):
            assert float(proj[view, sample, 0]) == sample * 10 + view
