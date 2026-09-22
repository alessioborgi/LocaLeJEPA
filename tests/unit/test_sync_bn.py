"""Synchronized BatchNorm for escnn models (F48).

Every D4-RN50 launch on more than one GPU died before its first step: Lightning's
``trainer.sync_batchnorm`` calls torch's converter, which re-adds every child through
``add_module``, and escnn's ``SequentialModule`` refuses the re-admission with a FieldType
assertion. One GPU never converts, so nothing local saw it. These pin the in-place conversion:
same statistics, same outputs in eval mode, plain torch containers converted as torch does, and
the module's hook doing it only under DDP.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.utils.sync_bn import convert_sync_batchnorm_inplace

escnn = pytest.importorskip("escnn", reason="escnn is a post-create install (environment-aicr.yml)")

from src.models.components import e2resnet  # noqa: E402


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def build_d4():
    """A fresh D4 ResNet-18 with running statistics to carry (one training forward). Built per
    test: escnn modules hold non-leaf tensors after a forward, so they cannot be deep-copied."""
    torch.manual_seed(0)
    model = e2resnet.build_e2resnet(group="d4", depth=18, num_classes=16, pool="equivariant")
    model.train()
    with torch.no_grad():
        model(torch.randn(4, 3, 33, 33, generator=torch.Generator().manual_seed(3)))
    return model.eval()


def test_torch_s_own_converter_refuses_the_escnn_model():
    """The defect itself, kept as a test so an escnn release that fixes it is noticed."""
    with pytest.raises(AssertionError, match="regular"):
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(build_d4())


def test_in_place_conversion_keeps_the_escnn_model_and_its_outputs():
    model = build_d4()
    x = torch.randn(2, 3, 33, 33, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        before = model(x)
    n_before = len(bn_modules(model))
    converted = convert_sync_batchnorm_inplace(model)          # in place: the same object
    n_after = len(bn_modules(converted))
    assert converted is model and n_before > 0 and n_after == n_before
    assert all(type(m) is torch.nn.SyncBatchNorm for m in bn_modules(converted))
    assert not any(type(m) is torch.nn.BatchNorm3d for m in bn_modules(converted))
    with torch.no_grad():
        after = converted.eval()(x)
    torch.testing.assert_close(after, before, rtol=0, atol=0), "eval-mode outputs use the copied running statistics"


def test_plain_containers_are_converted_exactly_as_torch_converts_them():
    torch.manual_seed(1)
    mlp = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.BatchNorm1d(16), torch.nn.ReLU(), torch.nn.Linear(16, 4),
                              torch.nn.BatchNorm1d(4, affine=False))          # one layer without affine parameters
    mlp[1].qconfig = "kept-as-is"                                            # torch's converter carries qconfig along; so must ours
    mlp.train()
    with torch.no_grad():
        mlp(torch.randn(6, 8))
    import copy

    theirs = torch.nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(mlp))
    ours = convert_sync_batchnorm_inplace(copy.deepcopy(mlp))
    assert [type(m) for m in ours.modules()] == [type(m) for m in theirs.modules()]
    for a, b in zip(ours.state_dict().values(), theirs.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(m.training for m in bn_modules(ours)), "training flags travel with the module"
    assert ours[1].qconfig == "kept-as-is" and ours[4].weight is None


def test_the_module_converts_only_under_ddp_and_only_when_asked():
    from src.models.lejepa_module import LeJEPAModule

    def make(sync: bool, world_size: int):
        torch.manual_seed(0)
        module = LeJEPAModule(
            encoder=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 32), torch.nn.BatchNorm1d(32)),
            optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
            projector=torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.BatchNorm1d(16)),
            embedding_dim=32, projection_dim=16, num_classes=10, sync_batchnorm=sync,
        )
        module._trainer = SimpleNamespace(world_size=world_size)
        module.setup("fit")
        return module

    assert not any(isinstance(m, torch.nn.SyncBatchNorm) for m in make(True, 1).modules()), "one GPU: nothing to sync"
    assert not any(isinstance(m, torch.nn.SyncBatchNorm) for m in make(False, 4).modules()), "asked not to"
    converted = make(True, 4)
    assert sum(isinstance(m, torch.nn.SyncBatchNorm) for m in converted.modules()) == 2, "encoder and projector alike"
