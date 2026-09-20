"""The learning-rate schedule.

``lr_eta_min_ratio`` expresses the cosine floor as a fraction of the peak learning rate. It was
introduced to reproduce runs that had used an absolute ``eta_min`` of 1e-3, and the pinned ratios
therefore produce **three different schedule shapes** across the campaign's configs. That is the
parameterisation behind the "same schedule" wording in the README, so it is pinned here in code
rather than described in prose.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from conftest import FakeTrainer, TagEncoder
from src.models.lejepa_module import LeJEPAModule


def build(lr: float, ratio: float, warmup_epochs: int = 0, probe: bool = True) -> LeJEPAModule:
    module = LeJEPAModule(
        encoder=TagEncoder(4),
        optimizer=lambda params: torch.optim.AdamW(params, lr=lr),
        projector=nn.Linear(4, 4),
        online_probe=nn.Linear(4, 3) if probe else None,
        embedding_dim=4,
        projection_dim=4,
        use_online_probe=probe,
        warmup_epochs=warmup_epochs,
        lr_eta_min_ratio=ratio,
    )
    return module


def lr_curve(module: LeJEPAModule, total_steps: int, max_epochs: int) -> list[float]:
    """Drive the scheduler for the whole run and record the learning rate at every step."""
    module.trainer = FakeTrainer(total_steps, max_epochs)
    config = module.configure_optimizers()
    optimizer, scheduler = config["optimizer"], config["lr_scheduler"]["scheduler"]
    curve = []
    for _ in range(total_steps):
        curve.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return curve


# --------------------------------------------------------------------------------------
# eta_min is a ratio of the peak
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("lr, ratio", [(2e-3, 0.5), (1e-3, 1.0), (5e-4, 2.0)])
def test_eta_min_is_the_ratio_times_the_peak_learning_rate(lr, ratio):
    """All three campaign settings resolve to the same absolute floor of 1e-3, which is the
    historical value the ratios were pinned to reproduce."""
    module = build(lr, ratio)
    module.trainer = FakeTrainer(100, 1)
    scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert scheduler.eta_min == pytest.approx(ratio * lr)
    assert scheduler.eta_min == pytest.approx(1e-3), "all three pin the same absolute floor"


def test_eta_min_comes_from_the_first_param_group():
    """The probe is a second group with its own ``probe_lr``. ``eta_min`` must be derived from
    the encoder/projector group; taking it from the probe's group would change the schedule for
    every run that uses an online probe -- which is all of them."""
    module = build(lr=2e-3, ratio=0.5, probe=True)
    module.probe_lr = 7e-3
    module.trainer = FakeTrainer(100, 1)
    config = module.configure_optimizers()
    assert len(config["optimizer"].param_groups) == 2, "encoder/projector, then probe"
    assert config["lr_scheduler"]["scheduler"].eta_min == pytest.approx(0.5 * 2e-3)


# --------------------------------------------------------------------------------------
# the three shapes the campaign actually uses
# --------------------------------------------------------------------------------------

def test_ratio_below_one_decays():
    """`vit_in1k_sigreg`: lr 2e-3, ratio 0.5 -- the baseline arm. Halves over the run."""
    curve = lr_curve(build(2e-3, 0.5), total_steps=100, max_epochs=1)
    assert curve[0] == pytest.approx(2e-3, rel=1e-6)
    assert curve[-1] < curve[0]
    assert curve[-1] == pytest.approx(1e-3, rel=0.02)


def test_ratio_of_exactly_one_is_a_constant_learning_rate():
    """`vit_in1k_localballreg_l01_k4`: lr 1e-3, ratio 1.0 -- **our** arm.

    With eta_min equal to the peak, the cosine term has zero amplitude, so there is no decay at
    all. Pinned at tight tolerance because it is exactly constant, not approximately so.
    """
    curve = lr_curve(build(1e-3, 1.0), total_steps=100, max_epochs=1)
    assert max(curve) - min(curve) < 1e-12, "a ratio of 1.0 removes the schedule entirely"
    assert curve[0] == pytest.approx(1e-3)


def test_ratio_above_one_rises():
    """`vitl14_in1k_*_paper`: lr 5e-4, ratio 2.0 -- the ViT-L/14 paper-setting arms.

    eta_min above the peak inverts the cosine, so the learning rate **increases** from 5e-4 to
    1e-3 over the run. Legitimate as a record of what was launched, but it is not a decaying
    cosine, which is what the README describes.
    """
    curve = lr_curve(build(5e-4, 2.0), total_steps=100, max_epochs=1)
    assert curve[-1] > curve[0], "this arm's learning rate rises over training"
    assert curve[0] == pytest.approx(5e-4, rel=1e-6)
    assert curve[-1] == pytest.approx(1e-3, rel=0.02)


def test_the_three_campaign_arms_do_not_share_a_schedule_shape():
    """Stated as a single assertion because it is the claim under dispute: the baseline decays,
    ours is flat, the paper arms rise. "Same schedule" is false in the only sense that matters."""
    shapes = {}
    for name, (lr, ratio) in {
        "baseline": (2e-3, 0.5), "ours": (1e-3, 1.0), "paper": (5e-4, 2.0)
    }.items():
        curve = lr_curve(build(lr, ratio), total_steps=50, max_epochs=1)
        drift = curve[-1] - curve[0]
        shapes[name] = "flat" if abs(drift) < 1e-12 else ("down" if drift < 0 else "up")
    assert shapes == {"baseline": "down", "ours": "flat", "paper": "up"}


# --------------------------------------------------------------------------------------
# warm-up
# --------------------------------------------------------------------------------------

def test_warmup_starts_at_one_percent_and_climbs_to_the_peak():
    """LinearLR with start_factor=0.01, then cosine. The milestone is the warm-up step count."""
    module = build(1e-3, 0.5, warmup_epochs=2)
    curve = lr_curve(module, total_steps=100, max_epochs=10)   # 10 steps/epoch -> 20 warm-up
    assert curve[0] == pytest.approx(1e-3 * 0.01, rel=1e-3)
    assert curve[20] == pytest.approx(1e-3, rel=0.05), "peak reached at the milestone"
    assert curve[19] < curve[20]


def test_warmup_longer_than_the_run_degenerates_to_pure_warmup():
    """The edge case a short debug run hits: ``warmup_steps >= total_steps`` returns the warm-up
    schedule alone, with no cosine phase. Without this branch ``SequentialLR`` would be built
    with a milestone past the end of the run.

    It is also what a 3-batch smoke test experiences -- the effective learning rate there is one
    percent of the configured one, which is why smoke-test metrics mean nothing.
    """
    module = build(1e-3, 0.5, warmup_epochs=50)
    curve = lr_curve(module, total_steps=10, max_epochs=1)
    assert curve[0] == pytest.approx(1e-5, rel=1e-3), "1% of 1e-3"
    assert curve[-1] < 1e-3, "never reaches the peak"
    assert curve[-1] > curve[0], "but is still climbing"


def test_no_warmup_gives_a_pure_cosine():
    module = build(1e-3, 0.5, warmup_epochs=0)
    curve = lr_curve(module, total_steps=50, max_epochs=1)
    assert curve[0] == pytest.approx(1e-3, rel=1e-6), "starts at the peak, no ramp"


# --------------------------------------------------------------------------------------
# optimizer construction
# --------------------------------------------------------------------------------------

def test_probe_gets_its_own_learning_rate_and_weight_decay():
    module = build(2e-3, 0.5, probe=True)
    module.probe_lr, module.probe_weight_decay = 3e-4, 1e-7
    module.trainer = FakeTrainer(10, 1)
    groups = module.configure_optimizers()["optimizer"].param_groups
    assert groups[1]["lr"] == pytest.approx(3e-4)
    assert groups[1]["weight_decay"] == pytest.approx(1e-7)


def test_without_a_probe_there_is_a_single_param_group():
    module = build(2e-3, 0.5, probe=False)
    module.trainer = FakeTrainer(10, 1)
    assert len(module.configure_optimizers()["optimizer"].param_groups) == 1


def test_the_scheduler_is_stepped_per_step_not_per_epoch():
    """``interval: step``. Per-epoch stepping would stretch the cosine over 100x the intended
    horizon and leave the learning rate near its peak for the whole run."""
    module = build(1e-3, 0.5)
    module.trainer = FakeTrainer(10, 1)
    assert module.configure_optimizers()["lr_scheduler"]["interval"] == "step"
