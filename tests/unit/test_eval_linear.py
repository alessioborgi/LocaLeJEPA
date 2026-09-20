"""The linear probe -- the ``Linear (%)`` column of every reported table.

Two properties carry the reported numbers: the learning-rate search must actually select on
held-out data (not on what it then reports), and the whole thing must be reproducible from a
seed, or two runs of the same evaluation disagree.
"""

from __future__ import annotations

import pytest
import torch

from src.eval.linear_probe import _evaluate, _standardize_stats, _train_head, run_linear_probe


def separable(n_per_class: int = 60, num_classes: int = 4, dim: int = 12):
    centres = torch.eye(num_classes, dim) * 5.0
    feats = torch.cat([centres[c] + 0.3 * torch.randn(n_per_class, dim)
                       for c in range(num_classes)])
    labels = torch.arange(num_classes).repeat_interleave(n_per_class)
    perm = torch.randperm(labels.numel())
    return feats[perm], labels[perm]


# --------------------------------------------------------------------------------------
# standardisation
# --------------------------------------------------------------------------------------

def test_standardize_stats_are_the_per_dimension_mean_and_std():
    x = torch.randn(500, 6) * torch.tensor([1.0, 5.0, 0.1, 2.0, 3.0, 0.5]) + 7.0
    mean, std = _standardize_stats(x)
    torch.testing.assert_close(mean, x.mean(0), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(std, x.std(0), rtol=1e-5, atol=1e-6)


def test_standardize_std_is_floored_to_avoid_dividing_by_zero():
    """A constant feature dimension has zero variance. Without the clamp the probe would divide
    by zero and every downstream accuracy would be NaN -- which a results table would render as
    a blank cell rather than an error."""
    x = torch.randn(100, 4)
    x[:, 2] = 3.0
    _, std = _standardize_stats(x)
    assert float(std[2]) > 0.0 and torch.isfinite(std).all()


# --------------------------------------------------------------------------------------
# training and evaluation
# --------------------------------------------------------------------------------------

def test_train_head_is_not_reproducible_from_its_seed_alone():
    """FINDING, pinned deliberately.

    ``_train_head`` takes a ``seed``, but that seed only feeds the ``torch.Generator`` used for
    the shuffle permutation. The head itself -- ``nn.Linear(...)`` followed by
    ``trunc_normal_`` -- draws from the **global** RNG, which the seed never touches. So two
    calls with the same seed in the same process start from different weights.

    Measured consequence at the probe level: repeated ``run_linear_probe(seed=0)`` calls span
    **0.417 percentage points** of top-1 on easy synthetic data. The reported tables quote
    Linear (%) to 0.1 pp, and ``eval_indomain.py`` probes backbones in a loop, so the Nth
    backbone's initialisation depends on how many ran before it -- meaning ``--only <name>``
    does not reproduce the number the full sweep produced for that backbone.

    Pinned rather than fixed: seeding the global RNG inside ``_train_head`` would change every
    reported Linear (%) number, so it belongs with the experimental-settings discussion.
    """
    x, y = separable()
    args = dict(num_classes=4, lr=0.1, epochs=3, batch_size=64, weight_decay=0.0,
                momentum=0.9, mean=None, std=None, device=torch.device("cpu"))
    first, _ = _train_head(x, y, seed=0, **args)
    second, _ = _train_head(x, y, seed=0, **args)
    assert not torch.allclose(first, second), (
        "if this now passes, the head initialisation has been brought under the seed -- a "
        "welcome fix, but one that moves every published Linear (%) number"
    )


def test_train_head_is_reproducible_once_the_global_rng_is_fixed():
    """The counterpart, and the workaround: seeding globally before the call makes the probe
    exactly reproducible. Any caller that needs a reproducible number must do this."""
    x, y = separable()
    args = dict(num_classes=4, lr=0.1, epochs=3, batch_size=64, weight_decay=0.0,
                momentum=0.9, mean=None, std=None, device=torch.device("cpu"))
    torch.manual_seed(999)
    first, first_bias = _train_head(x, y, seed=0, **args)
    torch.manual_seed(999)
    second, second_bias = _train_head(x, y, seed=0, **args)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_bias, second_bias, rtol=0, atol=0)


def test_the_seed_does_control_the_shuffle_order():
    """The seed is not inert -- it drives the minibatch permutation. With the global RNG pinned,
    two different seeds must still diverge, which isolates the generator's actual effect."""
    x, y = separable()
    args = dict(num_classes=4, lr=0.3, epochs=5, batch_size=16, weight_decay=0.0,
                momentum=0.9, mean=None, std=None, device=torch.device("cpu"))
    torch.manual_seed(7)
    a, _ = _train_head(x, y, seed=0, **args)
    torch.manual_seed(7)
    b, _ = _train_head(x, y, seed=1, **args)
    assert not torch.allclose(a, b)


def test_evaluate_returns_percentages_and_top5_dominates_top1():
    x, y = separable(num_classes=6)
    w, b = _train_head(x, y, num_classes=6, lr=0.3, epochs=10, batch_size=64,
                       weight_decay=0.0, momentum=0.9, mean=None, std=None,
                       device=torch.device("cpu"), seed=0)
    top1, top5 = _evaluate(w, b, x, y, None, None, torch.device("cpu"))
    assert 0.0 <= top1 <= 100.0 and top5 >= top1


def test_evaluate_batching_does_not_change_the_result():
    x, y = separable()
    w, b = _train_head(x, y, num_classes=4, lr=0.3, epochs=5, batch_size=64, weight_decay=0.0,
                       momentum=0.9, mean=None, std=None, device=torch.device("cpu"), seed=0)
    a = _evaluate(w, b, x, y, None, None, torch.device("cpu"), batch_size=13)
    c = _evaluate(w, b, x, y, None, None, torch.device("cpu"), batch_size=10_000)
    assert a == pytest.approx(c)


# --------------------------------------------------------------------------------------
# the full probe
# --------------------------------------------------------------------------------------

def test_separable_data_is_classified_well():
    x, y = separable()
    out = run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[0.3], epochs=20,
                           batch_size=64, device="cpu", seed=0)
    assert out["top1"] > 90.0


def test_a_single_lr_skips_the_search_entirely():
    """With one candidate there is nothing to select, so no held-out split is carved out and
    ``search`` stays empty. That also means the full train set is used for the single fit."""
    x, y = separable()
    out = run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[0.1], epochs=3,
                           batch_size=64, device="cpu", seed=0)
    assert out["search"] == [] and out["best_lr"] == 0.1


def test_the_search_records_every_candidate_and_picks_the_best_held_out():
    """The selected lr must be the argmax of the recorded held-out accuracies -- not the last
    tried, and not the best on the reported set."""
    x, y = separable()
    out = run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[0.001, 0.1, 1.0], epochs=8,
                           batch_size=64, device="cpu", seed=0, val_frac=0.2)
    assert len(out["search"]) == 3
    best = max(out["search"], key=lambda row: row["heldout_top1"])
    assert out["best_lr"] == best["lr"]


def test_the_reported_split_is_never_used_for_selection():
    """Selection happens on a slice of *train*. Pass a test set of pure noise: the chosen lr must
    be unchanged, because the test set has no say in it."""
    x, y = separable()
    noise_x, noise_y = torch.randn_like(x), torch.randint(0, 4, y.shape)
    common = dict(num_classes=4, lr_grid=[0.01, 0.3], epochs=6, batch_size=64,
                  device="cpu", seed=0, val_frac=0.2)
    honest = run_linear_probe(x, y, x, y, **common)
    with_noise = run_linear_probe(x, y, noise_x, noise_y, **common)
    assert honest["best_lr"] == with_noise["best_lr"]
    assert with_noise["top1"] < honest["top1"], "but the reported number does reflect the test set"


def test_standardisation_can_be_turned_off():
    x, y = separable()
    out = run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[0.1], epochs=3,
                           batch_size=64, device="cpu", seed=0, standardize=False)
    assert out["standardize"] is False and 0.0 <= out["top1"] <= 100.0


def test_the_result_records_the_settings_it_was_produced_with():
    """Provenance: the returned dict is what lands in ``per_backbone/*.json``, so it has to carry
    enough to reconstruct the protocol."""
    x, y = separable()
    out = run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[0.1], epochs=7,
                           batch_size=32, weight_decay=1e-4, device="cpu", seed=0)
    assert out["epochs"] == 7 and out["batch_size"] == 32
    assert out["weight_decay"] == pytest.approx(1e-4)
    assert {"top1", "top5", "train_top1", "best_lr", "search"} <= set(out)
