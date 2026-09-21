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

HEAD_ARGS = dict(num_classes=4, lr=0.1, epochs=3, batch_size=64, weight_decay=0.0,
                 momentum=0.9, mean=None, std=None, device=torch.device("cpu"))


def test_train_head_is_reproducible_from_its_seed_alone():
    """WAS A FINDING (ledger T.suite-P1-5), fixed. This test is its inverse.

    ``_train_head``'s ``seed`` used to feed only the shuffle ``Generator``; the head itself --
    ``nn.Linear(...)`` then ``trunc_normal_`` -- drew from the **global** RNG, and because
    ``.to(device)`` ran first, on CUDA that was the global *CUDA* RNG, which nothing in the repo
    ever seeds. Two calls with the same seed started from different weights.

    Measured before the fix: repeated ``run_linear_probe(seed=0)`` spanned **0.417 pp** of top-1
    on easy synthetic data, on CPU. Published tables quote Linear (%) to 0.1 pp, and
    ``eval_indomain.py`` probes backbones in a loop, so the Nth backbone's initialisation
    depended on how many ran before it -- ``--only <name>`` did not reproduce the sweep's number.

    Now exact, with no global seeding by the caller. Zero tolerance deliberately: this is either
    reproducible or it is not.
    """
    x, y = separable()
    first, first_bias = _train_head(x, y, seed=0, **HEAD_ARGS)
    second, second_bias = _train_head(x, y, seed=0, **HEAD_ARGS)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_bias, second_bias, rtol=0, atol=0)


def test_the_probe_does_not_depend_on_the_callers_global_rng():
    """The property that makes ``--only <name>`` reproduce the full sweep.

    Two calls with the same seed but wildly different global RNG states must agree exactly. This
    is the test that would have caught the original defect, and the one that catches it coming
    back: the old code's result was a function of how much global randomness had been consumed
    before it, which is to say of the backbone's *position in the loop*.
    """
    x, y = separable()
    torch.manual_seed(1)
    first, _ = _train_head(x, y, seed=0, **HEAD_ARGS)

    torch.manual_seed(12345)
    _ = torch.randn(1000)                       # advance the global stream arbitrarily
    second, _ = _train_head(x, y, seed=0, **HEAD_ARGS)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_the_probe_leaves_the_callers_global_rng_alone():
    """The converse courtesy: the probe draws from its own generator, so it must not perturb the
    stream its caller is using. ``eval_indomain`` runs probes between other seeded work."""
    x, y = separable()
    torch.manual_seed(4)
    before = torch.randn(3)

    torch.manual_seed(4)
    _train_head(x, y, seed=0, **HEAD_ARGS)
    after = torch.randn(3)

    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_every_learning_rate_in_the_search_starts_from_the_same_initialisation():
    """A quieter consequence of the fix, and a real improvement to the protocol.

    ``run_linear_probe`` calls ``_train_head`` once per grid point with the same ``seed``. Before
    the fix each call drew a *different* init from the advancing global RNG, so the search was
    comparing (learning rate x initialisation) pairs and attributing the whole difference to the
    learning rate. Now the init is a function of the seed alone, so the grid varies the one thing
    it is supposed to vary.
    """
    x, y = separable()
    at_lr = {lr: _train_head(x, y, seed=0, **{**HEAD_ARGS, "lr": lr, "epochs": 0})[0]
             for lr in (0.01, 0.1, 1.0)}

    reference = at_lr[0.01]
    for lr, weight in at_lr.items():
        torch.testing.assert_close(weight, reference, rtol=0, atol=0,
                                   msg=f"lr={lr} started from a different initialisation")


def test_the_initialisation_is_a_small_centred_normal():
    """Not merely deterministic -- deterministic *and* the intended distribution. std=0.01 keeps
    the initial logits near zero so the cosine schedule has somewhere to go; a defect that seeded
    the head but changed its scale would pass every test above.

    Note on the name ``trunc_normal_``: its ``a``/``b`` bounds are **absolute**, defaulting to
    +/-2.0, not multiples of ``std``. At std=0.01 they are 200 sigma away, so no truncation ever
    occurs and this is a plain normal. Asserted as such rather than as a truncation, because a
    test that expected clipping at 2*std would be testing something the code does not do.
    """
    x, y = separable(n_per_class=5)
    weight, bias = _train_head(x, y, seed=0, **{**HEAD_ARGS, "epochs": 0})

    assert float(bias.abs().max()) == 0.0, "the bias starts at exactly zero"
    assert abs(float(weight.mean())) < 0.005, "centred"
    assert 0.005 < float(weight.std()) < 0.02, f"std {float(weight.std()):.4f} is not ~0.01"


def test_different_seeds_still_give_different_results():
    """The seed is not inert. Without this, a "fix" that hard-coded one initialisation -- or
    ignored the seed entirely -- would satisfy every reproducibility test above while making the
    multi-seed spread identically zero."""
    x, y = separable()
    args = {**HEAD_ARGS, "lr": 0.3, "epochs": 5, "batch_size": 16}
    a, _ = _train_head(x, y, seed=0, **args)
    b, _ = _train_head(x, y, seed=1, **args)
    assert not torch.allclose(a, b)


def test_the_seed_controls_the_shuffle_as_well_as_the_initialisation():
    """Both, not just the init. At ``epochs=0`` no shuffle happens, so two seeds differ only in
    their init; with epochs > 0 they differ in both. Pinning the init and asserting the seeds
    still diverge shows the permutation is genuinely seed-driven too."""
    x, y = separable()
    at_zero_epochs = {seed: _train_head(x, y, seed=seed, **{**HEAD_ARGS, "epochs": 0})[0]
                      for seed in (0, 1)}
    assert not torch.allclose(at_zero_epochs[0], at_zero_epochs[1]), "inits differ"

    trained = {seed: _train_head(x, y, seed=seed, **{**HEAD_ARGS, "lr": 0.3, "epochs": 5})[0]
               for seed in (0, 1)}
    assert not torch.allclose(trained[0], trained[1])


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


# --------------------------------------------------------------------------------------
# the LeJEPA-protocol probe (campaign C4, decision 3)
# --------------------------------------------------------------------------------------

LEJEPA_ARGS = dict(num_classes=4, lr=1e-3, epochs=5, batch_size=64, weight_decay=1e-6, momentum=0.9,
                   mean=None, std=None, device=torch.device("cpu"))


def test_the_lejepa_probe_is_reproducible_from_its_seed_alone():
    torch.manual_seed(0)
    x, y = separable()
    first, first_bias = _train_head(x, y, seed=0, probe="lejepa", **LEJEPA_ARGS)
    second, second_bias = _train_head(x, y, seed=0, probe="lejepa", **LEJEPA_ARGS)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_bias, second_bias, rtol=0, atol=0)


def test_the_lejepa_probe_is_scale_invariant_because_of_its_layer_norm():
    """LayerNorm before the linear layer makes the probe blind to a global rescaling of the
    features -- the property that distinguishes it from ours (which standardises per dimension
    instead). Multiplying every feature by 10 must leave every prediction unchanged."""
    torch.manual_seed(0)
    x, y = separable()
    weight, bias = _train_head(x, y, seed=0, probe="lejepa", **LEJEPA_ARGS)
    top1, _ = _evaluate(weight, bias, x, y, None, None, torch.device("cpu"), layer_norm=True)
    scaled, _ = _evaluate(weight, bias, 10.0 * x, y, None, None, torch.device("cpu"), layer_norm=True)
    assert top1 == scaled and top1 > 90.0


def test_run_linear_probe_reports_which_protocol_it_used():
    torch.manual_seed(0)
    x, y = separable()
    ours = run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[0.1], epochs=3, batch_size=64,
                            device="cpu", seed=0)
    theirs = run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[1e-3], epochs=5, batch_size=64,
                              weight_decay=1e-6, standardize=False, device="cpu", seed=0, probe="lejepa")
    assert ours["probe"] == "sgd" and theirs["probe"] == "lejepa"
    assert theirs["top1"] > 90.0, "AdamW + LayerNorm must learn the separable toy problem"


def test_an_unknown_probe_is_refused():
    torch.manual_seed(0)
    x, y = separable()
    with pytest.raises(ValueError, match="probe must be"):
        run_linear_probe(x, y, x, y, num_classes=4, lr_grid=[0.1], epochs=1, device="cpu", probe="ridge")
