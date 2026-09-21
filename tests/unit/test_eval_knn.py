"""The weighted k-NN probe -- the ``k-NN (%)`` column of every reported table.

The decisive test here is the **label-shuffle control**: separable features scoring 100% proves
little on its own, because a probe that accidentally indexed the bank by position rather than by
similarity could also score well on sorted data. Destroying the label-feature correspondence must
destroy the accuracy. That control already exists in ``scripts/verify_indomain_result.py``; this
promotes it into the suite.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.eval.knn_probe import weighted_knn


def separable(n_per_class: int = 20, num_classes: int = 5, dim: int = 16):
    """Each class on its own orthogonal axis, so cosine similarity is 1 within a class and 0
    across classes -- as unambiguous as a classification problem can be."""
    centres = torch.eye(num_classes, dim)
    feats = torch.cat([centres[c] + 0.01 * torch.randn(n_per_class, dim)
                       for c in range(num_classes)])
    labels = torch.arange(num_classes).repeat_interleave(n_per_class)
    return feats, labels


def test_separable_features_are_classified_perfectly():
    x, y = separable()
    out = weighted_knn(x, y, x, y, num_classes=5, k_values=[5], device="cpu")
    assert out[5]["top1"] == pytest.approx(100.0)


def test_shuffling_the_bank_labels_collapses_accuracy_to_chance():
    """THE control. With the bank's labels permuted, the features no longer predict anything, so
    top-1 must fall to roughly 1/num_classes. A probe scoring well here would be reading position,
    not similarity."""
    x, y = separable(n_per_class=40, num_classes=5)
    shuffled = y[torch.randperm(y.numel())]
    out = weighted_knn(x, shuffled, x, y, num_classes=5, k_values=[20], device="cpu")
    assert out[20]["top1"] < 40.0, "chance is 20%; anything near 100% means the labels are unused"


def test_top5_is_at_least_top1():
    x, y = separable(num_classes=10)
    out = weighted_knn(x, y, x, y, num_classes=10, k_values=[10], device="cpu")
    assert out[10]["top5"] >= out[10]["top1"]


def test_every_requested_k_is_reported_in_one_pass():
    x, y = separable(n_per_class=30)
    out = weighted_knn(x, y, x, y, num_classes=5, k_values=[1, 5, 20, 100], device="cpu")
    assert sorted(out) == [1, 5, 20, 100]


def test_duplicate_and_unsorted_k_values_are_normalised():
    """``k_values`` is de-duplicated and sorted internally, so a careless config cannot produce
    two differently-keyed copies of the same k."""
    x, y = separable()
    out = weighted_knn(x, y, x, y, num_classes=5, k_values=[20, 5, 20, 5], device="cpu")
    assert sorted(out) == [5, 20]


def test_k_larger_than_the_bank_raises():
    """Silently clamping would report a k that was never used -- the reported number and the
    heading would disagree."""
    x, y = separable(n_per_class=2, num_classes=2)
    with pytest.raises(ValueError, match="exceeds the .* feature bank"):
        weighted_knn(x, y, x, y, num_classes=2, k_values=[100], device="cpu")


def test_chunking_does_not_change_the_result():
    """``chunk_size`` is a memory knob. Different chunkings must give identical accuracy, or the
    per-chunk accumulation is wrong."""
    x, y = separable(n_per_class=25, num_classes=4)
    a = weighted_knn(x, y, x, y, num_classes=4, k_values=[10], device="cpu", chunk_size=7)
    b = weighted_knn(x, y, x, y, num_classes=4, k_values=[10], device="cpu", chunk_size=1000)
    assert a[10]["top1"] == pytest.approx(b[10]["top1"])


def test_cosine_similarity_ignores_feature_magnitude():
    """Features are L2-normalised before the vote, so scaling a query cannot change its
    neighbours. If magnitude leaked in, a backbone producing larger norms would appear better."""
    x, y = separable()
    out_plain = weighted_knn(x, y, x, y, num_classes=5, k_values=[5], device="cpu")
    out_scaled = weighted_knn(x, y, x * 100.0, y, num_classes=5, k_values=[5], device="cpu")
    assert out_plain[5]["top1"] == pytest.approx(out_scaled[5]["top1"])


def test_the_query_norm_cannot_change_the_vote():
    """ADDED AFTER A SURVIVING MUTANT. Deleting ``F.normalize`` from the *query* survived the
    test above, because separable data votes unanimously and no weighting can overturn a
    unanimous vote.

    The query's norm does not change its own ranking -- ``topk`` is per-row -- so the defect is
    invisible in the neighbour *set*. It changes the weights: ``(sim/T).exp()`` with an
    unnormalised query is ``(|q| * cos / T).exp()``, i.e. the query's magnitude silently acts as
    a per-sample temperature. A backbone whose features happen to be large votes as if it were
    much colder.

    The construction makes that decisive: one class-0 neighbour almost exactly on the query, and
    four class-1 neighbours in a wedge 0.6 rad away, with k=5. A sharp vote follows the single
    close neighbour; a flat vote is outnumbered four to one. The correct code normalises, so all
    three magnitudes give class 0. The defect flips to class 1 at |q| = 0.02 -- verified.

    This is not academic: it is the difference between a reported k-NN accuracy that measures the
    representation and one that measures its scale.
    """
    bank = torch.zeros(5, 2)
    bank[0] = torch.tensor([1.0, 0.0])                             # class 0, on top of the query
    for i, angle in enumerate([0.60, 0.62, 0.64, 0.66]):           # class 1, a wedge further out
        bank[1 + i] = torch.tensor([math.cos(angle), math.sin(angle)])
    labels = torch.tensor([0, 1, 1, 1, 1])
    query = torch.tensor([[1.0, 0.02]])

    for scale in (0.02, 1.0, 50.0):
        out = weighted_knn(bank, labels, query * scale, torch.tensor([0]),
                           num_classes=2, k_values=[5], device="cpu")
        assert out[5]["top1"] == pytest.approx(100.0), (
            f"|q| = {scale} changed the vote: the query is not being L2-normalised"
        )


def test_temperature_sharpens_the_vote():
    """exp(sim/T): a low temperature concentrates the vote on the nearest neighbour. With
    deliberately noisy data the two temperatures must not agree exactly, or T is being ignored."""
    torch.manual_seed(0)
    x = torch.randn(200, 8)
    y = torch.randint(0, 4, (200,))
    hot = weighted_knn(x, y, x, y, num_classes=4, k_values=[50], temperature=1.0, device="cpu")
    cold = weighted_knn(x, y, x, y, num_classes=4, k_values=[50], temperature=0.01, device="cpu")
    assert hot[50]["top1"] != cold[50]["top1"]


def test_accuracies_are_percentages_not_fractions():
    """The reported tables are in percent; returning fractions would silently divide every
    published number by 100."""
    x, y = separable()
    out = weighted_knn(x, y, x, y, num_classes=5, k_values=[5], device="cpu")
    assert 0.0 <= out[5]["top1"] <= 100.0 and out[5]["top1"] > 1.0
