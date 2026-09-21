"""The ImageNet staging pre-flight check.

Tested against a miniature tree with the *shape* of ImageNet-1K rather than its size: the class
counts are monkeypatched down so a 6-class fixture exercises every branch. What is under test is
the checking logic, not the constants -- and the constants are asserted separately against the
published figures.

Every defect here is one that a campaign would otherwise discover hours in, or not at all.
"""

from __future__ import annotations

import pytest

import scripts.verify_imagenet_staging as V


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """A 3-class tree in ImageNet's layout: 4 train images per class, 2 val images per class."""
    monkeypatch.setattr(V, "EXPECTED_CLASSES", 3)
    monkeypatch.setattr(V, "EXPECTED", {"train": 12, "val": 6})
    monkeypatch.setattr(V, "VAL_PER_CLASS", 2)
    monkeypatch.setattr(V, "TRAIN_PER_CLASS_RANGE", (3, 5))

    for split, per_class in (("train", 4), ("val", 2)):
        for label in range(3):
            d = tmp_path / split / f"{label:04d}"
            d.mkdir(parents=True)
            for i in range(per_class):
                (d / f"{split}_{i}.JPEG").write_bytes(V.JPEG_MAGIC + b"body")
    return tmp_path


def test_a_healthy_tree_reports_nothing(mini):
    assert V.verify(mini) == []


def test_a_missing_split_is_caught(mini):
    import shutil

    shutil.rmtree(mini / "val")
    assert any("missing split directory" in p for p in V.verify(mini))


def test_a_missing_class_directory_is_caught(mini):
    import shutil

    shutil.rmtree(mini / "train" / "0001")
    problems = V.verify(mini)
    assert any("class dirs, expected 3" in p for p in problems)
    assert any("different class sets" in p for p in problems), (
        "a class present in val but not train breaks get_indomain_splits' class_to_idx assert"
    )


def test_class_directories_that_are_not_zero_padded_are_caught(mini):
    """ImageFolder sorts class dirs as STRINGS, so `10` sorts before `2` and every label
    permutes. The model trains perfectly and evaluates to noise -- the defect this catches is
    invisible in any loss curve."""
    (mini / "train" / "0002").rename(mini / "train" / "2")
    assert any("zero-padded" in p for p in V.verify(mini))


def test_a_short_validation_class_is_caught(mini):
    """THE SHARPEST CHECK. ImageNet-1K's val split is exactly 50 per class. A download
    interrupted after touching every class -- the likely interruption, since shards are not
    ordered by class -- passes a class-directory count and fails this."""
    (mini / "val" / "0001" / "val_1.JPEG").unlink()
    problems = V.verify(mini)
    assert any("do not have exactly 2 images" in p for p in problems)


def test_a_wrong_total_is_caught_with_the_signed_shortfall(mini):
    """The message carries how many are missing, because "1,281,100 vs 1,281,167" is diagnosable
    and "counts differ" is not."""
    (mini / "train" / "0000" / "train_0.JPEG").unlink()
    assert any("-1" in p and "train:" in p for p in V.verify(mini))


def test_a_train_class_outside_the_plausible_range_is_caught(mini):
    for i in range(4, 9):
        (mini / "train" / "0000" / f"train_{i}.JPEG").write_bytes(V.JPEG_MAGIC)
    assert any("outside [3, 5]" in p for p in V.verify(mini))


def test_zero_byte_files_are_caught(mini):
    """What an interrupted write leaves behind. They pass a file count."""
    (mini / "train" / "0000" / "train_0.JPEG").write_bytes(b"")
    assert any("zero-byte" in p for p in V.verify(mini))


def test_a_file_that_is_not_a_jpeg_is_caught(mini):
    """Magic-number check on a sample: catches truncation and garbage without decoding 1.28M
    files. `sample` is large so every class is drawn; EVERY file of the class is corrupted so the
    assertion does not depend on which file the draw picks (F44: it used to depend on readdir
    order, and was green on APFS and red on the cluster's NFS with the same seed)."""
    for f in (mini / "val" / "0000").iterdir():
        f.write_bytes(b"NOT A JPEG AT ALL")
    assert any("not JPEG" in p for p in V.verify(mini, sample=100))


def test_the_sampled_file_does_not_depend_on_readdir_order(mini, monkeypatch):
    """F44 pinned: the file a seeded draw picks must be the same whatever order the filesystem
    lists the directory in. `os.scandir` is reversed here and the pick must not move."""
    import os as _os

    real_scandir = _os.scandir

    def reversed_scandir(path):
        return iter(sorted(real_scandir(path), key=lambda e: e.name, reverse=True))

    ordered = V._files_in_order(mini / "val" / "0000")
    monkeypatch.setattr(V.os, "scandir", reversed_scandir)
    assert V._files_in_order(mini / "val" / "0000") == ordered
    assert ordered == sorted(ordered) and len(ordered) == 2


def test_decode_catches_a_file_that_is_jpeg_headed_but_truncated(mini):
    """The case the magic number cannot see: correct first three bytes, unreadable body. Opt-in
    because it is slow, and worth running once before a campaign."""
    (mini / "val" / "0000" / "val_0.JPEG").write_bytes(V.JPEG_MAGIC + b"\x00" * 32)
    assert any("failed to decode" in p for p in V.verify(mini, decode=100))
    assert not any("failed to decode" in p for p in V.verify(mini, decode=0)), \
        "decoding must stay opt-in"


def test_the_class_sets_must_match_between_splits(mini):
    """`get_indomain_splits` asserts `train.class_to_idx == val.class_to_idx`. Detecting it here
    costs a set difference; detecting it later costs a feature extraction."""
    (mini / "val" / "0002").rename(mini / "val" / "0009")
    assert any("different class sets" in p for p in V.verify(mini))


def test_the_published_constants_are_right():
    """Asserted against the dataset, not against the code: ILSVRC2012 has 1000 classes,
    1,281,167 train images and exactly 50 val images per class. If someone 'fixes' a failing
    check by editing a constant, this fails instead."""
    assert V.EXPECTED_CLASSES == 1000
    assert V.EXPECTED == {"train": 1_281_167, "val": 50_000}
    assert V.VAL_PER_CLASS == 50
    assert V.EXPECTED["val"] == V.EXPECTED_CLASSES * V.VAL_PER_CLASS


def test_verification_is_deterministic(mini):
    """The sampling is seeded, so two runs on the same tree agree -- otherwise a flaky pre-flight
    gate gets ignored, which is worse than not having one."""
    (mini / "val" / "0000" / "val_0.JPEG").write_bytes(b"NOT A JPEG")
    assert V.verify(mini, sample=100) == V.verify(mini, sample=100)
