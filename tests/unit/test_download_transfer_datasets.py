"""``scripts/download_transfer_datasets.py --verify``: the structural gate that replaced the
existence-only ``.ready`` marker (research_rules/05: a file on disk proves a process started)."""

from __future__ import annotations

import json

import pytest

import scripts.download_transfer_datasets as D


def fake_splits(sizes: dict[str, tuple[int, int, int]]):
    def _get(key, root, crop_size=128, download=False):
        if key not in sizes:
            raise FileNotFoundError(f"{key} not downloaded")
        train, test, classes = sizes[key]
        return list(range(train)), list(range(test)), classes
    return _get


def test_verify_records_every_dataset_and_writes_the_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "get_transfer_splits", fake_splits({"dtd": (3760, 1880, 47), "pets": (3680, 3669, 37)}))
    report = D.verify(str(tmp_path), ["dtd", "pets"])
    assert report == {"dtd": {"train": 3760, "test": 1880, "classes": 47},
                      "pets": {"train": 3680, "test": 3669, "classes": 37}}


def test_verify_refuses_a_missing_or_empty_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "get_transfer_splits", fake_splits({"dtd": (0, 1880, 47)}))
    with pytest.raises(SystemExit, match="empty split"):
        D.verify(str(tmp_path), ["dtd"])
    with pytest.raises(SystemExit, match="not downloaded"):
        D.verify(str(tmp_path), ["aircraft"])


def test_main_verify_writes_verified_json_only_for_the_full_set(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(D, "get_transfer_splits", fake_splits({k: (10, 5, n) for k, (_, n) in D.TRANSFER_DATASETS.items()}))
    monkeypatch.setattr("sys.argv", ["x", "--root", str(tmp_path), "--verify"])
    D.main()
    marker = json.loads((tmp_path / ".verified").read_text())
    assert set(marker["datasets"]) == set(D.TRANSFER_DATASETS)
    assert not (tmp_path / ".ready").exists(), "the existence-only marker is gone"


def test_download_collects_failures_instead_of_stopping(monkeypatch, capsys):
    monkeypatch.setattr(D, "get_transfer_splits", fake_splits({"dtd": (1, 1, 47)}))
    assert D.download("/r", ["dtd", "pets"]) == ["pets"]


def test_the_docstring_sends_people_to_the_batch_job_not_the_login_node():
    assert "batch job" in D.__doc__ and "Run on a login node" not in D.__doc__
