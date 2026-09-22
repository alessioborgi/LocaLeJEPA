"""Modification times are the scratch purge's clock (F49).

AICR deletes files whose mtime is older than 30 days, daily; extraction preserves the archives'
old mtimes; Imagenette vanished twice and the transfer datasets once. Every extraction onto
scratch ends with a ``touch_tree``, and the two extraction paths in the code do it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from src.utils.touch import touch_tree

OLD = time.time() - 400 * 86400          # 400 days: past any purge threshold


def old_tree(root: Path) -> list[Path]:
    files = [root / "a" / "x.JPEG", root / "a" / "b" / "y.JPEG", root / "z.txt"]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"1")
    for p in [*files, root / "a", root / "a" / "b", root]:
        os.utime(p, (OLD, OLD))
    return files


def test_touch_tree_resets_every_file_and_directory_to_now(tmp_path):
    root = tmp_path / "ds"
    files = old_tree(root)
    before = time.time() - 5
    n = touch_tree(root)
    assert n == 3 + 2 + 1, "three files, two subdirectories, the root"
    for p in [*files, root / "a", root / "a" / "b", root]:
        assert p.stat().st_mtime >= before, p


def test_touch_tree_of_a_missing_root_is_zero_not_an_error(tmp_path):
    assert touch_tree(tmp_path / "never-downloaded") == 0


def test_touch_tree_touches_a_symlink_itself_not_its_target(tmp_path):
    target = tmp_path / "elsewhere.bin"; target.write_bytes(b"x"); os.utime(target, (OLD, OLD))
    root = tmp_path / "ds"; root.mkdir(); (root / "link").symlink_to(target)
    touch_tree(root)
    assert target.stat().st_mtime < time.time() - 300 * 86400, "the target outside the tree is untouched"


def test_the_transfer_download_touches_its_root(tmp_path, monkeypatch):
    import scripts.download_transfer_datasets as dl

    root = tmp_path / "transfer"
    def fake_splits(key, root_dir, crop_size, download):
        files = old_tree(Path(root_dir) / key)
        return list(range(3)), list(range(2)), 10
    monkeypatch.setattr(dl, "get_transfer_splits", fake_splits)
    before = time.time() - 5
    assert dl.download(str(root), ["cifar10"]) == []
    for p in (root / "cifar10").rglob("*"):
        assert p.stat().st_mtime >= before, p


def test_the_imagenette_datamodule_touches_what_it_extracts(tmp_path, monkeypatch):
    import src.data.imagenette_datamodule as m

    root = tmp_path / "imagenette"
    def fake_download(url, download_root, filename):
        old_tree(Path(download_root) / "imagenette2-160")
    monkeypatch.setattr(m, "download_and_extract_archive", fake_download)
    dm = m.ImagenetteDataModule(train_transform=lambda x: x, eval_transform=lambda x: x, data_dir=str(root), variant="160px")
    before = time.time() - 5
    dm.prepare_data()
    extracted = root / "imagenette2-160"
    assert extracted.is_dir() and all(p.stat().st_mtime >= before for p in extracted.rglob("*"))
    marker = extracted.stat().st_mtime
    dm.prepare_data()                        # already there: no download, no second touch of the root
    assert extracted.stat().st_mtime == marker
