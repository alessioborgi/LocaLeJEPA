"""Crash-safe writes and self-healing cache loads (``src/eval/io.py``).

Written against the failure these helpers exist for: a job killed with no warning. Every
evaluation in this project runs under that threat -- the `preemptable` partition kills without
notice and every other partition has a 24-hour wall -- so "what does a half-finished write leave
behind" is an operational question with a measured answer, not a hypothetical.

Ledger C0-3 predicted that a truncated feature cache would be "happily reused", i.e. silent
corruption. Reproduced 2026-09-19: it is not silent, it raises -- and because the writer sits
after the cache-hit branch, the bad file is never replaced, so the backbone is wedged until a
human deletes it. The reproduction refuted half the claim and made the other half worse; these
tests pin the observed behaviour, not the predicted one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from src.eval.io import atomic_torch_save, atomic_write, checkpoint_digest, load_tensor_cache


# ======================================================================================
# atomic_write
# ======================================================================================

def test_a_completed_write_lands_at_the_destination(tmp_path):
    target = tmp_path / "results.json"
    with atomic_write(target, "w", encoding="utf-8") as file:
        json.dump({"top1": 42.0}, file)

    assert json.loads(target.read_text()) == {"top1": 42.0}


def test_no_temporary_file_survives_a_successful_write(tmp_path):
    """A leftover ``.tmp`` is not harmless: ``_collect_results`` scans the per-backbone directory,
    and while it filters on ``.json`` today, a stray file there is a standing invitation for the
    next person to glob it in."""
    target = tmp_path / "results.json"
    with atomic_write(target, "w", encoding="utf-8") as file:
        file.write("{}")

    assert list(tmp_path.iterdir()) == [target]


def test_an_interrupted_write_leaves_the_previous_file_untouched(tmp_path):
    """THE POINT OF THE WHOLE MODULE.

    A job killed partway through writing its results must lose only the new results, never the
    old ones. Before this, the destination was opened directly in ``"w"`` mode -- which truncates
    it *before* a single byte of the new content is written, so an interruption destroyed the
    previous run's output as well as the current one.
    """
    target = tmp_path / "results.json"
    target.write_text('{"from": "the previous run"}')

    with pytest.raises(RuntimeError, match="killed"):
        with atomic_write(target, "w", encoding="utf-8") as file:
            file.write('{"half')
            raise RuntimeError("killed mid-write")

    assert json.loads(target.read_text()) == {"from": "the previous run"}
    assert list(tmp_path.iterdir()) == [target], "the temporary must be cleaned up"


def test_an_interrupted_first_write_leaves_nothing_behind(tmp_path):
    """No previous file to preserve, so the requirement is only that no partial file is left
    where a later reader would find it and trust it."""
    target = tmp_path / "results.json"

    with pytest.raises(ValueError):
        with atomic_write(target, "w", encoding="utf-8") as file:
            file.write("partial")
            raise ValueError("boom")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_keyboard_interrupt_is_cleaned_up_too(tmp_path):
    """``except BaseException``, not ``except Exception``. SIGINT and SystemExit do not inherit
    from Exception, and a Slurm cancellation surfaces as exactly that -- which is the case the
    module is named for."""
    target = tmp_path / "results.json"

    with pytest.raises(KeyboardInterrupt):
        with atomic_write(target, "w", encoding="utf-8") as file:
            file.write("partial")
            raise KeyboardInterrupt

    assert not target.exists() and list(tmp_path.iterdir()) == []


def test_parent_directories_are_created(tmp_path):
    """Per-backbone results go into ``<out>/per_backbone/<name>.json``, and on a fresh output
    directory that folder does not exist yet."""
    target = tmp_path / "deep" / "nested" / "results.json"
    with atomic_write(target, "w", encoding="utf-8") as file:
        file.write("{}")
    assert target.is_file()


def test_the_temporary_uses_the_json_tmp_suffix(tmp_path):
    """``<name>.json.tmp``, not ``<name>.tmp``. ``_collect_results`` filters on
    ``endswith(".json")``, so the suffix ordering is what makes a leftover skipped rather than
    parsed as a result. Pinned because it looks like a cosmetic choice and is not."""
    target = tmp_path / "backbone.json"
    seen = []
    with atomic_write(target, "w", encoding="utf-8") as file:
        seen.extend(p.name for p in tmp_path.iterdir())
        file.write("{}")

    assert seen == ["backbone.json.tmp"]
    assert not seen[0].endswith(".json")


def test_csv_writers_work_through_it_unchanged(tmp_path):
    """``newline=""`` has to reach ``open``, or csv writes \\r\\r\\n on some platforms. The
    helper forwards **kwargs precisely so converting a site is a one-word change."""
    import csv

    target = tmp_path / "results.csv"
    with atomic_write(target, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows([["backbone", "top1"], ["vit", "42.0"]])

    assert target.read_text().splitlines() == ["backbone,top1", "vit,42.0"]


# ======================================================================================
# atomic_torch_save
# ======================================================================================

def test_a_tensor_payload_round_trips(tmp_path):
    target = tmp_path / "feats.pt"
    payload = {"X": torch.randn(4, 3), "y": torch.arange(4)}
    atomic_torch_save(payload, target)

    loaded = torch.load(target, map_location="cpu")
    torch.testing.assert_close(loaded["X"], payload["X"], rtol=0, atol=0)
    assert list(tmp_path.iterdir()) == [target]


def test_a_failed_save_leaves_the_previous_cache_intact(tmp_path):
    """The feature cache is expensive -- ~1.4 GB and GPU-hours per ImageNet backbone. Losing a
    good one because a re-extraction died partway is the worst possible trade."""
    target = tmp_path / "feats.pt"
    atomic_torch_save({"X": torch.ones(2, 2), "y": torch.zeros(2)}, target)

    class Unpicklable:
        def __reduce__(self):
            raise RuntimeError("killed mid-save")

    with pytest.raises(RuntimeError, match="killed mid-save"):
        atomic_torch_save({"X": Unpicklable()}, target)

    torch.testing.assert_close(torch.load(target, map_location="cpu")["X"], torch.ones(2, 2))
    assert list(tmp_path.iterdir()) == [target]


# ======================================================================================
# load_tensor_cache -- turning a wedge into a re-extraction
# ======================================================================================

def test_a_healthy_cache_is_returned(tmp_path):
    target = tmp_path / "feats.pt"
    atomic_torch_save({"X": torch.randn(3, 2), "y": torch.arange(3)}, target)
    assert load_tensor_cache(target) is not None


@pytest.mark.parametrize("path", [None])
def test_an_absent_cache_is_a_miss(tmp_path, path):
    assert load_tensor_cache(path) is None
    assert load_tensor_cache(tmp_path / "never_written.pt") is None


def test_a_truncated_cache_is_a_miss_not_an_exception(tmp_path, caplog):
    """LEDGER C0-3, as reproduced rather than as predicted.

    The candidate said a truncated cache would be silently reused. It would not -- ``torch.load``
    raises ``PytorchStreamReader failed reading zip archive``. The real defect is that the write
    sits *after* the cache-hit branch, so the bad file is never replaced and every retry dies
    identically. On ``preemptable`` a single badly-timed kill wedged that backbone permanently,
    until someone noticed and deleted the file by hand.

    Returning ``None`` makes it self-healing: the caller re-extracts and overwrites. Expensive,
    but it completes without intervention, which is the difference that matters at 3am.
    """
    import logging

    target = tmp_path / "feats.pt"
    atomic_torch_save({"X": torch.randn(50, 8), "y": torch.arange(50)}, target)
    size = target.stat().st_size
    with open(target, "r+b") as handle:
        handle.truncate(size // 2)

    with caplog.at_level(logging.WARNING):
        assert load_tensor_cache(target) is None
    assert "unreadable feature cache" in caplog.text, "a silent miss would hide a real problem"


def test_an_empty_cache_file_is_a_miss(tmp_path):
    """A kill between ``open`` and the first write. Zero bytes raises a different exception than
    a truncated archive, which is why the handler is broad."""
    target = tmp_path / "feats.pt"
    target.write_bytes(b"")
    assert load_tensor_cache(target) is None


def test_a_cache_without_the_expected_keys_is_a_miss(tmp_path, caplog):
    """A readable ``.pt`` that is not a feature cache -- someone's checkpoint saved to the wrong
    path, or an older format. Loading it would blow up on ``blob["X"]`` at the call site with a
    ``KeyError`` naming nothing useful."""
    import logging

    target = tmp_path / "feats.pt"
    atomic_torch_save({"state_dict": {}, "epoch": 3}, target)

    with caplog.at_level(logging.WARNING):
        assert load_tensor_cache(target) is None
    assert "unexpected contents" in caplog.text


def test_the_cache_is_rewritten_after_a_miss(tmp_path):
    """Self-healing end to end, through the real extraction function: a damaged cache must be
    replaced by a good one, not merely bypassed. Otherwise every run pays the extraction cost
    forever and nobody notices."""
    from src.eval.indomain import extract_classification_features

    class Tiny(torch.utils.data.Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, index):
            return torch.full((3, 4, 4), float(index)), index % 3

    cache = tmp_path / "feats.pt"
    extract = lambda images: images.flatten(1)[:, :4].float()  # noqa: E731

    first, _ = extract_classification_features(extract, Tiny(), batch_size=2, num_workers=0,
                                               cache_path=str(cache))
    with open(cache, "r+b") as handle:
        handle.truncate(cache.stat().st_size // 2)

    second, _ = extract_classification_features(extract, Tiny(), batch_size=2, num_workers=0,
                                                cache_path=str(cache))
    torch.testing.assert_close(first, second, rtol=0, atol=0)

    third = load_tensor_cache(cache)
    assert third is not None, "the damaged cache must have been overwritten, not just skipped"
    torch.testing.assert_close(third["X"], first, rtol=0, atol=0)


# ======================================================================================
# checkpoint_digest
# ======================================================================================

def test_the_same_checkpoint_gives_the_same_digest(tmp_path):
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"weights")
    assert checkpoint_digest(ckpt) == checkpoint_digest(ckpt)
    assert len(checkpoint_digest(ckpt)) == 8


def test_a_different_checkpoint_gives_a_different_digest(tmp_path):
    """The whole purpose: two runs that share a backbone ``name`` and an ``epoch`` must not share
    a cache entry."""
    first, second = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    first.write_bytes(b"weights from run one")
    second.write_bytes(b"entirely different weights")
    assert checkpoint_digest(first) != checkpoint_digest(second)


def test_a_modified_checkpoint_gives_a_different_digest(tmp_path):
    """Same path, new contents -- a re-run that overwrote ``last.ckpt``. The size changes here;
    the mtime would catch it even at identical size."""
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"epoch 50")
    before = checkpoint_digest(ckpt)
    ckpt.write_bytes(b"epoch 100 -- longer contents")
    assert checkpoint_digest(ckpt) != before


def test_a_rewritten_checkpoint_of_identical_size_still_changes_the_digest(tmp_path):
    """The case that ``st_size`` alone cannot see, and the one that actually happens: a run
    restarted from scratch overwrites ``last.ckpt`` with a checkpoint of exactly the same shape
    and therefore the same byte count. Without ``st_mtime_ns`` in the key, the new run would
    silently inherit the old run's cached features -- which is ledger C0-4 with an extra twist.
    """
    ckpt = tmp_path / "last.ckpt"
    ckpt.write_bytes(b"A" * 128)
    before = checkpoint_digest(ckpt)

    os.utime(ckpt, ns=(0, 0))                    # a different mtime, byte-identical size
    assert ckpt.stat().st_size == 128
    assert checkpoint_digest(ckpt) != before


def test_the_digest_follows_the_resolved_path(tmp_path):
    """``realpath``: the campaign symlinks checkpoint trees into per-job run roots, so the same
    file is reachable under several names. Those must share a cache entry, or every job
    re-extracts."""
    real = tmp_path / "real.ckpt"
    real.write_bytes(b"weights")
    link = tmp_path / "link.ckpt"
    link.symlink_to(real)

    assert checkpoint_digest(link) == checkpoint_digest(real)


def test_a_missing_checkpoint_raises_rather_than_returning_a_digest(tmp_path):
    """A digest of a file that is not there would be a cache key for nothing. Fail where the
    problem is."""
    with pytest.raises(OSError):
        checkpoint_digest(tmp_path / "absent.ckpt")


# ======================================================================================
# the edges: a bare filename, and a cleanup that itself fails
# ======================================================================================

def test_a_bare_filename_with_no_directory_works(tmp_path, monkeypatch):
    """``os.path.dirname("results.json")`` is ``""``, and ``os.makedirs("")`` raises. Guarded --
    and reachable, because several scripts are run with the output directory as cwd."""
    monkeypatch.chdir(tmp_path)
    with atomic_write("results.json", "w", encoding="utf-8") as file:
        file.write("{}")
    assert (tmp_path / "results.json").read_text() == "{}"

    atomic_torch_save({"X": torch.zeros(1), "y": torch.zeros(1)}, "feats.pt")
    assert (tmp_path / "feats.pt").is_file()


def test_a_failure_before_the_temporary_exists_is_still_reported(tmp_path):
    """The cleanup does ``os.remove(tmp)`` inside a try/except OSError, because the failure may
    have happened *in* ``open`` -- in which case there is no temporary to remove. The original
    exception must survive that, not be replaced by a FileNotFoundError from the cleanup.
    """
    target = tmp_path / "sub" / "results.json"

    with pytest.raises(IsADirectoryError):
        # the temporary path is a directory, so open() fails before creating anything
        os.makedirs(str(target) + ".tmp")
        with atomic_write(target, "w", encoding="utf-8") as file:
            file.write("never reached")

    assert not target.exists()


def test_a_torch_save_failure_before_the_temporary_exists_is_still_reported(tmp_path):
    target = tmp_path / "feats.pt"
    os.makedirs(str(target) + ".tmp")

    # torch.save wraps the OS error in its own RuntimeError rather than letting it through
    with pytest.raises(RuntimeError, match="Is a directory"):
        atomic_torch_save({"X": torch.zeros(1)}, target)

    assert not target.is_file()


# ======================================================================================
# feature_cache_path -- one builder, so a writer and a reader cannot drift apart
# ======================================================================================

def cache_args(tmp_path, ckpt, **over):
    base = dict(cache_dir=tmp_path / "cache", backbone="vit_sigreg", dataset="imagenet1k",
                split="train", crop_size=128, epoch=99, checkpoint=ckpt)
    return {**base, **over}


@pytest.fixture
def ckpt(tmp_path):
    path = tmp_path / "a.ckpt"
    path.write_bytes(b"checkpoint A")
    return path


def test_the_path_has_the_documented_shape(tmp_path, ckpt):
    from src.eval.io import checkpoint_digest, feature_cache_path

    built = feature_cache_path(**cache_args(tmp_path, ckpt))
    assert built.endswith(f"imagenet1k_train_128_ep99_{checkpoint_digest(ckpt)}.pt")
    assert Path(built).parent.name == "vit_sigreg", "per-backbone subdirectory is preserved"


def test_two_different_checkpoints_never_share_a_cache_entry(tmp_path, ckpt):
    """LEDGER C0-4, reproduced 2026-09-20 and confirmed EXACTLY as claimed -- unlike C0-3, this
    one is genuinely silent.

    The key carried the backbone's config *name* and its stored *epoch*. Two different checkpoints
    sharing both -- a re-run that overwrote `last.ckpt`, or a `name` reused across configs -- hit
    the same file, and the second evaluation reported features computed from the first. Nothing
    raised, nothing warned, the job completed and the table was wrong.
    """
    from src.eval.io import feature_cache_path

    other = tmp_path / "b.ckpt"
    other.write_bytes(b"checkpoint B -- a different run entirely")

    assert feature_cache_path(**cache_args(tmp_path, ckpt)) != \
        feature_cache_path(**cache_args(tmp_path, other))


def test_forgetting_the_checkpoint_is_an_error_not_a_silent_fallback(tmp_path):
    """The argument is mandatory on purpose. If it defaulted to None the key would quietly revert
    to the unprotected form -- reintroducing C0-4 in exactly the way that made it invisible."""
    from src.eval.io import feature_cache_path

    with pytest.raises(TypeError):
        feature_cache_path(tmp_path, "vit", "imagenet1k", "train", 128, 99)


def test_a_writer_and_a_reader_agree_given_the_same_inputs(tmp_path, ckpt):
    """THE PROPERTY THE WHOLE COMMIT EXISTS FOR.

    ``eval_rankme`` does not write the in-domain or transfer caches -- it reconstructs their
    filenames and reads them. A divergence does not raise: rankme logs "no cached features --
    skipping this target" and continues, dropping the target and re-extracting on GPU. Silent,
    expensive degradation. Two independent calls must produce byte-identical paths.
    """
    from src.eval.io import feature_cache_path

    writer = feature_cache_path(**cache_args(tmp_path, ckpt))
    reader = feature_cache_path(**cache_args(tmp_path, ckpt))
    assert writer == reader


@pytest.mark.parametrize("field, value", [
    ("dataset", "cifar10"), ("split", "val"), ("crop_size", 224), ("epoch", 50),
    ("backbone", "vit_localballreg"),
])
def test_every_component_separates_the_cache(tmp_path, ckpt, field, value):
    """Each field must actually discriminate. A component that silently dropped out of the key
    would make two different extractions collide -- which is C0-4 in a new costume."""
    from src.eval.io import feature_cache_path

    assert feature_cache_path(**cache_args(tmp_path, ckpt)) != \
        feature_cache_path(**cache_args(tmp_path, ckpt, **{field: value}))


def test_no_cache_dir_means_no_caching(tmp_path, ckpt):
    """Every call site already treats None as "do not cache", and passing None must not be an
    error -- `cache_dir` is optional in every eval config."""
    from src.eval.io import feature_cache_path

    assert feature_cache_path(**cache_args(tmp_path, ckpt, cache_dir=None)) is None
    assert feature_cache_path(**cache_args(tmp_path, ckpt, cache_dir="")) is None


def test_an_omitted_epoch_is_allowed(tmp_path, ckpt):
    """`eval_conformal` never loads the epoch. It still gets checkpoint identity, which is what
    matters -- the digest subsumes the epoch anyway."""
    from src.eval.io import feature_cache_path

    built = feature_cache_path(**cache_args(tmp_path, ckpt, epoch=None))
    assert "_ep" not in Path(built).name
    assert built != feature_cache_path(**cache_args(tmp_path, ckpt))
