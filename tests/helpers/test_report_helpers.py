"""Tier B: the separable, correctness-bearing helpers inside the entry points and scripts.

Their ``main()`` bodies are orchestration over checkpoints, datasets and the network, and faking
those at scale costs more than it protects -- so Tier B carries no coverage floor. But several of
the helpers *decide what number gets printed*, and those are pure functions over dicts. They are
tested here, at the cost of a few dozen lines, because a defect in one of them corrupts a
published table while every run completes successfully.

Four of these tests are ``xfail(strict=True)``. They are not aspirational: they encode confirmed
defects, they fail on the tree as it stands, and ``strict`` means that the day the code is fixed
they turn into failures demanding to be flipped to ordinary asserts. That is how a finding is kept
alive without a permanently red suite.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    """Import ``scripts/<name>.py`` as a module.

    ``scripts/`` has no ``__init__.py`` and is not on the path, so a plain import will not find
    it. These modules are import-safe -- their module level is constants only -- with the single
    exception of ``verify_plainvit_result.py``, which does ``json.load(open(...))`` at import and
    is therefore not loaded anywhere in this suite.
    """
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def transfer_report():
    return load_script("make_transfer_results")


@pytest.fixture(scope="module")
def rankme_report():
    return load_script("make_rankme_results")


@pytest.fixture(scope="module")
def in1k_report():
    return load_script("make_imagenet1k_results")


@pytest.fixture(scope="module")
def plainvit_report():
    return load_script("make_plainvit_report")


def knn_entry(**by_k) -> dict:
    """A results entry whose k-NN block has exactly the given k values."""
    return {
        "knn": {k: {"top1": v, "top5": v + 5.0} for k, v in by_k.items()},
        "linear": {"top1": 50.0, "top5": 75.0, "best_lr": 0.1},
        "epoch": 99,
    }


# ======================================================================================
# the all-or-nothing transfer average
# ======================================================================================

def test_transfer_average_is_none_unless_every_dataset_is_present(transfer_report):
    """``_avg`` returns None unless all seven datasets reported.

    This guard is the reason the transfer column can be trusted: averaging over whichever subset
    happens to have finished would produce a number that moves as jobs land, and a backbone that
    failed on the two hard datasets would appear to *improve*. The all-or-nothing rule makes a
    partial result visibly absent instead of quietly optimistic.
    """
    all_seven = {k: {"top1": 10.0} for k, _ in transfer_report.DATASETS}
    assert transfer_report._avg({"datasets": all_seven}) == pytest.approx(10.0)

    missing_one = dict(all_seven)
    missing_one.pop(transfer_report.DATASETS[0][0])
    assert transfer_report._avg({"datasets": missing_one}) is None


def test_transfer_average_ignores_datasets_outside_the_declared_seven(transfer_report):
    """An extra key must not stand in for a missing one: the count is over DATASETS, not over
    whatever the file happens to contain."""
    entry = {k: {"top1": 10.0} for k, _ in transfer_report.DATASETS[:-1]}
    entry["some_other_dataset"] = {"top1": 90.0}
    assert transfer_report._avg({"datasets": entry}) is None


def test_rankme_transfer_average_matches_the_transfer_reports_rule(rankme_report, transfer_report):
    """Two scripts, two copies of the same average. They must agree, or the RankMe correlation is
    computed against a different quantity than the transfer table reports."""
    all_seven = {k: {"top1": 12.5} for k, _ in transfer_report.DATASETS}
    assert rankme_report._transfer_avg({"datasets": all_seven}) == pytest.approx(
        transfer_report._avg({"datasets": all_seven})
    )
    assert rankme_report._transfer_avg(None) is None, "a missing entry must not raise"


# ======================================================================================
# epoch-matched accuracies
# ======================================================================================

def test_accuracies_are_only_returned_when_the_epochs_match(rankme_report):
    """``_acc`` pairs RankMe at epoch E with accuracies at epoch E, or reports nothing.

    Without this the correlation would silently pair a rank computed at epoch 100 against a probe
    run on an earlier best.ckpt -- and the resulting rho would be reported as if it came from
    matched checkpoints. It is the clearest example in the repo of a guard that exists to keep a
    *statistic* honest rather than to keep the code from crashing.
    """
    ind = {"epoch": 99, "knn": {rankme_report.KNN_K: {"top1": 40.0}}, "linear": {"top1": 50.0}}
    transfer = {"epoch": 99, "datasets": {k: {"top1": 10.0} for k, _ in rankme_report.DATASETS}}

    assert rankme_report._acc(ind, transfer, 99) == (40.0, 50.0, pytest.approx(10.0))
    assert rankme_report._acc(ind, transfer, 50) == (None, None, None), "epoch mismatch"
    assert rankme_report._acc(None, None, 99) == (None, None, None), "nothing reported"

    # a mismatch on one side must not suppress the other
    knn, lin, avg = rankme_report._acc(ind, {**transfer, "epoch": 12}, 99)
    assert (knn, lin) == (40.0, 50.0) and avg is None


# ======================================================================================
# the four-point floor on Spearman
# ======================================================================================

def test_spearman_refuses_fewer_than_four_paired_points(rankme_report):
    """research_rules/08: a rank correlation on three points takes only a handful of values and
    |rho| = 1 among them is unremarkable. The floor is what keeps a two-point coincidence from
    being reported as a relationship."""
    assert rankme_report._spearman([1, 2, 3], [1, 2, 3]) == (None, 3)

    rho, n = rankme_report._spearman([1, 2, 3, 4], [1, 2, 3, 4])
    assert n == 4 and rho == pytest.approx(1.0)


def test_spearman_counts_pairs_after_dropping_incomplete_ones(rankme_report):
    """Five inputs but two unpaired: the effective n is 3, so it must refuse. Counting before
    the drop would let a mostly-empty column clear the floor."""
    assert rankme_report._spearman([1, 2, None, 4, None], [1, 2, 3, 4, 5]) == (None, 3)


def test_spearman_is_monotone_not_linear(rankme_report):
    """A rank correlation, not a Pearson one: a perfectly monotone but strongly non-linear
    relation must still score 1.0."""
    rho, n = rankme_report._spearman([1, 2, 3, 4, 5], [1, 4, 9, 100, 10000])
    assert n == 5 and rho == pytest.approx(1.0)


# ======================================================================================
# THE SILENT-k SUBSTITUTION -- three copies of one defect
# ======================================================================================
#
# All three do ``entry["knn"].get(str(k)) or next(iter(entry["knn"].values()))``. When the
# requested k is absent, the fallback reports a DIFFERENT k's accuracy under the requested k's
# heading, with no marker. The column header says "k-NN (k=20)"; the number can be k=10.
#
# The expression is not even a correct presence check: ``.get(...)`` returning a falsy value
# would also trigger the fallback, though in practice these are always dicts.
#
# The contrast is ``eval_indomain._write_results``, which writes "" for a missing k in the CSV.
# The machine-readable output is honest; the human-readable one is not. Pinned below.

XFAIL_SILENT_K = pytest.mark.xfail(
    strict=True,
    reason="CONFIRMED DEFECT: a missing k falls back to an arbitrary other k under the requested "
           "heading. Pinned, not endorsed; the fix is a separate decision.",
)


@XFAIL_SILENT_K
def test_in1k_report_does_not_substitute_another_k(in1k_report):
    """Requested k=20, available k=10 only. Reporting 90.0 under a 'k=20' heading is wrong."""
    top1, _, _, _ = in1k_report._cell(knn_entry(**{"10": 90.0}), k=20)
    assert top1 != "90.0", "k=10's accuracy is being reported as k=20"


@XFAIL_SILENT_K
def test_plainvit_report_does_not_substitute_another_k(plainvit_report):
    assert plainvit_report._knn(knn_entry(**{"10": 90.0}), k="20")["top1"] != 90.0


@XFAIL_SILENT_K
def test_rankme_acc_does_not_substitute_another_k(rankme_report):
    """The third copy, and the one that reaches furthest: this number feeds the RankMe-vs-accuracy
    correlation, so a substituted k does not merely misprint a cell -- it enters a statistic."""
    ind = {"epoch": 9, "knn": {"10": {"top1": 90.0}}, "linear": {"top1": 50.0}}
    assert rankme_report._acc(ind, None, 9)[0] != 90.0


def test_all_three_copies_agree_when_the_requested_k_is_present(in1k_report, plainvit_report,
                                                                rankme_report):
    """The happy path, which is what makes the defect survivable in normal use -- and what makes
    it worth pinning, since nothing in ordinary operation reveals it."""
    entry = knn_entry(**{"10": 30.0, "20": 40.0, "100": 50.0})
    assert in1k_report._cell(entry, k=20)[0] == "40.0"
    assert plainvit_report._knn(entry, k="20")["top1"] == 40.0
    assert rankme_report._acc({**entry, "epoch": 9}, None, 9)[0] == 40.0


def test_the_csv_writer_leaves_a_missing_k_blank(tmp_path):
    """The honest sibling of the three above: ``_write_results`` emits "" rather than borrowing.
    Included so that if someone ever 'harmonises' the two behaviours, they harmonise toward this
    one and this test says which way round that is."""
    import csv

    from src.eval_indomain import _write_results

    results = {
        "has_both": {**knn_entry(**{"10": 10.0, "20": 20.0}), "experiment": "e",
                     "checkpoint": "c", "feat_dim": 8, "n_train": 1, "n_val": 1},
        "has_only_10": {**knn_entry(**{"10": 11.0}), "experiment": "e",
                        "checkpoint": "c", "feat_dim": 8, "n_train": 1, "n_val": 1},
    }
    _write_results(results, str(tmp_path))

    with open(tmp_path / "results.csv", encoding="utf-8") as file:
        rows = {row["backbone"]: row for row in csv.DictReader(file)}

    assert rows["has_both"]["knn20_top1"] == "20.00"
    assert rows["has_only_10"]["knn20_top1"] == "", "a missing k must be blank, not borrowed"


def test_write_results_does_nothing_when_there_is_nothing_to_write(tmp_path):
    """The early return matters on the cluster: a merge job that finds no per-backbone files must
    leave the previous results.json alone rather than truncating it to '{}'."""
    from src.eval_indomain import _write_results

    (tmp_path / "results.json").write_text('{"previous": "run"}')
    _write_results({}, str(tmp_path))
    assert json.loads((tmp_path / "results.json").read_text()) == {"previous": "run"}


# ======================================================================================
# per-backbone collection and ordering
# ======================================================================================

def test_collected_results_follow_config_order_then_append_the_rest(tmp_path):
    """Files are read in ``sorted(os.listdir(...))`` order, which is alphabetical and has nothing
    to do with the order the table wants. The reordering step is what makes the printed table
    match the config; the append step is what stops an unexpected file from being dropped."""
    from src.eval_indomain import _collect_results, _save_backbone

    for name in ("zebra", "alpha", "surprise"):
        _save_backbone(name, {"name": name}, str(tmp_path))

    collected = _collect_results(str(tmp_path), order=["zebra", "alpha", "absent"])
    assert list(collected) == ["zebra", "alpha", "surprise"]


def test_collection_ignores_non_json_files(tmp_path):
    """The per-backbone directory also receives ``.tmp`` files from the atomic writers, and on a
    preempted job one can be left behind. It must not be collected as a result."""
    from src.eval_indomain import _collect_results, _per_backbone_dir, _save_backbone

    _save_backbone("real", {"name": "real"}, str(tmp_path))
    (Path(_per_backbone_dir(str(tmp_path))) / "leftover.json.tmp").write_text("{ truncated")
    (Path(_per_backbone_dir(str(tmp_path))) / "notes.txt").write_text("hello")

    assert list(_collect_results(str(tmp_path), order=[])) == ["real"]


@pytest.mark.parametrize("module_name", ["src.eval_transfer", "src.eval_rankme"])
def test_the_atomic_writers_never_expose_a_partial_file(tmp_path, module_name):
    """``.tmp`` + ``os.replace`` -- and ``os.replace`` is atomic within a filesystem.

    This is not decoration: these jobs run on the ``preemptable`` partition and are killed with
    no warning. A plain ``open(path, "w")`` that is interrupted leaves a truncated JSON that the
    next merge cannot parse, and the evidence of the completed work is gone.
    """
    import importlib

    module = importlib.import_module(module_name)
    module._save_backbone("bb", {"epoch": 99, "value": 1.0}, str(tmp_path))

    written = Path(tmp_path) / "per_backbone" / "bb.json"
    assert json.loads(written.read_text()) == {"epoch": 99, "value": 1.0}
    assert list(written.parent.glob("*.tmp")) == [], "the temporary file must not be left behind"


def test_the_indomain_writer_is_not_atomic_unlike_its_two_siblings():
    """FINDING, pinned as a test because it is invisible otherwise.

    ``eval_transfer`` and ``eval_rankme`` both write ``.tmp`` then ``os.replace``.
    ``eval_indomain._save_backbone`` -- the one whose numbers appear in the headline table --
    writes the destination directly. Same three-line function, same preemptable partition, one
    of the three unprotected.

    Asserted on the source text rather than by simulating a kill, which is what makes it cheap.
    When ``eval_indomain`` is brought into line this test fails and should be deleted; that is
    the intended end of it.
    """
    source = (REPO / "src" / "eval_indomain.py").read_text()
    body = source.split("def _save_backbone")[1].split("\ndef ")[0]
    assert "os.replace" not in body, (
        "eval_indomain._save_backbone now looks atomic -- confirm it, then delete this test"
    )


# ======================================================================================
# the loaders and the progress strings
# ======================================================================================

@pytest.mark.parametrize("script", ["make_transfer_results", "make_rankme_results",
                                    "make_imagenet1k_results", "make_plainvit_report"])
def test_every_report_load_returns_none_for_a_missing_file(tmp_path, script):
    """Four copies of ``_load``. All four must return None rather than raise: these scripts are
    run while the campaign is still producing files, and a missing result is the normal case."""
    module = load_script(script)
    assert module._load(str(tmp_path / "absent.json")) is None

    (tmp_path / "present.json").write_text('{"epoch": 3}')
    assert module._load(str(tmp_path / "present.json")) == {"epoch": 3}


def test_load_first_takes_the_earliest_directory_that_has_the_file(tmp_path, rankme_report):
    """``IND_DIRS`` is ordered: the 100-epoch evaluation shadows the older plain-ViT one. If the
    precedence reversed, the table would silently revert to superseded numbers."""
    preferred, fallback = tmp_path / "a", tmp_path / "b"
    preferred.mkdir()
    fallback.mkdir()
    (preferred / "x.json").write_text('{"from": "preferred"}')
    (fallback / "x.json").write_text('{"from": "fallback"}')
    (fallback / "y.json").write_text('{"from": "fallback"}')

    dirs = [str(preferred), str(fallback)]
    assert rankme_report._load_first(dirs, "x") == {"from": "preferred"}
    assert rankme_report._load_first(dirs, "y") == {"from": "fallback"}
    assert rankme_report._load_first(dirs, "z") is None


@pytest.mark.parametrize("script", ["make_transfer_results", "make_rankme_results",
                                    "make_imagenet1k_results", "make_plainvit_report"])
def test_checkpoint_epoch_is_none_for_a_missing_or_unreadable_file(tmp_path, script):
    """The ``except Exception: return None`` is deliberate here and worth pinning: these scripts
    read ``last.ckpt`` files that a running job may be rewriting, so a partial read must report
    'unknown' rather than crash the report generator."""
    module = load_script(script)
    assert module._ckpt_epoch(str(tmp_path / "absent.ckpt")) is None

    not_a_checkpoint = tmp_path / "junk.ckpt"
    not_a_checkpoint.write_bytes(b"this is not a torch archive")
    assert module._ckpt_epoch(str(not_a_checkpoint)) is None


def test_status_boundary_is_the_last_epoch_index_not_the_count(plainvit_report, monkeypatch):
    """Lightning's ``epoch`` is zero-based, so a 100-epoch run finishes at ``epoch == 99``.
    Comparing against MAX_EPOCHS rather than MAX_EPOCHS - 1 would report every finished run as
    still pretraining, forever."""
    calls = {}

    def fake_epoch(path):
        return calls["value"]

    monkeypatch.setattr(plainvit_report, "_ckpt_epoch", fake_epoch)

    calls["value"] = None
    assert plainvit_report._status("any") == "*not started*"

    calls["value"] = plainvit_report.MAX_EPOCHS - 2
    assert "pretraining, epoch 99/100" in plainvit_report._status("any")

    calls["value"] = plainvit_report.MAX_EPOCHS - 1
    assert "finished" in plainvit_report._status("any")


def test_progress_reports_an_unknown_epoch_without_arithmetic_on_none(rankme_report, monkeypatch):
    """``last_ep + 1 if last_ep is not None else '?'`` -- the guard exists because a not-yet-
    started run is normal, and ``None + 1`` would take the whole report down with it."""
    monkeypatch.setattr(rankme_report, "_ckpt_epoch", lambda path: None)
    assert "epoch ?/100" in rankme_report._progress("any")


@pytest.mark.parametrize("value, expected", [(None, "—"), (12.345, "12.3"), (0.0, "0.0")])
def test_format_renders_a_missing_value_as_an_em_dash(rankme_report, value, expected):
    """0.0 must render as "0.0", not as a dash: a genuine zero and a missing measurement are
    different claims, and ``if not v`` instead of ``if v is None`` would conflate them."""
    assert rankme_report._fmt(value) == expected


def test_format_bolds_only_when_asked(rankme_report):
    assert rankme_report._fmt(12.345, bold=True) == "**12.3**"
    assert rankme_report._fmt(None, bold=True) == "—", "a missing value is never bolded"
    assert rankme_report._fmt(12.345, nd=3) == "12.345"


# ======================================================================================
# ImageNet-1K staging: byte fidelity and idempotence
# ======================================================================================

@pytest.fixture(scope="module")
def stage():
    return load_script("stage_imagenet1k_hf")


def jpeg_bytes(colour=(10, 20, 30), size=(8, 8)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def png_bytes(colour=(10, 20, 30), size=(8, 8)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_a_jpeg_source_is_copied_byte_for_byte(stage, tmp_path):
    """The magic-number branch. Re-encoding every image would apply a second lossy pass to all
    1.28M training files -- invisible in any metric, permanent, and it would mean the pretraining
    corpus is not ImageNet. The verbatim copy is the whole point of the branch."""
    source = jpeg_bytes()
    split = tmp_path / "train"
    assert stage._write_example({"bytes": source}, 7, split, 0) is True

    written = split / "0007" / "train_0.JPEG"
    assert written.read_bytes() == source


def test_a_non_jpeg_source_is_decoded_and_re_saved_as_jpeg(stage, tmp_path):
    """ImageNet-1K contains a handful of PNG and CMYK files. They cannot be copied verbatim into
    a ``.JPEG`` name that the datamodule will hand to a JPEG-expecting loader."""
    from PIL import Image

    split = tmp_path / "train"
    assert stage._write_example({"bytes": png_bytes()}, 3, split, 5) is True

    written = split / "0003" / "train_5.JPEG"
    with Image.open(written) as image:
        assert image.format == "JPEG" and image.mode == "RGB"


def test_writing_the_same_example_twice_is_a_no_op(stage, tmp_path):
    """Idempotence is what makes staging resumable, and staging ImageNet-1K takes hours on a
    walltime-limited partition -- so a resumed job re-walks files that already exist. It must not
    rewrite them: it returns False, and the counter the caller keeps is a count of *new* files.
    """
    split = tmp_path / "train"
    stage._write_example({"bytes": jpeg_bytes()}, 1, split, 0)
    written = split / "0001" / "train_0.JPEG"
    first_mtime = written.stat().st_mtime_ns

    assert stage._write_example({"bytes": jpeg_bytes((99, 99, 99))}, 1, split, 0) is False
    assert written.stat().st_mtime_ns == first_mtime, "the existing file must not be touched"


def test_an_example_with_a_path_instead_of_bytes_is_read_from_disk(stage, tmp_path):
    """HuggingFace hands back either inline bytes or a path, depending on how the dataset was
    materialised. Both must work; only one is exercised by a normal streaming run."""
    source_file = tmp_path / "source.JPEG"
    source_file.write_bytes(jpeg_bytes())

    split = tmp_path / "val"
    assert stage._write_example({"bytes": None, "path": str(source_file)}, 2, split, 0) is True
    assert (split / "0002" / "val_0.JPEG").read_bytes() == source_file.read_bytes()


def test_an_example_with_neither_bytes_nor_path_raises(stage, tmp_path):
    """Loudly, rather than writing a zero-byte JPEG that only fails 40 minutes into the first
    training epoch."""
    with pytest.raises(ValueError, match="neither image bytes nor path"):
        stage._write_example({"bytes": None, "path": None}, 0, tmp_path / "train", 0)


def test_the_class_directory_is_zero_padded_to_four_digits(stage, tmp_path):
    """ImageFolder sorts class directories as strings and assigns indices in that order, so
    ``10`` would sort before ``2`` and every label would be permuted -- a defect that trains
    perfectly and evaluates to noise."""
    split = tmp_path / "train"
    for label in (2, 10, 999):
        stage._write_example({"bytes": jpeg_bytes()}, label, split, label)

    assert sorted(p.name for p in split.iterdir()) == ["0002", "0010", "0999"]
