"""Every campaign eval config is derived from the manifests, and says so by being in sync.

F35: three checkpoint-naming conventions coexisted and a mismatch was a silently dropped row.
The manifest is the single source of run names; ``scripts/aicr/sync_eval_configs.py`` renders
every campaign eval config from it, and this file fails the moment a rendered config on disk
differs from what the manifests say -- which is exactly what happens when Phase 4 declares
(lambda, K) and the script has not been re-run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

import scripts.aicr.campaign as C

REPO = Path(__file__).resolve().parents[2]


def load_sync():
    spec = importlib.util.spec_from_file_location("_sync_eval_configs", REPO / "scripts/aicr/sync_eval_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYNC = load_sync()
RENDERED = SYNC.render_all()


def manifest_run_names() -> set[str]:
    names = set()
    for manifest in (REPO / "configs" / "campaign").glob("*.yaml"):
        names |= {run.run_name for run in C.expand(C.load_manifest(manifest))}
    return names


@pytest.mark.parametrize("rel", sorted(RENDERED))
def test_the_generated_config_on_disk_is_what_the_manifests_say(rel):
    path = REPO / rel
    assert path.is_file(), f"{rel} missing: run scripts/aicr/sync_eval_configs.py"
    assert path.read_text(encoding="utf-8") == RENDERED[rel], f"{rel} is stale: re-run sync_eval_configs.py"


@pytest.mark.parametrize("rel", sorted(RENDERED))
def test_every_backbone_names_a_manifest_run_and_its_own_directory(rel):
    names = manifest_run_names()
    cfg = yaml.safe_load(RENDERED[rel])
    assert cfg["backbones"], rel
    for entry in cfg["backbones"]:
        name = entry["name"]
        run = name[:-5] if name.endswith("_best") else name
        assert run in names, f"{rel}: {name} is not a manifest run name"
        expected = f"checkpoints/{run}/{'best' if name.endswith('_best') else 'last'}.ckpt"
        assert entry["checkpoint"] == expected, entry


def test_the_lejepa_protocol_config_declares_both_discriminators():
    cfg = yaml.safe_load(RENDERED["configs/eval/indomain_in1k_vitl14_lejepaprobe.yaml"])
    assert cfg["feature"] == "cls2" and cfg["linear"]["probe"] == "lejepa"
    assert cfg["linear"]["standardize"] is False and cfg["linear"]["weight_decay"] == 1e-6
    assert cfg["crop_size"] == 238
    assert all(not e["name"].endswith("_best") for e in cfg["backbones"]), "one protocol, last.ckpt only"


def test_our_protocol_configs_carry_best_twins_and_the_others_do_not():
    ours = yaml.safe_load(RENDERED["configs/eval/indomain_in1k_campaign.yaml"])
    names = [e["name"] for e in ours["backbones"]]
    assert sum(n.endswith("_best") for n in names) == len(names) // 2
    for rel in ("configs/eval/transfer_in1k_campaign.yaml", "configs/eval/rankme_in1k_campaign.yaml",
                "configs/eval/yoda/torus_campaign.yaml"):
        assert not any(e["name"].endswith("_best") for e in yaml.safe_load(RENDERED[rel])["backbones"]), rel


def test_rankme_entries_carry_a_family_and_the_vitl_pair_evaluates_at_238():
    rankme = yaml.safe_load(RENDERED["configs/eval/rankme_in1k_campaign.yaml"])
    assert {e["family"] for e in rankme["backbones"]} == {"vit", "d4rn50", "hvit"}
    for rel in ("configs/eval/indomain_in1k_vitl14_campaign.yaml", "configs/eval/transfer_in1k_vitl14_campaign.yaml",
                "configs/eval/yoda/vitl14_campaign.yaml"):
        assert yaml.safe_load(RENDERED[rel])["crop_size"] == 238, rel


def test_check_mode_reports_a_stale_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(SYNC, "REPO", tmp_path)
    assert SYNC.main(["--check"]) == 1
    assert "STALE" in capsys.readouterr().out
    assert SYNC.main([]) == 0
    assert SYNC.main(["--check"]) == 0


# ----- F71's guard: no manifest may be silently unevaluated -----
#
# F71 (2026-09-28): `configs/campaign/in1k_global_ball.yaml` was added and the suite stayed green at
# 3173 tests, yet its three runs would have trained for ~24 h each and finished with no eval config
# naming them -- the F35 silently-dropped-row failure this generator exists to prevent. The reason no
# test caught it is that `test_every_rendered_config_matches_the_manifests` compares configs against
# the manifests IN sync_eval_configs' lists, so a manifest absent from those lists is invisible to
# the staleness check rather than flagged by it.
#
# The fix cannot be "every manifest must be in the lists": Imagenette-scale grids are deliberately
# NOT there (sync_eval_configs.py:43-47) -- they are evaluated through
# `make_ablation_eval_config.py` from their own grid JSON, at run time, because they only include
# cells that already carry a `.verified` marker. And calibration/smoke manifests are never evaluated
# at all.
#
# So the guard forces a DECISION instead of inferring intent: every manifest must appear in exactly
# one bucket below. Adding a manifest without classifying it fails the test and makes you answer the
# question F71 went unasked -- "when these runs finish, what will evaluate them?"

GRID_EVALUATED = {
    # Imagenette-scale. Evaluated by `make_ablation_eval_config.py --grid <grid>.json`, which is run
    # on the cluster against the checkpoint root and includes only verified cells.
    "ablation_grid", "ablation_batch128", "ablation_grid_ep800", "ablation_grid_ep800_d4",
    "target_and_centering", "capacity_grid_ep800",
}

NOT_EVALUATED = {
    # Throwaway checkpoints: calibration measures throughput, the smoke grids rehearse a geometry
    # before a wave. Neither produces a number that reaches a table.
    "calibration_1gpu", "calibration_4gpu", "calibration_8gpu",
    "smoke_1gpu", "smoke_4gpu", "smoke_8gpu",
}


def test_every_manifest_is_classified_as_synced_grid_evaluated_or_not_evaluated():
    """F71's guard. A manifest in no bucket is a manifest nothing will evaluate.

    The synced set is READ from `sync_eval_configs.py`'s own lists rather than duplicated here, so
    drift in either direction fails: removing a manifest from the generator's lists without
    reclassifying it here, or adding one here without adding it there.

    INVERTED once: deleting "capacity_grid_ep800" from GRID_EVALUATED fails with that name in the
    unclassified set, which is exactly the report F71 should have produced.
    """
    on_disk = {p.stem for p in (REPO / "configs" / "campaign").glob("*.yaml")}
    assert on_disk, "the manifest glob matched nothing"

    synced = {Path(name).stem for name in (*SYNC.IN1K_128, *SYNC.VITL, *SYNC.TORUS)}

    overlaps = (synced & GRID_EVALUATED) | (synced & NOT_EVALUATED) | (GRID_EVALUATED & NOT_EVALUATED)
    assert not overlaps, f"a manifest is in two buckets: {sorted(overlaps)}"

    classified = synced | GRID_EVALUATED | NOT_EVALUATED
    unclassified = on_disk - classified
    assert not unclassified, (
        f"unclassified manifest(s): {sorted(unclassified)}. Ask what will evaluate their runs: add "
        f"IN1K-scale ones to sync_eval_configs.py's lists, Imagenette-scale ones to GRID_EVALUATED, "
        f"and throwaway ones to NOT_EVALUATED."
    )
    stale = classified - on_disk
    assert not stale, f"classified but no longer on disk: {sorted(stale)}"


def test_the_imagenette_grids_are_deliberately_absent_from_the_generator_lists():
    """The other half of F71, and the reason the guard above is not simply 'list everything'.

    An Imagenette grid in `sync_eval_configs.py`'s lists would render a campaign eval config naming
    every one of its cells whether verified or not, defeating the `.verified` gate that
    `make_ablation_eval_config.py` applies.

    HOW MUCH OF THAT THIS TEST ACTUALLY CATCHES, measured by inverting it: adding
    `capacity_grid_ep800.yaml` to `IN1K_128` does NOT reach this assertion. It raises
    `ValueError: no backbone family for tag 'cap_fixk'` inside `family()` during module import, when
    `RENDERED = SYNC.render_all()` runs, so the whole file errors at collection instead. That is a
    perfectly good failure and it is louder than an assertion -- but it is `family()`'s, not this
    test's.

    What this test adds is the case `family()` cannot see: a grid-evaluated manifest whose block
    tags happen to start with a known backbone prefix (`vit`, `hvit`, `d4rn50`, `svit`, `nvit`,
    `vitl14`). Such a manifest would render cleanly and silently produce a config naming unverified
    cells. That is the subtler and more dangerous leak, and it is the one this assertion covers.
    """
    synced = {Path(name).stem for name in (*SYNC.IN1K_128, *SYNC.VITL, *SYNC.TORUS)}
    assert not (GRID_EVALUATED & synced), \
        f"grid-evaluated manifest(s) leaked into the generator's lists: {sorted(GRID_EVALUATED & synced)}"


# ----- shape-bearing overrides must survive into the GENERATED configs -----


def test_a_generated_entry_replays_the_overrides_that_rebuild_its_architecture():
    """CAUGHT BEFORE LAUNCH, 2026-10-02, and it would have failed all eight capacity evaluations.

    `load_backbone` and `load_projector` instantiate from the composed config ALONE -- experiment
    plus crop size. `capacity_in1k` trains at `model.encoder.depth` 6 and 3, supplied as a MANIFEST
    override, while `lejepa/vit_in1k_localballreg_depth` composes to its default 12. So without the
    override replayed in the entry, every capacity checkpoint would be loaded into a 12-layer module
    and `src/eval/extract.py:134-140` would raise on the strict key check -- loudly, but only after
    the eval had queued, staged and started.

    The same gap had already been fixed in `make_ablation_eval_config.py` for the Imagenette grids;
    this generator is a SECOND one and still had it. The keys now live once, in
    `campaign.MODEL_SHAPE_KEYS`, and both generators call `campaign.shape_overrides`.

    INVERTED once by deleting `{_shape(run)}` from `backbone_entries`: the entries lose their
    overrides and this fails for all eight.
    """
    wanted = {}
    for run in C.expand(C.load_manifest(REPO / "configs" / "campaign" / "capacity_in1k.yaml")):
        depth = next(o for o in run.overrides if o.startswith("model.encoder.depth="))
        wanted[run.run_name] = [depth]
    assert len(wanted) == 4

    seen = 0
    for rel, text in RENDERED.items():
        cfg = yaml.safe_load(text)
        for entry in cfg.get("backbones", []):
            base = entry["name"][:-5] if entry["name"].endswith("_best") else entry["name"]
            if base not in wanted:
                continue
            seen += 1
            assert entry.get("overrides") == wanted[base], f"{rel}:{entry['name']} -> {entry.get('overrides')}"
    assert seen >= 8, f"expected the capacity cells in several generated configs, saw {seen}"


def test_only_runs_that_NEED_an_override_carry_one():
    """The converse, so the fix cannot degrade into "emit everything". lambda, K, the batch, the
    logger and the epoch count cannot reach an instantiated encoder or projector, and putting them
    in an eval config would compose a logger the eval never uses and make the file churn whenever a
    training detail changed.

    Every reference arm -- the baselines, the control, the global ball, the local arms -- takes its
    architecture entirely from its experiment config, so none of them should carry overrides.
    """
    offenders = []
    for rel, text in RENDERED.items():
        cfg = yaml.safe_load(text)
        for entry in cfg.get("backbones", []):
            if entry.get("overrides") and not entry["name"].startswith("vit_capdep_"):
                offenders.append((rel, entry["name"], entry["overrides"]))
    assert not offenders, f"non-capacity entries carrying overrides: {offenders}"
