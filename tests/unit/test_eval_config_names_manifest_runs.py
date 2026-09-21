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
