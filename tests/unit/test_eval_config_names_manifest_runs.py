"""Every checkpoint an eval config names must be a run the campaign manifests produce.

F35: three checkpoint-naming conventions coexisted and a mismatch was a silently dropped row.
The manifest is now the single source of run names; an eval config that names anything else is
caught here, on the workstation, before a GPU is allocated for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import scripts.aicr.campaign as C

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN_EVAL_CONFIGS = ["configs/eval/indomain_in1k_vitl14_lejepaprobe.yaml"]


def manifest_run_names() -> set[str]:
    names = set()
    for manifest in (REPO / "configs" / "campaign").glob("*.yaml"):
        names |= {run.run_name for run in C.expand(C.load_manifest(manifest))}
    return names


@pytest.mark.parametrize("config", CAMPAIGN_EVAL_CONFIGS)
def test_every_backbone_names_a_manifest_run(config):
    names = manifest_run_names()
    with open(REPO / config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for entry in cfg["backbones"]:
        assert entry["name"] in names, f"{config}: {entry['name']} is not a manifest run name"
        assert entry["checkpoint"] == f"checkpoints/{entry['name']}/last.ckpt", entry


def test_the_lejepa_protocol_config_declares_both_discriminators():
    with open(REPO / CAMPAIGN_EVAL_CONFIGS[0], encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["feature"] == "cls2" and cfg["linear"]["probe"] == "lejepa"
    assert cfg["linear"]["standardize"] is False and cfg["linear"]["weight_decay"] == 1e-6
    assert cfg["crop_size"] == 238
