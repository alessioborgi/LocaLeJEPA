"""C5c: the torus/rotation report on manifest rows, the features index the figure scripts read,
the audit's torus cells, and the generalised verifier's config reading."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report():
    return load(REPO / "scripts/make_equivariance_report.py", "_equiv_report")


@pytest.fixture(scope="module")
def audit():
    return load(REPO / "scripts/audit_tables.py", "_audit_c5c")


@pytest.fixture(scope="module")
def verifier():
    return load(REPO / "scripts/verify_indomain_result.py", "_verify_indomain")


def multiseed(names: list[str], r2: float = 0.99, mae: float = 3.2) -> dict:
    return {n: {"n_seeds": 10, "n_frames": 8097, "angles": {
        f"angle{k}": {"label": f"figure {k}", "r2_mean": r2 - 0.01 * k, "r2_std": 0.0003,
                      "mae_mean": mae + k, "mae_std": 0.05} for k in range(2)}} for n in names}


def test_torus_rows_come_from_the_manifest_in_order_and_read_by_run_name(report, tmp_path):
    torus = [r.run_name for r in report.C.expand(report.C.load_manifest(REPO / "configs/campaign/torus.yaml"))]
    root = tmp_path / "root"
    (root / report.TORUS_DIR).mkdir(parents=True)
    (root / report.TORUS_DIR / "results_multiseed.json").write_text(json.dumps(multiseed(torus[:3])))
    body, payload, dirs = report.render(root)
    lines = [l for l in body.splitlines() if l.startswith("| ") and "figurine" not in l and "---" not in l]
    assert lines[0].startswith("| Spherical ViT | SUSReg |") and "0.9900±0.0003 / 3.20±0.05°" in lines[0]
    assert any(l.startswith("| ViT | LocaLeJEPA (λ=0.1, K=2) |") for l in lines), "labels name λ and K"
    assert lines[3].startswith("| ViT | SIGReg |") and "*pending*" in lines[3], "the fourth row has no result yet"
    assert payload["cells"][f"torus/{torus[0]}/angle0"]["r2_mean"] == pytest.approx(0.99)
    assert payload["cells"][f"torus/{torus[0]}/angle0"]["source"] == "eval_angle_probe_torus_campaign/results_multiseed.json"
    assert payload["n_done"] == 3


def test_main_writes_the_report_and_the_audit_re_derives_its_torus_cells(report, audit, tmp_path, monkeypatch):
    torus = [r.run_name for r in report.C.expand(report.C.load_manifest(REPO / "configs/campaign/torus.yaml"))]
    root = tmp_path / "root"
    (root / report.TORUS_DIR).mkdir(parents=True)
    (root / report.TORUS_DIR / "results_multiseed.json").write_text(json.dumps(multiseed(torus)))
    monkeypatch.chdir(tmp_path)
    report.main(["--root", str(root)])
    table = Path(report.OUT).read_text()
    payload = json.loads(Path(report.CELLS_OUT).read_text())
    assert audit.audit(table, payload, Path("results/raw")) == []
    edited = table.replace("0.9900±0.0003", "0.9910±0.0003", 1)
    assert any("not in its row" in p for p in audit.audit(edited, payload, Path("results/raw")))


def test_the_figure_scripts_read_the_features_index(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "features_index.json").write_text(json.dumps({
        "vit_local_l01_k2_s0": {"cache": "logs/eval/cache/x.pt", "checkpoint": "c", "experiment": "e", "crop_size": 128},
        "vit_sigreg_s0": {"cache": str(tmp_path / "abs.pt"), "checkpoint": "c", "experiment": "e", "crop_size": 128},
    }))
    monkeypatch.setenv("YODA_EVAL_DIR", str(eval_dir))
    monkeypatch.syspath_prepend(str(REPO / "visualisations"))
    torus = load(REPO / "visualisations/plot_yoda_torus.py", "_plot_yoda_torus_c5c")
    index = torus.load_index(str(eval_dir))
    assert index["vit_sigreg_s0"] == str(tmp_path / "abs.pt")
    assert index["vit_local_l01_k2_s0"].endswith("logs/eval/cache/x.pt") and Path(index["vit_local_l01_k2_s0"]).is_absolute()
    assert torus.load_index(str(tmp_path / "nowhere")) == {}
    assert set(torus.BACKBONES) == {"vit_local_l01_k2_s0", "vit_sigreg_s0"}


def test_the_verifier_reads_its_rows_from_the_eval_config(verifier):
    cfg = REPO / "configs/eval/indomain_in1k_campaign.yaml"
    rows = verifier.backbones_from_config(cfg)
    assert rows and all(not n.endswith("_best") for n in rows), "headline rows only by default"
    assert all(v[2] == "encoder" for v in rows.values())
    name = next(iter(rows))
    assert verifier.backbones_from_config(cfg, [f"{name}_best"]) == {f"{name}_best": (rows[name][0], rows[name][1].replace("last", "best"), "encoder")}
    with pytest.raises(ValueError, match="not in"):
        verifier.backbones_from_config(cfg, ["nope"])
    lejepa = verifier.backbones_from_config(REPO / "configs/eval/indomain_in1k_vitl14_lejepaprobe.yaml")
    assert all(v[2] == "cls2" for v in lejepa.values())
    settings = verifier.eval_settings(cfg)
    assert settings["report_k"] == 20 and settings["crop_size"] == 128 and settings["cache_dir"]
