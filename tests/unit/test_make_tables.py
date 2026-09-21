"""The campaign's table generators and their audit (plan C5b).

``make_imagenet1k_results`` fills a rows-x-columns table from the manifests' run names;
``audit_tables`` re-derives every rendered number from the archive with code that shares nothing
with the generator; ``make_ablation_tables`` aggregates the online-probe accuracy over seeds,
prints the realised K beside every LocaLeJEPA cell and applies the declaration rule. All three
are exercised on synthetic inputs whose every number is known in advance.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def in1k():
    return load_script("make_imagenet1k_results")


@pytest.fixture(scope="module")
def audit():
    return load_script("audit_tables")


@pytest.fixture(scope="module")
def ablations():
    return load_script("make_ablation_tables")


def entry(knn20: float, linear: float, epoch: int = 99, **extra) -> dict:
    return {"epoch": epoch, "knn": {"10": {"top1": knn20 - 1, "top5": 90.0}, "20": {"top1": knn20, "top5": 91.0}},
            "linear": {"top1": linear, "top5": 92.0, "best_lr": 0.1}, **extra}


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# --------------------------------------------------------------------------------------
# the main table
# --------------------------------------------------------------------------------------

def test_the_cell_map_comes_from_the_manifests_and_is_unique(in1k):
    runs = in1k.campaign_runs()
    assert {("vit", "lejepa"), ("vit", "rdmreg"), ("vit", "local"), ("vit", "control"),
            ("vitl14", "lejepa"), ("vitl14", "local"), ("d4rn50", "lejepa"), ("hvit", "local")} <= set(runs)
    assert ("vitl14", "rdmreg") not in runs and ("d4rn50", "control") not in runs
    assert len({r.run_name for r in runs.values()}) == len(runs)


def eval_root(tmp_path: Path, in1k) -> tuple[Path, dict]:
    root = tmp_path / "root"
    runs = in1k.campaign_runs()
    write(root / in1k.OURS_128 / f"{runs[('vit', 'lejepa')].run_name}.json", entry(41.2, 55.3))
    write(root / in1k.OURS_128 / f"{runs[('vit', 'lejepa')].run_name}_best.json", entry(41.0, 55.0, epoch=87))
    write(root / in1k.OURS_128 / f"{runs[('vit', 'local')].run_name}.json", entry(45.7, 58.1))
    write(root / in1k.OURS_VITL / f"{runs[('vitl14', 'lejepa')].run_name}.json", entry(60.0, 70.5))
    write(root / in1k.LEJEPA_PROTOCOL_VITL / f"{runs[('vitl14', 'lejepa')].run_name}.json",
          entry(60.5, 72.9, feature="cls2", probe="lejepa"))
    return root, runs


def test_render_places_each_run_in_its_row_and_column(in1k, tmp_path):
    root, runs = eval_root(tmp_path, in1k)
    body, payload, tex = in1k.render(root)
    vit_row = next(l for l in body.splitlines() if l.startswith("| ViT-S/8"))
    assert "41.2 / 55.3" in vit_row and "45.7 / 58.1" in vit_row
    assert vit_row.split("|")[2].strip() == "—", "no published number for ViT-S/8"
    vitl_row = next(l for l in body.splitlines() if l.startswith("| ViT-L/14"))
    assert "— / 74.24†" in vitl_row and "60.0 / 70.5" in vitl_row and "n/a" in vitl_row
    assert "*pretraining, epoch ?/100*" in vit_row, "an unrun cell says so instead of showing a number"
    assert payload["cells"]["vit/lejepa/last"]["knn20"] == 41.2
    assert payload["cells"]["vit/lejepa/best"]["epoch"] == 87
    assert payload["cells"]["vitl14/lejepa/lejepa_protocol"]["linear"] == 72.9
    assert "linear 72.9" in body
    assert "k-NN +4.5, linear +2.8" in body, "the gap line is computed, never typed"
    assert r"60.0 / 70.5" in tex and r"74.24$^\dagger$" in tex


def test_the_published_number_never_enters_the_best_table(in1k, tmp_path):
    root, _ = eval_root(tmp_path, in1k)
    body, _, _ = in1k.render(root)
    best_section = body.split("## Secondary")[1].split("## ViT-L/14 under")[0]
    vitl_best = next(l for l in best_section.splitlines() if l.startswith("| ViT-L/14"))
    assert "74.24" not in vitl_best and vitl_best.split("|")[2].strip() == "—"


def test_the_lejepa_protocol_block_refuses_a_result_from_the_wrong_protocol(in1k, tmp_path):
    root, runs = eval_root(tmp_path, in1k)
    write(root / in1k.LEJEPA_PROTOCOL_VITL / f"{runs[('vitl14', 'lejepa')].run_name}.json",
          entry(60.5, 72.9, feature="encoder", probe="sgd"))
    with pytest.raises(AssertionError, match="encoder/sgd"):
        in1k.render(root)


def test_main_writes_the_table_the_cells_and_the_archive_and_the_audit_passes(in1k, audit, tmp_path, monkeypatch):
    root, _ = eval_root(tmp_path, in1k)
    monkeypatch.chdir(tmp_path)
    in1k.main(["--root", str(root)])
    table = Path(in1k.OUT).read_text()
    payload = json.loads(Path(in1k.CELLS_OUT).read_text())
    assert "## Provenance" in table and Path("results/raw").is_dir()
    assert audit.audit(table, payload, Path("results/raw")) == []
    assert audit.main(["--table", in1k.OUT, "--cells", in1k.CELLS_OUT, "--raw", "results/raw"]) == 0


def test_the_audit_catches_a_hand_edited_digit_and_a_stale_archive(in1k, audit, tmp_path, monkeypatch):
    root, runs = eval_root(tmp_path, in1k)
    monkeypatch.chdir(tmp_path)
    in1k.main(["--root", str(root)])
    table = Path(in1k.OUT).read_text()
    payload = json.loads(Path(in1k.CELLS_OUT).read_text())

    edited = table.replace("41.2 / 55.3", "41.2 / 56.3")
    problems = audit.audit(edited, payload, Path("results/raw"))
    assert any("vit/lejepa/last" in p and "not in its row" in p for p in problems)

    archived = Path("results/raw") / payload["cells"]["vit/local/last"]["source"]
    blob = json.loads(archived.read_text()); blob["linear"]["top1"] = 58.2
    archived.write_text(json.dumps(blob))
    problems = audit.audit(table, payload, Path("results/raw"))
    assert any("vit/local/last.linear" in p for p in problems)
    assert audit.main(["--table", in1k.OUT, "--cells", in1k.CELLS_OUT, "--raw", "results/raw"]) == 1


# --------------------------------------------------------------------------------------
# the ablation tables
# --------------------------------------------------------------------------------------

def metrics_csv(path: Path, *, epochs: int, final_acc: float, local: bool, realised_k: float | None,
                fallback_epochs: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "step", "train/loss_epoch", "val/probe_acc"]
    if local:
        cols += ["train/local/effective_clusters_epoch", "train/local/fallback_epoch", "train/local/configured_clusters_epoch"]
    rows = [",".join(cols)]
    for e in range(epochs):
        acc = final_acc if e == epochs - 1 else 0.5 * final_acc
        row = [str(e), str(e * 10), "0.4", f"{acc / 100:.4f}"]
        if local:
            row += [f"{realised_k}", "1.0" if e < fallback_epochs else "0.0", "8.0"]
        rows.append(",".join(row))
    path.write_text("\n".join(rows) + "\n")


def ablation_root(tmp_path: Path, ablations) -> Path:
    log_root = tmp_path / "logs"
    runs = [r for m in ablations.MANIFESTS for r in ablations.C.expand(ablations.C.load_manifest(REPO / "configs/campaign" / m))]
    by_name = {r.run_name: r for r in runs}
    accs = {  # (tag, lam, k, b): final acc; K=8 wins at 256, fwd wins at 64; lambda 0.1 and 0.2 tie at their best
        ("fixk", 0.1, 1, 256): 80.0, ("fixk", 0.1, 8, 256): 85.0, ("fixk", 0.2, 8, 256): 85.0,
        ("fixk", 0.1, 1, 64): 79.0, ("fixk", 0.1, 8, 64): 82.0,
        ("fwd", 0.1, None, 64): 83.0, ("rev", 0.1, None, 64): 81.0,
        ("sigreg", None, None, 256): 84.0,
    }
    for run in runs:
        o = run.override_map()
        key = (run.tag, float(o["model.lamb"]) if "model.lamb" in o else None,
               int(o["model.reg_loss.num_clusters"]) if "model.reg_loss.num_clusters" in o else None,
               int(o["data.batch_size"]))
        if key not in accs:
            continue
        local = run.tag in ("fixk", "fwd", "rev")
        metrics_csv(log_root / "train" / "runs" / run.run_name / "csv" / "version_0" / "metrics.csv",
                    epochs=run.max_epochs, final_acc=accs[key] + 0.5 * run.seed, local=local,
                    realised_k=(7.6 if key[2] == 8 else 1.0) if local else None,
                    fallback_epochs=3 if key == ("fixk", 0.1, 1, 64) else 0)
    # one unfinished run: 50 of 100 epochs at a cell that would otherwise be filled
    unfinished = next(r for r in runs if r.tag == "fixk" and r.overrides == by_name[r.run_name].overrides
                      and "model.lamb=0.3" in r.overrides and "model.reg_loss.num_clusters=1" in r.overrides
                      and "data.batch_size=256" in r.overrides and r.seed == 0)
    metrics_csv(log_root / "train" / "runs" / unfinished.run_name / "csv" / "version_0" / "metrics.csv",
                epochs=50, final_acc=99.0, local=True, realised_k=1.0)
    return log_root


def test_ablation_cells_aggregate_over_seeds_with_the_realised_k(ablations, tmp_path):
    cells, per_run, runs = ablations.build(ablation_root(tmp_path, ablations))
    agg = ablations.lookup(cells, runs, tag="fixk", lam=0.1, k=8, batch=(256, 1))
    assert agg["n"] == 3 and agg["mean"] == pytest.approx(85.5) and agg["std"] == pytest.approx(0.5)
    assert agg["realised_k"] == pytest.approx(7.6)
    assert ablations.fmt(agg, with_k=True) == "85.5 ± 0.5 (n=3), K̂=7.6"
    assert "⚠fallback×3" in ablations.fmt(ablations.lookup(cells, runs, tag="fixk", lam=0.1, k=1, batch=(64, 4)), with_k=True)
    assert ablations.lookup(cells, runs, tag="fixk", lam=0.3, k=1, batch=(256, 1)) is None, "an unfinished run is not a number"
    assert ablations.fmt(None, with_k=True) == "—"


def test_the_declaration_rule_breaks_ties_towards_the_smaller_lambda_and_picks_k_per_batch(ablations, tmp_path):
    cells, _, runs = ablations.build(ablation_root(tmp_path, ablations))
    declaration = ablations.declare(cells, runs)
    assert declaration["lambda"] == 0.1, "0.1 and 0.2 tie at 85.5; the rule takes the smaller"
    assert declaration["K_by_batch"]["256×1"]["K"] == "8"
    assert declaration["K_by_batch"]["64×4"]["K"] == "fwd", "the forward ramp beats every fixed K at batch 64"
    assert "128×2" not in declaration["K_by_batch"], "no finished run at the middle batch"


def test_the_six_tables_render_and_main_writes_declaration_and_raw_inputs(ablations, tmp_path, monkeypatch):
    log_root = ablation_root(tmp_path, ablations)
    monkeypatch.chdir(tmp_path)
    text = ablations.main(["--log-root", str(log_root)])
    assert text.split("## Provenance")[0].count("\n## ") == 6, "six tables, then the provenance block"
    assert "85.5 ± 0.5 (n=3), K̂=7.6" in text and "| LeJEPA (SIGReg) | 84.5 ± 0.5 (n=3) |" in text
    declaration = json.loads(Path("results/ablations/declaration.json").read_text())
    assert declaration["lambda"] == 0.1
    raw = list(Path("results/raw/ablations").glob("*.json"))
    assert len(raw) == 8 * 3, "one archived summary per finished run"
    assert "## Provenance" in Path("results/ablations/ablation_tables.md").read_text()
