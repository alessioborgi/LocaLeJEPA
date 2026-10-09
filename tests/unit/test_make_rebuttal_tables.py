"""The rebuttal's LaTeX tables (decision 32) and their audit.

``make_rebuttal_tables`` renders the paper's ImageNet-1K tables from per-run evaluation JSONs;
``audit_tables --latex`` re-derives every cell from the archive with code of its own and finds it at
its exact place in its row. Every number in these synthetic inputs is known in advance, and each
defect the audit exists for -- an edited digit, a stale archive, swapped rows, an extra number,
misplaced bold, an epoch-98 source, a 6/7 transfer, an edited label -- is planted once and must be
named.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DATASETS = ["dtd", "aircraft", "cifar10", "cifar100", "flowers", "food", "pets"]


def load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return load_script("make_rebuttal_tables")


@pytest.fixture(scope="module")
def audit():
    return load_script("audit_tables")


def indomain(knn20: float, linear: float, epoch: int = 99, **extra) -> dict:
    return {"epoch": epoch, "checkpoint_kind": "last", "probe_select": "train_slice",
            "knn": {str(k): {"top1": knn20 + d, "top5": 90.0} for k, d in ((10, -0.5), (20, 0.0), (100, -1.0), (200, -2.0))},
            "linear": {"top1": linear, "top5": linear + 20.0}, **extra}


def transfer(base: float, epoch: int = 99, drop: tuple[str, ...] = ()) -> dict:
    ds = {d: {"top1": base + i} for i, d in enumerate(DATASETS) if d not in drop}
    complete = len(ds) == len(DATASETS)
    return {"epoch": epoch, "datasets": ds, "avg_top1": (sum(v["top1"] for v in ds.values()) / 7) if complete else None}


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


IND = "evalroot/logs/eval_indomain_in1k_campaign/per_backbone"
TR = "evalroot/logs/eval_transfer_in1k_campaign/per_backbone"
IND_L = "evalroot/logs/eval_indomain_in1k_vitl14_campaign/per_backbone"

# ViT-S, three seeds each (knn20, linear): chosen so k4 is best on every column
VIT = {"sigreg": [(32.0, 44.0), (33.0, 45.0), (33.6, 44.9)],
       "control_lam0.1": [(38.9, 51.6), (39.0, 51.5), (38.8, 51.7)],
       "globalball_lam0.1_k1": [(45.4, 59.7), (45.2, 59.9), (45.6, 59.5)],
       "strmatch_lam0.025_k1": [(47.9, 60.0), (47.8, 60.1), (48.0, 59.9)],
       "local_lam0.1_k4": [(50.1, 62.1), (50.0, 62.3), (50.2, 61.9)]}


@pytest.fixture()
def root(tmp_path):
    r = tmp_path / "scr"
    for tag, seeds in VIT.items():
        for s, (k, lin) in enumerate(seeds):
            write(r / IND / f"vit_{tag}_s{s}.json", indomain(k, lin))
            write(r / TR / f"vit_{tag}_s{s}.json", transfer(50.0 + k / 10 + s / 100))
    write(r / IND / "vit_rdmreg_s0.json", indomain(12.6, 25.7))
    # H-ViT: k4 and the paper's dynamic K tie on linear at 1 dp (61.94 vs 61.86 -> both 61.9)
    write(r / IND / "hvit_local_lam0.1_k4_s0.json", indomain(50.6, 61.94))
    write(r / IND / "hvit_dynk_lam0.1_s0.json", indomain(49.0, 61.86))
    write(r / IND / "hvit_sigreg_s0.json", indomain(32.6, 44.0))
    write(r / TR / "hvit_local_lam0.1_k4_s0.json", transfer(60.0))
    write(r / TR / "hvit_sigreg_s0.json", transfer(55.0, drop=("flowers", "food", "pets")))     # 4/7: pending
    write(r / IND_L / "vitl14_sigreg_s0.json", indomain(46.0, 54.7))
    write(r / IND_L / "vitl14_local_k4_s0.json", indomain(47.4, 61.0))
    return r


@pytest.fixture()
def built(gen, root, tmp_path):
    out, raw = tmp_path / "out", tmp_path / "raw"
    assert gen.main(["--root", str(root), "--out", str(out), "--raw", str(raw)]) == 0
    return out, raw


def table(out: Path, name: str) -> tuple[str, dict]:
    return (out / f"{name}.tex").read_text(), json.loads((out / f"{name}.cells.json").read_text())


def row(tex: str, marker: str) -> str:
    return next(line for line in tex.splitlines() if line.rstrip().endswith(marker))


# --------------------------------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------------------------------

def test_a_three_seed_cell_is_the_mean_and_sample_sd(built):
    out, _ = built
    tex, payload = table(out, "tab_in1k_main")
    cell = payload["cells"]["tab_in1k_main/vit/k4/2"]          # k-NN@20, ViT-S, three seeds
    values = [k for k, _ in VIT["local_lam0.1_k4"]]
    assert cell["values"] == values
    assert cell["mean"] == pytest.approx(statistics.fmean(values))
    assert cell["sd"] == pytest.approx(statistics.stdev(values))     # n-1, not the population sd
    assert f"$\\mathbf{{50.1 \\pm {statistics.stdev(values):.2f}}}$" in row(tex, "% row:tab_in1k_main/vit/k4")


def test_a_single_run_has_no_plus_minus_and_rdmreg_is_seed_0_only(built):
    out, _ = built
    tex, payload = table(out, "tab_in1k_main")
    assert payload["cells"]["tab_in1k_main/vit/rdmreg/0"]["runs"] == ["vit_rdmreg_s0"]
    assert "$25.7$" in row(tex, "% row:tab_in1k_main/vit/rdmreg") and "\\pm" not in row(tex, "% row:tab_in1k_main/vit/rdmreg")


def test_bold_is_the_best_per_family_and_column_and_a_tie_bolds_both(built):
    out, _ = built
    tex, _ = table(out, "tab_in1k_main")
    assert "$\\mathbf{61.9}$" in row(tex, "% row:tab_in1k_main/hvit/k4")
    assert "$\\mathbf{61.9}$" in row(tex, "% row:tab_in1k_main/hvit/dynk"), "61.94 and 61.86 both show 61.9: both bold"
    assert "$\\mathbf{50.6}$" in row(tex, "% row:tab_in1k_main/hvit/k4")
    assert "$49.0$" in row(tex, "% row:tab_in1k_main/hvit/dynk")


def test_a_run_without_an_evaluation_renders_as_a_dash_never_a_number(built):
    out, _ = built
    tex, payload = table(out, "tab_in1k_main")
    assert row(tex, "% row:tab_in1k_main/hvit/localsigreg").startswith(r"LocaLeJEPA-SIGReg ($\lambda{=}0.1$, $K{=}4$) & -- & -- & -- & --")
    assert payload["pending_cells"] > 0


def test_a_partial_transfer_is_pending_and_never_averaged(built):
    out, _ = built
    tex, payload = table(out, "tab_in1k_transfer_main")
    assert row(tex, "% row:tab_in1k_transfer_main/hvit/lejepa").count("--") == 8
    assert not any(k.startswith("tab_in1k_transfer_main/hvit/lejepa/") for k in payload["cells"])
    assert "$\\mathbf{63.0}$" in row(tex, "% row:tab_in1k_transfer_main/hvit/k4"), "avg of 60..66 = 63.0, the only full row"


def test_the_transfer_main_table_keeps_its_compact_one_decimal_sd(built):
    out, _ = built
    tex, _ = table(out, "tab_in1k_transfer_main")
    assert "{\\pm}0." in row(tex, "% row:tab_in1k_transfer_main/vit/k4")


def test_an_epoch_98_evaluation_stops_the_generator(gen, root, tmp_path):
    write(root / IND / "vit_rdmreg_s0.json", indomain(12.6, 25.7, epoch=98))
    with pytest.raises(gen.ProtocolError, match="epoch 98"):
        gen.main(["--root", str(root), "--out", str(tmp_path / "o"), "--raw", str(tmp_path / "r")])


def test_an_eval_selected_probe_stops_the_generator(gen, root, tmp_path):
    write(root / IND / "vit_rdmreg_s0.json", indomain(12.6, 25.7, probe_select="eval"))
    with pytest.raises(gen.ProtocolError, match="probe_select"):
        gen.main(["--root", str(root), "--out", str(tmp_path / "o"), "--raw", str(tmp_path / "r")])


def test_a_row_template_that_names_no_campaign_run_is_an_error(gen, root, tmp_path, monkeypatch):
    monkeypatch.setitem(gen.ARMS, "rdmreg", ("RDMReg", "{f}_rdmreg_typo_s{s}"))
    with pytest.raises(SystemExit, match="not a run of any campaign manifest"):
        gen.main(["--root", str(root), "--out", str(tmp_path / "o"), "--raw", str(tmp_path / "r")])


def test_require_complete_refuses_while_any_cell_is_pending(gen, root, tmp_path):
    with pytest.raises(SystemExit, match="require-complete"):
        gen.main(["--root", str(root), "--out", str(tmp_path / "o"), "--raw", str(tmp_path / "r"), "--require-complete"])


def test_the_presentation_rule_no_centred_control_is_ever_read(gen):
    templates = [t for _, t in gen.ARMS.values()] + [t for _, t in gen.VITL_ARMS.values()]
    assert not any("centred" in t or "localgauss" in t for t in templates), "decision 32: never separate centring from the ball"


def test_every_row_of_every_table_names_real_campaign_runs(gen):
    known = gen.manifest_run_names()
    for spec in gen.TABLES.values():
        for family, arms in spec["rows"].items():
            for arm in arms:
                for name in gen.run_names(family, arm):
                    assert name in known, f"{name} is not a campaign run"
    for _, runs in gen.DEPTH_ARMS.values():
        assert set(runs.values()) <= known


def test_the_text_numbers_are_differences_of_cell_means_with_definitions(built):
    out, _ = built
    numbers = json.loads((out / "text_numbers.json").read_text())
    entry = numbers["k4_over_ours/vit/knn20"]
    expected = statistics.fmean(k for k, _ in VIT["local_lam0.1_k4"]) - statistics.fmean(k for k, _ in VIT["control_lam0.1"])
    assert entry["value"] == pytest.approx(expected)
    assert entry["rendered_1dp"] == f"{expected:+.1f}"
    assert "matched settings" in entry["definition"] and "3 pretraining seeds" in entry["definition"]
    assert "grouping/vit/linear" in numbers and "locality/vitl14/knn20" not in numbers, "ViT-L K=1 is not in the fixture"


def test_only_the_files_a_table_reads_are_archived(built):
    _, raw = built
    archived = sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*.json"))
    assert "eval_indomain_in1k_campaign/vit_local_lam0.1_k4_s2.json" in archived
    assert "eval_transfer_in1k_campaign/hvit_sigreg_s0.json" not in archived, "a pending partial is never archived"


# --------------------------------------------------------------------------------------------------
# the audit: passes on the generated tables, names every planted defect
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["tab_in1k_main", "tab_in1k_transfer_main", "tab_in1k_indomain_full",
                                  "tab_in1k_transfer_full", "tab_depth"])
def test_the_audit_passes_every_generated_table(audit, built, name):
    out, raw = built
    tex, payload = table(out, name)
    assert audit.audit_latex(tex, payload, raw) == []


def test_an_edited_digit_is_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_main")
    edited = tex.replace("$25.7$", "$25.8$")
    problems = audit.audit_latex(edited, payload, raw)
    assert any("tab_in1k_main/vit/rdmreg/0" in p and "25.8" in p for p in problems)


def test_a_stale_archive_is_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_main")
    write(raw / "eval_indomain_in1k_campaign" / "vit_rdmreg_s0.json", indomain(12.6, 26.7))
    problems = audit.audit_latex(tex, payload, raw)
    assert any("changed since the table was generated" in p for p in problems)
    assert any("tab_in1k_main/vit/rdmreg/0.mean" in p for p in problems)


def test_swapped_rows_are_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_main")
    a, b = row(tex, "% row:tab_in1k_main/vit/k1"), row(tex, "% row:tab_in1k_main/vit/k4")
    da, db = a.split(" & ", 1)[1].rsplit(" % row:", 1)[0], b.split(" & ", 1)[1].rsplit(" % row:", 1)[0]
    swapped = tex.replace(a, a.replace(da, db)).replace(b, b.replace(db, da))
    problems = audit.audit_latex(swapped, payload, raw)
    assert any("tab_in1k_main/vit/k1/" in p for p in problems) and any("tab_in1k_main/vit/k4/" in p for p in problems)


def test_an_extra_number_in_a_row_is_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_main")
    line = row(tex, "% row:tab_in1k_main/vit/rdmreg")
    problems = audit.audit_latex(tex.replace(line, line.replace("$25.7$", "$25.7$ (1.3)")), payload, raw)
    assert any("unclaimed" in p and "1.3" in p for p in problems)


def test_misplaced_bold_is_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_main")
    line = row(tex, "% row:tab_in1k_main/hvit/k4")
    problems = audit.audit_latex(tex.replace(line, line.replace("$\\mathbf{50.6}$", "$50.6$")), payload, raw)
    assert any("tab_in1k_main/hvit/k4/2" in p for p in problems)


def test_an_epoch_98_archive_is_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_main")
    write(raw / "eval_indomain_in1k_campaign" / "vit_rdmreg_s0.json", indomain(12.6, 25.7, epoch=98))
    assert any("epoch 98" in p for p in audit.audit_latex(tex, payload, raw))


def test_a_six_of_seven_transfer_archive_is_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_transfer_main")
    write(raw / "eval_transfer_in1k_campaign" / "hvit_local_lam0.1_k4_s0.json", transfer(60.0, drop=("pets",)))
    assert any("6/7" in p for p in audit.audit_latex(tex, payload, raw))


def test_an_edited_row_label_is_named(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_in1k_main")
    edited = tex.replace("Rectified LpJEPA (RDMReg) &", "RDMReg &", 1)
    assert any("label" in p and "rdmreg" in p for p in audit.audit_latex(edited, payload, raw))


def test_the_depth_table_checks_each_half_of_a_knn_slash_linear_cell(audit, built):
    out, raw = built
    tex, payload = table(out, "tab_depth")
    assert payload["cells"], "the depth runs at 12 blocks are in the fixture"
    line = row(tex, "% row:tab_depth/vit/depth12")
    first = line.split(" & ")[1]
    knn, lin = first.split(" / ")
    swapped = tex.replace(line, line.replace(first, f"{lin} / {knn}", 1))
    problems = audit.audit_latex(swapped, payload, raw)
    assert any("column 0.0" in p for p in problems) and any("column 0.1" in p for p in problems)


# --------------------------------------------------------------------------------------------------
# Appendix B: the Imagenette K axis and the effective-strength bands (800-epoch grid)
# --------------------------------------------------------------------------------------------------

def metrics_csv(path: Path, epochs: range, acc: float, khat: float, drift: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["epoch,step,val/probe_acc,train/local/effective_clusters_epoch,train/loss_epoch"]
    lines += [f"{e},{10 * e},{acc + drift * e / 799:.6f},{khat},1.0" for e in epochs]
    path.write_text("\n".join(lines) + "\n")


def imagenette_run(root: Path, name: str, acc: float, khat: float, linear: float, *, resumed: bool = False) -> None:
    csv_dir = root / "logs/train/runs" / name / "csv"
    if resumed:     # a wall at epoch 400: the second launch re-trains 390-400 and supersedes them (F52)
        metrics_csv(csv_dir / "version_0/metrics.csv", range(0, 401), acc - 0.5, khat)
        metrics_csv(csv_dir / "version_1/metrics.csv", range(390, 800), acc, khat)
    else:
        metrics_csv(csv_dir / "version_0/metrics.csv", range(0, 800), acc, khat)
    write(root / "logs/eval_ablation_ep800/per_backbone" / f"{name}.json",
          {**indomain(linear - 1.0, linear), "epoch": 799})


IM_CELLS = {  # (b, lam, k, x) -> per seed (online acc fraction, realised K, post-hoc linear)
    (256, "0.1", 1, 1): [(0.874, 1.0, 87.0), (0.879, 1.0, 87.4), (0.870, 1.0, 86.9)],
    (256, "0.1", 8, 1): [(0.896, 7.9, 89.6), (0.895, 7.8, 89.4), (0.897, 8.0, 89.8)],
    (256, "0.1", 32, 1): [(0.890, 27.7, 89.0), (0.891, 27.5, 89.1), (0.889, 27.9, 88.9)],
    (64, "0.02", 32, 4): [(0.20, 3.0, 33.0), (0.40, 3.0, 50.0), (0.60, 3.0, 66.0)],       # 0.04: collapse
    (64, "0.02", 2, 4): [(0.88, 2.0, 88.0), (0.88, 2.0, 88.0), (0.88, 2.0, 88.0)],        # EXACTLY 0.64: healthy
}


@pytest.fixture()
def im_root(root):
    for (b, lam, k, x), seeds in IM_CELLS.items():
        for s, (acc, khat, linear) in enumerate(seeds):
            imagenette_run(root, f"ep800_fixk_b{b}_lam{lam}_k{k}_x{x}_s{s}", acc, khat, linear,
                           resumed=(b, k, s) == (256, 8, 1))
    return root


@pytest.fixture()
def im_built(gen, im_root, tmp_path):
    out, raw = tmp_path / "out", tmp_path / "raw"
    assert gen.main(["--root", str(im_root), "--out", str(out), "--raw", str(raw)]) == 0
    return out, raw


def test_the_k_axis_reads_the_final_epoch_of_every_launch_merged(im_built):
    out, _ = im_built
    tex, payload = table(out, "tab_k_ablation_256")
    cell = payload["cells"]["tab_k_ablation_256/k8/acc"]
    assert cell["values"] == pytest.approx([89.6, 89.5, 89.7]), "the resumed seed's final epoch is from its LAST launch"
    assert cell["sd"] == pytest.approx(statistics.stdev([89.6, 89.5, 89.7]))
    line = row(tex, "% row:tab_k_ablation_256/acc")
    assert r"\cellcolor{ablationgold}\makecell{$89.60 \pm 0.10$}" in line
    assert r"\cellcolor{ablationsilver}\makecell{$89.00 \pm 0.10$}" in line, "K=32"
    assert r"\cellcolor{ablationbronze}\makecell{$87.43 \pm 0.45$}" in line, "K=1 (sd of 87.4 / 87.9 / 87.0 = 0.451)"
    assert line.count("--") == 3, "K=2, 4, 16 have no runs in this fixture"
    assert "$7.9$" in row(tex, "% row:tab_k_ablation_256/khat")


def test_the_bands_place_each_cell_by_lambda_n_over_k(im_built):
    """lambda N/K: (64, 0.02, 32) -> 0.04 collapse; (64, 0.02, 2) -> exactly 0.64, the healthy band's closed
    lower edge; (256, 0.1, 32) -> 0.8 and (256, 0.1, 8) -> 3.2 healthy; (256, 0.1, 1) -> 25.6
    over-regularized. Each band shows the min / median / max of its CELL MEANS."""
    out, _ = im_built
    tex, _ = table(out, "tab_strength_bands")
    expect = {
        "collapse": "$<0.4$ & collapse & 1 & $40.00$ & $40.00$ & $40.00$ & $49.67$ & $49.67$ & $49.67$",
        "degraded": "$0.4$--$0.64$ & degraded & -- & -- & -- & -- & -- & -- & --",
        "healthy": "$0.64$--$8$ & healthy & 3 & $88.00$ & $89.00$ & $89.60$ & $88.00$ & $89.00$ & $89.60$",
        "declining": "$8$--$20$ & declining & -- & -- & -- & -- & -- & -- & --",
        "over-regularized": "$>20$ & over-regularized & 1 & $87.43$ & $87.43$ & $87.43$ & $87.10$ & $87.10$ & $87.10$",
    }
    for band, text in expect.items():
        assert row(tex, f"% row:tab_strength_bands/{band}").startswith(text + r" \\"), band


def test_the_audit_passes_both_appendix_b_tables(audit, im_built):
    out, raw = im_built
    for name in ("tab_k_ablation_256", "tab_strength_bands"):
        tex, payload = table(out, name)
        assert audit.audit_latex(tex, payload, raw) == [], name


def test_a_truncated_online_curve_is_named(audit, im_built):
    out, raw = im_built
    tex, payload = table(out, "tab_k_ablation_256")
    curve = raw / "online_ep800" / "ep800_fixk_b256_lam0.1_k1_x1_s0" / "version_0.csv"
    curve.write_text("\n".join(curve.read_text().splitlines()[:-1]) + "\n")      # drop epoch 799
    assert any("online-probe epochs" in p for p in audit.audit_latex(tex, payload, raw))


def test_a_moved_medal_is_named(audit, im_built):
    out, raw = im_built
    tex, payload = table(out, "tab_k_ablation_256")
    edited = tex.replace(r"\cellcolor{ablationgold}\makecell{$89.60", r"\makecell{$89.60", 1)
    assert any("tab_k_ablation_256/k8/acc" in p for p in audit.audit_latex(edited, payload, raw))


def test_a_band_with_a_cell_left_out_is_named(audit, im_built):
    out, raw = im_built
    tex, payload = table(out, "tab_strength_bands")
    key = next(k for k in payload["cells"] if k.endswith("/healthy/cells"))
    payload["cells"][key]["members"] = payload["cells"][key]["members"][1:]
    assert any("not the archive's" in p for p in audit.audit_latex(tex, payload, raw))
