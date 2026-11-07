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
import re
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


def test_the_four_published_tables_read_no_separating_control_and_only_the_further_table_does(gen):
    """Decision 36 ended decision 32's rule for the new table, and the tables already in the paper keep
    the rows they had, so their numbers cannot move. Inverted by adding a new arm to MAIN_ROWS."""
    new_arms = {"ours_quarter", "centred_k1", "centred_k4", "randpart", "subreg_pub", "subreg_ours", "susreg_pub",
                "susreg_ours", "susreg_ours_l005", "subreg_ours_l0025", "susreg_ours_l0025"}
    for name, spec in gen.TABLES.items():
        used = {arm for arms in spec["rows"].values() for arm in arms}
        if name == "tab_further_controls":
            assert new_arms <= used
        else:
            assert not (used & new_arms), f"{name} reads {sorted(used & new_arms)}"


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
                                  "tab_in1k_transfer_full", "tab_depth", "tab_further_controls"])
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


# --------------------------------------------------------------------------------------------------
# the numbers the prose quotes beyond the tables (batch 2 of the edit list)
# --------------------------------------------------------------------------------------------------

def numbers(out: Path) -> dict:
    return json.loads((out / "text_numbers.json").read_text())


def test_the_smallest_per_seed_difference_is_recorded_for_a_three_seed_step(built):
    out, _ = built
    n = numbers(out)
    per_seed = [o - l for (o, _), (l, _) in zip(VIT["control_lam0.1"], VIT["sigreg"])]     # 6.9, 6.0, 5.2
    assert n["recipe_seedmin/vit/knn20"]["per_seed"] == pytest.approx(per_seed)
    assert n["recipe_seedmin/vit/knn20"]["value"] == pytest.approx(min(per_seed))
    assert n["recipe_seedmin/vit/knn20"]["rendered_1dp"] == "+5.2"


def test_rdmreg_gap_is_lejepa_minus_rdmreg(built):
    out, _ = built
    expected = statistics.fmean(k for k, _ in VIT["sigreg"]) - 12.6
    assert numbers(out)["lejepa_over_rdmreg/vit/knn20"]["value"] == pytest.approx(expected)


def test_the_depth_steps_use_seed_0_at_every_depth(built):
    out, _ = built
    n = numbers(out)
    k1, k1q, k4 = VIT["globalball_lam0.1_k1"][0][0], VIT["strmatch_lam0.025_k1"][0][0], VIT["local_lam0.1_k4"][0][0]
    assert n["depth_grouping/d12/knn20"]["value"] == pytest.approx(k4 - k1q)
    assert n["depth_strength/d12/knn20"]["value"] == pytest.approx(k1q - k1)
    assert n["depth_locality/d12/knn20"]["value"] == pytest.approx(k4 - k1)
    assert "depth_grouping/d6/knn20" not in n, "no depth-6 run in the fixture: absent, never zero"


def test_the_knn_k_claim_reports_the_largest_gap_and_the_ordering(gen, root, tmp_path):
    # one run whose best k is 10, 0.9 above its k=20
    e = indomain(50.6, 61.94)
    e["knn"]["10"]["top1"] = 51.5
    write(root / IND / "hvit_local_lam0.1_k4_s0.json", e)
    out = tmp_path / "o"
    assert gen.main(["--root", str(root), "--out", str(out), "--raw", str(tmp_path / "r")]) == 0
    n = numbers(out)
    assert n["knn_k20_maxgap/all/knn"]["value"] == pytest.approx(0.9)
    assert n["knn_order_same/vit/knn"]["value"] == 1.0


def test_the_vitl_matched_control_reads_its_values_and_its_early_online_probe(gen, root, tmp_path):
    write(root / IND_L / "vitl14_strmatch_lam0.0125_k1_s0.json", indomain(25.06, 35.14))
    metrics_csv(root / "logs/train/runs/vitl14_strmatch_lam0.0125_k1_s0/csv/version_0/metrics.csv", range(0, 100), 0.0027, 1.0)
    out = tmp_path / "o"
    assert gen.main(["--root", str(root), "--out", str(out), "--raw", str(tmp_path / "r")]) == 0
    n = numbers(out)
    assert n["vitl14_matched/value/linear"]["rendered_2dp"] == "35.14"
    assert n["vitl14_online/matched/ep5"]["value"] == pytest.approx(0.27)


def test_the_imagenette_tie_takes_the_best_lambda_per_k(im_built):
    out, _ = im_built
    n = numbers(out)
    k1 = statistics.fmean([87.4, 87.9, 87.0])
    assert n["imagenette_bestk_minus_k1/b256/online"]["value"] == pytest.approx(89.6 - k1)
    assert "imagenette_bestk_minus_k1/b64/online" not in n, "no K=1 cell at B=64 in the fixture"


def test_transfer_per_dataset_differences_and_wins(built):
    out, _ = built
    n = numbers(out)
    k4 = statistics.fmean(50.0 + k / 10 + s / 100 for s, (k, _) in enumerate(VIT["local_lam0.1_k4"]))
    le = statistics.fmean(50.0 + k / 10 + s / 100 for s, (k, _) in enumerate(VIT["sigreg"]))
    assert n["transfer_k4_vs_lejepa/vit/dtd"]["value"] == pytest.approx(k4 - le)
    assert n["transfer_wins_vs_lejepa/vit/count"]["value"] == 7.0


# --------------------------------------------------------------------------------------------------
# what a reader sees of the floats: the user's prose rules (2026-10-09) -- no semicolon unless really
# needed (none is, in a caption or a row label), and none of the AI-writing tells the curated lists name
# (Wikipedia's "Signs of AI writing", Graphite 2026, GPTZero, Juzek & Ward, Liang et al., Kobak et al.)
# --------------------------------------------------------------------------------------------------

TELLS = re.compile(
    r"\b(additionally|align(s|ed|ing)?|crucial(ly)?|consequently|delv(e|es|ed|ing)|enhanc\w*|furthermore|highlight\w*|"
    r"interplay|intricat\w*|landscape|leverag\w*|meaningful|meticulous\w*|moreover|notabl[ey]|pivotal|practical|"
    r"robust\w*|seamless\w*|showcas\w*|underscor\w*|valuable|comprehensive|ultimately|rather\s+than|not\s+only|"
    r"not\s+just)\b", re.I)


def reader_text(tex: str) -> str:
    """A generated float as a reader sees it: caption, headers and row labels -- no math, commands or markers."""
    text = re.sub(r"(?<!\\)%[^\n]*", " ", tex)                  # the % row: markers
    text = re.sub(r"(?<!\\)\$[^$]*(?<!\\)\$", " ", text)          # math (where a semicolon can be notation)
    return re.sub(r"\\[A-Za-z]+\*?", " ", text)                    # command names


@pytest.mark.parametrize("name", ["tab_in1k_main", "tab_in1k_transfer_main", "tab_in1k_indomain_full",
                                  "tab_in1k_transfer_full", "tab_depth", "tab_further_controls", "tab_k_ablation_256", "tab_strength_bands"])
def test_no_semicolon_and_no_ai_writing_tell_reaches_the_paper(im_built, name):
    out, _ = im_built
    tex, _ = table(out, name)
    seen = reader_text(tex)
    assert ";" not in seen, f"{name}: a semicolon in what the reader sees: …{seen[max(0, seen.find(';') - 60):seen.find(';') + 30]}…"
    tell = TELLS.search(seen)
    assert tell is None, f"{name}: an AI-writing tell in what the reader sees: {tell.group(0)!r}"


# --------------------------------------------------------------------------------------------------
# decision 36: the further controls, their table and their read-outs
# --------------------------------------------------------------------------------------------------
FURTHER_FIXTURE = {  # run: (knn20, linear). Chosen so each read-out lands where a test below expects it.
    "vit_randpart_lam0.1_k4_s0": (47.9, 59.8), "vit_randpart_lam0.1_k4_s1": (47.7, 60.0), "vit_randpart_lam0.1_k4_s2": (47.8, 59.9),
    "vit_control_lam0.025_s0": (41.0, 51.7), "vit_control_lam0.025_s1": (41.2, 51.8), "vit_control_lam0.025_s2": (41.1, 51.6),
    "vit_centred_lam0.1_s0": (45.7, 59.3), "vit_localgauss_lam0.1_k4_s0": (50.3, 62.1),
    "hvit_control_lam0.1_s0": (39.4, 51.6), "hvit_control_lam0.025_s0": (42.0, 52.0), "hvit_centred_lam0.1_s0": (46.1, 59.9),
    "hvit_globalball_lam0.1_k1_s0": (45.7, 59.5), "hvit_strmatch_lam0.025_k1_s0": (48.0, 59.9),
    "hvit_randpart_lam0.1_k4_s0": (50.0, 61.0),        # K-means minus random: +0.6 / +0.94, within
    "hvit_localgauss_lam0.1_k4_s0": (50.4, 62.0), "hvit_localsigreg_lam0.1_k4_s0": (35.4, 48.2),
    "d4rn50_sigreg_s0": (34.5, 43.4), "d4rn50_control_lam0.1_s0": (39.7, 48.8), "d4rn50_centred_lam0.1_s0": (44.4, 52.4),
    "d4rn50_globalball_lam0.1_k1_s0": (43.9, 52.4), "d4rn50_strmatch_lam0.025_k1_s0": (44.5, 52.4),
    "d4rn50_local_lam0.1_k4_s0": (47.5, 55.2), "d4rn50_randpart_lam0.1_k4_s0": (45.0, 55.0),  # +2.5 / +0.2, mixed
    "vit_in1k_subreg_ours_lam0.1_s0": (40.0, 50.0), "vit_in1k_susreg_ours_lam0.1_s0": (41.0, 51.0),
    "vit_in1k_susreg_ours_lam0.05_s0": (42.0, 52.0), "vit_in1k_subreg_pub_s0": (31.0, 43.0), "vit_in1k_susreg_pub_s0": (30.0, 42.0),
    "vit_in1k_subreg_ours_lam0.025_s0": (43.0, 53.0), "vit_in1k_susreg_ours_lam0.025_s0": (44.0, 54.0),
}


@pytest.fixture()
def further_built(gen, root, tmp_path):
    for run, (k, lin) in FURTHER_FIXTURE.items():
        write(root / IND / f"{run}.json", indomain(k, lin))
    out, raw = tmp_path / "out", tmp_path / "raw"
    assert gen.main(["--root", str(root), "--out", str(out), "--raw", str(raw)]) == 0
    return out, raw, json.loads((out / "text_numbers.json").read_text())


def test_the_further_table_shows_each_new_row_with_its_own_value(further_built):
    out, _, _ = further_built
    tex, payload = table(out, "tab_further_controls")
    assert "50.3" in row(tex, "% row:tab_further_controls/vit/centred_k4")
    assert "$45.7$" in row(tex, "% row:tab_further_controls/vit/centred_k1")
    assert r"\pm" in row(tex, "% row:tab_further_controls/vit/randpart"), "ViT-S/8's random partition has three seeds"
    assert r"\pm" not in row(tex, "% row:tab_further_controls/hvit/randpart"), "one seed on the other encoders"
    cell = payload["cells"]["tab_further_controls/vit/randpart/1"]
    assert cell["runs"] == [f"vit_randpart_lam0.1_k4_s{s}" for s in range(3)]
    assert cell["mean"] == pytest.approx(statistics.fmean(FURTHER_FIXTURE[f"vit_randpart_lam0.1_k4_s{s}"][0] for s in range(3)))


def test_a_further_run_without_an_evaluation_renders_as_a_dash(gen, built):
    """The table exists before its runs do, and a pending run must never read as a number. No fixture run for these."""
    tex, payload = table(built[0], "tab_further_controls")
    assert "--" in row(tex, "% row:tab_further_controls/vit/randpart")
    assert payload["pending_cells"] > 0


def test_the_further_table_only_offers_each_arm_where_it_was_run(gen):
    rows = gen.FURTHER_ROWS
    assert "centred_k4" in rows["vit"] and "centred_k4" in rows["hvit"] and "centred_k4" not in rows["d4rn50"]
    assert "localsigreg" not in rows["vit"] and "localsigreg" in rows["hvit"] and "localsigreg" in rows["d4rn50"]
    assert not any(a.startswith(("subreg", "susreg")) for f in ("hvit", "d4rn50") for a in rows[f])


@pytest.mark.parametrize("knn, linear, want", [(1.1, 1.1, "beyond"), (2.3, 2.3, "beyond"), (1.0, 1.1, "mixed"), (1.1, 1.0, "mixed"),
                                               (1.0, 1.0, "within"), (-1.0, 1.0, "within"), (0.0, 0.0, "within"), (-1.1, 0.0, "mixed"),
                                               (-2.0, -2.0, "mixed"), (2.0, -2.0, "mixed")])
def test_the_readout_rule_is_the_one_fixed_in_decision_36(gen, knn, linear, want):
    """Beyond is STRICTLY more than 1.0 on both rulers, within is at most 1.0 either way on both, anything else is mixed."""
    assert gen.readout_verdict(knn, linear) == want


def test_run_a_reads_kmeans_minus_random_with_each_verdict(further_built):
    _, _, n = further_built
    expected = statistics.fmean(k for k, _ in VIT["local_lam0.1_k4"]) - statistics.fmean(
        FURTHER_FIXTURE[f"vit_randpart_lam0.1_k4_s{s}"][0] for s in range(3))
    assert n["randpart_readout/vit/knn20"]["value"] == pytest.approx(expected)
    assert "randpart_readout_seedmin/vit/knn20" in n, "ViT-S/8 has three seeds on both sides: every seed is reported"
    assert n["readout_A_randpart/vit/verdict"]["rendered_1dp"] == "beyond"
    assert n["readout_A_randpart/hvit/verdict"]["rendered_1dp"] == "within"
    assert n["readout_A_randpart/d4rn50/verdict"]["rendered_1dp"] == "mixed"


def test_a_pending_run_gives_neither_a_difference_nor_a_verdict(built):
    n = json.loads((built[0] / "text_numbers.json").read_text())
    assert not [k for k in n if k.startswith(("randpart_readout", "readout_A", "k4_over_ours_quarter", "readout_D"))]


def test_run_d_reads_k4_minus_the_control_at_the_same_strength(further_built):
    _, _, n = further_built
    expected = statistics.fmean(k for k, _ in VIT["local_lam0.1_k4"]) - statistics.fmean(
        FURTHER_FIXTURE[f"vit_control_lam0.025_s{s}"][0] for s in range(3))
    assert n["k4_over_ours_quarter/vit/knn20"]["value"] == pytest.approx(expected)
    assert n["readout_D_same_strength/vit/verdict"]["rendered_1dp"] == "beyond"


@pytest.mark.parametrize("family", ["vit", "hvit", "d4rn50"])
def test_the_seed_0_ladder_sums_exactly_to_k4_minus_lejepa(further_built, family):
    """The steps are settings, centring, target, weight and locality. They must telescope, or the table misstates the split."""
    _, _, n = further_built
    for ruler in ("knn20", "linear"):
        steps = [n[f"ladder_{s}/{family}/{ruler}"]["value"] for s in ("settings", "centring", "target", "weight", "locality")]
        assert sum(steps) == pytest.approx(n[f"ladder_total/{family}/{ruler}"]["value"])


def test_the_ladder_reads_seed_0_on_vit_even_where_three_seeds_exist(further_built):
    """The separating controls have one seed, so a three-seed mean on one side would compare unlike things."""
    _, _, n = further_built
    assert n["ladder_settings/vit/knn20"]["value"] == pytest.approx(VIT["control_lam0.1"][0][0] - VIT["sigreg"][0][0])


def test_run_e_gives_hvit_both_comparisons(further_built):
    _, _, n = further_built
    assert n["centred_minus_uncentred_k4/hvit/knn20"]["value"] == pytest.approx(50.4 - 35.4)
    assert n["readout_E_centring_k4/hvit/verdict"]["rendered_1dp"] == "beyond"
    assert n["ball_minus_centred_k4/hvit/knn20"]["value"] == pytest.approx(50.6 - 50.4)
    assert n["readout_E_target_k4/hvit/verdict"]["rendered_1dp"] == "within"


def test_run_b_compares_each_baseline_at_its_own_settings(further_built):
    _, _, n = further_built
    assert n["k4_over_subreg_ours/vit/knn20"]["value"] == pytest.approx(VIT["local_lam0.1_k4"][0][0] - 40.0)
    assert n["k4_over_susreg_ours_l005/vit/linear"]["value"] == pytest.approx(VIT["local_lam0.1_k4"][0][1] - 52.0)
    assert n["lejepa_over_subreg_pub/vit/knn20"]["value"] == pytest.approx(VIT["sigreg"][0][0] - 31.0)


def test_run_b_reads_each_baseline_at_the_strength_of_k4_from_its_own_run(further_built):
    """Addition B': SubReg and SUSReg at lambda 0.025. Inverted by pointing either read-out or row at the lambda 0.1 or
    0.05 run, which the fixture gives other values."""
    out, _, n = further_built
    assert n["k4_over_subreg_ours_l0025/vit/knn20"]["value"] == pytest.approx(VIT["local_lam0.1_k4"][0][0] - 43.0)
    assert n["k4_over_subreg_ours_l0025/vit/linear"]["value"] == pytest.approx(VIT["local_lam0.1_k4"][0][1] - 53.0)
    assert n["k4_over_susreg_ours_l0025/vit/knn20"]["value"] == pytest.approx(VIT["local_lam0.1_k4"][0][0] - 44.0)
    assert n["k4_over_susreg_ours_l0025/vit/linear"]["value"] == pytest.approx(VIT["local_lam0.1_k4"][0][1] - 54.0)
    tex, _ = table(out, "tab_further_controls")
    assert "$43.0$" in row(tex, "% row:tab_further_controls/vit/subreg_ours_l0025")
    assert "$54.0$" in row(tex, "% row:tab_further_controls/vit/susreg_ours_l0025")


def test_a_pending_baseline_at_the_strength_of_k4_gives_no_number(gen, root, tmp_path):
    """A pair with a missing run is left out, never filled: before the lambda 0.025 runs are evaluated, neither
    read-out exists, and the lambda 0.1 one is unaffected."""
    for run, (k, lin) in FURTHER_FIXTURE.items():
        if "lam0.025_s0" in run and "_in1k_su" in run:
            continue
        write(root / IND / f"{run}.json", indomain(k, lin))
    out = tmp_path / "out"
    assert gen.main(["--root", str(root), "--out", str(out), "--raw", str(tmp_path / "raw")]) == 0
    n = json.loads((out / "text_numbers.json").read_text())
    assert not any(k.startswith(("k4_over_subreg_ours_l0025/", "k4_over_susreg_ours_l0025/")) for k in n)
    assert n["k4_over_subreg_ours/vit/knn20"]["value"] == pytest.approx(VIT["local_lam0.1_k4"][0][0] - 40.0)
