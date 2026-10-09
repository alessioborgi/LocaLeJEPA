#!/usr/bin/env python
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Re-derive every number a generated table shows from its archived inputs, and find it in the
table -- the numeral gate of research_rules/06, both directions.

A generator that writes ``<table>.cells.json`` (every rendered number with the archived file it
came from) is audited here with code that shares nothing with the generator: each cell's value is
read again from ``results/raw/<source>``, compared to what the generator claims it rendered, and
its formatted string is searched for in the table's own row. A hand edit, a stale archive, a
rounding change or a swapped row is a non-zero exit with the cell named.

    python scripts/audit_tables.py                       # results/Imagenet1K_results.md
    python scripts/audit_tables.py --table <md> --cells <cells.json> --raw results/raw
    python scripts/audit_tables.py --latex --table results/rebuttal/<t>.tex --cells results/rebuttal/<t>.cells.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

DEFAULT_TABLE = "results/Imagenet1K_results.md"
DEFAULT_CELLS = "results/Imagenet1K_results.cells.json"


def rederive(raw_root: Path, cell: dict, report_k: int) -> dict:
    """The cell's numbers, read again from the archive, with no help from the generator."""
    path = raw_root / cell["source"]
    if not path.is_file():
        raise FileNotFoundError(f"archived input {path} missing")
    entry = json.loads(path.read_text(encoding="utf-8"))
    if cell.get("kind") == "torus":
        # results_multiseed.json is keyed by run name; the cell names its run and its angle
        angle = entry[cell["run"]]["angles"][cell_angle(cell)]
        return {"r2_mean": angle["r2_mean"], "mae_mean": angle["mae_mean"]}
    knn = entry.get("knn", {}).get(str(report_k))
    # decision 16: the archive must say what the column claims -- an eval-selected result under a
    # strict key, or a strict one under an eval-selected key, is a mislabelled table
    selected = entry.get("probe_select", "train_slice")
    expected = "eval" if "evalsel" in cell["key"] else "train_slice"
    if selected != expected:
        raise ValueError(f"archive says probe_select={selected!r}, the cell is {expected!r}")
    return {"knn20": knn["top1"] if knn else None, "linear": entry["linear"]["top1"], "epoch": entry.get("epoch")}


def rederive_online(raw_root: Path, cell: dict) -> dict:
    """Appendix B, read again from the archived CSVs with a parser of its own: the per-epoch
    ``val/probe_acc`` rows across launches, final = the last, max = the first epoch reaching it."""
    import csv

    folder = raw_root / cell["source"]
    files = sorted(folder.glob("version_*.csv"), key=lambda f: int(f.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"archived online curve {folder} missing")
    by_epoch: dict[int, list[float]] = {}
    for file in files:
        launch: dict[int, list[float]] = {}
        with open(file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = row.get("val/probe_acc") or row.get("val/probe_acc_epoch") or ""
                if value == "":
                    continue
                epoch = int(float(row.get("epoch") or len(launch)))
                launch.setdefault(epoch, []).append(round(100.0 * float(value), 6))
        by_epoch.update(launch)      # a later launch supersedes the epochs it re-trained (F52)
    points = [(epoch, value) for epoch in sorted(by_epoch) for value in by_epoch[epoch]]
    if not points:
        raise ValueError(f"no online-probe rows in {folder}")
    best = max(v for _, v in points)
    argmax = min(e for e, v in points if v == best)
    return {"final": points[-1][1], "max": best, "argmax_epoch": argmax}


def cell_angle(cell: dict) -> str:
    return cell.get("angle") or cell["key"].rsplit("/", 1)[-1]


def rendered_text(cell: dict, fresh: dict, key: str) -> str:
    if cell.get("kind") == "torus":
        return f"{fresh['r2_mean']:.4f}" if key.endswith("angle0") or key.endswith("angle1") else ""
    if cell.get("kind") == "online":
        return f"{fresh['final']:.1f} / {fresh['max']:.1f} (ep {fresh['argmax_epoch']})"
    if "lejepa_protocol" in key:
        return f"linear {fresh['linear']:.1f}"
    if "evalsel" in key:
        return f"{fresh['linear']:.1f}"
    knn_text = f"{fresh['knn20']:.1f}" if fresh["knn20"] is not None else "—"
    return f"{knn_text} / {fresh['linear']:.1f}"


def audit(table_text: str, cells_payload: dict, raw_root: Path) -> list[str]:
    problems: list[str] = []
    report_k = int(cells_payload.get("report_k", 20))
    rows = [line for line in table_text.splitlines() if line.startswith("|")]
    for key, cell in cells_payload["cells"].items():
        cell = {**cell, "key": key}
        try:
            fresh = rederive_online(raw_root, cell) if cell.get("kind") == "online" else rederive(raw_root, cell, report_k)
        except (FileNotFoundError, KeyError, ValueError) as error:
            problems.append(f"{key}: cannot re-derive ({error})")
            continue
        fields = {"torus": ("r2_mean", "mae_mean"), "online": ("final", "max", "argmax_epoch")}.get(cell.get("kind"), ("knn20", "linear"))
        for field in fields:
            claimed, actual = cell.get(field), fresh.get(field)
            if claimed is None and actual is None:
                continue
            if claimed is None or actual is None or not math.isclose(claimed, actual, rel_tol=0, abs_tol=1e-9):
                problems.append(f"{key}.{field}: cells.json says {claimed!r}, the archive says {actual!r}")
        if cell.get("kind") not in ("torus", "online") and cell.get("epoch") != fresh.get("epoch"):
            problems.append(f"{key}.epoch: cells.json says {cell.get('epoch')!r}, the archive says {fresh.get('epoch')!r}")
        # the rendered string must appear in the table -- the grep direction -- and in the
        # cell's OWN row when the generator recorded one: a value that happens to appear in
        # another row (eleven identical torus rows in a test; two backbones tied in real life)
        # must not vouch for this cell.
        rendered = rendered_text(cell, fresh, key)
        candidates = [row for row in rows if cell.get("row", "") in row] if cell.get("row") else rows
        if not candidates:
            problems.append(f"{key}: no table row carries the label {cell.get('row')!r}")
        elif not any(rendered in row for row in candidates):
            problems.append(f"{key}: rendered value {rendered!r} is not in its row")
    return problems


# --------------------------------------------------------------------------------------------------
# LaTeX mode: the rebuttal tables (scripts/make_rebuttal_tables.py). Its own reader, protocol check,
# statistics and formatter -- nothing imported from the generator.
# --------------------------------------------------------------------------------------------------

LATEX_DATASETS = ("dtd", "aircraft", "cifar10", "cifar100", "flowers", "food", "pets")
LATEX_MARKER = re.compile(r"(% row:\S+)\s*$")
NUMERAL = re.compile(r"\d+(?:\.\d+)?")


def _latex_value(entry: dict, field: str) -> float:
    if field == "avg":
        return sum(float(entry["datasets"][d]["top1"]) for d in LATEX_DATASETS) / len(LATEX_DATASETS)
    node = entry
    for part in field.split("."):
        node = node[part]
    return float(node)


def _latex_protocol(entry: dict, kind: str) -> str | None:
    """Why this archived evaluation cannot fill a strict, final-epoch cell -- or None."""
    if entry.get("epoch") != 99:
        return f"epoch {entry.get('epoch')!r}, not 99"
    if kind == "indomain":
        if entry.get("checkpoint_kind", "last") != "last":
            return f"checkpoint_kind {entry.get('checkpoint_kind')!r}, not last"
        if entry.get("probe_select", "train_slice") != "train_slice":
            return f"probe_select {entry.get('probe_select')!r}, not train_slice"
        return None
    got = entry.get("datasets") or {}
    have = [d for d in LATEX_DATASETS if isinstance(got.get(d), dict) and "top1" in got[d]]
    if len(have) != len(LATEX_DATASETS):
        return f"transfer has {len(have)}/{len(LATEX_DATASETS)} datasets"
    stored = entry.get("avg_top1")
    mean = sum(float(got[d]["top1"]) for d in LATEX_DATASETS) / len(LATEX_DATASETS)
    if stored is not None and abs(float(stored) - mean) > 1e-6:
        return f"stored avg_top1 {stored} is not the mean of the seven ({mean})"
    return None


def _latex_stats(values: list[float]) -> tuple[float, float | None]:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, None
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _latex_render(mean: float, sd: float | None, sd_digits: int, compact: bool) -> str:
    if sd is None:
        return f"{mean:.1f}"
    return f"{mean:.1f}{{\\pm}}{sd:.{sd_digits}f}" if compact else f"{mean:.1f} \\pm {sd:.{sd_digits}f}"


def _online_series(folder: Path) -> dict[int, dict[str, str]]:
    """Per epoch, the row of the LAST launch that logged it (a resumed run writes one CSV per launch)."""
    import csv

    files = sorted(folder.glob("version_*.csv"), key=lambda f: int(f.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"archived online curve {folder} missing")
    by_epoch: dict[int, dict[str, str]] = {}
    for file in files:
        launch: dict[int, dict[str, str]] = {}
        with open(file, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("epoch") in ("", None):
                    continue
                merged = launch.setdefault(int(float(row["epoch"])), {})
                merged.update({k: v for k, v in row.items() if v not in ("", None)})
        by_epoch.update(launch)
    return by_epoch


def _imagenette_run(raw_root: Path, run: str, ruler: str) -> float:
    if ruler == "linear":
        entry = json.loads((raw_root / "eval_ablation_ep800" / f"{run}.json").read_text(encoding="utf-8"))
        if entry.get("epoch") != 799 or entry.get("checkpoint_kind", "last") != "last" \
                or entry.get("probe_select", "train_slice") != "train_slice":
            raise ValueError(f"{run}: post-hoc result is not strict / epoch 799 ({entry.get('epoch')!r})")
        return float(entry["linear"]["top1"])
    series = _online_series(raw_root / "online_ep800" / run)
    acc = [(e, float(r["val/probe_acc"])) for e, r in sorted(series.items()) if "val/probe_acc" in r]
    if len(acc) != 800 or acc[-1][0] != 799:
        raise ValueError(f"{run}: {len(acc)} online-probe epochs, last {acc[-1][0] if acc else None}; expected 800 to 799")
    if ruler == "online":
        value = acc[-1][1]
        return 100.0 * value if value <= 1.0 else value
    khat = [float(r["train/local/effective_clusters_epoch"]) for _, r in sorted(series.items())
            if "train/local/effective_clusters_epoch" in r]
    tail = khat[-10:]
    return sum(tail) / len(tail)


def _imagenette_cells(raw_root: Path) -> dict[tuple[int, float, int], set[str]]:
    """Every archived fixk run, grouped by (batch, lambda, K) parsed from its own name."""
    cells: dict[tuple[int, float, int], set[str]] = {}
    for folder in sorted((raw_root / "online_ep800").glob("ep800_fixk_*")):
        found = re.fullmatch(r"ep800_fixk_b(\d+)_lam([0-9.]+)_k(\d+)_x\d+_s\d+", folder.name)
        if found:
            key = (int(found.group(1)), float(found.group(2)), int(found.group(3)))
            cells.setdefault(key, set()).add(folder.name)
    return cells


def _band_of(strength: float) -> str:
    s = round(strength, 9)
    return ("collapse" if s < 0.4 else "degraded" if s < 0.64 else "healthy" if s <= 8
            else "declining" if s <= 20 else "over-regularized")


def _rederive(cell: dict, raw_root: Path) -> dict:
    """The cell's numbers from the archive alone. Raises on anything the archive cannot vouch for."""
    import statistics as _st

    kind = cell["kind"]
    if kind in ("indomain", "transfer"):
        values = []
        for source in cell["sources"]:
            path = raw_root / source
            if not path.is_file():
                raise FileNotFoundError(f"archived input {source} missing")
            entry = json.loads(path.read_text(encoding="utf-8"))
            why = _latex_protocol(entry, kind)
            if why:
                raise ValueError(f"{source}: {why}")
            values.append(_latex_value(entry, cell["field"]))
        mean, sd = _latex_stats(values)
        return {"mean": mean, "sd": sd}
    if kind in ("imagenette_online", "imagenette_khat"):
        values = [_imagenette_run(raw_root, run, kind.split("_")[1]) for run in cell["runs"]]
        mean, sd = _latex_stats(values)
        return {"mean": mean, "sd": sd if kind == "imagenette_online" else None}
    if kind in ("band_count", "band_stat"):
        band = cell["row"].rsplit("/", 1)[-1]
        truth = {frozenset(runs) for key, runs in _imagenette_cells(raw_root).items()
                 if _band_of(key[1] * key[0] / key[2]) == band}
        claimed = {frozenset(runs) for runs in cell["members"]}
        if claimed != truth:
            raise ValueError(f"band {band}: the table's cells are not the archive's ({len(claimed)} vs {len(truth)})")
        if kind == "band_count":
            return {"value": len(truth)}
        means = [_st.fmean(_imagenette_run(raw_root, run, cell["ruler"]) for run in runs) for runs in cell["members"]]
        return {"value": {"min": min(means), "median": _st.median(means), "max": max(means)}[cell["stat"]]}
    raise ValueError(f"unknown cell kind {kind!r}")


def _claimed(cell: dict) -> dict:
    if cell["kind"] in ("band_count", "band_stat"):
        return {"value": cell.get("value")}
    return {"mean": cell.get("mean"), "sd": cell.get("sd")}


def audit_latex(table_text: str, payload: dict, raw_root: Path) -> list[str]:
    """Every claimed cell re-derived from ``raw_root`` and found at its exact place in its own row; the
    bold and the medals re-decided from the data; no numeral in a row that no cell claims; every row
    label as recorded; for the strength bands, the membership rebuilt from the archived run names."""
    problems: list[str] = []
    rows: dict[str, tuple[str, list[str]]] = {}
    for line in table_text.splitlines():
        found = LATEX_MARKER.search(line)
        if not found:
            continue
        body = line[:found.start()].rstrip()
        if body.endswith("\\\\"):
            body = body[:-2].rstrip()
        label, *data = [part.strip() for part in body.split(" & ")]
        if found.group(1) in rows:
            problems.append(f"{found.group(1)}: the marker appears on two rows")
        rows[found.group(1)] = (label, data)
    for marker, label in (payload.get("row_labels") or {}).items():
        if marker not in rows:
            problems.append(f"{marker}: row missing from the table")
        elif rows[marker][0] != label:
            problems.append(f"{marker}: label {rows[marker][0]!r} is not the recorded {label!r}")

    import hashlib

    fresh: dict[str, dict] = {}
    for key, cell in payload["cells"].items():
        for source, digest in (cell.get("blake2b") or {}).items():
            path = raw_root / source
            if path.is_file() and hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest() != digest:
                problems.append(f"{key}: archive {source} changed since the table was generated")
        try:
            got = _rederive(cell, raw_root)
        except (FileNotFoundError, KeyError, ValueError) as error:
            problems.append(f"{key}: {error}")
            continue
        for name, actual in got.items():
            claimed = _claimed(cell).get(name)
            if (claimed is None) != (actual is None) or (actual is not None and abs(claimed - actual) > 1e-9):
                problems.append(f"{key}.{name}: cells.json says {claimed!r}, the archive says {actual!r}")
        fresh[key] = got

    expected_bold: set[str] = set()
    if payload.get("bold_rule") == "best_in_family_column":
        groups: dict[tuple[str, int], list[tuple[str, float]]] = {}
        for key, got in fresh.items():
            cell = payload["cells"][key]
            groups.setdefault((cell["row"].split("/")[1], cell["latex_col"]), []).append((key, float(f"{got['mean']:.1f}")))
        for members in groups.values():
            top = max(v for _, v in members)
            expected_bold.update(k for k, v in members if v == top)
    medals: dict[str, str] = {}
    if payload.get("medal_rule") == "top3_by_2dp_mean":
        shown = {k: float(f"{g['mean']:.2f}") for k, g in fresh.items() if payload["cells"][k]["kind"] == "imagenette_online"}
        ranked = sorted(set(shown.values()), reverse=True)[:3]
        medals = {k: ("ablationgold", "ablationsilver", "ablationbronze")[ranked.index(v)] for k, v in shown.items() if v in ranked}

    claimed_numerals: dict[str, list[str]] = {}
    claimed_slots: set[tuple[str, int]] = set()
    for key, got in fresh.items():
        cell = payload["cells"][key]
        kind = cell["kind"]
        if kind in ("indomain", "transfer"):
            text = _latex_render(got["mean"], got["sd"], cell.get("sd_digits", 2), cell.get("compact_pm", False))
            want = f"$\\mathbf{{{text}}}$" if key in expected_bold else f"${text}$"
        elif kind == "imagenette_online":
            want = (f"\\cellcolor{{{medals[key]}}}" if key in medals else "") + f"\\makecell{{${got['mean']:.2f} \\pm {got['sd']:.2f}$}}"
        elif kind == "imagenette_khat":
            want = f"${got['mean']:.1f}$"
        elif kind == "band_count":
            want = str(got["value"])
        else:
            want = f"${got['value']:.2f}$"
        if cell["row"] not in rows:
            problems.append(f"{key}: no table row carries {cell['row']!r}")
            continue
        data = rows[cell["row"]][1]
        col, part = cell["latex_col"], cell.get("part")
        if col >= len(data):
            problems.append(f"{key}: row has {len(data)} columns, the cell claims column {col}")
            continue
        shown = data[col] if part is None else (data[col].split(" / ") + [""] * 2)[part].strip()
        if shown != want:
            problems.append(f"{key}: expected {want!r} at column {col}{'' if part is None else f'.{part}'}, the table has {shown!r}")
        claimed_numerals.setdefault(cell["row"], []).extend(NUMERAL.findall(re.sub(r"\\cellcolor\{[a-z]+\}", "", want)))
        claimed_slots.add((cell["row"], col))
    for marker, (_, data) in rows.items():
        present = sorted(n for text in data for n in NUMERAL.findall(re.sub(r"\\cellcolor\{[a-z]+\}", "", text)))
        if present != sorted(claimed_numerals.get(marker, [])):
            extra = sorted(set(present) - set(claimed_numerals.get(marker, [])))
            problems.append(f"{marker}: numerals {present} are not exactly the claimed ones"
                            + (f" (unclaimed: {extra})" if extra else ""))
        for col, text in enumerate(data):
            if (marker, col) not in claimed_slots and text != "--" and NUMERAL.search(text):
                problems.append(f"{marker}: column {col} shows {text!r} but no cell claims it")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default=DEFAULT_TABLE, type=Path)
    parser.add_argument("--cells", default=DEFAULT_CELLS, type=Path)
    parser.add_argument("--raw", default="results/raw", type=Path)
    parser.add_argument("--latex", action="store_true",
                        help="a LaTeX table from scripts/make_rebuttal_tables.py (row markers, exact columns, bold)")
    args = parser.parse_args(argv)
    if not args.table.is_file() or not args.cells.is_file():
        print(f"nothing to audit: {args.table} or {args.cells} missing", file=sys.stderr)
        return 2
    payload = json.loads(args.cells.read_text(encoding="utf-8"))
    checker = audit_latex if args.latex else audit
    problems = checker(args.table.read_text(encoding="utf-8"), payload, args.raw)
    n = len(payload["cells"])
    if problems:
        print(f"AUDIT FAILED: {len(problems)} problem(s) over {n} cell(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"AUDIT PASSED: {n} cell(s) re-derived from {args.raw} and found in {args.table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
