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
"""

from __future__ import annotations

import argparse
import json
import math
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", default=DEFAULT_TABLE, type=Path)
    parser.add_argument("--cells", default=DEFAULT_CELLS, type=Path)
    parser.add_argument("--raw", default="results/raw", type=Path)
    args = parser.parse_args(argv)
    if not args.table.is_file() or not args.cells.is_file():
        print(f"nothing to audit: {args.table} or {args.cells} missing", file=sys.stderr)
        return 2
    payload = json.loads(args.cells.read_text(encoding="utf-8"))
    problems = audit(args.table.read_text(encoding="utf-8"), payload, args.raw)
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
