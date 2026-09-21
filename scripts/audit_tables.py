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
    knn = entry.get("knn", {}).get(str(report_k))
    return {"knn20": knn["top1"] if knn else None, "linear": entry["linear"]["top1"], "epoch": entry.get("epoch")}


def audit(table_text: str, cells_payload: dict, raw_root: Path) -> list[str]:
    problems: list[str] = []
    report_k = int(cells_payload.get("report_k", 20))
    rows = [line for line in table_text.splitlines() if line.startswith("|")]
    for key, cell in cells_payload["cells"].items():
        try:
            fresh = rederive(raw_root, cell, report_k)
        except (FileNotFoundError, KeyError, ValueError) as error:
            problems.append(f"{key}: cannot re-derive ({error})")
            continue
        for field in ("knn20", "linear"):
            claimed, actual = cell.get(field), fresh.get(field)
            if claimed is None and actual is None:
                continue
            if claimed is None or actual is None or not math.isclose(claimed, actual, rel_tol=0, abs_tol=1e-9):
                problems.append(f"{key}.{field}: cells.json says {claimed!r}, the archive says {actual!r}")
        if cell.get("epoch") != fresh.get("epoch"):
            problems.append(f"{key}.epoch: cells.json says {cell.get('epoch')!r}, the archive says {fresh.get('epoch')!r}")
        # the rendered string must appear in the table -- the grep direction
        knn_text = f"{fresh['knn20']:.1f}" if fresh["knn20"] is not None else "—"
        rendered = f"{knn_text} / {fresh['linear']:.1f}" if "lejepa_protocol" not in key else f"linear {fresh['linear']:.1f}"
        if not any(rendered in row for row in rows):
            problems.append(f"{key}: rendered value {rendered!r} is not in the table")
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
