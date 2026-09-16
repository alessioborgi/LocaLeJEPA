# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Read a learning curve at chosen epoch horizons.

Answers one question: does a gap between two arms exist EARLY and close later?
The manuscript reports its local-vs-global BallReg gains at 100 epochs; our runs
go to 800 and tie there. Both horizons come off the same snapshot series, so the
comparison needs no extra pretraining -- only this readout.

Unlike ``plot_embedor_learning_curves.py`` this makes no assumption about run
names (that one parses ``ball_l01_k<N>_s<seed>`` and cannot see the geometric
arms at all), so it works on any ``eval_snapshots.py`` output directory.

    python scripts/snapshot_horizon_table.py --data logs/eval_snapshots_geometric
    python scripts/snapshot_horizon_table.py --data logs/eval_snapshots_geometric \
        --horizons 100 200 800 --metric orc_gap
    python scripts/snapshot_horizon_table.py --data logs/eval_snapshots_geometric \
        --baseline hvit_sigreg_matched          # add a delta-vs-baseline column

Columns: the metric at each horizon, AULC over 0..--aulc-epochs (normalised, so
it reads on the metric's own scale and measures how fast the curve rose), and
the best value reached at any snapshot.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np

# np.trapezoid is numpy >= 2.0; np.trapz is the <2.0 spelling and is deprecated
# in >= 2.0. Bind once so the script runs under either (the cluster venv has
# numpy 2.4, a stock conda python may still be on 1.26).
_trapz = getattr(np, "trapezoid", None) or np.trapz


def load(data_dir: str) -> dict[str, list[dict]]:
    """{run: [row, ...]} from the per-run CSVs, ignoring the merged file."""
    runs: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        if os.path.basename(path) == "learning_curves.csv":
            continue
        with open(path, encoding="utf-8") as fh:
            rows = []
            for r in csv.DictReader(fh):
                out = {"run": r["run"], "epoch": int(r["epoch"])}
                for k, v in r.items():
                    if k not in ("run", "experiment", "epoch"):
                        try:
                            out[k] = float(v)
                        except (TypeError, ValueError):
                            out[k] = None
                rows.append(out)
        if rows:
            runs[rows[0]["run"]] = sorted(rows, key=lambda r: r["epoch"])
    return runs


def at_horizon(rows: list[dict], metric: str, horizon: int) -> tuple[float | None, int | None]:
    """Value at the LAST snapshot at or before `horizon`, and that snapshot's epoch.

    No interpolation: a snapshot is a real evaluated checkpoint, and inventing a
    value between two of them would put a number in the table that was never
    measured. The epoch actually used is reported alongside.
    """
    usable = [r for r in rows if r["epoch"] <= horizon and r.get(metric) is not None]
    if not usable:
        return None, None
    return usable[-1][metric], usable[-1]["epoch"]


def aulc(rows: list[dict], metric: str, cap: int) -> float | None:
    """Trapezoidal area under the curve up to `cap`, divided by the epoch span."""
    pts = [(r["epoch"], r[metric]) for r in rows if r["epoch"] <= cap and r.get(metric) is not None]
    if len(pts) < 2:
        return None
    ep, val = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
    return float(_trapz(val, ep) / (ep[-1] - ep[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="logs/eval_snapshots_geometric")
    ap.add_argument("--metric", default="knn_loo_top1")
    ap.add_argument("--horizons", type=int, nargs="+", default=[100, 800])
    ap.add_argument("--aulc-epochs", type=int, default=200)
    ap.add_argument("--baseline", default=None,
                    help="run name to difference every other run against")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    args = ap.parse_args()

    runs = load(args.data)
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        runs = {k: v for k, v in runs.items() if k in want}
    if not runs:
        sys.exit(f"no per-run CSVs in {args.data} -- run eval_snapshots.py first")

    horizons = sorted(args.horizons)
    head = f'{"run":24s}' + "".join(f'{"ep" + str(h):>12}' for h in horizons)
    head += f'{"AULC(0-" + str(args.aulc_epochs) + ")":>14}{"best":>9}'
    print(f"\n{args.metric}   ({args.data})")
    print(head)
    print("-" * len(head))

    table = {}
    for name, rows in sorted(runs.items()):
        cells, used = [], []
        for h in horizons:
            v, ep = at_horizon(rows, args.metric, h)
            cells.append(v)
            used.append(ep)
        area = aulc(rows, args.metric, args.aulc_epochs)
        best = max((r[args.metric] for r in rows if r.get(args.metric) is not None), default=None)
        table[name] = (cells, area, best)

        line = f"{name:24s}"
        for v, ep, h in zip(cells, used, horizons):
            line += "           -" if v is None else (
                f"{v:>12.4f}" if ep == h else f"{v:>11.4f}*")
        line += "             -" if area is None else f"{area:>14.4f}"
        line += "        -" if best is None else f"{best:>9.4f}"
        print(line)

    if any(ep != h for name, rows in runs.items()
           for h in horizons for _, ep in [at_horizon(rows, args.metric, h)] if ep is not None):
        print("\n  * value taken from the last snapshot at or before that horizon, "
              "not exactly at it (no interpolation).")

    if args.baseline:
        if args.baseline not in table:
            sys.exit(f"baseline {args.baseline!r} not among {sorted(table)}")
        base_cells, base_area, _ = table[args.baseline]
        print(f"\ndifference vs {args.baseline}")
        print(f'{"run":24s}' + "".join(f'{"ep" + str(h):>12}' for h in horizons)
              + f'{"AULC":>14}')
        print("-" * len(head))
        for name, (cells, area, _) in sorted(table.items()):
            if name == args.baseline:
                continue
            line = f"{name:24s}"
            for v, b in zip(cells, base_cells):
                line += "           -" if (v is None or b is None) else f"{v - b:>+12.4f}"
            line += "             -" if (area is None or base_area is None) else f"{area - base_area:>+14.4f}"
            print(line)
        print("\n  A gap that is positive at the early horizon and ~zero at the late one is a "
              "convergence-SPEED\n  effect. One that is ~zero at both was never there on this "
              "dataset. Single runs: no error bars,\n  so read only differences that clear the "
              "seed spread you measured on the plain-ViT sweep (~0.3pp).")
    print()


if __name__ == "__main__":
    main()
