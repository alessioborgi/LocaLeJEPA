#!/usr/bin/env python
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The six Imagenette ablation tables, with the realised-K column, from the campaign's runs.

Inputs are the training runs themselves -- ``<log_root>/train/runs/<run_name>/csv/version_*/
metrics.csv`` -- because the ablation metric is the online probe's final-epoch validation
accuracy (LeJEPA's own Imagenette reading, `MINIMAL.md`); no separate evaluation exists. Every
cell is mean ± std over the manifest's seeds, and every LocaLeJEPA cell carries the realised
number of clusters (`train/local/effective_clusters`, mean over the last 10 epochs): K is a
MAXIMUM, and at batch 64 the large-K cells run with far fewer (src/losses.py::local_loss).

The declaration rule of campaign decision 14 is applied here and written to
``results/ablations/declaration.json``: λ = best mean accuracy over the fixed-K block at
B=256×1, K = best per batch at that λ. Phase 4 copies those values into the manifests by hand
(a committed edit is the gate, G6) -- this script never edits a manifest.

Per-run summaries are first written to ``results/raw/ablations/<run_name>.json`` (final
accuracy, realised K, epochs), so the tables have archived inputs and
``scripts/audit_tables.py`` can re-derive them.

    python scripts/make_ablation_tables.py --log-root <LOG_ROOT> [--out results/ablations]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eval.io import atomic_write, provenance_block, refuse_to_replace_with_placeholders  # noqa: E402
import scripts.aicr.campaign as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = ["ablation_grid.yaml", "ablation_batch128.yaml"]
# Which number each cell reports. "online" is the probe trained DURING pretraining (LeJEPA's own
# Imagenette reading); the other two are the draft's post-hoc probes on frozen features and need
# --eval-dir. A cell is absent for a metric it has no value for, never zero and never a fallback.
METRIC_KEYS = {"online": "final_probe_acc", "knn": "posthoc_knn_top1", "linear": "posthoc_linear_top1"}
ACC = "val/probe_acc"
BATCHES = [(64, 4), (128, 2), (256, 1)]
K_AXIS = [1, 2, 4, 8, 16, 32]


def batch_label(b: int, x: int) -> str:
    return f"{b}×{x}"


def posthoc_of(run_name: str, eval_dir: Path | None, report_k: int) -> tuple[float | None, float | None]:
    """``(knn_top1, linear_top1)`` from the evaluation's per-backbone JSON, or ``(None, None)``.

    The JSON is written incrementally by ``src/eval_indomain.py``, one file per backbone, so this
    needs no merge step and a cell whose evaluation has not run yet simply has no value -- which
    ``aggregate`` turns into an absent cell rather than a number.
    """
    if eval_dir is None:
        return None, None
    path = eval_dir / "per_backbone" / f"{run_name}.json"
    if not path.is_file():
        return None, None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    knn = (entry.get("knn") or {}).get(str(report_k)) or {}
    linear = entry.get("linear") or {}
    return knn.get("top1"), linear.get("top1")


def summarise_run(run: C.RunSpec, log_root: Path, eval_dir: Path | None = None,
                  report_k: int = 20) -> dict | None:
    """Final-epoch online-probe accuracy and the realised K of one run, or None if it has no
    complete metrics.csv. A run that is not finished is 'pending', never a partial number."""
    csv_dir = log_root / "train" / "runs" / run.run_name / "csv"
    if not C.metrics_versions(csv_dir):
        return None
    # every launch of a resumed run, each epoch from the last launch that logged it (F52)
    _, rows = C.merged_metrics_rows(csv_dir)
    final_acc, epochs = None, set()
    for row in rows:
        if row.get(ACC) not in ("", None):
            final_acc = float(row[ACC])
            epochs.add(int(float(row["epoch"])))
    if final_acc is None or len(epochs) < run.max_epochs:
        return None
    realised = C.realised_k(csv_dir)
    knn_top1, linear_top1 = posthoc_of(run.run_name, eval_dir, report_k)
    return {"run": run.run_name, "tag": run.tag, "seed": run.seed, "overrides": list(run.overrides),
            "posthoc_knn_top1": knn_top1, "posthoc_linear_top1": linear_top1, "posthoc_report_k": report_k,
            "final_probe_acc": 100.0 * final_acc if final_acc <= 1.0 else final_acc,
            "epochs": len(epochs), "realised_k_last10": realised["last_mean"],
            "realised_k_mean": realised["mean"], "fallback_epochs": realised["fallback_epochs"],
            "configured_k": realised["configured"]}


def cell_key(run: C.RunSpec) -> tuple:
    """Everything that identifies a cell: the tag and the overrides minus the seed."""
    return (run.tag, run.overrides)


def aggregate(summaries: list[dict], metric: str = "online") -> dict | None:
    """``None`` when no seed of the cell has a value for this metric: an unevaluated cell must render
    as absent, not as a number taken from a different ruler."""
    accs = [s[METRIC_KEYS[metric]] for s in summaries if s.get(METRIC_KEYS[metric]) is not None]
    if not accs:
        return None
    ks = [s["realised_k_last10"] for s in summaries if s["realised_k_last10"] is not None]
    mean = sum(accs) / len(accs)
    std = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) if len(accs) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(accs),
            "realised_k": (sum(ks) / len(ks)) if ks else None,
            "fallback_epochs": max((s["fallback_epochs"] or 0) for s in summaries)}


def fmt(agg: dict | None, *, with_k: bool) -> str:
    if agg is None:
        return "—"
    text = f"{agg['mean']:.1f} ± {agg['std']:.1f} (n={agg['n']})"
    if with_k and agg["realised_k"] is not None:
        text += f", K̂={agg['realised_k']:.1f}"
    if agg["fallback_epochs"]:
        text += f" ⚠fallback×{agg['fallback_epochs']}"
    return text


def overrides_of(run: C.RunSpec) -> dict[str, str]:
    return run.override_map()


def build(log_root: Path, manifests: list[str] | None = None, eval_dir: Path | None = None,
          metric: str = "online", report_k: int = 20) -> tuple[dict[tuple, dict], list[dict], list[C.RunSpec]]:
    runs = [r for m in (manifests or DEFAULT_MANIFESTS)
            for r in C.expand(C.load_manifest(REPO / "configs" / "campaign" / m))]
    per_run = []
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        summary = summarise_run(run, log_root, eval_dir, report_k)
        if summary is not None:
            per_run.append(summary)
            grouped[cell_key(run)].append(summary)
    cells = {}
    for key, items in grouped.items():
        agg = aggregate(items, metric)
        if agg is not None:
            cells[key] = agg
    return cells, per_run, runs


def lookup(cells, runs, *, tag: str, lam: float | None = None, k: int | None = None,
           batch: tuple[int, int] | None = None, prefix: str = ""):
    """``tag`` is the BARE block name; ``prefix`` is the manifest's (``ep800_`` for the 800-epoch
    grid). Without it the 800-epoch tags match nothing and every cell renders as an em dash."""
    for run in runs:
        if run.tag != prefix + tag:
            continue
        o = overrides_of(run)
        if lam is not None and not math.isclose(float(o.get("model.lamb", "nan")), lam):
            continue
        if k is not None and int(o.get("model.reg_loss.num_clusters", -1)) != k:
            continue
        if batch is not None and (int(o.get("data.batch_size", -1)), int(o.get("trainer.accumulate_grad_batches", -1))) != batch:
            continue
        return cells.get(cell_key(run))
    return None


def declare(cells, runs, prefix: str = "", metric: str = "online") -> dict:
    """Decision 14's rule, from the data: λ = best mean over the fixed-K block at 256×1; K = best
    per batch at that λ (ramps compete as 'fwd'/'rev'). Ties go to the smaller λ / smaller K."""
    lambdas = sorted({float(overrides_of(r)["model.lamb"]) for r in runs if r.tag == prefix + "fixk"})
    best_lam, best = None, -1.0
    for lam in lambdas:
        scores = [lookup(cells, runs, tag="fixk", lam=lam, k=k, batch=(256, 1), prefix=prefix) for k in K_AXIS]
        scores = [s["mean"] for s in scores if s is not None]
        if scores and max(scores) > best:
            best_lam, best = lam, max(scores)
    k_by_batch = {}
    if best_lam is not None:
        for b in BATCHES:
            options = {str(k): lookup(cells, runs, tag="fixk", lam=best_lam, k=k, batch=b, prefix=prefix) for k in K_AXIS}
            options.update({"fwd": lookup(cells, runs, tag="fwd", lam=best_lam, batch=b, prefix=prefix),
                            "rev": lookup(cells, runs, tag="rev", lam=best_lam, batch=b, prefix=prefix)})
            options = {k: v for k, v in options.items() if v is not None}
            if options:
                winner = max(options, key=lambda k: options[k]["mean"])
                k_by_batch[batch_label(*b)] = {"K": winner, "mean": options[winner]["mean"], "n": options[winner]["n"]}
    return {"rule": f"lambda = argmax mean {metric} acc over the fixed-K block at B=256x1; "
                    "K = argmax per batch at that lambda (fwd/rev ramps compete); ties to the smaller value",
            "metric": metric, "tag_prefix": prefix,
            "lambda": best_lam, "lambda_best_mean": best if best_lam is not None else None, "K_by_batch": k_by_batch}


def tables(cells, runs, declared_lam: float | None, prefix: str = "") -> list[tuple[str, list[str]]]:
    out = []
    lambdas = sorted({float(overrides_of(r)["model.lamb"]) for r in runs if r.tag == prefix + "fixk"})
    # 1-2: lambda axis at 256x1, fixed K and dynamic K
    lines = ["| λ | " + " | ".join(f"K={k}" for k in K_AXIS) + " |", "|---|" + "---|" * len(K_AXIS)]
    for lam in lambdas:
        lines.append(f"| {lam:g} | " + " | ".join(fmt(lookup(cells, runs, tag="fixk", lam=lam, k=k, batch=(256, 1), prefix=prefix), with_k=True) for k in K_AXIS) + " |")
    out.append(("λ axis, fixed K, B=256×1", lines))
    lines = ["| λ | K 2→16 (fwd) | K 16→2 (rev) |", "|---|---|---|"]
    for lam in lambdas:
        lines.append(f"| {lam:g} | {fmt(lookup(cells, runs, tag='fwd', lam=lam, batch=(256, 1), prefix=prefix), with_k=True)} | "
                     f"{fmt(lookup(cells, runs, tag='rev', lam=lam, batch=(256, 1), prefix=prefix), with_k=True)} |")
    out.append(("λ axis, dynamic K, B=256×1", lines))
    # 3-4: K axis at 64x4 and 256x1, at the declared lambda, with the scale rows
    for b in ((64, 4), (256, 1)):
        lines = ["| K | LocaLeJEPA |", "|---|---|"]
        for k in K_AXIS:
            lines.append(f"| {k} | {fmt(lookup(cells, runs, tag='fixk', lam=declared_lam, k=k, batch=b, prefix=prefix), with_k=True)} |")
        lines.append(f"| 2→16 (fwd) | {fmt(lookup(cells, runs, tag='fwd', lam=declared_lam, batch=b, prefix=prefix), with_k=True)} |")
        lines.append(f"| 16→2 (rev) | {fmt(lookup(cells, runs, tag='rev', lam=declared_lam, batch=b, prefix=prefix), with_k=True)} |")
        lines.append(f"| LeJEPA (SIGReg) | {fmt(lookup(cells, runs, tag='sigreg', batch=b, prefix=prefix), with_k=False)} |")
        lines.append(f"| RDMReg | {fmt(lookup(cells, runs, tag='rdmreg', batch=b, prefix=prefix), with_k=False)} |")
        out.append((f"K axis, B={batch_label(*b)}, λ={declared_lam if declared_lam is not None else '?'}", lines))
    # 5-6: batch axis at the declared lambda, fixed K and dynamic K
    lines = ["| per-step batch | " + " | ".join(f"K={k}" for k in K_AXIS) + " |", "|---|" + "---|" * len(K_AXIS)]
    for b in BATCHES:
        lines.append(f"| {batch_label(*b)} | " + " | ".join(fmt(lookup(cells, runs, tag="fixk", lam=declared_lam, k=k, batch=b, prefix=prefix), with_k=True) for k in K_AXIS) + " |")
    out.append((f"batch axis, fixed K, λ={declared_lam if declared_lam is not None else '?'}", lines))
    lines = ["| per-step batch | K 2→16 (fwd) | K 16→2 (rev) |", "|---|---|---|"]
    for b in BATCHES:
        lines.append(f"| {batch_label(*b)} | {fmt(lookup(cells, runs, tag='fwd', lam=declared_lam, batch=b, prefix=prefix), with_k=True)} | "
                     f"{fmt(lookup(cells, runs, tag='rev', lam=declared_lam, batch=b, prefix=prefix), with_k=True)} |")
    out.append((f"batch axis, dynamic K, λ={declared_lam if declared_lam is not None else '?'}", lines))
    return out


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--out", default="results/ablations", type=Path)
    parser.add_argument("--raw", default="results/raw/ablations", type=Path)
    parser.add_argument("--manifests", nargs="+", default=DEFAULT_MANIFESTS,
                        help="campaign manifests to read; default is the 100-epoch pair")
    parser.add_argument("--tag-prefix", default="",
                        help="prefix every block tag carries, e.g. ep800_ for the 800-epoch grid")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="an eval output_folder; enables the post-hoc knn/linear metrics")
    parser.add_argument("--metric", choices=sorted(METRIC_KEYS), default="online",
                        help="which number the cells report; knn and linear need --eval-dir")
    parser.add_argument("--report-k", type=int, default=20, help="the k of the reported k-NN")
    args = parser.parse_args(argv)
    if args.metric != "online" and args.eval_dir is None:
        parser.error(f"--metric {args.metric} needs --eval-dir: it is read from the evaluation, not the training log")

    cells, per_run, runs = build(args.log_root, args.manifests, args.eval_dir, args.metric, args.report_k)
    args.raw.mkdir(parents=True, exist_ok=True)
    for summary in per_run:
        with atomic_write(args.raw / f"{summary['run']}.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
    declaration = declare(cells, runs, args.tag_prefix, args.metric)
    args.out.mkdir(parents=True, exist_ok=True)
    with atomic_write(args.out / "declaration.json", "w", encoding="utf-8") as f:
        json.dump(declaration, f, indent=1)

    lengths = sorted({r.max_epochs for r in runs})
    epochs_text = " / ".join(str(e) for e in lengths) + (" epochs" if len(lengths) == 1 else " epochs (mixed)")
    metric_text = {"online": "online-probe accuracy (trained during pretraining)",
                   "knn": f"post-hoc k-NN@{args.report_k} top-1 on frozen features",
                   "linear": "post-hoc linear-probe top-1 on frozen features"}[args.metric]
    doc = [f"# Imagenette ablations (ViT-S/8, {epochs_text}, {metric_text}, mean ± std over seeds)", "",
           f"Auto-generated by `scripts/make_ablation_tables.py` from {len(per_run)} finished runs of "
           f"{len(runs)} in {', '.join(args.manifests)}. K̂ = realised clusters (mean of the last 10 epochs); ⚠ marks a",
           "cell where some epoch fell back to the global statistic. `—` = no finished run, or none evaluated yet.", "",
           f"**Declared by rule** (decision 14): λ = {declaration['lambda']}, K per batch = "
           f"{ {b: v['K'] for b, v in declaration['K_by_batch'].items()} }", ""]
    for title, lines in tables(cells, runs, declaration["lambda"], args.tag_prefix):
        doc += [f"## {title}", "", *lines, ""]
    text = "\n".join(doc)
    out_md = args.out / "ablation_tables.md"
    refuse_to_replace_with_placeholders(str(out_md), [str(args.raw)])
    text += provenance_block(os.path.relpath(__file__, REPO), [str(args.raw)], archive_to=None)
    with atomic_write(out_md, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return text


if __name__ == "__main__":
    main()
