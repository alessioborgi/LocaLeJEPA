#!/usr/bin/env python
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The rebuttal revision's ImageNet-1K tables (decision 32), as LaTeX, from the campaign's evaluations.

The revision replaces the submitted paper's ImageNet-1K tables with campaign numbers, so every
cell here comes from one evaluation path: the run's ``last.ckpt`` (epoch 99), the strict linear
probe (learning rate chosen on a 2% slice of the training set, probe seed 0), k-NN with k=20 for
the headline. Four tables keep the submitted layouts and labels (``tab:in1k-main``,
``tab:in1k-transfer-main``, ``tab:in1k-indomain-full``, ``tab:in1k-transfer-full``); a fifth is
the depth study (ViT-S/8 at 12 / 6 / 3 blocks). ViT-S/8 cells average the three pretraining
seeds where three exist (mean ± sample sd); every other cell is a single run. A cell whose run
has no evaluation yet renders as ``--``, never as a number from another run or another ruler.

**Presentation rule (decision 32).** Mean-subtraction and the ball target appear only together;
no row separates them, so the centred-SIGReg and centred-local-Gaussian runs are never read here.

Which run fills which row comes from a fixed name template per arm, and every name must be a run
of some campaign manifest (``scripts/aicr/campaign.py``) -- a typo is an error, not an empty
row. Each JSON a table reads is checked against the protocol it claims (epoch 99, ``last``,
``train_slice``; transfer: all seven datasets, and a stored average equal to their mean) and
copied to ``results/raw/<eval name>/`` beside the output, so ``scripts/audit_tables.py --latex``
can re-derive every cell with code of its own. Alongside each ``<table>.tex`` this writes
``<table>.cells.json``; ``text_numbers.json`` holds every difference the prose quotes, each with
its definition.

    python scripts/make_rebuttal_tables.py --root <dir laid out like $SCR> [--out results/rebuttal]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eval.io import atomic_write, provenance_block  # noqa: E402
import scripts.aicr.campaign as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REPORT_K = 20
LAST_EPOCH = 99
DATASETS = ["dtd", "aircraft", "cifar10", "cifar100", "flowers", "food", "pets"]
DATASET_LABEL = {"dtd": "DTD", "aircraft": "Aircraft", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
                 "flowers": "Flowers", "food": "Food", "pets": "Pets"}

# eval output folders under <root> (the rendered campaign eval configs' output_folder, under evalroot/)
EVAL_DIRS = {
    ("indomain", "128"): "evalroot/logs/eval_indomain_in1k_campaign/per_backbone",
    ("indomain", "vitl14"): "evalroot/logs/eval_indomain_in1k_vitl14_campaign/per_backbone",
    ("transfer", "128"): "evalroot/logs/eval_transfer_in1k_campaign/per_backbone",
    ("transfer", "vitl14"): "evalroot/logs/eval_transfer_in1k_vitl14_campaign/per_backbone",
}

FAMILIES = {
    "vit": r"Plain ViT-S/8 (Euclidean, 21.7M)",
    "hvit": r"Hyperbolic ViT-S/8 (Poincar\'e ball, 21.7M)",
    "d4rn50": r"$D_4$-equivariant ResNet-50 (24.4M)",
    "vitl14": r"Plain ViT-L/14 (Euclidean, 303.2M; LeJEPA's ViT-L settings)",
}

# arm -> (row label, run-name template). The 128 px arms; ViT-L has its own three below.
ARMS = {
    "lejepa": (r"LeJEPA", "{f}_sigreg_s{s}"),
    "ours": (r"LeJEPA (our settings)", "{f}_control_lam0.1_s{s}"),
    "rdmreg": (r"Rectified LpJEPA (RDMReg)", "{f}_rdmreg_s{s}"),
    "k1": (r"LocaLeJEPA ($\lambda{=}0.1$, $K{=}1$)", "{f}_globalball_lam0.1_k1_s{s}"),
    "k1_quarter": (r"LocaLeJEPA ($\lambda{=}0.025$, $K{=}1$)", "{f}_strmatch_lam0.025_k1_s{s}"),
    "k2": (r"LocaLeJEPA ($\lambda{=}0.1$, $K{=}2$)", "{f}_local_lam0.1_k2_s{s}"),
    "k4": (r"LocaLeJEPA ($\lambda{=}0.1$, $K{=}4$)", "{f}_local_lam0.1_k4_s{s}"),
    "dynk": (r"LocaLeJEPA ($\lambda{=}0.1$, $K{=}1{\to}10$)", "{f}_dynk_lam0.1_s{s}"),
    "k2_l03": (r"LocaLeJEPA ($\lambda{=}0.3$, $K{=}2$)", "{f}_local_lam0.3_k2_s{s}"),
    "localsigreg": (r"LocaLeJEPA-SIGReg ($\lambda{=}0.1$, $K{=}4$)", "{f}_localsigreg_lam0.1_k4_s{s}"),
}
VITL_ARMS = {
    "lejepa": (r"LeJEPA", "vitl14_sigreg_s0"),
    "k1": (r"LocaLeJEPA ($\lambda{=}0.05$, $K{=}1$)", "vitl14_globalball_k1_s0"),
    "k4": (r"LocaLeJEPA ($\lambda{=}0.05$, $K{=}4$)", "vitl14_local_k4_s0"),
}
# the ViT-S/8 cells with three pretraining seeds (decision 29); every other cell is seed 0 only
THREE_SEEDS = {"lejepa", "ours", "k1", "k1_quarter", "k4"}

ALL_128 = ["lejepa", "ours", "rdmreg", "k1", "k1_quarter", "k2", "k4", "dynk", "k2_l03", "localsigreg"]
NOT_ON_VIT = {"k2", "dynk", "k2_l03", "localsigreg"}       # the paper ran these on H-ViT and D4 only


def full_rows(family: str) -> list[str]:
    if family == "vitl14":
        return list(VITL_ARMS)
    return [a for a in ALL_128 if not (family == "vit" and a in NOT_ON_VIT)]


MAIN_ROWS = {"vit": ["lejepa", "ours", "rdmreg", "k1", "k4"],
             "hvit": ["lejepa", "ours", "rdmreg", "k1", "k4", "dynk", "localsigreg"],
             "d4rn50": ["lejepa", "ours", "rdmreg", "k1", "k4", "dynk", "localsigreg"],
             "vitl14": ["lejepa", "k1", "k4"]}
TRANSFER_MAIN_ROWS = {f: [a for a in rows if a != "rdmreg"] for f, rows in MAIN_ROWS.items()}
FULL_ROWS = {f: full_rows(f) for f in FAMILIES}

# column = (kind, field, header). kind indomain: field "knn.<k>.top1" / "linear.top1|top5"; kind
# transfer: "datasets.<d>.top1" or "avg" (the unweighted mean of the seven).
TABLES = {
    "tab_in1k_main": {
        "label": "tab:in1k-main", "rows": MAIN_ROWS, "sd_digits": 2,
        "cols": [("indomain", "linear.top1"), ("indomain", "linear.top5"), ("indomain", "knn.20.top1"),
                 ("transfer", "avg")],
        "head": [r" & \multicolumn{2}{c}{ImageNet-1K linear (\%)} & $k$-NN (\%) & Transfer (\%) \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-5}",
                 r"Method & top-1($\uparrow$) & top-5($\uparrow$) & $k{=}20$($\uparrow$) & 7-dataset avg($\uparrow$) \\"],
        "colspec": "l cc c c", "size": r"\tiny", "tabcolsep": "4pt",
        "caption": (r"\emph{ImageNet-1K Frozen-feature Evaluation across Representation Geometries.} "
                    r"ImageNet classification and seven-dataset transfer for the Euclidean, hyperbolic and "
                    r"$D_4$-equivariant encoders at 128 px, and ViT-L/14 at 238 px. Final-epoch checkpoints; one "
                    r"linear-probe seed, learning rate selected on a held-out slice of the training set. "
                    r"ViT-S/8 rows other than RDMReg are mean $\pm$ SD over three pretraining seeds; all other rows "
                    r"are single runs. \emph{LeJEPA} uses LeJEPA's published settings; \emph{LeJEPA (our settings)} "
                    r"uses the optimization settings of LocaLeJEPA. Full sweeps in \Cref{tab:in1k-indomain-full}. "
                    r"\textbf{Bold} denotes the best result within each encoder family and metric."),
    },
    "tab_in1k_transfer_main": {
        "label": "tab:in1k-transfer-main", "rows": TRANSFER_MAIN_ROWS, "sd_digits": 1, "compact_pm": True,
        "cols": [("transfer", f"datasets.{d}.top1") for d in DATASETS] + [("transfer", "avg")],
        "head": ["Method & " + " & ".join(DATASET_LABEL[d] for d in DATASETS) + r" & Avg \\"],
        "colspec": "l ccccccc c", "size": r"\small", "tabcolsep": "4pt",
        "caption": (r"\emph{Linear Transfer across Seven Downstream Datasets.} "
                    r"Frozen-feature top-1 accuracy (\%); \emph{Avg} is the unweighted mean. One linear-probe seed. "
                    r"ViT-S/8 rows are mean $\pm$ SD over three pretraining seeds; all other rows are single runs. "
                    r"\textbf{Bold} denotes the best mean within each encoder family and column."),
    },
    "tab_in1k_indomain_full": {
        "label": "tab:in1k-indomain-full", "rows": FULL_ROWS, "sd_digits": 2,
        "cols": [("indomain", f"knn.{k}.top1") for k in (10, 20, 100, 200)]
                + [("indomain", "linear.top1"), ("indomain", "linear.top5")],
        "head": [r" & \multicolumn{4}{c}{$k$-NN top-1 (\%)} & \multicolumn{2}{c}{Linear (\%)} \\",
                 r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}",
                 r"Method & $k{=}10$ & $k{=}20$ & $k{=}100$ & $k{=}200$ & top-1 & top-5 \\"],
        "colspec": "l cccc cc", "size": r"\small", "tabcolsep": "4pt",
        "caption": (r"\emph{ImageNet-1K Frozen-feature Classification across All Configurations.} "
                    r"Top-1 $k$-NN and linear-probe accuracy (\%) on the ImageNet-1K validation set using the "
                    r"final-epoch checkpoints; one linear-probe seed. ViT-S/8 rows other than RDMReg are mean $\pm$ SD "
                    r"over three pretraining seeds; all other rows are single runs. "
                    r"\textbf{Bold} denotes the best mean within each encoder family."),
    },
    "tab_in1k_transfer_full": {
        "label": "tab:in1k-transfer-full", "rows": FULL_ROWS, "sd_digits": 2,
        "cols": [("transfer", f"datasets.{d}.top1") for d in DATASETS] + [("transfer", "avg")],
        "head": ["Method & " + " & ".join(DATASET_LABEL[d] for d in DATASETS) + r" & Avg \\"],
        "colspec": "l ccccccc c", "size": r"\small", "tabcolsep": "3pt",
        "caption": (r"\emph{Transfer Linear Probing across All Configurations.} "
                    r"Top-1 accuracy (\%) on the official test splits of seven downstream datasets using frozen "
                    r"final-epoch ImageNet-1K representations; one linear-probe seed. ViT-S/8 rows other than RDMReg "
                    r"are mean $\pm$ SD over three pretraining seeds; all other rows are single runs. "
                    r"\textbf{Bold} marks the best mean within each encoder family and column."),
    },
}

# the depth study (capacity campaign, decision 25 + the strength control): ViT-S/8 seed 0
DEPTH_ARMS = {"k1": (r"$K{=}1$, $\lambda{=}0.1$", {12: "vit_globalball_lam0.1_k1_s0", 6: "vit_capdep_dep6_lam0.1_k1_s0",
                                                    3: "vit_capdep_dep3_lam0.1_k1_s0"}),
              "k1_quarter": (r"$K{=}1$, $\lambda{=}0.025$", {12: "vit_strmatch_lam0.025_k1_s0",
                                                              6: "vit_strmatch_dep6_lam0.025_k1_s0",
                                                              3: "vit_strmatch_dep3_lam0.025_k1_s0"}),
              "k4": (r"$K{=}4$, $\lambda{=}0.1$", {12: "vit_local_lam0.1_k4_s0", 6: "vit_capdep_dep6_lam0.1_k4_s0",
                                                   3: "vit_capdep_dep3_lam0.1_k4_s0"})}
DEPTHS = (12, 6, 3)


class ProtocolError(ValueError):
    """An evaluation file that does not say what its cell claims (wrong epoch, checkpoint, probe, or a
    partial transfer). Raised, never rendered: a table built on it would be mislabelled."""


def manifest_run_names() -> set[str]:
    names: set[str] = set()
    for manifest in sorted((REPO / "configs" / "campaign").glob("*.yaml")):
        names.update(run.run_name for run in C.expand(C.load_manifest(manifest)))
    return names


def run_names(family: str, arm: str) -> list[str]:
    if family == "vitl14":
        return [VITL_ARMS[arm][1]]
    seeds = (0, 1, 2) if family == "vit" and arm in THREE_SEEDS else (0,)
    return [ARMS[arm][1].format(f=family, s=s) for s in seeds]


def row_label(family: str, arm: str) -> str:
    return VITL_ARMS[arm][0] if family == "vitl14" else ARMS[arm][0]


def eval_dir(kind: str, family: str) -> str:
    return EVAL_DIRS[(kind, "vitl14" if family == "vitl14" else "128")]


def blake2b(path: Path) -> str:
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def check_indomain(entry: dict, name: str) -> None:
    if entry.get("epoch") != LAST_EPOCH:
        raise ProtocolError(f"{name}: epoch {entry.get('epoch')!r}, the tables report epoch {LAST_EPOCH}")
    if entry.get("checkpoint_kind", "last") != "last":
        raise ProtocolError(f"{name}: checkpoint_kind {entry.get('checkpoint_kind')!r}, the tables report last.ckpt")
    if entry.get("probe_select", "train_slice") != "train_slice":
        raise ProtocolError(f"{name}: probe_select {entry.get('probe_select')!r}, the tables are strict (train_slice)")
    if str(REPORT_K) not in (entry.get("knn") or {}):
        raise ProtocolError(f"{name}: no k={REPORT_K} k-NN result")


def transfer_missing(entry: dict) -> list[str]:
    got = entry.get("datasets") or {}
    return [d for d in DATASETS if not (isinstance(got.get(d), dict) and "top1" in got[d])]


def check_transfer(entry: dict, name: str) -> None:
    """A COMPLETE transfer result only (the loader treats a partial one as pending before this runs)."""
    if entry.get("epoch") != LAST_EPOCH:
        raise ProtocolError(f"{name}: transfer at epoch {entry.get('epoch')!r}, the tables report epoch {LAST_EPOCH}")
    missing = transfer_missing(entry)
    if missing:
        raise ProtocolError(f"{name}: transfer is missing {', '.join(missing)} ({7 - len(missing)}/7)")
    got = entry["datasets"]
    stored = entry.get("avg_top1")
    mean = sum(got[d]["top1"] for d in DATASETS) / len(DATASETS)
    if stored is not None and not math.isclose(stored, mean, rel_tol=0, abs_tol=1e-6):
        raise ProtocolError(f"{name}: stored avg_top1 {stored} is not the mean of the seven ({mean})")


def field_value(entry: dict, field: str) -> float:
    if field == "avg":
        return sum(entry["datasets"][d]["top1"] for d in DATASETS) / len(DATASETS)
    node = entry
    for part in field.split("."):
        node = node[part]
    return float(node)


class Source:
    """The evaluation files under ``root``: loaded once, protocol-checked once, archived once."""

    def __init__(self, root: Path, known: set[str]):
        self.root, self.known = root, known
        self.used: dict[str, Path] = {}          # archive-relative name -> source file
        self.pending: dict[str, str] = {}        # run -> why it renders as -- (no file yet, or a partial transfer)

    def load(self, kind: str, family: str, name: str) -> tuple[dict | None, str]:
        if name not in self.known:
            raise SystemExit(f"{name}: not a run of any campaign manifest -- a typo in a row template?")
        folder = eval_dir(kind, family)
        path = self.root / folder / f"{name}.json"
        rel = f"{Path(folder).parent.name}/{name}.json"
        if not path.is_file():
            self.pending[f"{kind}:{name}"] = "no evaluation yet"
            return None, rel
        entry = json.loads(path.read_text(encoding="utf-8"))
        if kind == "transfer" and transfer_missing(entry):
            # an evaluation still running (or one to resubmit): pending, never a partial average
            self.pending[f"{kind}:{name}"] = f"partial transfer {7 - len(transfer_missing(entry))}/7"
            return None, rel
        (check_indomain if kind == "indomain" else check_transfer)(entry, name)
        self.used[rel] = path
        return entry, rel


def fmt(x: float, digits: int) -> str:
    return f"{x:.{digits}f}"


def cell_text(mean: float, sd: float | None, sd_digits: int, compact: bool, bold: bool) -> str:
    body = fmt(mean, 1) if sd is None else (f"{fmt(mean, 1)}{{\\pm}}{fmt(sd, sd_digits)}" if compact
                                            else f"{fmt(mean, 1)} \\pm {fmt(sd, sd_digits)}")
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def build_table(name: str, spec: dict, src: Source) -> tuple[str, dict]:
    cells: dict[str, dict] = {}
    rows_out: list[str] = []
    labels: dict[str, str] = {}
    blanks = 0
    for family, arms in spec["rows"].items():
        # values first, so bold can be decided per family and column from the data
        grid: dict[tuple[str, int], dict] = {}
        for arm in arms:
            names = run_names(family, arm)
            for j, (kind, field) in enumerate(spec["cols"]):
                values, sources, digests = [], [], {}
                for run in names:
                    entry, rel = src.load(kind, family, run)
                    if entry is None:
                        values = None
                        break
                    values.append(field_value(entry, field))
                    sources.append(rel)
                    digests[rel] = blake2b(src.root / eval_dir(kind, family) / f"{run}.json")
                if values is None:
                    continue
                mean = statistics.fmean(values)
                sd = statistics.stdev(values) if len(values) > 1 else None
                grid[(arm, j)] = {"runs": names, "sources": sources, "blake2b": digests, "kind": kind,
                                  "field": field, "values": values, "mean": mean, "sd": sd}
        best = {}
        for j in range(len(spec["cols"])):
            shown = [round(float(fmt(c["mean"], 1)), 1) for (a, jj), c in grid.items() if jj == j]
            best[j] = max(shown) if shown else None
        rows_out.append(rf"\multicolumn{{{len(spec['cols']) + 1}}}{{l}}{{\emph{{{FAMILIES[family]}}}}} \\ % block:{family}")
        for arm in arms:
            marker = f"% row:{name}/{family}/{arm}"
            parts = [row_label(family, arm)]
            for j in range(len(spec["cols"])):
                c = grid.get((arm, j))
                if c is None:
                    parts.append("--")
                    blanks += 1
                    continue
                bold = best[j] is not None and round(float(fmt(c["mean"], 1)), 1) == best[j]
                text = cell_text(c["mean"], c["sd"], spec["sd_digits"], spec.get("compact_pm", False), bold)
                parts.append(text)
                cells[f"{name}/{family}/{arm}/{j}"] = {**c, "row": marker, "rendered": text, "bold": bold,
                                                       "col": j, "latex_col": j, "part": None,
                                                       "sd_digits": spec["sd_digits"],
                                                       "compact_pm": spec.get("compact_pm", False)}
            labels[marker] = parts[0]
            rows_out.append(" & ".join(parts) + r" \\ " + marker)
        rows_out.append(r"\midrule")
    rows_out = rows_out[:-1]                     # the last \midrule is the \bottomrule's job
    tex = [r"\begin{table}[t]", r"\revstart", r"\centering", rf"\caption{{{spec['caption']}}}",
           rf"\label{{{spec['label']}}}", spec["size"], rf"\setlength{{\tabcolsep}}{{{spec['tabcolsep']}}}",
           rf"\resizebox{{\linewidth}}{{!}}{{\begin{{tabular}}{{{spec['colspec']}}}", r"\toprule", *spec["head"],
           r"\midrule", *rows_out, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(tex), {"table": name, "report_k": REPORT_K, "cells": cells, "pending_cells": blanks,
                             "bold_rule": "best_in_family_column", "row_labels": labels}


def build_depth(src: Source) -> tuple[str, dict]:
    name = "tab_depth"
    cells: dict[str, dict] = {}
    labels: dict[str, str] = {}
    lines = []
    blanks = 0
    for depth in DEPTHS:
        marker = f"% row:{name}/vit/depth{depth}"
        parts = [str(depth)]
        labels[marker] = parts[0]
        for col, (arm, (_, runs)) in enumerate(DEPTH_ARMS.items()):
            entry, rel = src.load("indomain", "vit", runs[depth])
            if entry is None:
                parts.append("--")
                blanks += 1
                continue
            knn, lin = field_value(entry, "knn.20.top1"), field_value(entry, "linear.top1")
            text = f"${fmt(knn, 1)}$ / ${fmt(lin, 1)}$"
            parts.append(text)
            digest = {rel: blake2b(src.root / eval_dir("indomain", "vit") / f"{runs[depth]}.json")}
            for j, (field, value) in enumerate((("knn.20.top1", knn), ("linear.top1", lin))):
                cells[f"{name}/vit/depth{depth}/{arm}/{field}"] = {
                    "runs": [runs[depth]], "sources": [rel], "blake2b": digest, "kind": "indomain", "field": field,
                    "values": [value], "mean": value, "sd": None, "row": marker, "rendered": f"${fmt(value, 1)}$",
                    "bold": False, "col": col, "latex_col": col, "part": j, "sd_digits": 2, "compact_pm": False}
        lines.append(" & ".join(parts) + r" \\ " + marker)
    tex = [r"\begin{table}[t]", r"\revstart", r"\centering",
           r"\caption{\emph{Locality across encoder depth (ImageNet-1K, ViT-S/8, 100 epochs).} "
           r"$k$-NN ($k{=}20$) / linear top-1 (\%) on the ImageNet-1K validation set, final-epoch checkpoints, "
           r"pretraining seed 0. $K{=}1$ at $\lambda{=}0.025$ matches the effective regularization strength of "
           r"$K{=}4$ at $\lambda{=}0.1$.}",
           r"\label{tab:depth}", r"\small", r"\begin{tabular}{l ccc}", r"\toprule",
           r"Depth (blocks) & " + " & ".join(label for label, _ in DEPTH_ARMS.values()) + r" \\", r"\midrule",
           *lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(tex), {"table": name, "report_k": REPORT_K, "cells": cells, "pending_cells": blanks,
                             "bold_rule": None, "row_labels": labels}


# ---------------------------------------------------------------------------------------------------
# Appendix B (Imagenette, the campaign's 800-epoch grid, three seeds per cell): the K axis at B=256x1
# and the effective-strength bands. Read through make_ablation_tables (the online probe's final
# epoch and the realised K from metrics.csv, every launch merged -- F52) and the post-hoc JSONs.
# ---------------------------------------------------------------------------------------------------

IMAGENETTE_MANIFEST = "ablation_grid_ep800.yaml"
IMAGENETTE_PREFIX = "ep800_"
IMAGENETTE_POSTHOC = "logs/eval_ablation_ep800/per_backbone"
IMAGENETTE_LAST_EPOCH = 799
K_AXIS = (1, 2, 4, 8, 16, 32)
ONLINE_COLUMNS = ("epoch", "val/probe_acc", "train/local/effective_clusters_epoch")
MEDALS = ("ablationgold", "ablationsilver", "ablationbronze")
# effective strength lambda * N / K, N = the per-step batch (one GPU); edges as recorded in state/10
BANDS = [("collapse", r"$<0.4$", lambda s: s < 0.4), ("degraded", r"$0.4$--$0.64$", lambda s: 0.4 <= s < 0.64),
         ("healthy", r"$0.64$--$8$", lambda s: 0.64 <= s <= 8), ("declining", r"$8$--$20$", lambda s: 8 < s <= 20),
         ("over-regularized", r"$>20$", lambda s: s > 20)]


def _ablations():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_make_ablation_tables", REPO / "scripts" / "make_ablation_tables.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Imagenette:
    """The fixk block of the 800-epoch grid: per run, the online final accuracy, the realised K and the
    post-hoc linear accuracy, each with the archive-relative file the audit re-derives it from."""

    def __init__(self, root: Path):
        self.root = root
        self.A = _ablations()
        self.runs = [r for r in C.expand(C.load_manifest(REPO / "configs" / "campaign" / IMAGENETTE_MANIFEST))
                     if r.tag == IMAGENETTE_PREFIX + "fixk"]
        self.used_csv: dict[str, list[Path]] = {}      # run -> its launches' metrics.csv
        self.used_json: dict[str, Path] = {}
        self.summary: dict[str, dict] = {}
        for run in self.runs:
            s = self.A.summarise_run(run, root / "logs", root / IMAGENETTE_POSTHOC.rsplit("/", 1)[0], REPORT_K)
            if s is None:
                continue
            post = root / IMAGENETTE_POSTHOC / f"{run.run_name}.json"
            if not post.is_file():
                continue
            entry = json.loads(post.read_text(encoding="utf-8"))
            if entry.get("epoch") != IMAGENETTE_LAST_EPOCH or entry.get("checkpoint_kind", "last") != "last" \
                    or entry.get("probe_select", "train_slice") != "train_slice":
                raise ProtocolError(f"{run.run_name}: post-hoc result is not strict / final-epoch ({entry.get('epoch')})")
            self.summary[run.run_name] = s
            self.used_csv[run.run_name] = C.metrics_versions(root / "logs" / "train" / "runs" / run.run_name / "csv")
            self.used_json[run.run_name] = post

    def cell_runs(self, b: int, lam: float, k: int) -> list[str]:
        out = []
        for run in self.runs:
            o = run.override_map()
            if (int(o["data.batch_size"]), float(o["model.lamb"]), int(o["model.reg_loss.num_clusters"])) == (b, lam, k):
                out.append(run.run_name)
        return out

    def cells(self) -> dict[tuple[int, float, int], list[str]]:
        keyed: dict[tuple[int, float, int], list[str]] = {}
        for run in self.runs:
            o = run.override_map()
            keyed.setdefault((int(o["data.batch_size"]), float(o["model.lamb"]), int(o["model.reg_loss.num_clusters"])),
                             []).append(run.run_name)
        return keyed

    def sources(self, run: str, ruler: str) -> list[str]:
        if ruler == "linear":
            return [f"eval_ablation_ep800/{run}.json"]
        return [f"online_ep800/{run}/version_{i}.csv" for i in range(len(self.used_csv[run]))]

    def value(self, run: str, ruler: str) -> float:
        s = self.summary[run]
        return {"online": s["final_probe_acc"], "linear": s["posthoc_linear_top1"], "khat": s["realised_k_last10"]}[ruler]

    def archive(self, raw: Path) -> int:
        n = 0
        for run, files in self.used_csv.items():
            for i, path in enumerate(files):
                target = raw / "online_ep800" / run / f"version_{i}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                with open(path, newline="", encoding="utf-8") as src, open(target, "w", newline="", encoding="utf-8") as dst:
                    import csv as _csv
                    reader = _csv.DictReader(src)
                    writer = _csv.writer(dst)
                    writer.writerow(ONLINE_COLUMNS)
                    for row in reader:
                        if any(row.get(c) not in ("", None) for c in ONLINE_COLUMNS[1:]):
                            writer.writerow([row.get(c, "") for c in ONLINE_COLUMNS])
                n += 1
        for run, path in self.used_json.items():
            target = raw / "eval_ablation_ep800" / f"{run}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            n += 1
        return n


def build_k_axis(im: Imagenette) -> tuple[str, dict]:
    """``tab:k-ablation-online`` at B=256x1, lambda 0.1, K = 1..32: online final accuracy, mean +/- SD over
    three seeds, the top three marked as the submitted table marked them; the realised K beneath."""
    name = "tab_k_ablation_256"
    cells: dict[str, dict] = {}
    stats = {}
    for k in K_AXIS:
        runs = im.cell_runs(256, 0.1, k)
        if not runs or any(r not in im.summary for r in runs):
            stats[k] = None
            continue
        acc = [im.value(r, "online") for r in runs]
        khat = [im.value(r, "khat") for r in runs]
        stats[k] = (runs, acc, khat)
    shown = {k: round(float(f"{statistics.fmean(v[1]):.2f}"), 2) for k, v in stats.items() if v}
    ranked = sorted(set(shown.values()), reverse=True)
    medal = {k: MEDALS[ranked.index(v)] for k, v in shown.items() if ranked.index(v) < len(MEDALS)}
    acc_row, khat_row = ["0.1"], [r"$\hat K$"]
    for j, k in enumerate(K_AXIS):
        if stats[k] is None:
            acc_row.append("--")
            khat_row.append("--")
            continue
        runs, acc, khat = stats[k]
        mean, sd = statistics.fmean(acc), statistics.stdev(acc)
        body = rf"\makecell{{${mean:.2f} \pm {sd:.2f}$}}"
        acc_row.append((rf"\cellcolor{{{medal[k]}}}" if k in medal else "") + body)
        khat_row.append(f"${statistics.fmean(khat):.1f}$")
        for row_key, ruler, values, text, digits in (("acc", "online", acc, acc_row[-1], 2), ("khat", "khat", khat, khat_row[-1], 1)):
            cells[f"{name}/k{k}/{row_key}"] = {
                "runs": runs, "sources": [s for r in runs for s in im.sources(r, "online")], "kind": f"imagenette_{ruler}",
                "values": values, "mean": statistics.fmean(values), "sd": statistics.stdev(values) if ruler == "online" else None,
                "row": f"% row:{name}/{row_key}", "rendered": text, "medal": medal.get(k) if ruler == "online" else None,
                "latex_col": j, "part": None, "digits": digits}
    marker_acc, marker_khat = f"% row:{name}/acc", f"% row:{name}/khat"
    tex = [r"\begin{table}[htbp]", r"\revstart", r"\centering",
           r"\caption{\textit{Neighborhood-count ablation on Imagenette, 800 epochs, $B=256\times1$, $\lambda=0.1$.} "
           r"Final-epoch online linear-probe accuracy (\%): mean $\pm$ SD over three pretraining seeds; $\hat K$ is the "
           r"realized number of neighborhoods (mean of the last ten epochs, averaged over seeds).}",
           r"\label{tab:k-ablation-online}", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{l" + "c" * len(K_AXIS) + "}", r"\toprule",
           r"$\lambda$ & " + " & ".join(f"$K={k}$" for k in K_AXIS) + r" \\", r"\midrule",
           " & ".join(acc_row) + r" \\ " + marker_acc, " & ".join(khat_row) + r" \\ " + marker_khat,
           r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(tex), {"table": name, "cells": cells, "bold_rule": None, "medal_rule": "top3_by_2dp_mean",
                            "row_labels": {marker_acc: "0.1", marker_khat: r"$\hat K$"},
                            "pending_cells": sum(1 for v in stats.values() if v is None) * 2}


def build_bands(im: Imagenette) -> tuple[str, dict]:
    """The effective-strength law: every fixk cell (three seeds) grouped by lambda*N/K; per band the number
    of cells and the min / median / max of the cell means, on the online and the post-hoc linear ruler."""
    name = "tab_strength_bands"
    keyed = im.cells()
    complete = {key: runs for key, runs in keyed.items() if all(r in im.summary for r in runs)}
    cells: dict[str, dict] = {}
    rows, labels = [], {}
    for band, label, member in BANDS:
        marker = f"% row:{name}/{band}"
        labels[marker] = label
        members = sorted(key for key in complete if member(round(key[1] * key[0] / key[2], 9)))
        parts = [label, band]
        if not members:
            parts += ["--"] * 7
            rows.append(" & ".join(parts) + r" \\ " + marker)
            continue
        parts.append(str(len(members)))
        cells[f"{name}/{band}/cells"] = {"kind": "band_count", "members": [complete[m] for m in members],
                                         "value": len(members), "row": marker, "latex_col": 1, "part": None,
                                         "sources": [], "rendered": str(len(members))}
        for j, ruler in enumerate(("online", "linear")):
            means = [statistics.fmean(im.value(r, ruler) for r in complete[m]) for m in members]
            for i, (stat, value) in enumerate((("min", min(means)), ("median", statistics.median(means)), ("max", max(means)))):
                text = f"${value:.2f}$"
                parts.append(text)
                cells[f"{name}/{band}/{ruler}/{stat}"] = {
                    "kind": "band_stat", "ruler": ruler, "stat": stat, "members": [complete[m] for m in members],
                    "value": value, "row": marker, "latex_col": 2 + 3 * j + i, "part": None, "rendered": text,
                    "sources": sorted({s for m in members for r in complete[m] for s in im.sources(r, ruler)})}
        rows.append(" & ".join(parts) + r" \\ " + marker)
    tex = [r"\begin{table}[htbp]", r"\revstart", r"\centering",
           r"\caption{\textit{Effective regularization strength orders the Imagenette grid (800 epochs, both per-step "
           r"batches).} Each cell of the $(B, \lambda, K)$ grid (three pretraining seeds) is placed by its effective "
           r"strength $\lambda N/K$, with $N$ the per-step batch; per band, the number of cells and the minimum / median "
           r"/ maximum of the cell means (\%), on the final-epoch online probe and on the post-hoc linear probe.}",
           r"\label{tab:strength-bands}", r"\small", r"\setlength{\tabcolsep}{3pt}",
           r"\resizebox{\linewidth}{!}{\begin{tabular}{l l c ccc ccc}", r"\toprule",
           r" & & & \multicolumn{3}{c}{Online probe (\%)} & \multicolumn{3}{c}{Linear probe (\%)} \\",
           r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
           r"$\lambda N/K$ & regime & cells & min & median & max & min & median & max \\", r"\midrule",
           *rows, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(tex), {"table": name, "cells": cells, "bold_rule": None, "row_labels": labels,
                            "pending_cells": 0 if len(complete) == len(keyed) else len(keyed) - len(complete)}


# ---------------------------------------------------------------------------------------------------
# the numbers the prose quotes: differences of cell means, each with its definition
# ---------------------------------------------------------------------------------------------------

DIFFS = [  # (key, minuend arm, subtrahend arm, families, definition)
    ("recipe", "ours", "lejepa", ("vit", "hvit", "d4rn50"),
     "LeJEPA (our settings) minus LeJEPA: the optimization settings alone"),
    ("k4_over_lejepa", "k4", "lejepa", ("vit", "hvit", "d4rn50", "vitl14"), "LocaLeJEPA K=4 minus LeJEPA"),
    ("k4_over_ours", "k4", "ours", ("vit", "hvit", "d4rn50"),
     "LocaLeJEPA K=4 minus LeJEPA (our settings): the method at matched settings"),
    ("k1_over_ours", "k1", "ours", ("vit", "hvit", "d4rn50"),
     "LocaLeJEPA K=1 minus LeJEPA (our settings): the centred ball target with one neighbourhood"),
    ("k1_over_lejepa", "k1", "lejepa", ("vitl14",), "ViT-L: LocaLeJEPA K=1 minus LeJEPA"),
    ("locality", "k4", "k1", ("vit", "hvit", "d4rn50", "vitl14"), "LocaLeJEPA K=4 minus K=1 at the same lambda"),
    ("grouping", "k4", "k1_quarter", ("vit", "hvit", "d4rn50"),
     "LocaLeJEPA K=4 (lambda 0.1) minus K=1 at lambda 0.025: locality at matched effective strength"),
    ("dynk_over_ours", "dynk", "ours", ("hvit", "d4rn50"), "dynamic K (1->10) minus LeJEPA (our settings)"),
    ("dynk_vs_k4", "dynk", "k4", ("hvit", "d4rn50"), "dynamic K (1->10) minus LocaLeJEPA K=4"),
    ("k4_over_localsigreg", "k4", "localsigreg", ("hvit", "d4rn50"),
     "LocaLeJEPA K=4 minus LocaLeJEPA-SIGReg (same neighbourhoods; the centred ball target vs the uncentred Gaussian)"),
    ("k2_vs_k4", "k2", "k4", ("hvit", "d4rn50"), "K=2 minus K=4, both at lambda 0.1"),
    ("k2l03_vs_k2", "k2_l03", "k2", ("hvit", "d4rn50"), "K=2 at lambda 0.3 minus K=2 at lambda 0.1"),
    ("lejepa_over_rdmreg", "lejepa", "rdmreg", ("vit", "hvit", "d4rn50"),
     "LeJEPA minus RDMReg, both at LeJEPA's published settings (ViT-S: LeJEPA over 3 seeds, RDMReg seed 0)"),
]
RULERS = {"knn20": ("indomain", "knn.20.top1"), "linear": ("indomain", "linear.top1"), "transfer": ("transfer", "avg")}


def text_numbers(src: Source) -> dict:
    out: dict[str, dict] = {}
    for key, a, b, families, definition in DIFFS:
        for family in families:
            if family == "vitl14" and (a not in VITL_ARMS or b not in VITL_ARMS):
                continue
            for ruler, (kind, field) in RULERS.items():
                means = []
                for arm in (a, b):
                    values = []
                    for run in run_names(family, arm):
                        entry, _ = src.load(kind, family, run)
                        if entry is None:
                            values = None
                            break
                        values.append(field_value(entry, field))
                    means.append(statistics.fmean(values) if values else None)
                if None in means:
                    continue
                value = means[0] - means[1]
                n = len(run_names(family, a))
                out[f"{key}/{family}/{ruler}"] = {
                    "definition": definition + f"; {ruler}; " + ("mean over 3 pretraining seeds of each arm"
                                                                 if n == 3 else "single runs"),
                    "value": value, "rendered_1dp": f"{value:+.1f}", "rendered_2dp": f"{value:+.2f}",
                    "minuend": run_names(family, a), "subtrahend": run_names(family, b)}
                if n == 3 and len(run_names(family, b)) == 3:
                    per_seed = [field_value(src.load(kind, family, ra)[0], field) - field_value(src.load(kind, family, rb)[0], field)
                                for ra, rb in zip(run_names(family, a), run_names(family, b))]
                    out[f"{key}_seedmin/{family}/{ruler}"] = {
                        "definition": definition + f"; {ruler}; the SMALLEST of the three per-seed differences (seed s minus seed s)",
                        "value": min(per_seed), "per_seed": per_seed, "rendered_1dp": f"{min(per_seed):+.1f}",
                        "rendered_2dp": f"{min(per_seed):+.2f}"}
    out.update(depth_numbers(src))
    out.update(knn_k_numbers(src))
    out.update(transfer_detail_numbers(src))
    return out


def _value_entry(definition: str, value: float, signed: bool) -> dict:
    fmt1, fmt2 = ("{:+.1f}", "{:+.2f}") if signed else ("{:.1f}", "{:.2f}")
    return {"definition": definition, "value": value, "rendered_1dp": fmt1.format(value), "rendered_2dp": fmt2.format(value)}


def depth_numbers(src: Source) -> dict:
    """The depth study's steps, seed 0 at every depth (depth 12 included, so the three depths compare like for like):
    grouping = K=4 minus K=1 at lambda/4; strength = K=1 at lambda/4 minus K=1; locality = K=4 minus K=1."""
    out = {}
    for depth in DEPTHS:
        got = {}
        for arm, (_, runs) in DEPTH_ARMS.items():
            entry, _ = src.load("indomain", "vit", runs[depth])
            got[arm] = entry
        if None in got.values():
            continue
        for ruler, field in (("knn20", "knn.20.top1"), ("linear", "linear.top1")):
            v = {arm: field_value(e, field) for arm, e in got.items()}
            for step, value in (("grouping", v["k4"] - v["k1_quarter"]), ("strength", v["k1_quarter"] - v["k1"]),
                                ("locality", v["k4"] - v["k1"])):
                out[f"depth_{step}/d{depth}/{ruler}"] = _value_entry(
                    f"ViT-S/8 at depth {depth}, seed 0: the {step} step; {ruler}", value, signed=True)
    return out


def knn_k_numbers(src: Source) -> dict:
    """App C's claim about the k of k-NN, over every row of tab:in1k-indomain-full: the largest gap between the best k
    and k=20 (row means), and whether the ordering of a family's rows is the same at every k."""
    out, gaps, same = {}, [], {}
    ks = (10, 20, 100, 200)
    for family, arms in FULL_ROWS.items():
        means = {}
        for arm in arms:
            vals = {k: [] for k in ks}
            for run in run_names(family, arm):
                entry, _ = src.load("indomain", family, run)
                if entry is None:
                    vals = None
                    break
                for k in ks:
                    vals[k].append(field_value(entry, f"knn.{k}.top1"))
            if vals:
                means[arm] = {k: statistics.fmean(v) for k, v in vals.items()}
        for arm, m in means.items():
            gaps.append(max(m.values()) - m[20])
        if len(means) > 1:
            orders = {tuple(sorted(means, key=lambda a: -means[a][k])) for k in ks}
            same[family] = len(orders) == 1
    if gaps:
        out["knn_k20_maxgap/all/knn"] = _value_entry("over every row of tab:in1k-indomain-full with data: the largest "
                                                     "(best-k minus k=20) k-NN top-1 gap", max(gaps), signed=False)
    for family, ok in same.items():
        out[f"knn_order_same/{family}/knn"] = {"definition": f"{family}: the rows' k-NN ordering is identical at k = 10, 20, "
                                               "100 and 200 (1 = yes)", "value": float(ok), "rendered_1dp": str(int(ok)),
                                               "rendered_2dp": str(int(ok))}
    return out


def transfer_detail_numbers(src: Source) -> dict:
    """Per dataset, LocaLeJEPA K=4 minus each LeJEPA arm; and how many of the seven it wins (mean > mean)."""
    out = {}
    for family in ("vit", "hvit", "d4rn50", "vitl14"):
        for vs in ("lejepa", "ours"):
            if family == "vitl14" and vs == "ours":
                continue
            per = {}
            for arm in ("k4", vs):
                rows = []
                for run in run_names(family, arm):
                    entry, _ = src.load("transfer", family, run)
                    if entry is None:
                        rows = None
                        break
                    rows.append(entry)
                per[arm] = rows
            if not per["k4"] or not per[vs]:
                continue
            wins = 0
            for d in DATASETS:
                diff = (statistics.fmean(field_value(e, f"datasets.{d}.top1") for e in per["k4"])
                        - statistics.fmean(field_value(e, f"datasets.{d}.top1") for e in per[vs]))
                wins += diff > 0
                out[f"transfer_k4_vs_{vs}/{family}/{d}"] = _value_entry(
                    f"{family}: LocaLeJEPA K=4 minus {vs} on {d} (top-1)", diff, signed=True)
            out[f"transfer_wins_vs_{vs}/{family}/count"] = {
                "definition": f"{family}: datasets (of 7) where LocaLeJEPA K=4's mean beats {vs}'s", "value": float(wins),
                "rendered_1dp": str(wins), "rendered_2dp": str(wins)}
    return out


def vitl_matched_numbers(root: Path, src: Source) -> dict:
    """ViT-L/14's strength-matched control (K=1 at lambda 0.0125) -- not a table row, it trained unstably: its final strict
    k-NN / linear, and every ViT-L arm's online probe at epoch index 5 (the sixth epoch) and at the last epoch."""
    out = {}
    entry, _ = src.load("indomain", "vitl14", "vitl14_strmatch_lam0.0125_k1_s0")
    if entry is not None:
        for ruler, field in (("knn20", "knn.20.top1"), ("linear", "linear.top1")):
            out[f"vitl14_matched/value/{ruler}"] = _value_entry(
                f"ViT-L/14, K=1 at lambda 0.0125 (strength-matched to K=4): final strict {ruler}", field_value(entry, field),
                signed=False)
    for name, tag in (("vitl14_strmatch_lam0.0125_k1_s0", "matched"), ("vitl14_sigreg_s0", "lejepa"),
                      ("vitl14_globalball_k1_s0", "k1"), ("vitl14_local_k4_s0", "k4")):
        csv_dir = root / "logs" / "train" / "runs" / name / "csv"
        if not C.metrics_versions(csv_dir):
            continue
        _, rows = C.merged_metrics_rows(csv_dir)
        acc = {}
        for row in rows:
            if row.get("val/probe_acc") not in ("", None) and row.get("epoch") not in ("", None):
                acc[int(float(row["epoch"]))] = 100.0 * float(row["val/probe_acc"])
        if 5 in acc:
            out[f"vitl14_online/{tag}/ep5"] = _value_entry(f"ViT-L/14 {tag}: online probe top-1 (%) at epoch index 5",
                                                           acc[5], signed=False)
        if acc:
            out[f"vitl14_online/{tag}/last"] = _value_entry(f"ViT-L/14 {tag}: online probe top-1 (%) at its last epoch "
                                                            f"({max(acc)})", acc[max(acc)], signed=False)
    return out


def imagenette_tie_numbers(im: "Imagenette") -> dict:
    """With lambda chosen per K (the best cell mean over lambda), the best K>1 minus K=1, per batch and ruler."""
    out = {}
    keyed = im.cells()
    for b in (64, 256):
        for ruler in ("online", "linear"):
            best = {}
            for (bb, lam, k), runs in keyed.items():
                if bb != b or not all(r in im.summary for r in runs):
                    continue
                m = statistics.fmean(im.value(r, ruler) for r in runs)
                best[k] = max(best.get(k, -1.0), m)
            if 1 in best and len(best) > 1:
                top_k = max((k for k in best if k > 1), key=lambda k: best[k])
                out[f"imagenette_bestk_minus_k1/b{b}/{ruler}"] = _value_entry(
                    f"Imagenette 800 ep, B={b}: with lambda chosen per K, the best K>1 (K={top_k}) minus K=1; {ruler}",
                    best[top_k] - best[1], signed=True)
                for k, v in sorted(best.items()):
                    out[f"imagenette_bestlam/b{b}k{k}/{ruler}"] = _value_entry(
                        f"Imagenette 800 ep, B={b}, K={k}: best cell mean over lambda; {ruler}", v, signed=False)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, type=Path, help="a directory laid out like $SCR (or an unpacked bundle)")
    parser.add_argument("--out", default="results/rebuttal", type=Path)
    parser.add_argument("--raw", default="results/raw", type=Path, help="where every JSON a table reads is archived")
    parser.add_argument("--require-complete", action="store_true",
                        help="refuse to write if any cell would render as --")
    args = parser.parse_args(argv)

    src = Source(args.root, manifest_run_names())
    outputs = {name: build_table(name, spec, src) for name, spec in TABLES.items()}
    outputs["tab_depth"] = build_depth(src)
    numbers = text_numbers(src)
    numbers.update(vitl_matched_numbers(args.root, src))
    im = Imagenette(args.root) if (args.root / "logs" / "train" / "runs").is_dir() else None
    if im is not None:
        outputs["tab_k_ablation_256"] = build_k_axis(im)
        outputs["tab_strength_bands"] = build_bands(im)
        numbers.update(imagenette_tie_numbers(im))
    else:
        print(f"Imagenette tables SKIPPED: no logs/train/runs under {args.root}")

    if args.require_complete:
        blanks = {n: p["pending_cells"] for n, (_, p) in outputs.items() if p["pending_cells"]}
        if blanks or im is None:
            raise SystemExit(f"REFUSED (--require-complete): pending cells {blanks}; pending runs {src.pending}"
                             + ("; the Imagenette tables were skipped" if im is None else ""))

    args.out.mkdir(parents=True, exist_ok=True)
    for rel, path in sorted(src.used.items()):
        target = args.raw / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    n_imagenette = im.archive(args.raw) if im is not None else 0
    for name, (tex, payload) in outputs.items():
        with atomic_write(args.out / f"{name}.tex", "w", encoding="utf-8") as f:
            f.write(tex)
        with atomic_write(args.out / f"{name}.cells.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, sort_keys=True)
    with atomic_write(args.out / "text_numbers.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=1, sort_keys=True)
    dirs = sorted({str(args.root / d) for d in EVAL_DIRS.values()} | {str(args.root / IMAGENETTE_POSTHOC)})
    with atomic_write(args.out / "PROVENANCE.md", "w", encoding="utf-8") as f:
        f.write("# Rebuttal tables -- provenance\n" + provenance_block(os.path.relpath(__file__, REPO), dirs, archive_to=None)
                + f"\nArchived to `{args.raw}` (only the files the tables read): {len(src.used)} ImageNet-1K JSON(s); "
                  f"{n_imagenette} Imagenette file(s) (online-probe extracts of every launch + post-hoc JSONs).\n")
    n_cells = sum(len(p["cells"]) for _, p in outputs.values())
    print(f"wrote {len(outputs)} tables ({n_cells} cells) and {len(numbers)} text numbers to {args.out}; "
          f"archived {len(src.used)} + {n_imagenette} input file(s) to {args.raw}")
    pending = {n: p["pending_cells"] for n, (_, p) in outputs.items() if p["pending_cells"]}
    if pending:
        print(f"pending (rendered --): {pending}")
        for run, why in sorted(src.pending.items()):
            print(f"  {run}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
