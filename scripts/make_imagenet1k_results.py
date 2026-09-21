#!/usr/bin/env python
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The paper's main ImageNet-1K table, from the campaign's in-domain evaluations.

Rows are backbones, columns are methods (campaign design, state/05-DECISIONS.md):

    | backbone | LeJEPA published | LeJEPA reproduced | RDMReg | LocaLeJEPA | control |

Every cell is "k-NN (k=20) / Linear" top-1 on the 50k validation split from the run's
``last.ckpt`` -- no selection on the reported split (PD-4). The online probe's ``best.ckpt`` is
a second, labelled table, and the ViT-L/14 pair under LeJeJEPA's probe protocol (cls2 features,
LayerNorm -> Linear, AdamW) a third. Which run fills which cell comes from the campaign
manifests (``scripts/aicr/campaign.py``), so a renamed run cannot be silently dropped (F35);
the published LeJEPA number is a constant with its citation, never in a comparison column.

Alongside the markdown this writes ``results/Imagenet1K_results.cells.json`` -- every rendered
number with the archived file it came from -- which ``scripts/audit_tables.py`` re-derives
independently (research_rules/06: the gate checks both directions).

    python scripts/make_imagenet1k_results.py [--root <eval run root>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eval.io import _eval_name, atomic_write, provenance_block, refuse_to_replace_with_placeholders  # noqa: E402
import scripts.aicr.campaign as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = "results/Imagenet1K_results.md"
CELLS_OUT = "results/Imagenet1K_results.cells.json"
TEX_OUT = "paper/imagenet1k_results.tex"
MAX_EPOCHS = 100
REPORT_K = 20

# eval output folders (relative to the eval run root), from the rendered campaign configs
OURS_128 = "logs/eval_indomain_in1k_campaign/per_backbone"
OURS_VITL = "logs/eval_indomain_in1k_vitl14_campaign/per_backbone"
LEJEPA_PROTOCOL_VITL = "logs/eval_indomain_in1k_vitl14_lejepaprobe/per_backbone"
INPUT_DIRS = [OURS_128, OURS_VITL, LEJEPA_PROTOCOL_VITL]

FAMILIES = [
    ("vit", "ViT-S/8 (128 px, 4 views)"),
    ("vitl14", "ViT-L/14 (238 px, 2+6 views, LeJEPA's settings, our protocol)"),
    ("d4rn50", "D4-equivariant ResNet-50 (128 px)"),
    ("hvit", "Hyperbolic ViT-S/8 (128 px)"),
]
ARMS = [("lejepa", "LeJEPA (SIGReg)"), ("rdmreg", "Rectified LpJEPA (RDMReg)"),
        ("local", "LocaLeJEPA (declared config)"), ("control", "control: SIGReg, local recipe")]
ARM_TAG = {"lejepa": "sigreg", "rdmreg": "rdmreg", "local": "local", "control": "control"}

# LeJEPA (Balestriero & LeCun, 2025) Table 1, ViT-L/14, 100 epochs ImageNet-1K, linear probe --
# THEIR training setup (EMA teacher, patch masking, 2x238 + 6x98 views, batch 512). Cited, not
# compared (campaign decision 13). k-NN is not reported in the paper.
PUBLISHED = {"vitl14": {"linear": 74.24, "knn": None, "note": "LeJEPA Table 1, their setup (EMA + masking)"}}

MANIFESTS = ["in1k_baselines.yaml", "in1k_vitl14.yaml", "in1k_local.yaml"]


def campaign_runs() -> dict[tuple[str, str], C.RunSpec]:
    """(family, arm) -> the manifest run that fills that cell; exactly one or none."""
    cells: dict[tuple[str, str], C.RunSpec] = {}
    for manifest in MANIFESTS:
        for run in C.expand(C.load_manifest(REPO / "configs" / "campaign" / manifest)):
            family = next((f for f, _ in FAMILIES if run.tag == f or run.tag.startswith(f + "_")), None)
            arm = next((a for a, t in ARM_TAG.items() if run.tag.endswith("_" + t) or run.tag == t), None)
            if family is None or arm is None:
                continue
            key = (family, arm)
            if key in cells:
                raise SystemExit(f"two manifest runs fill the same cell {key}: {cells[key].run_name}, {run.run_name}")
            cells[key] = run
    return cells


def _load(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _ckpt_epoch(path):
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("epoch")
    except Exception:  # noqa: BLE001
        return None


def _cell(entry, k=REPORT_K):
    # A missing k renders as an em dash. It must NEVER fall back to another k: the previous
    # `.get(k) or next(iter(...))` reported the SMALLEST available k (the dict is built in
    # ascending order) under the requested k's heading, with no marker. Ledger T.suite-P2-7.
    knn = entry["knn"].get(str(k))
    knn_top1 = f'{knn["top1"]:.1f}' if knn else "—"
    knn_top5 = f'{knn["top5"]:.1f}' if knn else "—"
    return knn_top1, f'{entry["linear"]["top1"]:.1f}', knn_top5, f'{entry["linear"]["top5"]:.1f}'


def _progress(run: C.RunSpec, root: Path) -> str:
    last_ep = _ckpt_epoch(root / "checkpoints" / run.run_name / "last.ckpt")
    if last_ep is not None and last_ep >= MAX_EPOCHS - 1:
        return "*finished, eval pending*"
    return f"*pretraining, epoch {last_ep + 1 if last_ep is not None else '?'}/{MAX_EPOCHS}*"


def render(root: Path) -> tuple[str, dict[str, dict], str]:
    """(markdown body, cells.json payload, latex) from the eval outputs under ``root``."""
    runs = campaign_runs()
    cells: dict[str, dict] = {}
    n_done = 0

    def table(kind: str, suffix: str, eval_dir_of) -> list[str]:
        nonlocal n_done
        cols = [label for _, label in ARMS]
        lines = ["| Backbone | LeJEPA published | " + " | ".join(cols) + " |",
                 "|---|---|" + "---|" * len(cols)]
        for family, title in FAMILIES:
            row = [title]
            pub = PUBLISHED.get(family)
            row.append(f'— / {pub["linear"]:.2f}†' if pub and kind == "last" else "—")
            for arm, _ in ARMS:
                run = runs.get((family, arm))
                if run is None:
                    row.append("n/a")
                    continue
                eval_dir = eval_dir_of(family)
                path = root / eval_dir / f"{run.run_name}{suffix}.json"
                entry = _load(path)
                if entry is None:
                    row.append(_progress(run, root) if kind == "last" else "—")
                    continue
                k1, l1, _, _ = _cell(entry)
                if kind == "last":
                    n_done += 1
                row.append(f"{k1} / {l1}")
                cells[f"{family}/{arm}/{kind}"] = {
                    "run": run.run_name, "knn20": entry["knn"].get(str(REPORT_K), {}).get("top1"),
                    "linear": entry["linear"]["top1"], "epoch": entry.get("epoch"),
                    # where provenance_block archives it: results/raw/<eval name>/<file>
                    "source": f"{_eval_name(eval_dir)}/{path.name}",
                }
            lines.append("| " + " | ".join(row) + " |")
        return lines

    def ours(family):
        return OURS_VITL if family == "vitl14" else OURS_128

    body = ["## Headline: `last.ckpt`, our protocol (k-NN k=20 / Linear, top-1 %)", ""]
    body += table("last", "", ours)
    body += ["", "† LeJEPA's published ViT-L/14 linear result under *their* training setup (EMA teacher,",
             "patch masking); cited for reference, not a comparison (decision 13).", "",
             "## Secondary: the online probe's `best.ckpt` (same protocol)", ""]
    body += table("best", "_best", ours)

    # the LeJEPA-protocol probe on the ViT-L pair
    body += ["", "## ViT-L/14 under LeJEPA's probe protocol (cls2 features, LayerNorm → Linear, AdamW)", "",
             "| Backbone | LeJEPA reproduced | LocaLeJEPA |", "|---|---|---|"]
    row = ["ViT-L/14"]
    for arm in ("lejepa", "local"):
        run = runs.get(("vitl14", arm))
        entry = _load(root / LEJEPA_PROTOCOL_VITL / f"{run.run_name}.json") if run else None
        if entry is None:
            row.append("—")
            continue
        assert entry.get("probe") == "lejepa" and entry.get("feature") == "cls2", (
            f"{run.run_name}: the LeJEPA-protocol folder holds a {entry.get('feature')}/{entry.get('probe')} result")
        _, l1, _, _ = _cell(entry)
        row.append(f"linear {l1}")
        cells[f"vitl14/{arm}/lejepa_protocol"] = {
            "run": run.run_name, "knn20": entry["knn"].get(str(REPORT_K), {}).get("top1"),
            "linear": entry["linear"]["top1"], "epoch": entry.get("epoch"),
            "source": f"{_eval_name(LEJEPA_PROTOCOL_VITL)}/{run.run_name}.json",
        }
    body.append("| " + " | ".join(row) + " |")

    # data-driven gaps, nothing typed by hand
    body += ["", "## Gaps (LocaLeJEPA − LeJEPA reproduced, `last.ckpt`)", ""]
    for family, title in FAMILIES:
        a, b = cells.get(f"{family}/local/last"), cells.get(f"{family}/lejepa/last")
        if a and b and a["knn20"] is not None and b["knn20"] is not None:
            body.append(f"* {title}: k-NN {a['knn20'] - b['knn20']:+.1f}, linear {a['linear'] - b['linear']:+.1f}")
    c, b = cells.get("vit/control/last"), cells.get("vit/lejepa/last")
    if c and b and c["knn20"] is not None and b["knn20"] is not None:
        body.append(f"* recipe alone (control − LeJEPA, ViT-S/8): k-NN {c['knn20'] - b['knn20']:+.1f}, "
                    f"linear {c['linear'] - b['linear']:+.1f}")

    tex = make_tex(cells, runs)
    return "\n".join(body), {"report_k": REPORT_K, "max_epochs": MAX_EPOCHS, "cells": cells,
                             "n_done": n_done, "n_cells": len(runs)}, tex


def make_tex(cells: dict[str, dict], runs) -> str:
    rows_out = []
    for family, title in FAMILIES:
        parts = [title.split(" (")[0]]
        pub = PUBLISHED.get(family)
        parts.append(f"-- / {pub['linear']:.2f}$^\\dagger$" if pub else "--")
        for arm, _ in ARMS:
            cell = cells.get(f"{family}/{arm}/last")
            if cell is None:
                parts.append("--" if (family, arm) in runs else "n/a")
            else:
                knn = f"{cell['knn20']:.1f}" if cell["knn20"] is not None else "--"
                parts.append(f"{knn} / {cell['linear']:.1f}")
        rows_out.append(" & ".join(parts) + r" \\")
    return "\n".join([
        r"%% Auto-generated by scripts/make_imagenet1k_results.py -- do not edit by hand.",
        r"\begin{table}[t]\centering",
        r"\caption{ImageNet-1K, 100 epochs: $k$-NN ($k{=}20$) / linear top-1 (\%) on the validation split,",
        r"frozen features of the final checkpoint. $^\dagger$LeJEPA's published ViT-L/14 result under their",
        r"training setup (EMA teacher, patch masking), cited for reference only.}",
        r"\label{tab:imagenet-classification}",
        r"\small\begin{tabular}{l c c c c c}\toprule",
        r"Backbone & LeJEPA (published) & LeJEPA & RDMReg & LocaLeJEPA & control \\ \midrule",
        *rows_out,
        r"\bottomrule\end{tabular}\end{table}",
    ]) + "\n"


HEADER = """# ImageNet-1K in-domain evaluation of the campaign's 100-epoch pretrains

Auto-generated by `scripts/make_imagenet1k_results.py` from the campaign's in-domain
evaluations ({n_done}/{n_cells} headline cells filled). Rows are backbones, columns are methods;
each cell is **k-NN (k=20) / Linear** top-1 (%) on the 50k validation split.

## Protocol

* **Pretraining**: 100 epochs on ImageNet-1K. The 128 px rows: 4 views, effective batch 256
  (64 per GPU), embedding 512, projector [2048, 2048, 16]. Baselines at LeJEPA's `MINIMAL.md`
  recipe (lr 2e-3, cosine 2e-3 → 1e-3); LocaLeJEPA at its recipe (lr 1e-3, 10 warm-up epochs,
  grad-clip 1.0, cosine to peak/1000, λ and K declared by the Imagenette ablation before the run);
  the **control** is SIGReg under the LocaLeJEPA recipe (ViT-S/8 only). The ViT-L/14 pair uses
  LeJEPA's ImageNet-1K optimizer and loss settings in our protocol (no EMA, no masking) and is
  identical across arms except the regularizer.
* **Frozen features**: the encoder output before the projector; deterministic resize + centre
  crop at the pretraining resolution.
* **k-NN**: full train split as the bank, cosine similarity, weighted vote `exp(sim/0.07)`, k=20.
* **Linear**: one linear layer on standardized features, SGD, 50 epochs, learning rate selected
  on a 2% held-out slice of *train*, refit on the full train split.
* **Checkpoint**: `last.ckpt` (epoch 99) is the headline; the online probe's `best.ckpt` is
  the secondary table; no selection on the reported split.

"""


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="the eval run root the logs/ and checkpoints/ paths resolve against")
    args = parser.parse_args(argv)
    root = Path(args.root)

    body, payload, tex = render(root)
    doc = HEADER.format(n_done=payload["n_done"], n_cells=payload["n_cells"]) + body + "\n"
    input_dirs = [str(root / d) for d in INPUT_DIRS]
    refuse_to_replace_with_placeholders(OUT, input_dirs)
    doc += provenance_block(os.path.relpath(__file__, REPO), input_dirs, archive_to="results/raw")
    with atomic_write(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    with atomic_write(CELLS_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    with atomic_write(TEX_OUT, "w", encoding="utf-8") as f:
        f.write(tex)
    print(body)
    print(f"\n{payload['n_done']}/{payload['n_cells']} headline cells filled -> {OUT}, {CELLS_OUT}, {TEX_OUT}", file=sys.stderr)
    return body


if __name__ == "__main__":
    main()
