#!/usr/bin/env python
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The paper's main ImageNet-1K table, from the campaign's in-domain evaluations.

Rows are backbones, columns are methods (campaign design, state/05-DECISIONS.md):

    | backbone | LeJEPA published | LeJEPA reproduced | RDMReg | LocaLeJEPA | control |

Every headline cell is "k-NN (k=20) / Linear" top-1 on the 50k validation split from the run's
``last.ckpt`` -- no selection on the reported split (PD-4). Beside each arm's headline cell sits
ONE labelled column (decision 16): the linear probe selected on the evaluation split itself (lr x
wd x epoch, best reported) on ``best.ckpt`` -- an upper bound of LeJEPA's published protocol, which
its author described as exactly that selection. Columns are labelled from the DATA: every
per-backbone JSON carries ``probe_select`` and ``checkpoint_kind``, and a result that says
``train_slice`` can never fill an eval-selected cell or vice versa. The online probe's
``best.ckpt`` under the strict probe is a second, labelled table; the ViT-L/14 pair under
LeJEPA's probe protocol (cls2 features, LayerNorm -> Linear, AdamW) a third; the appendix holds
the two narrower readings (the eval-selected probe on ``last.ckpt``, and the online probe's own
curve: final epoch / maximum). Which run fills which cell comes from the campaign
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
# decision 16: the eval-selected probe (upper bound of the published protocol), last + best twins
EVALSEL_128 = "logs/eval_indomain_in1k_evalsel/per_backbone"
EVALSEL_VITL = "logs/eval_indomain_in1k_vitl14_evalsel/per_backbone"
LEJEPA_PROTOCOL_VITL_EVALSEL = "logs/eval_indomain_in1k_vitl14_lejepaprobe_evalsel/per_backbone"
INPUT_DIRS = [OURS_128, OURS_VITL, LEJEPA_PROTOCOL_VITL, EVALSEL_128, EVALSEL_VITL, LEJEPA_PROTOCOL_VITL_EVALSEL]
ONLINE_RAW = "online"                     # results/raw/online/<run>/version_k.csv: the online probe curves
TEX_ARM = {"lejepa": "LeJEPA", "rdmreg": "RDMReg", "local": "LocaLeJEPA", "control": "control"}

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


def _check_protocol(entry: dict, run_name: str, *, select: str, kind: str) -> None:
    """A column is labelled from the data (decision 16): the JSON must say what the cell claims.

    ``probe_select`` defaults to ``train_slice`` when absent (results written before decision 16
    were all strict); ``checkpoint_kind`` is checked when present. A strict result in an
    eval-selected cell, or the reverse, is a hard stop, not a footnote."""
    got_select = entry.get("probe_select", "train_slice")
    assert got_select == select, f"{run_name}: a {got_select!r}-selected result cannot fill a {select!r} cell"
    got_kind = entry.get("checkpoint_kind")
    assert got_kind in (None, kind), f"{run_name}: checkpoint_kind {got_kind!r} in a {kind!r} column"


def _online_curve(run: C.RunSpec, root: Path) -> tuple[dict | None, Path | None]:
    """(curve, csv dir) of the run's online probe from ``logs/train/runs/<run>/csv``, or (None, None)."""
    csv_dir = root / "logs" / "train" / "runs" / run.run_name / "csv"
    if not csv_dir.is_dir() or not any(csv_dir.glob("version_*/metrics.csv")):
        return None, None
    curve = C.online_probe_curve(csv_dir)
    return (curve if curve["epochs"] else None), csv_dir


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

    def record(key: str, run: C.RunSpec, title: str, entry: dict, eval_dir: str, filename: str) -> None:
        cells[key] = {
            "run": run.run_name, "row": f"| {title} |",
            "knn20": entry["knn"].get(str(REPORT_K), {}).get("top1"),
            "linear": entry["linear"]["top1"], "epoch": entry.get("epoch"),
            "probe_select": entry.get("probe_select", "train_slice"),
            "best_lr": entry["linear"].get("best_lr"), "best_weight_decay": entry["linear"].get("best_weight_decay"),
            "best_epoch": entry["linear"].get("best_epoch"),
            # where provenance_block archives it: results/raw/<eval name>/<file>
            "source": f"{_eval_name(eval_dir)}/{filename}",
        }

    def table(kind: str, suffix: str, eval_dir_of, *, evalsel_dir_of=None) -> list[str]:
        """The strict cells of ``kind`` (last/best); with ``evalsel_dir_of`` each arm also gets the
        eval-selected column (decision 16), read from ``<run>_best.json`` of that folder."""
        nonlocal n_done
        cols = []
        for _, label in ARMS:
            cols.append(f"{label}: strict" if evalsel_dir_of else label)
            if evalsel_dir_of:
                cols.append(f"{label}: eval-sel.‡")
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
                    if evalsel_dir_of:
                        row.append("n/a")
                    continue
                eval_dir = eval_dir_of(family)
                path = root / eval_dir / f"{run.run_name}{suffix}.json"
                entry = _load(path)
                if entry is None:
                    row.append(_progress(run, root) if kind == "last" else "—")
                else:
                    _check_protocol(entry, run.run_name, select="train_slice", kind=kind)
                    k1, l1, _, _ = _cell(entry)
                    if kind == "last":
                        n_done += 1
                    row.append(f"{k1} / {l1}")
                    record(f"{family}/{arm}/{kind}", run, title, entry, eval_dir, path.name)
                if evalsel_dir_of:
                    sel_dir = evalsel_dir_of(family)
                    sel_path = root / sel_dir / f"{run.run_name}_best.json"
                    sel = _load(sel_path)
                    if sel is None:
                        row.append("—")
                    else:
                        _check_protocol(sel, run.run_name, select="eval", kind="best")
                        row.append(f'{sel["linear"]["top1"]:.1f}')
                        record(f"{family}/{arm}/evalsel_best", run, title, sel, sel_dir, sel_path.name)
            lines.append("| " + " | ".join(row) + " |")
        return lines

    def ours(family):
        return OURS_VITL if family == "vitl14" else OURS_128

    def evalsel(family):
        return EVALSEL_VITL if family == "vitl14" else EVALSEL_128

    body = ["## Headline: `last.ckpt`, our protocol (k-NN k=20 / Linear, top-1 %) — with the eval-selected column‡", ""]
    body += table("last", "", ours, evalsel_dir_of=evalsel)
    body += ["", "† LeJEPA's published ViT-L/14 linear result under *their* training setup (EMA teacher,",
             "patch masking); cited for reference, not a comparison (decision 13).",
             "‡ eval-selected (decision 16): the linear probe's learning rate, weight decay and epoch chosen on the",
             "validation split itself, best reported, on `best.ckpt` — an upper bound of the published protocol.",
             "Selection on the reported split: never a headline number.", "",
             "## Secondary: the online probe's `best.ckpt` (same strict protocol)", ""]
    body += table("best", "_best", ours)

    # the LeJEPA-protocol probe on the ViT-L pair: strict selection, then eval-selected on last / best
    body += ["", "## ViT-L/14 under LeJEPA's probe protocol (cls2 features, LayerNorm → Linear, AdamW)", "",
             "| Backbone | LeJEPA reproduced: strict | LocaLeJEPA: strict | LeJEPA reproduced: eval-sel. last / best‡ | LocaLeJEPA: eval-sel. last / best‡ |",
             "|---|---|---|---|---|"]
    row = ["ViT-L/14"]
    for arm in ("lejepa", "local"):
        run = runs.get(("vitl14", arm))
        entry = _load(root / LEJEPA_PROTOCOL_VITL / f"{run.run_name}.json") if run else None
        if entry is None:
            row.append("—")
            continue
        assert entry.get("probe") == "lejepa" and entry.get("feature") == "cls2", (
            f"{run.run_name}: the LeJEPA-protocol folder holds a {entry.get('feature')}/{entry.get('probe')} result")
        _check_protocol(entry, run.run_name, select="train_slice", kind="last")
        _, l1, _, _ = _cell(entry)
        row.append(f"linear {l1}")
        record(f"vitl14/{arm}/lejepa_protocol", run, "ViT-L/14", entry, LEJEPA_PROTOCOL_VITL, f"{run.run_name}.json")
    for arm in ("lejepa", "local"):
        run = runs.get(("vitl14", arm))
        parts = []
        for kind, suffix in (("last", ""), ("best", "_best")):
            sel = _load(root / LEJEPA_PROTOCOL_VITL_EVALSEL / f"{run.run_name}{suffix}.json") if run else None
            if sel is None:
                parts.append("—")
                continue
            assert sel.get("probe") == "lejepa" and sel.get("feature") == "cls2", (
                f"{run.run_name}: the LeJEPA-protocol eval-selected folder holds a {sel.get('feature')}/{sel.get('probe')} result")
            _check_protocol(sel, run.run_name, select="eval", kind=kind)
            _, l1, _, _ = _cell(sel)
            parts.append(f"linear {l1}")
            record(f"vitl14/{arm}/lejepa_protocol_evalsel_{kind}", run, "ViT-L/14", sel,
                   LEJEPA_PROTOCOL_VITL_EVALSEL, f"{run.run_name}{suffix}.json")
        row.append(" / ".join(parts))
    body.append("| " + " | ".join(row) + " |")

    # data-driven gaps, nothing typed by hand
    body += ["", "## Gaps (LocaLeJEPA − LeJEPA reproduced)", ""]
    for family, title in FAMILIES:
        a, b = cells.get(f"{family}/local/last"), cells.get(f"{family}/lejepa/last")
        if a and b and a["knn20"] is not None and b["knn20"] is not None:
            body.append(f"* {title}, `last.ckpt`: k-NN {a['knn20'] - b['knn20']:+.1f}, linear {a['linear'] - b['linear']:+.1f}")
        a, b = cells.get(f"{family}/local/evalsel_best"), cells.get(f"{family}/lejepa/evalsel_best")
        if a and b:
            body.append(f"* {title}, eval-selected‡: linear {a['linear'] - b['linear']:+.1f}")
    c, b = cells.get("vit/control/last"), cells.get("vit/lejepa/last")
    if c and b and c["knn20"] is not None and b["knn20"] is not None:
        body.append(f"* recipe alone (control − LeJEPA, ViT-S/8, `last.ckpt`): k-NN {c['knn20'] - b['knn20']:+.1f}, "
                    f"linear {c['linear'] - b['linear']:+.1f}")

    # appendix: the two narrower readings of the published protocol (decision 16)
    body += ["", "## Appendix A: eval-selected probe on `last.ckpt` (probe selection only, no checkpoint selection)", "",
             "| Backbone | " + " | ".join(label for _, label in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for family, title in FAMILIES:
        row = [title]
        for arm, _ in ARMS:
            run = runs.get((family, arm))
            sel_dir = evalsel(family)
            sel = _load(root / sel_dir / f"{run.run_name}.json") if run else None
            if run is None:
                row.append("n/a")
            elif sel is None:
                row.append("—")
            else:
                _check_protocol(sel, run.run_name, select="eval", kind="last")
                row.append(f'{sel["linear"]["top1"]:.1f}')
                record(f"{family}/{arm}/evalsel_last", run, title, sel, sel_dir, f"{run.run_name}.json")
        body.append("| " + " | ".join(row) + " |")

    body += ["", "## Appendix B: the online probe's own curve (`val/probe_acc`): final epoch / maximum (epoch)", "",
             "| Backbone | " + " | ".join(label for _, label in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for family, title in FAMILIES:
        row = [title]
        for arm, _ in ARMS:
            run = runs.get((family, arm))
            curve, csv_dir = _online_curve(run, root) if run else (None, None)
            if run is None:
                row.append("n/a")
            elif curve is None:
                row.append("—")
            else:
                row.append(f'{curve["final"]:.1f} / {curve["max"]:.1f} (ep {curve["argmax_epoch"]})')
                cells[f"{family}/{arm}/online"] = {
                    "run": run.run_name, "row": f"| {title} |", "kind": "online",
                    "final": curve["final"], "max": curve["max"], "argmax_epoch": curve["argmax_epoch"],
                    "epochs": curve["epochs"], "csv_dir": str(csv_dir), "source": f"{ONLINE_RAW}/{run.run_name}",
                }
        body.append("| " + " | ".join(row) + " |")

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
            sel = cells.get(f"{family}/{arm}/evalsel_best")
            parts.append(f"{sel['linear']:.1f}" if sel else ("--" if (family, arm) in runs else "n/a"))
        rows_out.append(" & ".join(parts) + r" \\")
    arm_heads = " & ".join(r"\multicolumn{2}{c}{" + TEX_ARM[arm] + "}" for arm, _ in ARMS)
    sub_heads = " & ".join([r"$k$-NN / Lin & Lin$^\ddagger$"] * len(ARMS))
    return "\n".join([
        r"%% Auto-generated by scripts/make_imagenet1k_results.py -- do not edit by hand.",
        r"\begin{table}[t]\centering",
        r"\caption{ImageNet-1K, 100 epochs: $k$-NN ($k{=}20$) / linear top-1 (\%) on the validation split,",
        r"frozen features of the final checkpoint, nothing selected on the reported split. $^\ddagger$Linear",
        r"probe with learning rate, weight decay and epoch selected on the validation split itself, best",
        r"reported, on the online probe's best checkpoint: an upper bound of the published protocol.",
        r"$^\dagger$LeJEPA's published ViT-L/14 result under their training setup (EMA teacher, patch",
        r"masking), cited for reference only.}",
        r"\label{tab:imagenet-classification}",
        r"\small\begin{tabular}{l c cc cc cc cc}\toprule",
        r"Backbone & LeJEPA (publ.) & " + arm_heads + r" \\",
        r" & & " + sub_heads + r" \\ \midrule",
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
* **Eval-selected column (‡, decision 16)**: the same linear probe with learning rate, weight decay
  and epoch chosen on the validation split itself and the best reported, on `best.ckpt` — the
  protocol LeJEPA's author described for the published numbers, made at least as permissive on
  every axis. It is labelled from the data (`probe_select`, `checkpoint_kind` in every JSON) and
  is never a headline number; Appendix A gives it on `last.ckpt`, Appendix B the online probe's
  own curve (final / maximum), the two narrower readings.

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
    # the online curves are CSVs, which provenance_block does not archive: copied beside the JSONs
    # so audit_tables.py can re-derive Appendix B from results/raw like every other cell
    for cell in payload["cells"].values():
        if cell.get("kind") == "online":
            for src in sorted(Path(cell["csv_dir"]).glob("version_*/metrics.csv")):
                target = Path("results/raw") / cell["source"] / f"{src.parent.name}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(src.read_bytes())
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
