#!/usr/bin/env python
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""The torus table (main body) and the rotation-after-ImageNet table (appendix), from the
campaign's Yoda rotation-angle probes.

Rows come from the campaign manifests (``configs/campaign/torus.yaml`` for the torus lineup;
``in1k_baselines`` / ``in1k_local`` / ``in1k_vitl14`` for the appendix), so every row is a run
that exists under exactly that name, and a renamed run cannot be silently dropped (F35). Each
eval config is multi-backbone, so one ``results_multiseed.json`` per table is read *by run
name*. Every cell is R² / MAE (degrees), mean ± std over the 10 random 60/20/20 splits.

Alongside the markdown this writes ``results/report_equivariance.cells.json`` for
``scripts/audit_tables.py``.

    python scripts/make_equivariance_report.py [--root <eval run root>]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eval.io import atomic_write, provenance_block, refuse_to_replace_with_placeholders  # noqa: E402
import scripts.aicr.campaign as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = "results/report_equivariance.md"
CELLS_OUT = "results/report_equivariance.cells.json"
TORUS_DIR = "logs/eval_angle_probe_torus_campaign"
IN1K_DIRS = {"in1k_baselines.yaml": "logs/eval_angle_probe_in1k_campaign",
             "in1k_local.yaml": "logs/eval_angle_probe_in1k_campaign",
             "in1k_vitl14.yaml": "logs/eval_angle_probe_vitl14_campaign"}

LABELS = {  # torus lineup tags -> (encoder, regularizer), the draft's row labels
    "svit_susreg": ("Spherical ViT", "SUSReg"), "nvit_sigreg": ("nViT", "SIGReg"), "nvit_subreg": ("nViT", "SubReg"),
    # F56: the two Imagenette tags carry the dataset because `vit_sigreg`/`vit_rdmreg` are also block tags of
    # in1k_baselines.yaml and a run name is tag+seed; the old spellings stay so an existing run still labels.
    "vit_imagenette_sigreg": ("ViT", "SIGReg"), "vit_imagenette_rdmreg": ("ViT", "RDMReg"),
    "vit_sigreg": ("ViT", "SIGReg"), "vit_subreg": ("ViT", "SubReg"), "vit_rdmreg": ("ViT", "RDMReg"),
    "vit_local_l01_k2": ("ViT", "LocaLeJEPA (λ=0.1, K=2)"), "vit_local_l01_dynk": ("ViT", "LocaLeJEPA (λ=0.1, dynamic K)"),
    "vit_local_l01_k4": ("ViT", "LocaLeJEPA (λ=0.1, K=4)"), "vit_local_l03_k2": ("ViT", "LocaLeJEPA (λ=0.3, K=2)"),
    "vit_localsigreg_l01_k4": ("ViT", "local SIGReg (λ=0.1, K=4)"),
}


def _load_multiseed(folder: Path) -> dict:
    path = folder / "results_multiseed.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cell(angle: dict | None) -> str:
    if angle is None:
        return "—"
    return f'{angle["r2_mean"]:.4f}±{angle["r2_std"]:.4f} / {angle["mae_mean"]:.2f}±{angle["mae_std"]:.2f}°'


def _label(run: C.RunSpec) -> tuple[str, str]:
    if run.tag in LABELS:
        return LABELS[run.tag]
    parts = run.tag.split("_", 1)
    encoder = {"vit": "ViT-S/8", "vitl14": "ViT-L/14", "d4rn50": "D4-ResNet-50", "hvit": "Hyperbolic ViT-S/8"}.get(parts[0], parts[0])
    reg = {"sigreg": "LeJEPA (SIGReg)", "rdmreg": "RDMReg", "local": "LocaLeJEPA", "control": "control (SIGReg, local recipe)"}
    key = parts[1] if len(parts) > 1 else ""
    extra = ", ".join(("λ=" if o.startswith("model.lamb=") else "K=") + o.split("=", 1)[1] for o in run.overrides
                      if o.startswith("model.lamb=") or o.startswith("model.reg_loss.num_clusters="))
    return encoder, (reg.get(key, key) + (f" ({extra})" if extra else ""))


def table(specs: list[C.RunSpec], results: dict, cells: dict, prefix: str, source: str) -> list[str]:
    lines = ["| Encoder | Regularizer | figurine 0 (R² / MAE°) | figurine 1 (R² / MAE°) |", "|---|---|---|---|"]
    for run in specs:
        encoder, reg = _label(run)
        entry = results.get(run.run_name)
        if entry is None:
            lines.append(f"| {encoder} | {reg} | *pending* | *pending* |")
            continue
        angles = entry.get("angles", {})
        a0, a1 = angles.get("angle0"), angles.get("angle1")
        lines.append(f"| {encoder} | {reg} | {_cell(a0)} | {_cell(a1)} |")
        for k, angle in (("angle0", a0), ("angle1", a1)):
            if angle is not None:
                cells[f"{prefix}/{run.run_name}/{k}"] = {
                    "kind": "torus", "run": run.run_name, "row": f"| {encoder} | {reg} |",
                    "r2_mean": angle["r2_mean"], "mae_mean": angle["mae_mean"], "source": source,
                }
    return lines


def render(root: Path) -> tuple[str, dict, list[str]]:
    cells: dict[str, dict] = {}
    torus_runs = C.expand(C.load_manifest(REPO / "configs/campaign/torus.yaml"))
    torus_results = _load_multiseed(root / TORUS_DIR)
    body = ["## Torus table: Imagenette-pretrained (800 epochs), rotation-angle linear probe", "",
            *table(torus_runs, torus_results, cells, "torus", f"{os.path.basename(TORUS_DIR)}/results_multiseed.json"), ""]
    body += ["## Appendix: rotation recovery after ImageNet-1K pretraining (100 epochs)", ""]
    in1k_specs, in1k_results, sources = [], {}, []
    for manifest, folder in IN1K_DIRS.items():
        specs = C.expand(C.load_manifest(REPO / "configs/campaign" / manifest))
        res = _load_multiseed(root / folder)
        for run in specs:
            in1k_specs.append(run)
            if run.run_name in res:
                in1k_results[run.run_name] = res[run.run_name]
        sources.append(folder)
    # one source per run: the folder its manifest maps to
    source_of = {run.run_name: f"{os.path.basename(IN1K_DIRS[m])}/results_multiseed.json"
                 for m in IN1K_DIRS for run in C.expand(C.load_manifest(REPO / "configs/campaign" / m))}
    lines = ["| Encoder | Regularizer | figurine 0 (R² / MAE°) | figurine 1 (R² / MAE°) |", "|---|---|---|---|"]
    for run in in1k_specs:
        encoder, reg = _label(run)
        entry = in1k_results.get(run.run_name)
        if entry is None:
            lines.append(f"| {encoder} | {reg} | *pending* | *pending* |")
            continue
        angles = entry.get("angles", {})
        a0, a1 = angles.get("angle0"), angles.get("angle1")
        lines.append(f"| {encoder} | {reg} | {_cell(a0)} | {_cell(a1)} |")
        for k, angle in (("angle0", a0), ("angle1", a1)):
            if angle is not None:
                cells[f"in1k/{run.run_name}/{k}"] = {"kind": "torus", "run": run.run_name, "row": f"| {encoder} | {reg} |",
                                                     "r2_mean": angle["r2_mean"], "mae_mean": angle["mae_mean"],
                                                     "source": source_of[run.run_name]}
    body += lines
    n_done = sum(1 for key in cells if key.endswith("/angle0"))
    return "\n".join(body), {"cells": cells, "n_done": n_done, "n_rows": len(torus_runs) + len(in1k_specs)}, \
        [TORUS_DIR, *sorted(set(IN1K_DIRS.values()))]


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root)
    body, payload, dirs = render(root)
    head = ("# Rotation-angle (Yoda bobblehead) linear probes of the campaign's pretrains\n\n"
            f"Auto-generated by `scripts/make_equivariance_report.py` ({payload['n_done']}/{payload['n_rows']} rows "
            "complete). Frozen features, ridge probe on the two figurines' rotation angles, mean ± std over 10 random "
            "60/20/20 splits; every row reads the run's `last.ckpt`.\n\n")
    input_dirs = [str(root / d) for d in dirs]
    refuse_to_replace_with_placeholders(OUT, input_dirs)
    doc = head + body + "\n" + provenance_block(os.path.relpath(__file__, REPO), input_dirs, archive_to="results/raw")
    with atomic_write(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    with atomic_write(CELLS_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(body)
    return body


if __name__ == "__main__":
    main()
