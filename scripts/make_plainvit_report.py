# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compose report_plainViT_LastExp.md for the plain ViT-S/8 ImageNet-1K runs.

Two evaluations of the frozen 100-epoch backbones:
  * in-domain ImageNet-1K classification: weighted k-NN (k=20, T=0.07) and a
    linear probe, from logs/eval_indomain_in1k_plainvit/per_backbone/*.json
    (src/eval_indomain.py, SPHERE-JEPA Table 3 protocol);
  * Yoda bobblehead angle *regression*: ridge linear probe on the frozen
    features, 10 random 60/20/20 splits, from
    logs/eval_angle_probe_yoda_vit_*_in1k/results_multiseed.json
    (src/eval_angle_probe_multiseed.py).

The hvit / D4-ResNet-50 counterparts trained with the same recipe are added as
reference rows whenever their results exist.

    python scripts/make_plainvit_report.py
"""

import json
import os
import sys

import torch

OUT = "results/report_plainViT_LastExp.md"
MAX_EPOCHS = 100
IND_DIR = "logs/eval_indomain_in1k_plainvit/per_backbone"
IND_DIR_REF = "logs/eval_indomain_in1k_ep100/per_backbone"

# (in-domain name, yoda log dir / backbone key, method label, ckpt dir)
MAIN = [
    ("vit_sigreg", "vit_sigreg_in1k", "LeJEPA (SIGReg)", "checkpoints/lejepa_vit_in1k_sigreg"),
    ("vit_localballreg_l01_k4", "vit_localballreg_l01_k4_in1k", "LeJEPA + local BallReg (λ0.1, k4)",
     "checkpoints/lejepa_vit_in1k_localballreg_l01_k4"),
]
REF = [
    ("hvit_sigreg", "hvit_sigreg_in1k", "Hyperbolic ViT-S/8 · LeJEPA (SIGReg)", "checkpoints/lejepa_hvit_in1k_sigreg"),
    ("hvit_localballreg_l01_k4", "hvit_localballreg_l01_k4_in1k", "Hyperbolic ViT-S/8 · local BallReg (λ0.1, k4)",
     "checkpoints/lejepa_hvit_in1k_localballreg_l01_k4"),
    ("d4rn50_sigreg", "d4rn50_sigreg_in1k", "D4-ResNet-50 · LeJEPA (SIGReg)", "checkpoints/lejepa_d4rn50_in1k_sigreg"),
    ("d4rn50_localballreg_l01_k4", "d4rn50_localballreg_l01_k4_in1k", "D4-ResNet-50 · local BallReg (λ0.1, k4)",
     "checkpoints/lejepa_d4rn50_in1k_localballreg_l01_k4"),
]


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


def _status(ckdir):
    ep = _ckpt_epoch(f"{ckdir}/last.ckpt")
    if ep is None:
        return "*not started*"
    if ep >= MAX_EPOCHS - 1:
        return f"*finished (ep {ep}), eval pending*"
    return f"*pretraining, epoch {ep + 1}/{MAX_EPOCHS}*"


def _knn(entry, k="20"):
    return entry["knn"].get(k) or next(iter(entry["knn"].values()))


def _yoda(key):
    d = _load(f"logs/eval_angle_probe_yoda_{key}/results_multiseed.json")
    if not d:
        return None
    return next(iter(d.values()))


def _ind_rows(rows, ind_dir, with_best):
    out, n = [], 0
    for name, _, label, ckdir in rows:
        final = _load(f"{ind_dir}/{name}.json")
        best = _load(f"{ind_dir}/{name}_best.json")
        if final is None:
            out.append(f"| {label} | {_status(ckdir)} | — | — | — | — |" + (" — |" if with_best else ""))
            continue
        n += 1
        kn, li = _knn(final), final["linear"]
        cell = ""
        if with_best:
            if best is None:
                cell = " — |"
            elif best["epoch"] == final["epoch"]:
                cell = f" same as final (ep {best['epoch']}) |"
            else:
                cell = f" ep {best['epoch']}: {_knn(best)['top1']:.1f} / {best['linear']['top1']:.1f} |"
        out.append(f"| {label} | final (ep {final['epoch']}) | **{kn['top1']:.1f}** | **{li['top1']:.1f}** "
                   f"| {kn['top5']:.1f} | {li['top5']:.1f} |{cell}")
    return out, n


def _yoda_rows(rows):
    out, n = [], 0
    for _, key, label, ckdir in rows:
        r = _yoda(key)
        if r is None:
            out.append(f"| {label} | {_status(ckdir)} | — | — | — | — |")
            continue
        n += 1
        a0, a1 = r["angles"]["angle0"], r["angles"]["angle1"]
        out.append(f"| {label} | best.ckpt | **{a0['r2_mean']:.4f}** ± {a0['r2_std']:.4f} | **{a0['mae_mean']:.2f}** ± {a0['mae_std']:.2f} "
                   f"| **{a1['r2_mean']:.4f}** ± {a1['r2_std']:.4f} | **{a1['mae_mean']:.2f}** ± {a1['mae_std']:.2f} |")
    return out, n


def main():
    ind_hdr = ["| Method | Ckpt (epoch) | k-NN (k=20) top-1, % | Linear top-1, % | k-NN top-5 | Linear top-5 | best-probe ckpt (epoch): k-NN / Linear |",
               "|---|---|---|---|---|---|---|"]
    ind_hdr_ref = ["| Backbone · method | Ckpt (epoch) | k-NN (k=20) top-1, % | Linear top-1, % | k-NN top-5 | Linear top-5 |",
                   "|---|---|---|---|---|---|"]
    yoda_hdr = ["| Method | Ckpt | fig 0 R² | fig 0 MAE (°) | fig 1 R² | fig 1 MAE (°) |", "|---|---|---|---|---|---|"]
    yoda_hdr_ref = ["| Backbone · method | Ckpt | fig 0 R² | fig 0 MAE (°) | fig 1 R² | fig 1 MAE (°) |", "|---|---|---|---|---|---|"]

    main_ind, n_ind = _ind_rows(MAIN, IND_DIR, True)
    ref_ind, _ = _ind_rows(REF, IND_DIR_REF, False)
    main_yoda, n_yoda = _yoda_rows(MAIN)
    ref_yoda, _ = _yoda_rows(REF)

    doc = f"""# Plain ViT-S/8 on ImageNet-1K (100 ep.): LeJEPA vs. LeJEPA + local BallReg

Last experiment of the ImageNet-1K suite: the two 100-epoch pretrains that use the
**plain timm ViT-S/8** (no hyperbolic head, no D4 equivariance) as backbone —
`lejepa/vit_in1k_sigreg` (LeJEPA baseline, global SIGReg) and
`lejepa/vit_in1k_localballreg_l01_k4` (our local-regularizer model: BallReg inside
k=4 k-means neighbourhoods, λ=0.1). Auto-generated by `scripts/make_plainvit_report.py`
({n_ind}/2 classification rows, {n_yoda}/2 regression rows filled).

## Setup

* **Pretraining**: 100 epochs on ImageNet-1K train (1.28M images), crop 128, 4 views/image,
  global batch 256 (64 x 4 grad-accum), AdamW (wd 5e-2; lr 2e-3 for SIGReg, lr 1e-3 +
  10-epoch warmup + grad-clip 1.0 for local BallReg), cosine schedule, embedding dim 512,
  projector [2048, 2048, 16], online linear probe for checkpoint selection. No EMA.
* **Frozen features**: the 512-d encoder output before the projector, deterministic eval
  transform (resize + centre crop 128). Headline rows use the **final checkpoint**
  (epoch 99 = 100 ep.); the last column re-evaluates the online-probe-selected `best.ckpt`.

## 1. Classification: k-NN and linear probe on ImageNet-1K (in-domain)

Protocol of SPHERE-JEPA Table 3 (`src/eval_indomain.py`, `configs/eval/indomain_in1k_plainvit.yaml`):
**k-NN** = full 1.28M train split as feature bank, cosine similarity, weighted vote
`exp(sim/T)` with T = 0.07, k = 20 (k = 10/100/200 in `results.csv`); **linear** = one
linear layer on standardized frozen features, SGD (momentum 0.9, 50 epochs, batch 4096),
lr picked on a 2% held-out slice from {{0.01 … 10}}, refit on the full train split.
Top-1 / top-5 on the 50k IN1K val split.

### Plain ViT-S/8 (this experiment)

{chr(10).join(ind_hdr + main_ind)}

### Reference: same recipe with the other backbones (from `Imagenet1K_results.md`)

{chr(10).join(ind_hdr_ref + ref_ind)}

## 2. Regression: linear probe of the Yoda bobblehead angles

Ridge-regression linear probe on the frozen features of the 8097 bobblehead frames
(`src/eval_angle_probe_multiseed.py`, `configs/eval/yoda/vit_*_in1k.yaml`):
targets are the two figure rotation angles, 60/20/20 train/val/test split, ridge α chosen
on val from {{0.01 … 1000}}, standardized features, **10 random splits** (mean ± std).
R² and mean absolute angular error on the test split.

### Plain ViT-S/8 (this experiment)

{chr(10).join(yoda_hdr + main_yoda)}

### Reference: other backbones

{chr(10).join(yoda_hdr_ref + ref_yoda)}

## Files

* in-domain results: `logs/eval_indomain_in1k_plainvit/` (`results.csv`, `per_backbone/*.json`)
* regression probe: `logs/eval_angle_probe_yoda_vit_sigreg_in1k/`, `logs/eval_angle_probe_yoda_vit_localballreg_l01_k4_in1k/`
* checkpoints: `checkpoints/lejepa_vit_in1k_sigreg/`, `checkpoints/lejepa_vit_in1k_localballreg_l01_k4/`
* pipeline: pretrain (`scripts/slurm_submit.sh`) → `src/eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml` → `scripts/make_plainvit_report.py`
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print(doc)
    print(f"\n{n_ind}/2 classification + {n_yoda}/2 regression rows filled -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# LaTeX version of the ImageNet-1K classification table
# (paper/plainvit_in1k_classification.tex)
# ---------------------------------------------------------------------------
TEX_OUT = "paper/plainvit_in1k_classification.tex"
K_COLS = ("10", "20", "100", "200")
TEX_LABEL = {
    "LeJEPA (SIGReg)": r"LeJEPA (SIGReg)",
    "LeJEPA + local BallReg (λ0.1, k4)": r"LeJEPA + local BallReg ($\lambda{=}0.1$, $k{=}4$)",
}
TEX_GROUPS = [  # (group title, rows, in-domain dir)
    (r"Plain ViT-S/8 (21.7M) --- this experiment", MAIN, IND_DIR),
    (r"Hyperbolic ViT-S/8 (21.7M)", [r for r in REF if r[0].startswith("hvit")], IND_DIR_REF),
    (r"D4-equivariant ResNet-50 (24.4M)", [r for r in REF if r[0].startswith("d4rn50")], IND_DIR_REF),
]
TEX_HEADER = r"""%% Auto-generated by scripts/make_plainvit_report.py -- do not edit by hand.
%% Requires: \usepackage{booktabs,amsmath}
\begin{table}[t]
\centering
\caption{ImageNet-1K in-domain classification (100 epochs, crop 128, no EMA) with a
frozen \textbf{plain ViT-S/8} backbone: LeJEPA (global SIGReg) versus LeJEPA with our
local BallReg regularizer ($\lambda{=}0.1$, $k{=}4$ $k$-means neighbourhoods), following
the protocol of SPHERE-JEPA Table~3. The full 1.28M-image train split is the $k$-NN
feature bank / linear-probe training set; top-1 accuracy (\%) on the 50k validation
split. $k$-NN: cosine similarity, vote weighted by $\exp(\mathrm{sim}/0.07)$, for
$k\in\{10,20,100,200\}$ neighbours ($k{=}20$ is the LeJEPA/DINO convention). Linear:
one linear layer on standardized frozen features (SGD, 50 epochs, learning rate selected
on a 2\% held-out slice of train); top-5 also shown. All rows use the final epoch-99
checkpoint; the same recipe with the Hyperbolic ViT-S/8 and the D4-equivariant ResNet-50
is reported for reference (\emph{Training} rows have not reached 100 epochs yet). Bold:
best per backbone. For reference, the paper reports $50.4$ / $59.2$ ($k$-NN / linear)
for LeJEPA and $50.4$ / $61.0$ for SPHERE-JEPA with a ViT-B/14 at 224\,px.}
\label{tab:plainvit-in1k-classification}
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l cccc c c}
\toprule
& \multicolumn{4}{c}{$k$-NN top-1 (\%)} & \multicolumn{2}{c}{Linear (\%)} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7}
Method & $k{=}10$ & $k{=}20$ & $k{=}100$ & $k{=}200$ & top-1 & top-5 \\"""


def make_tex():
    lines = [TEX_HEADER]
    for gi, (title, rows, ind_dir) in enumerate(TEX_GROUPS):
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}} \\")
        vals = {}
        for name, _, label, ckdir in rows:
            e = _load(f"{ind_dir}/{name}.json")
            if e is not None:
                vals[name] = [e["knn"][k]["top1"] for k in K_COLS] + [e["linear"]["top1"], e["linear"]["top5"]]
        best = [max(v[i] for v in vals.values()) for i in range(6)] if len(vals) > 1 else [None] * 6
        for name, _, label, ckdir in rows:
            short = label.split(" · ", 1)[-1]          # drop the "Backbone · " prefix of REF rows
            short = short.replace("local BallReg", "LeJEPA + local BallReg") if not short.startswith("LeJEPA") else short
            tex_label = TEX_LABEL.get(short, short.replace("λ0.1, k4", r"$\lambda{=}0.1$, $k{=}4$"))
            if name not in vals:
                ep = _ckpt_epoch(f"{ckdir}/last.ckpt")
                prog = f"epoch {ep + 1 if ep is not None else '?'}/{MAX_EPOCHS}"
                lines.append(rf"{tex_label} & \multicolumn{{6}}{{c}}{{\textit{{Training ({prog})}}}} \\")
                continue
            cells = []
            for i, v in enumerate(vals[name]):
                s = f"{v:.1f}"
                if best[i] is not None and v == best[i] and i != 5:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
            lines.append(f"{tex_label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    with open(TEX_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"-> {TEX_OUT}", file=sys.stderr)


if __name__ == "__main__":
    make_tex()
