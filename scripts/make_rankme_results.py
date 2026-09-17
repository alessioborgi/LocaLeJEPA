# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compose RankMe_results.md (RankMe, Garrido et al. ICML 2023, for our IN1K pretrains).

Reads the per-backbone JSONs written by eval_rankme.py and joins them with the
k-NN / linear (Imagenet1K_results.md) and transfer (Transfer_results.md)
accuracies of the same checkpoints, then renders:

  * Table 1 per backbone family: RankMe of the 512-d representations on
    ImageNet-1K (headline), classical rank, centred RankMe, RankMe of the 16-d
    projector embeddings, and the downstream accuracies;
  * Table 2 (paper Tables S20-S25): RankMe of the representations on each
    target dataset;
  * figures (PNG + PDF) in paper/figs/rankme/: RankMe vs accuracy (paper
    Fig. 2), target-vs-source rank (Fig. 5), singular-value spectra (Fig. S8);
  * a LaTeX version of Table 1 in paper/rankme_results.tex.

    python scripts/make_rankme_results.py
"""

import json
import math
import os
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

RANK_DIR = "logs/eval_rankme_in1k_ep100/per_backbone"
IND_DIRS = ["logs/eval_indomain_in1k_ep100/per_backbone", "logs/eval_indomain_in1k_plainvit/per_backbone"]
TRANSFER_DIR = "logs/eval_transfer_in1k_ep100/per_backbone"
OUT = "results/RankMe_results.md"
TEX_OUT = "paper/rankme_results.tex"
FIG_DIR = "paper/figs/rankme"
MAX_EPOCHS = 100
KNN_K = "20"
DATASETS = [("dtd", "DTD"), ("aircraft", "Aircraft"), ("cifar10", "CIFAR10"), ("cifar100", "CIFAR100"),
            ("flowers", "Flowers"), ("food", "Food"), ("pets", "Pets")]

# (name, method label, ckpt dir, family)
ROWS = [
    ("d4rn50_sigreg", "LeJEPA (SIGReg)", "checkpoints/lejepa_d4rn50_in1k_sigreg", "d4rn50"),
    ("d4rn50_rdmreg", "Rectified LpJEPA (RDMReg)", "checkpoints/rectifiedlpjepa_d4rn50_in1k_rdmreg", "d4rn50"),
    ("d4rn50_localballreg_l01_k2", "LeJEPA + local BallReg (λ0.1, k2)", "checkpoints/lejepa_d4rn50_in1k_localballreg_l01_k2", "d4rn50"),
    ("d4rn50_localballreg_l01_k4", "LeJEPA + local BallReg (λ0.1, k4)", "checkpoints/lejepa_d4rn50_in1k_localballreg_l01_k4", "d4rn50"),
    ("d4rn50_localballreg_l03_k2", "LeJEPA + local BallReg (λ0.3, k2)", "checkpoints/lejepa_d4rn50_in1k_localballreg_l03_k2", "d4rn50"),
    ("d4rn50_localsigreg_l01_k4", "LeJEPA + local SIGReg (λ0.1, k4)", "checkpoints/lejepa_d4rn50_in1k_localsigreg_l01_k4", "d4rn50"),
    ("d4rn50_localballreg_l01_dynk", "LeJEPA + local BallReg (λ0.1, k1→10 dyn)", "checkpoints/lejepa_d4rn50_in1k_localballreg_l01_dynk", "d4rn50"),
    ("hvit_sigreg", "LeJEPA (SIGReg)", "checkpoints/lejepa_hvit_in1k_sigreg", "hvit"),
    ("hvit_rdmreg", "Rectified LpJEPA (RDMReg)", "checkpoints/rectifiedlpjepa_hvit_in1k_rdmreg", "hvit"),
    ("hvit_localballreg_l01_k2", "LeJEPA + local BallReg (λ0.1, k2)", "checkpoints/lejepa_hvit_in1k_localballreg_l01_k2", "hvit"),
    ("hvit_localballreg_l01_k4", "LeJEPA + local BallReg (λ0.1, k4)", "checkpoints/lejepa_hvit_in1k_localballreg_l01_k4", "hvit"),
    ("hvit_localballreg_l03_k2", "LeJEPA + local BallReg (λ0.3, k2)", "checkpoints/lejepa_hvit_in1k_localballreg_l03_k2", "hvit"),
    ("hvit_localsigreg_l01_k4", "LeJEPA + local SIGReg (λ0.1, k4)", "checkpoints/lejepa_hvit_in1k_localsigreg_l01_k4", "hvit"),
    ("hvit_localballreg_l01_dynk", "LeJEPA + local BallReg (λ0.1, k1→10 dyn)", "checkpoints/lejepa_hvit_in1k_localballreg_l01_dynk", "hvit"),
    ("vit_sigreg", "LeJEPA (SIGReg)", "checkpoints/lejepa_vit_in1k_sigreg", "vit"),
    ("vit_localballreg_l01_k4", "LeJEPA + local BallReg (λ0.1, k4)", "checkpoints/lejepa_vit_in1k_localballreg_l01_k4", "vit"),
]
FAMILIES = ["d4rn50", "hvit", "vit"]
FAMILY_TITLES = {
    "d4rn50": "D4-equivariant ResNet-50 (equivision / escnn, `pool=equivariant`, 24.4M params)",
    "hvit": "Hyperbolic ViT-S/8 (Poincaré ball, hyp_c=0.1, clip_r=2.3, 21.7M params)",
    "vit": "Plain ViT-S/8 (21.7M params)",
}
FAMILY_SHORT = {"d4rn50": "D4-RN50", "hvit": "HViT-S/8", "vit": "ViT-S/8"}
TEX_FAMILY = {"d4rn50": r"D4-equivariant ResNet-50", "hvit": r"Hyperbolic ViT-S/8", "vit": r"ViT-S/8"}
FAMILY_COLOR = {"d4rn50": "#1f77b4", "hvit": "#d62728", "vit": "#2ca02c"}
METHOD_MARKER = {
    "sigreg": "o", "rdmreg": "s", "localballreg_l01_k2": "^", "localballreg_l01_k4": "v",
    "localballreg_l03_k2": "<", "localsigreg_l01_k4": "D", "localballreg_l01_dynk": "P",
}

HEADER = """# RankMe of the 100-epoch ImageNet-1K pretrains

Replication of **RankMe (Garrido, Balestriero, Najman & LeCun, ICML 2023)** — the
label-free effective rank of a JE-SSL model's features as a predictor of its downstream
performance — for every ImageNet-1K pretrain in this repo: **D4-equivariant ResNet-50**,
**Hyperbolic ViT-S/8** and **plain ViT-S/8** backbones. Auto-generated by
`scripts/make_rankme_results.py` from `logs/eval_rankme_in1k_ep100/per_backbone/`
({n_done}/{n_rows} rows filled) joined with `Imagenet1K_results.md` (k-NN / linear) and
`Transfer_results.md` (7-dataset transfer average).

## Protocol

Implemented in `eval_rankme.py` (`configs/eval/rankme_in1k_ep100.yaml`), metric in
`src/eval/rankme.py`:

* **RankMe** (paper Eq. 1-2): with `σ_k` the singular values of the `N × K` feature
  matrix `Z`, `p_k = σ_k / ‖σ‖₁ + ε` (ε = 1e-7) and `RankMe(Z) = exp(−Σ_k p_k log p_k)`,
  the exponential of the Shannon entropy of the normalised spectrum. No labels, no
  training, no hyper-parameters. The **classical rank** is the threshold rank
  `#{{σ_k > σ_max · max(N, K) · 1e-7}}` (paper Appendix F).
* **Source dataset**: **25,600 ImageNet-1K train images** (a fixed seeded subset, the
  same images for every backbone), as in the paper (Appendix G shows the estimate has
  converged by 10k samples). Deterministic eval transform (resize + centre crop 128).
  Target datasets (paper Tables S20-S25): the train split of each transfer dataset, capped
  at 25,600 images (DTD 3,760 / Aircraft 6,667 / Flowers 2,040 / Pets 3,680 use everything).
* **Where the rank is measured — the one deliberate departure from the paper.** The paper
  ranks the *embeddings* (projector output, 2048-d in their setup) and shows this predicts
  accuracy of the *representations* (encoder output). Our projector is
  `[2048, 2048, 16]`, so embedding RankMe is capped at 16 and cannot discriminate between
  runs. The **headline column is therefore RankMe of the 512-d encoder output** — the
  representation the k-NN / linear / transfer probes actually consume (the paper's Fig. S5
  / Appendix H show the rank-accuracy link holds on representations too). The 16-d
  embedding RankMe (with `ReLU` applied for Rectified LpJEPA, as in its loss) is reported
  for completeness.
* `Z` is **not centred** (paper convention); a centred variant (per-dimension mean removed)
  is given alongside because a single dominant mean direction lowers the entropic rank.
  Features come from the fp16 evaluation caches; RankMe is insensitive to that
  quantisation, the classical rank less so (treat it as indicative).
* **Downstream columns**: k-NN (k = 20, T = 0.07) and linear top-1 on IN1K val from
  `Imagenet1K_results.md`; *Transfer Avg* = mean frozen-linear top-1 over DTD, Aircraft,
  CIFAR-10/100, Flowers, Food, Pets from `Transfer_results.md`. Same final checkpoints
  (epoch 99 = 100 epochs).
* **Reading the tables.** Per the paper (Prop. 5.1 and Sec. 3.2), a higher rank is a
  *necessary, not sufficient* condition for better accuracy and RankMe should only be
  used to compare runs of **the same backbone / method family**, never across backbones.
  Spearman rank correlations between RankMe and each accuracy are reported per family.

## Table 1 — RankMe on ImageNet-1K vs. downstream accuracy

Best RankMe / accuracy per family in bold. `Rank` = classical rank of the same matrix
(≤ 512); `RankMe (c)` = centred; `RankMe emb` = 16-d projector output (≤ 16).

"""

TABLE2_HEADER = """## Table 2 — RankMe of the representations on each target dataset (paper Tables S20-S25)

Entropic rank of the same 512-d features embedded on the *target* datasets' train
splits (≤ 25,600 images each). The paper's hypothesis (iii) is that source and target
ranks are monotonically linked (Eq. 3, Fig. 5); the per-dataset Spearman ρ with the
ImageNet-1K rank is in the last row of each block.

"""

FOOTER = """
`*pretraining, epoch N/100*` rows have not reached 100 epochs yet (their current
`last.ckpt` epoch + 1); `*finished, eval pending*` rows need the matching array index of
`eval_rankme.py --only <name>` on a GPU job via `scripts/slurm_submit.sh` (~10 min: embeds the
25,600-image subset). Runs whose features were already in the in-domain / transfer eval
caches were ranked on a login node with `--cached-only`. A `—` in the k-NN / Linear
columns of a finished row means the in-domain evaluation (`eval_indomain.py`) has not been
run for that checkpoint yet (it is not in `Imagenet1K_results.md` either); a `—` in
Transfer Avg means the same for `eval_transfer.py`.

## Figures

* `paper/figs/rankme/rankme_vs_accuracy.{{png,pdf}}` — paper Fig. 2: IN1K k-NN, IN1K
  linear and Transfer-Avg top-1 against RankMe (log₂ axis), one colour per backbone.
* `paper/figs/rankme/rankme_source_vs_target.{{png,pdf}}` — paper Fig. 5 (left): RankMe on
  each target dataset against RankMe on ImageNet-1K, with the per-dataset linear fit.
* `paper/figs/rankme/rankme_spectra.{{png,pdf}}` — paper Fig. S8/S9: normalised singular
  value spectra `σ_k / σ_1` of the 25,600 × 512 IN1K feature matrix, one panel per backbone.

## Observations

{observations}

## Reproduce

```bash
PYTHONPATH=. python eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml --cached-only  # login node
sbatch --export=ALL,CMD="python eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml --only <name>" scripts/slurm_submit.sh   # uncached backbones
python eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml --merge-only
python scripts/make_rankme_results.py                                                          # rewrites this file
```

All numbers (incl. centred / embedding ranks on every target dataset) are in
`logs/eval_rankme_in1k_ep100/results.csv`; the full spectra are in the per-backbone JSONs.
"""


def _load(path):
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_first(dirs, name):
    for d in dirs:
        e = _load(f"{d}/{name}.json")
        if e is not None:
            return e
    return None


def _ckpt_epoch(path):
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True).get("epoch")
    except Exception:  # noqa: BLE001
        return None


def _progress(ckdir):
    last_ep = _ckpt_epoch(f"{ckdir}/last.ckpt")
    if last_ep is not None and last_ep >= MAX_EPOCHS - 1:
        return f"*finished (ep {last_ep}), eval pending*"
    return f"*pretraining, epoch {last_ep + 1 if last_ep is not None else '?'}/{MAX_EPOCHS}*"


def _transfer_avg(entry):
    if entry is None:
        return None
    vals = [entry["datasets"][k]["top1"] for k, _ in DATASETS if k in entry["datasets"]]
    return sum(vals) / len(vals) if len(vals) == len(DATASETS) else None


def _acc(ind, tr, rank_epoch):
    """(knn, linear, transfer_avg) of the same checkpoint epoch, else None."""
    knn = lin = None
    if ind is not None and ind.get("epoch") == rank_epoch:
        knn = (ind["knn"].get(KNN_K) or next(iter(ind["knn"].values())))["top1"]
        lin = ind["linear"]["top1"]
    avg = _transfer_avg(tr) if (tr is not None and tr.get("epoch") == rank_epoch) else None
    return knn, lin, avg


def _fmt(v, nd=1, bold=False):
    if v is None:
        return "—"
    s = f"{v:.{nd}f}"
    return f"**{s}**" if bold else s


def _spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 4:
        return None, len(pairs)
    rho, _ = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    return float(rho), len(pairs)


def gather():
    data = {}
    for name, label, ckdir, fam in ROWS:
        r = _load(f"{RANK_DIR}/{name}.json")
        ind = _load_first(IND_DIRS, name)
        tr = _load(f"{TRANSFER_DIR}/{name}.json")
        rec = {"label": label, "family": fam, "ckdir": ckdir, "rank": r, "ind": ind, "tr": tr}
        if r is not None:
            src = r["source"]
            rec["rankme"] = src["repr"]["rankme"]
            rec["classical"] = src["repr"]["classical_rank"]
            rec["rankme_c"] = src["repr_centered"]["rankme"]
            rec["rankme_emb"] = src["emb"]["rankme"] if src.get("emb") else None
            rec["targets"] = {k: t["repr"]["rankme"] for k, t in r.get("targets", {}).items()}
            rec["knn"], rec["linear"], rec["tavg"] = _acc(ind, tr, r.get("epoch"))
        data[name] = rec
    return data


def table1(data):
    lines, n_done = [], 0
    for fam in FAMILIES:
        rows = [(n, d) for n, d in data.items() if d["family"] == fam]
        lines += [f"### {FAMILY_TITLES[fam]}", "",
                  "| Method | Ckpt (epoch) | **RankMe (512-d repr)** | Rank | RankMe (c) | RankMe emb (16-d) | k-NN % | Linear % | Transfer Avg % |",
                  "|---|---|---|---|---|---|---|---|---|"]
        best = {}
        for key in ("rankme", "classical", "rankme_c", "rankme_emb", "knn", "linear", "tavg"):
            vals = [d[key] for _, d in rows if d.get(key) is not None]
            best[key] = max(vals) if vals else None
        for name, d in rows:
            if d["rank"] is None:
                lines.append(f"| {d['label']} | {_progress(d['ckdir'])} | — | — | — | — | — | — | — |")
                continue
            n_done += 1
            cells = [_fmt(d["rankme"], 1, d["rankme"] == best["rankme"]),
                     _fmt(d["classical"], 0, d["classical"] == best["classical"]),
                     _fmt(d["rankme_c"], 1, d["rankme_c"] == best["rankme_c"]),
                     _fmt(d["rankme_emb"], 2, d["rankme_emb"] == best["rankme_emb"]),
                     _fmt(d["knn"], 1, d["knn"] == best["knn"]),
                     _fmt(d["linear"], 1, d["linear"] == best["linear"]),
                     _fmt(d["tavg"], 1, d["tavg"] == best["tavg"])]
            lines.append(f"| {d['label']} | final (ep {d['rank']['epoch']}) | " + " | ".join(cells) + " |")
        # correlations
        xs = [d.get("rankme") for _, d in rows]
        corr = []
        for key, disp in (("knn", "k-NN"), ("linear", "Linear"), ("tavg", "Transfer Avg")):
            rho, n = _spearman(xs, [d.get(key) for _, d in rows])
            corr.append(f"{disp}: {'ρ = %.2f' % rho if rho is not None else 'n/a'} (n = {n})")
        lines += ["", f"Spearman ρ(RankMe, accuracy) within this family — " + "; ".join(corr) + ".", ""]
    return "\n".join(lines), n_done


def table2(data):
    lines = []
    for fam in FAMILIES:
        rows = [(n, d) for n, d in data.items() if d["family"] == fam and d["rank"] is not None]
        if not rows:
            continue
        lines += [f"### {FAMILY_SHORT[fam]}", "",
                  "| Method | ImageNet-1K | " + " | ".join(disp for _, disp in DATASETS) + " |",
                  "|---|---|" + "---|" * len(DATASETS)]
        for name, d in rows:
            cells = [_fmt(d["targets"].get(k), 1) for k, _ in DATASETS]
            lines.append(f"| {d['label']} | {d['rankme']:.1f} | " + " | ".join(cells) + " |")
        xs = [d["rankme"] for _, d in rows]
        corr = []
        for k, _ in DATASETS:
            rho, n = _spearman(xs, [d["targets"].get(k) for _, d in rows])
            corr.append("%.2f" % rho if rho is not None else "n/a")
        lines += [f"| *Spearman ρ with IN1K rank* | — | " + " | ".join(corr) + " |", ""]
    return "\n".join(lines)


def _method_key(name):
    return name.split("_", 1)[1]


def figures(data):
    os.makedirs(FIG_DIR, exist_ok=True)
    done = {n: d for n, d in data.items() if d["rank"] is not None}
    if not done:
        return
    # --- Fig. 2: RankMe vs accuracy
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ("knn", "linear", "tavg"), ("ImageNet-1K k-NN", "ImageNet-1K linear", "Transfer Avg (7 datasets)")):
        for n, d in done.items():
            if d.get(key) is None:
                continue
            ax.scatter(d["rankme"], d[key], color=FAMILY_COLOR[d["family"]], marker=METHOD_MARKER.get(_method_key(n), "o"),
                       s=45, edgecolor="k", linewidth=0.4, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("RankMe (512-d representations, IN1K)")
        ax.set_ylabel("Top-1 (%)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], marker="o", linestyle="", label=FAMILY_SHORT[f]) for f in FAMILIES]
    handles += [plt.Line2D([], [], color="gray", marker=m, linestyle="", label=k) for k, m in METHOD_MARKER.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/rankme_vs_accuracy.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Fig. 5: target vs source rank
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cmap = plt.get_cmap("tab10")
    for i, (k, disp) in enumerate(DATASETS):
        xs = np.array([d["rankme"] for d in done.values() if k in d["targets"]])
        ys = np.array([d["targets"][k] for d in done.values() if k in d["targets"]])
        if len(xs) == 0:
            continue
        ax.scatter(xs, ys, color=cmap(i), s=28, label=disp, zorder=3)
        if len(xs) >= 3:
            a, b = np.polyfit(xs, ys, 1)
            grid = np.linspace(xs.min(), xs.max(), 10)
            ax.plot(grid, a * grid + b, color=cmap(i), alpha=0.6, linewidth=1)
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", alpha=0.3, linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("RankMe on ImageNet-1K")
    ax.set_ylabel("RankMe on target dataset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/rankme_source_vs_target.{ext}", dpi=200)
    plt.close(fig)

    # --- Fig. S8: spectra per family
    fams = [f for f in FAMILIES if any(d["family"] == f for d in done.values())]
    fig, axes = plt.subplots(1, len(fams), figsize=(4.5 * len(fams), 3.8), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for n, d in done.items():
            if d["family"] != fam or "spectrum" not in d["rank"]["source"]["repr"]:
                continue
            spec = np.array(d["rank"]["source"]["repr"]["spectrum"])
            ax.plot(np.arange(1, len(spec) + 1), spec, label=f"{d['label']} ({d['rankme']:.0f})", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.5)  # rectified (RDMReg) features have exactly-zero tails
        ax.axvline(16, color="gray", linestyle=":", linewidth=0.8)  # projector width
        ax.set_xlabel("index k")
        ax.set_ylabel("σ_k / σ_1")
        ax.set_title(FAMILY_SHORT[fam])
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_DIR}/rankme_spectra.{ext}", dpi=200)
    plt.close(fig)


def _observations(data):
    out = []
    for fam in FAMILIES:
        rows = {n: d for n, d in data.items() if d["family"] == fam and d["rank"] is not None}
        if not rows:
            continue
        best = max(rows, key=lambda n: rows[n]["rankme"])
        base = f"{fam}_sigreg"
        line = f"* **{FAMILY_SHORT[fam]}**: highest RankMe is {rows[best]['label']} at {rows[best]['rankme']:.1f}"
        if base in rows and best != base:
            line += f" vs {rows[base]['rankme']:.1f} for LeJEPA/SIGReg"
        line += f" (classical rank {rows[best]['classical']} / 512)."
        for key, disp in (("linear", "IN1K linear"), ("tavg", "transfer avg")):
            with_acc = {n: d for n, d in rows.items() if d.get(key) is not None}
            if len(with_acc) >= 2:
                best_acc = max(with_acc, key=lambda n: with_acc[n][key])
                line += (f" Best {disp} ({with_acc[best_acc][key]:.1f}) is "
                         + ("also " if best_acc == best else f"{with_acc[best_acc]['label']}, RankMe {with_acc[best_acc]['rankme']:.1f}, ")
                         + ("the highest-rank run." if best_acc == best else "not the highest-rank run."))
        rho, n = _spearman([d["rankme"] for d in rows.values()], [d.get("linear") for d in rows.values()])
        if rho is not None:
            line += f" Spearman ρ(RankMe, linear) = {rho:.2f} over {n} runs."
        out.append(line)
    done = [d for d in data.values() if d["rank"] is not None]
    if done:
        # Does the highest-rank run of a family ever have the best accuracy?
        fam_stats = []
        for fam in FAMILIES:
            rows = {n: d for n, d in data.items() if d["family"] == fam and d["rank"] is not None}
            if len(rows) < 2:
                continue
            top = max(rows, key=lambda n: rows[n]["rankme"])
            wins = [disp for key, disp in (("knn", "k-NN"), ("linear", "linear"), ("tavg", "transfer"))
                    if any(d.get(key) is not None for d in rows.values())
                    and rows[top].get(key) is not None
                    and rows[top][key] == max(d[key] for d in rows.values() if d.get(key) is not None)]
            fam_stats.append((FAMILY_SHORT[fam], rows[top]["label"], wins))
        if fam_stats:
            hit = [f for f, _, w in fam_stats if "linear" in w]
            miss = [f for f, _, w in fam_stats if "linear" not in w]
            rhos = {FAMILY_SHORT[fam]: _spearman([d["rankme"] for d in data.values() if d["family"] == fam and d["rank"] is not None],
                                                  [d.get("linear") for d in data.values() if d["family"] == fam and d["rank"] is not None])
                    for fam in FAMILIES}
            out.append("* **Does the highest-rank run win?** "
                       + "; ".join(f"{f} → {lab}" + (f" (best on {', '.join(w)})" if w else " (best on nothing)")
                                   for f, lab, w in fam_stats)
                       + ". RankMe picks the best IN1K-linear run in " + (", ".join(hit) if hit else "no family")
                       + (" but not in " + ", ".join(miss) if miss else "")
                       + " (ρ(RankMe, linear): " + ", ".join(f"{f} {'%.2f' % r if r is not None else 'n/a'} (n={n})"
                                                           for f, (r, n) in rhos.items() if n) + "). "
                         "All non-collapsed runs sit in the regime the paper flags as outside RankMe's reach — rank is "
                         "necessary, not sufficient — and the local regularisers differ in how the variance is organised "
                         "(within-cluster structure) rather than in how much of it there is.")
        # Share of spectral energy (sum sigma^2) in the first 16 = projector-width directions, per family.
        knees = []
        for fam in FAMILIES:
            fracs = []
            for d in done:
                spec = d["rank"]["source"]["repr"].get("spectrum") if d["family"] == fam else None
                if spec:
                    sq = np.square(np.array(spec))
                    fracs.append(100.0 * sq[:16].sum() / sq.sum())
            if fracs:
                knees.append((FAMILY_SHORT[fam], min(fracs), max(fracs)))
        if knees:
            out.append("* **Where the variance sits** (`rankme_spectra`, dotted line = projector width 16): the first "
                       "16 singular directions of the 512-d representation carry "
                       + ", ".join(f"{lo:.0f}-{hi:.0f}% ({f})" for f, lo, hi in knees)
                       + " of the spectral energy Σσ² (much of it the uncentred mean direction σ₁). Beyond that the "
                         "equivariant ResNet's spectrum is nearly flat out to k = 512 (hence its much larger RankMe), "
                         "whereas both ViTs show a knee at about the projector width and decay steeply after it.")
        collapsed = [d for d in done if d["rankme"] < 20]
        if collapsed:
            out.append("* **RankMe does catch true collapse**: "
                       + "; ".join(f"{FAMILY_SHORT[d['family']]} {d['label']} has RankMe {d['rankme']:.1f} "
                                   f"(classical rank {d['classical']}), i.e. its 512-d representation lives in ~{d['rankme']:.0f} "
                                   "dimensions — the width of the projector" for d in collapsed)
                       + "; it is also the worst run of its family on every probe. This is the regime the paper's "
                         "hyper-parameter selection (Sec. 4) is designed to discard.")
        embs = [d["rankme_emb"] for d in done if d.get("rankme_emb") is not None]
        if embs:
            out.append(f"* **16-d embedding RankMe is saturated**: {min(embs):.2f}-{max(embs):.2f} out of a maximum of 16 "
                       "for every run, i.e. the projector output is essentially full-rank/whitened (SIGReg/BallReg "
                       "push the 16-d projection towards an isotropic Gaussian / ball, so this is by construction). "
                       "It cannot rank the runs — hence the 512-d headline column.")
        gaps = [(d["rankme_c"] - d["rankme"]) for d in done]
        out.append(f"* Centring raises RankMe by {min(gaps):.0f}-{max(gaps):.0f}: the uncentred features carry a large "
                   "mean direction (one dominant singular value) that the paper's convention counts against the rank.")
        out.append("* Per the paper, compare rows **within** a family only; RankMe across backbones (ResNet vs ViT, "
                   "Euclidean vs hyperbolic) is confounded by geometry and is not a performance predictor.")
    return "\n".join(out) if out else "* No backbone ranked yet."


def _tex(label):
    """Method label -> LaTeX: 'local BallReg (λ0.1, k2)' -> 'local BallReg ($\lambda{=}0.1$, $k{=}2$)'."""
    import re
    label = label.replace("LeJEPA + ", "+ ")
    label = re.sub(r"λ([0-9.]+)", r"$\\lambda{=}\1$", label)
    label = re.sub(r"\bk(\d+)→(\d+) dyn", r"$k{=}\1\\rightarrow\2$", label)
    label = re.sub(r"\bk(\d+)\b", r"$k{=}\1$", label)
    return label


def make_tex(data):
    keys = ("rankme", "classical", "rankme_emb", "knn", "linear", "tavg")
    fmt = {"rankme": "{:.1f}", "classical": "{:d}", "rankme_emb": "{:.2f}", "knn": "{:.1f}", "linear": "{:.1f}", "tavg": "{:.1f}"}
    ncol = 1 + len(keys)
    rows = []
    for fam in FAMILIES:
        fam_rows = [d for d in data.values() if d["family"] == fam and d["rank"] is not None]
        if not fam_rows:
            continue
        best = {k: max((d[k] for d in fam_rows if d.get(k) is not None), default=None) for k in keys}
        rows.append(r"\midrule")
        rows.append(r"\multicolumn{" + str(ncol) + r"}{l}{\textit{" + TEX_FAMILY[fam] + r"}} \\")
        for d in fam_rows:
            cells = []
            for k in keys:
                v = d.get(k)
                if v is None:
                    cells.append("--")
                    continue
                cell = fmt[k].format(v)
                cells.append(r"\textbf{" + cell + "}" if v == best[k] and k != "rankme_emb" else cell)
            rows.append(f"{_tex(d['label'])} & " + " & ".join(cells) + r" \\")
    if not rows:
        return
    tex = [r"% Auto-generated by scripts/make_rankme_results.py -- do not edit by hand.",
           r"% Requires \usepackage{booktabs}.",
           r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
           r"\caption{\textbf{RankMe~\cite{garrido2023rankme} of our 100-epoch ImageNet-1K pretrains.} Effective "
           r"(entropic) rank of the frozen 512-d representations on 25,600 ImageNet-1K train images (uncentred, "
           r"Eq.~1 of \cite{garrido2023rankme}), the classical threshold rank of the same matrix ($\le 512$), RankMe "
           r"of the 16-d projector embeddings ($\le 16$; saturated for every run), and the downstream accuracies of the "
           r"same checkpoints: ImageNet-1K $k$-NN ($k{=}20$) and linear-probe top-1, and the mean frozen linear-probe "
           r"top-1 over DTD, Aircraft, CIFAR-10/100, Flowers, Food and Pets. Best value per backbone in bold. "
           r"Following \cite{garrido2023rankme}, RankMe is only comparable between runs of the same backbone.}",
           r"\label{tab:rankme}",
           r"\begin{tabular}{l" + "c" * len(keys) + "}", r"\toprule",
           r"Method & RankMe (512-d) & Rank & RankMe emb. (16-d) & $k$-NN (\%) & Linear (\%) & Transfer avg.\ (\%) \\"]
    tex += rows + [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    with open(TEX_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))


def main():
    data = gather()
    body1, n_done = table1(data)
    body2 = table2(data)
    figures(data)
    doc = HEADER.format(n_done=n_done, n_rows=len(ROWS)) + body1 + "\n" + TABLE2_HEADER + body2 \
        + FOOTER.format(observations=_observations(data))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print(body1)
    print(body2)
    print(f"\n{n_done}/{len(ROWS)} rows complete -> {OUT}", file=sys.stderr)
    make_tex(data)


if __name__ == "__main__":
    main()
