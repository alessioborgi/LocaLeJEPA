# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Figures and summary table for ``eval_structure.py``.

  structure_layouts   rows = backbone (sorted by k-NN), columns = t-SNE / UMAP /
                      EmbedOR / EmbedOR+ORC-ManL, coloured by Imagenette class.
                      Each panel is captioned with the k-NN accuracy measured
                      *inside* that 2-D layout, so the reader can see how much
                      class structure the projection kept -- a layout that
                      scores far below the ambient number is reorganising the
                      data, not showing it.

                      Panels are Procrustes-aligned (rotation/reflection, via
                      class centroids) to the top row of their column: SNE-type
                      layouts have no canonical orientation, and without this a
                      rotation reads as a real difference between backbones.

  structure_metrics   purity vs neighbour rank, fragmentation vs graph k, k-NN
                      vs k, and bars for accuracy / clustering agreement /
                      curvature. No silhouette -- see eval_structure.py.

  structure_table.md  the numbers as a markdown table.

    python visualisations/plot_embedding_structure.py
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Sized for reading on a screen rather than in a printed column: a 13 pt base,
# and greys kept dark enough to stay legible against white.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})
MUTED = "#4a4a4a"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(_HERE, "figures")
DATA = os.path.join(_ROOT, "logs", "eval_structure")

CLASSES = ["tench", "English springer", "cassette player", "chain saw", "church",
           "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]
PALETTE = plt.get_cmap("tab10").colors

LAYOUT_LABEL = {"tsne": "t-SNE", "umap": "UMAP",
                "embedor": "EmbedOR", "embedor_pruned": "EmbedOR + ORC-ManL"}
LAYOUT_ORDER = ["tsne", "umap", "embedor", "embedor_pruned"]

# Stable colour per method family; seeds of one arm share a colour.
FAMILY = [("lejepa", "#8c8c8c", "LeJEPA (SIGReg)"),
          ("rectlpjepa", "#3d6fb5", "RectLpJEPA (RDMReg)"),
          ("k2", "#f4a259", "BallJEPA λ0.1 k2"),
          ("k4", "#e07a1f", "BallJEPA λ0.1 k4"),
          ("k8", "#2a9d8f", "BallJEPA λ0.1 k8")]


def _family(name: str) -> tuple[str, str]:
    for key, colour, label in FAMILY:
        if key in name:
            seed = name.split("_s")[-1] if "_s" in name else None
            return colour, (f"{label} · s{seed}" if seed and seed.isdigit() else label)
    return "#777777", name


def _load():
    path = os.path.join(DATA, "metrics.csv")
    if not os.path.exists(path):
        sys.exit(f"no metrics at {path} -- run eval_structure.py first")
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in list(r.items()):
            if k != "backbone":
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    r[k] = None
    with open(os.path.join(DATA, "purity.json"), encoding="utf-8") as fh:
        purity = json.load(fh)
    return rows, purity, np.load(os.path.join(DATA, "coords.npz"))


def _sort_key(r):
    """Rank by k-NN; fall back to LOO when Table-4 columns are absent."""
    return -(r.get("knn20_top1") or r.get("loo_knn20") or 0.0)


def _knn_2d(P: np.ndarray, y: np.ndarray, k: int = 20) -> float:
    """k-NN accuracy computed in the 2-D layout: how much survived the projection."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(P)
    _, idx = nn.kneighbors(P)
    return 100.0 * float((np.array([np.bincount(y[r]).argmax() for r in idx[:, 1:]]) == y).mean())


def _procrustes(src: np.ndarray, ref: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rotate/reflect/scale `src` onto `ref` using class centroids as correspondences."""
    from scipy.linalg import orthogonal_procrustes

    def cents(P):
        return np.vstack([P[labels == c].mean(0) for c in np.unique(labels)])

    s, r = src - src.mean(0), ref - ref.mean(0)
    scale = np.linalg.norm(cents(r)) / max(np.linalg.norm(cents(s)), 1e-12)
    R, _ = orthogonal_procrustes(cents(s * scale), cents(r))
    return (s * scale) @ R


def plot_layouts(rows, coords, transpose: bool = False) -> None:
    order = [r["backbone"] for r in sorted(rows, key=_sort_key)]
    methods = [m for m in LAYOUT_ORDER if f"{order[0]}__{m}" in coords]
    if not methods:
        print("skipping structure_layouts: no layouts in coords.npz")
        return

    nrow, ncol = (len(methods), len(order)) if transpose else (len(order), len(methods))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.95 * nrow), squeeze=False)
    for j, m in enumerate(methods):
        ref = None
        for i, name in enumerate(order):
            ax = axes[j][i] if transpose else axes[i][j]
            P = np.asarray(coords[f"{name}__{m}"], dtype=float)
            y = np.asarray(coords[f"{name}__labels"], dtype=int)
            if ref is None:
                P = ref = P - P.mean(0)
            else:
                P = _procrustes(P, ref, y)

            for c in range(len(CLASSES)):
                sel = y == c
                ax.scatter(P[sel, 0], P[sel, 1], s=4.0, alpha=0.7,
                           color=PALETTE[c], linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for sp in ax.spines.values():
                sp.set_color("#cccccc")
            ax.set_title(f"k-NN in 2-D: {_knn_2d(P, y):.1f}%", fontsize=13,
                         color=MUTED, pad=5)
            r = next(x for x in rows if x["backbone"] == name)
            amb = r.get("knn20_top1") or r.get("loo_knn20")
            backbone_label = f"{_family(name)[1]}\nambient k-NN {amb:.1f}%"
            if (j == 0) if transpose else (i == 0):
                ax.text(0.5, 1.24 if transpose else 1.13,
                        backbone_label if transpose else LAYOUT_LABEL[m],
                        transform=ax.transAxes, ha="center",
                        fontsize=16 if transpose else 17, fontweight="bold")
            if (i == 0) if transpose else (j == 0):
                ax.text(-0.09, 0.5, LAYOUT_LABEL[m] if transpose else backbone_label,
                        transform=ax.transAxes, rotation=90, va="center", ha="right",
                        fontsize=17 if transpose else 14, fontweight="bold")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=11, color=PALETTE[c], label=CLASSES[c])
               for c in range(len(CLASSES))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=15,
               bbox_to_anchor=(0.5, -0.012 - 0.004 * nrow))
    fig.suptitle("Imagenette val embeddings coloured by class — frozen ViT-S/8 encoder\n"
                 "panel titles give k-NN measured inside the 2-D layout; compare with the "
                 f"ambient value at the {'top' if transpose else 'left'}",
                 fontsize=18, y=1.0 + 0.006 * nrow)
    _save(fig, "structure_layouts")


def plot_metrics(rows, purity) -> None:
    rows = sorted(rows, key=_sort_key)
    names = [r["backbone"] for r in rows]
    colours = [_family(n)[0] for n in names]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.2))

    for r in rows:
        p = np.asarray(purity[r["backbone"]])
        ax[0][0].plot(np.arange(1, len(p) + 1), p, color=_family(r["backbone"])[0],
                      lw=1.8, label=_family(r["backbone"])[1])
    ax[0][0].axhline(0.1, color=MUTED, ls=":", lw=1.2)
    ax[0][0].annotate("chance", (100, 0.1), ha="right", va="bottom", fontsize=12, color=MUTED)
    ax[0][0].set(xlabel="neighbour rank", ylabel="P(same class)", xscale="log")
    ax[0][0].set_title("Neighbourhood purity profile", fontsize=15)
    ax[0][0].legend(frameon=False, fontsize=11)

    ks = sorted(int(c.split("_k")[1]) for c in rows[0] if c.startswith("frag_k"))
    for r in rows:
        ax[0][1].plot(ks, [r[f"frag_k{k}"] for k in ks], "o-", ms=6,
                      color=_family(r["backbone"])[0], lw=1.7)
    ax[0][1].axhline(1.0, color=MUTED, ls=":", lw=1.2)
    ax[0][1].annotate("one piece per class", (ks[-1], 1.0), ha="right", va="bottom",
                      fontsize=12, color=MUTED)
    ax[0][1].set(xlabel="k-NN graph k", ylabel="components per class", yscale="log")
    ax[0][1].set_title("Class fragmentation", fontsize=15)

    kk = [k for k in (10, 20, 100, 200) if rows[0].get(f"knn{k}_top1") is not None]
    for r in rows:
        if kk:
            ax[0][2].plot(kk, [r[f"knn{k}_top1"] for k in kk], "o-", ms=6,
                          color=_family(r["backbone"])[0], lw=1.7)
    ax[0][2].set(xlabel="k (neighbours)", ylabel="top-1 (%)", xscale="log")
    ax[0][2].set_title("Weighted k-NN vs k (train bank → val)", fontsize=15)
    if not kk:
        ax[0][2].axis("off")

    def bars(a, series, title, ylabel):
        present = [(k, h) for k, h in series if all(r.get(k) is not None for r in rows)]
        if not present:
            a.axis("off")
            return
        w, x = 0.8 / len(present), np.arange(len(names))
        for s, (key, hatch) in enumerate(present):
            xs = x + s * w - 0.4 + w / 2
            a.bar(xs, [r[key] for r in rows], w, hatch=hatch, edgecolor="white",
                  color=colours, label=key)
            for xi, r in zip(xs, rows):
                a.annotate(f"{r[key]:.2f}" if r[key] < 10 else f"{r[key]:.1f}",
                           (xi, r[key]), ha="center", va="bottom", fontsize=10)
        a.set_xticks(x)
        a.set_xticklabels([_family(n)[1].replace(" (", "\n(").replace(" · ", "\n")
                           for n in names], fontsize=11)
        a.set_title(title, fontsize=15)
        a.set_ylabel(ylabel)
        a.margins(y=0.18)
        a.legend(frameon=False, fontsize=12, loc="lower right")

    bars(ax[1][0], [("knn20_top1", ""), ("linear_top1", "//")],
         "Table 4 — k-NN and linear (train bank → val)", "top-1 (%)")
    bars(ax[1][1], [("kmeans_ari", ""), ("spectral_ari", "//")],
         "Clustering agreement — compactness vs connectivity", "adjusted Rand index")
    bars(ax[1][2], [("orc_gap", ""), ("orc_auroc", "//")],
         "Ollivier-Ricci curvature", "Δκ  /  AUROC(−κ → class-crossing)")

    for a in ax.ravel():
        a.grid(alpha=0.22, lw=0.5)
    fig.suptitle("Cluster structure of the Imagenette val embeddings — frozen ViT-S/8 encoder",
                 fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    _save(fig, "structure_metrics")


def write_table(rows, coords) -> None:
    """Markdown summary: the Table-4 columns plus the cluster-quality measures."""
    rows = sorted(rows, key=_sort_key)
    cols = [("knn20_top1", "k-NN@20"), ("linear_top1", "Linear"),
            ("kmeans_ari", "ARI k-means"), ("spectral_ari", "ARI spectral"),
            ("orc_gap", "Δκ"), ("orc_auroc", "κ-AUROC"), ("frag_k15", "frag@15")]
    present = [(k, h) for k, h in cols if any(r.get(k) is not None for r in rows)]

    lines = ["| Backbone | " + " | ".join(h for _, h in present) + " |",
             "|---" * (len(present) + 1) + "|"]
    for r in rows:
        vals = [f"{r[k]:.2f}" if r.get(k) is not None else "—" for k, _ in present]
        lines.append(f"| {_family(r['backbone'])[1]} | " + " | ".join(vals) + " |")

    methods = [m for m in LAYOUT_ORDER if f"{rows[0]['backbone']}__{m}" in coords]
    if methods:
        lines += ["", "**Class structure retained by each 2-D layout** "
                      "(k-NN@20 measured in the layout; ambient value in the last column)", "",
                  "| Backbone | " + " | ".join(LAYOUT_LABEL[m] for m in methods) + " | ambient |",
                  "|---" * (len(methods) + 2) + "|"]
        for r in rows:
            n = r["backbone"]
            y = np.asarray(coords[f"{n}__labels"], dtype=int)
            vals = [f"{_knn_2d(np.asarray(coords[f'{n}__{m}'], dtype=float), y):.1f}"
                    for m in methods]
            amb = r.get("knn20_top1") or r.get("loo_knn20")
            lines.append(f"| {_family(n)[1]} | " + " | ".join(vals) + f" | {amb:.1f} |")

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, "structure_table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote figures/structure_table.md")


def _save(fig, stem: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{stem}.{ext}"), dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{stem}.png / .pdf")


def _one_seed(rows, seed: str):
    """Drop every seeded arm except `seed`; unseeded backbones are always kept."""
    keep = [r for r in rows
            if "_s" not in r["backbone"] or r["backbone"].endswith(f"_s{seed}")]
    if not keep:
        sys.exit(f"--layout-seed {seed} matched no backbone")
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--transpose", action="store_true",
                    help="layout grid with methods as rows and backbones as columns")
    ap.add_argument("--layout-seed", default=None,
                    help="restrict the layout grid to this pretraining seed (e.g. 0); "
                         "the metrics panel and table keep every seed")
    args = ap.parse_args()
    rows, purity, coords = _load()
    plot_layouts(_one_seed(rows, args.layout_seed) if args.layout_seed else rows,
                 coords, transpose=args.transpose)
    plot_metrics(rows, purity)
    write_table(rows, coords)


if __name__ == "__main__":
    main()
