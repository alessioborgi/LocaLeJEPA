# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""DINOv2-style patch-PCA figure for the main paper.

One row per backbone family (Euclidean ViT-S/8, hyperbolic ViT-S/8,
D4-equivariant ResNet-50), each showing the LocaLeJEPA checkpoint on a few
ImageNet-1K validation crops as [input | PCA -> RGB | overlay] triplets. Reads
the group-scope npz/png outputs written by src/visualize_patch_pca.py. With
--with-baseline every family gets a second row for the LeJEPA (SIGReg)
checkpoint, which is the appendix variant.

    python visualisations/plot_patch_pca_main.py                 # main figure
    python visualisations/plot_patch_pca_main.py --with-baseline # appendix
"""
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
FIG_DIR = _HERE / "figures"

# family label -> (LocaLeJEPA dir, LeJEPA/SIGReg dir); D4 uses the 256-px crop
# run because its output stride is 32 (8x8 map at 256 px vs 4x4 at 128 px).
FAMILIES = [
    ("ViT-S/8 (Euclidean)",
     "logs/patch_pca_in1k/group/vit_localballreg_l01_k4",
     "logs/patch_pca_in1k/group/vit_sigreg"),
    ("H-ViT-S/8 (hyperbolic)",
     "logs/patch_pca_in1k/group/hvit_localballreg_l01_k2",
     "logs/patch_pca_in1k/group/hvit_sigreg"),
    ("D4-RN50 (equivariant)",
     "logs/patch_pca_in1k_d4rn50_256/group/d4rn50_localballreg_l01_k4",
     "logs/patch_pca_in1k_d4rn50_256/group/d4rn50_sigreg"),
]


def load_row(d, group, image_ids, size):
    d = _ROOT / d
    arc = np.load(d / f"group_{group}.npz")
    rgb, grid = arc["rgb"], tuple(int(v) for v in arc["grid"])
    out = []
    for i in image_ids:
        im = Image.open(d / f"group_{group}_input_{i}.png").convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        pca = Image.fromarray(np.uint8(255 * rgb[i].reshape(*grid, 3))).resize((size, size), Image.Resampling.NEAREST)
        out.append((np.asarray(im) / 255.0, np.asarray(pca) / 255.0, grid))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", default="00", help="image group (00 = golden retriever)")
    ap.add_argument("--images", type=int, nargs="+", default=[0, 4], help="indices within the group")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--size", type=int, default=256, help="display resolution of every tile")
    ap.add_argument("--with-baseline", action="store_true", help="add a LeJEPA (SIGReg) row per family")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or FIG_DIR / ("patch_pca_main_vs_sigreg" if args.with_baseline else "patch_pca_main")

    rows = []
    for label, ours, base in FAMILIES:
        rows.append((label, "LocaLeJEPA", ours))
        if args.with_baseline:
            rows.append((label, "LeJEPA (SIGReg)", base))

    n_img = len(args.images)
    ncol = 3 * n_img
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.55 * ncol, 1.62 * len(rows)), squeeze=False)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.075, right=0.995, top=0.94, bottom=0.01)
    col_titles = ["Input", "PCA", "Overlay"]
    for r, (label, method, d) in enumerate(rows):
        tiles = load_row(d, args.group, args.images, args.size)
        for j, (im, pca, grid) in enumerate(tiles):
            ax_in, ax_pca, ax_ov = axes[r, 3 * j], axes[r, 3 * j + 1], axes[r, 3 * j + 2]
            ax_in.imshow(im)
            ax_pca.imshow(pca)
            ax_ov.imshow(im)
            ax_ov.imshow(pca, alpha=args.alpha)
            if r == 0:
                for ax, t in zip((ax_in, ax_pca, ax_ov), col_titles):
                    ax.set_title(t, fontsize=9, pad=3)
        ylab = label if not args.with_baseline else f"{label}\n{method}"
        axes[r, 0].set_ylabel(ylab, fontsize=8.5, labelpad=4)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("#888")
    # thin separators between image triplets
    for j in range(1, n_img):
        for r in range(len(rows)):
            axes[r, 3 * j].set_position(_shift(axes[r, 3 * j].get_position(), 0.012))
            axes[r, 3 * j + 1].set_position(_shift(axes[r, 3 * j + 1].get_position(), 0.012))
            axes[r, 3 * j + 2].set_position(_shift(axes[r, 3 * j + 2].get_position(), 0.012))
    os.makedirs(out.parent, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}.pdf / .png ({len(rows)} rows x {ncol} cols)")


def _shift(bbox, dx):
    from matplotlib.transforms import Bbox
    return Bbox.from_bounds(bbox.x0 + dx, bbox.y0, bbox.width, bbox.height)


if __name__ == "__main__":
    main()
