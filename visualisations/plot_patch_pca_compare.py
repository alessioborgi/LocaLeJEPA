"""Side-by-side group-scope patch-PCA overlays across methods and backbones.

Reads the npz/png outputs written by src/visualize_patch_pca.py and stacks one
row per (backbone, method) on the same input crops. Usage:

    python visualisations/plot_patch_pca_compare.py \
        --rows "ViT-S/8 · LocaLeJEPA:logs/patch_pca_in1k/group/vit_localballreg_l01_k4" \
               "ViT-S/8 · LeJEPA:logs/patch_pca_in1k/group/vit_sigreg" \
        --out logs/patch_pca_in1k/compare/compare
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUP_NAMES = {'00': 'golden retriever (0207)', '01': 'English springer (0217)', '02': 'tabby (0281)'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows', nargs='+', required=True, help='"label:dir" pairs; dir holds group_XX.npz + inputs')
    ap.add_argument('--groups', nargs='+', default=list(GROUP_NAMES))
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--title', default='ImageNet-1K val · group-scope patch PCA → RGB · 100-ep checkpoints')
    ap.add_argument('--out', type=Path, required=True, help='output prefix; _group_XX.{png,pdf} is appended')
    args = ap.parse_args()
    rows = [(r.split(':', 1)[0], Path(r.split(':', 1)[1])) for r in args.rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for g in args.groups:
        first = rows[0][1]
        inputs = []
        while (first / f'group_{g}_input_{len(inputs)}.png').exists():
            inputs.append(Image.open(first / f'group_{g}_input_{len(inputs)}.png').convert('RGB'))
        n = len(inputs)
        fig, axes = plt.subplots(1 + len(rows), n, figsize=(2.6 * n, 2.6 * (1 + len(rows))), squeeze=False, layout='constrained')
        for i, im in enumerate(inputs):
            axes[0, i].imshow(im)
        axes[0, 0].set_ylabel('Input crop', fontsize=9)
        for r, (label, d) in enumerate(rows, 1):
            arc = np.load(d / f'group_{g}.npz'); rgb, grid = arc['rgb'], tuple(int(v) for v in arc['grid'])
            for i, im in enumerate(inputs):
                # Inputs may differ in resolution across rows (e.g. crop 256 CNN maps): display on the first row's crops.
                axes[r, i].imshow(im)
                ov = Image.fromarray(np.uint8(255 * rgb[i].reshape(*grid, 3))).resize(im.size, Image.Resampling.NEAREST)
                axes[r, i].imshow(np.asarray(ov) / 255, alpha=args.alpha)
            axes[r, 0].set_ylabel(f'{label}\n{grid[0]}×{grid[1]}', fontsize=8)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{args.title} · {GROUP_NAMES.get(g, g)}', fontsize=10)
        for ext in ('png', 'pdf'):
            fig.savefig(f'{args.out}_group_{g}.{ext}', dpi=160)
        plt.close(fig)
        print(f'wrote {args.out}_group_{g}.png', flush=True)


if __name__ == '__main__':
    main()
