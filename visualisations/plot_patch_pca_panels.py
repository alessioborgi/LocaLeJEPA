"""Method-comparison panels from src/visualize_patch_pca.py outputs.

Columns are backbones/methods, rows are: input crop, PCA -> RGB, PC1, PC2, PC3
(the three normalised principal-component scores that make up the RGB map) and
the RGB overlay. Two presets per backbone family:

  variants : the LocaLeJEPA variants (local BallReg k2 / k4 / lambda 0.3 k2 / dyn-k,
             local SIGReg k4)
  methods  : RDMReg, LeJEPA (SIGReg), best LocaLeJEPA (BallReg lambda 0.1 k4), LocaLeJEPA dyn-k

Maps are upsampled with a bicubic filter (smooth contours) onto the display crop
saved by the extraction run. Example:

  python visualisations/plot_patch_pca_panels.py --run-dir logs/patch_pca_in1k_256/group \
      --family hvit --figure methods --group 01 --image 1 --out paper/figs/patch_pca
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUP_NAMES = {'00': 'golden retriever', '01': 'English springer', '02': 'tabby cat'}
FAMILY = {'vit': 'ViT-S/8', 'hvit': 'H-ViT-S/8', 'd4rn50': 'D4-ResNet-50'}
# column label -> backbone name suffix (prefixed with the family)
PRESETS = {
    'variants': [
        ('BallReg\nλ=0.1, k=2', 'localballreg_l01_k2'),
        ('BallReg\nλ=0.1, k=4', 'localballreg_l01_k4'),
        ('BallReg\nλ=0.3, k=2', 'localballreg_l03_k2'),
        ('BallReg\nλ=0.1, dyn-k', 'localballreg_l01_dynk'),
        ('SIGReg (local)\nλ=0.1, k=4', 'localsigreg_l01_k4'),
    ],
    'methods': [
        ('Rectified LpJEPA\n(RDMReg)', 'rdmreg'),
        ('LeJEPA\n(SIGReg)', 'sigreg'),
        ('LocaLeJEPA\n(BallReg λ=0.1, k=4)', 'localballreg_l01_k4'),
        ('LocaLeJEPA\n(BallReg λ=0.1, dyn-k)', 'localballreg_l01_dynk'),
    ],
}
ROWS = ['Input', 'PCA → RGB', 'PC1', 'PC2', 'PC3', 'Overlay']
RESAMPLE = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR, 'bicubic': Image.Resampling.BICUBIC}


def upsample(values, size, interp):
    arr = np.asarray(values, dtype=np.float32)
    im = Image.fromarray(np.uint8(255 * np.clip(arr, 0, 1)), mode='RGB' if arr.ndim == 3 else 'L').resize(size, RESAMPLE[interp])
    return np.clip(np.asarray(im, dtype=np.float32) / 255, 0, 1)


def orient_center(rgb_all, grid):
    """Flip each PC (for the whole group, so images stay comparable) so the central
    half of the grid scores higher than the border: PCA signs are arbitrary."""
    maps = rgb_all.reshape(len(rgb_all), *grid, 3)
    h, w = grid; c = maps[:, h // 4: h - h // 4, w // 4: w - w // 4]
    flip = c.mean(axis=(0, 1, 2)) < maps.mean(axis=(0, 1, 2))
    return np.where(flip, 1 - rgb_all, rgb_all)


def load_column(run_dir, name, group, image, orient=True):
    d = run_dir / name
    arc = np.load(d / f'group_{group}.npz')
    grid = tuple(int(v) for v in arc['grid'])
    rgb_all = orient_center(arc['rgb'], grid) if orient else arc['rgb']
    rgb = rgb_all[image].reshape(*grid, 3)
    with Image.open(d / f'group_{group}_input_{image}.png') as im:
        crop = im.convert('RGB').copy()
    meta = json.loads((d / f'group_{group}.json').read_text())
    evr = None
    for rec in meta['pca']:
        if image in rec['image_indices']:
            evr = rec['explained_variance_ratio']
    return crop, rgb, grid, evr


def build(run_dir, columns, group, image, interp, alpha, title, out, orient=True):
    cols = [(label, name) for label, name in columns if (run_dir / name / f'group_{group}.npz').exists()]
    skipped = [name for _, name in columns if (run_dir / name / f'group_{group}.npz').exists() is False]
    if not cols:
        raise SystemExit(f'No columns found under {run_dir} for {[n for _, n in columns]}')
    n = len(cols)
    fig, axes = plt.subplots(len(ROWS), n, figsize=(2.4 * n, 2.4 * len(ROWS) + 0.4), squeeze=False, layout='constrained')
    for c, (label, name) in enumerate(cols):
        crop, rgb, grid, evr = load_column(run_dir, name, group, image, orient)
        maps = upsample(rgb, crop.size, interp)
        axes[0, c].imshow(crop)
        axes[1, c].imshow(maps, interpolation='nearest')
        for k in range(3):
            axes[2 + k, c].imshow(maps[..., k], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[5, c].imshow(crop); axes[5, c].imshow(maps, alpha=alpha)
        head = f'{label}\n{grid[0]}×{grid[1]}'
        if evr: head += f' · var {100 * sum(evr):.0f}%'
        axes[0, c].set_title(head, fontsize=8)
    for r, label in enumerate(ROWS):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(out.with_suffix('.' + ext), dpi=200)
    plt.close(fig)
    print(f'wrote {out}.png' + (f' (missing: {skipped})' if skipped else ''), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, required=True, help='e.g. logs/patch_pca_in1k_256/group')
    ap.add_argument('--family', choices=list(FAMILY), required=True)
    ap.add_argument('--figure', choices=list(PRESETS), required=True)
    ap.add_argument('--group', default='01', help='group index written by the extraction (00 retriever, 01 springer, 02 tabby)')
    ap.add_argument('--image', type=int, default=0, help='index of the image within the group')
    ap.add_argument('--label', help='group label for the title (default: from the group index, or the manifest group name)')
    ap.add_argument('--interp', choices=list(RESAMPLE), default='bicubic')
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--no-orient', action='store_true', help='keep the raw PCA signs instead of orienting each PC so the image centre scores high')
    ap.add_argument('--out', type=Path, required=True, help='output prefix (family/figure/group/image appended)')
    args = ap.parse_args()
    columns = [(label, f'{args.family}_{suffix}') for label, suffix in PRESETS[args.figure]]
    manifest = json.loads((args.run_dir / 'manifest.json').read_text()) if (args.run_dir / 'manifest.json').exists() else {}
    scope = manifest.get('options', {}).get('scope', '?'); crop = manifest.get('crop_size', '?')
    group_label = args.label or GROUP_NAMES.get(args.group) or next(iter(manifest.get('groups', {'?': 0}).keys()))
    title = (f'{FAMILY[args.family]} · {group_label} · ImageNet-1K, 100-ep checkpoints\n'
             f'{scope}-scope PCA of spatial features · {crop}px input · {args.interp} upsampling'
             + ('' if args.no_orient else ' · PCs oriented: centre high'))
    out = Path(f'{args.out}_{args.family}_{args.figure}_g{args.group}_i{args.image}')
    build(args.run_dir, columns, args.group, args.image, args.interp, args.alpha, title, out, not args.no_orient)


if __name__ == '__main__':
    main()
