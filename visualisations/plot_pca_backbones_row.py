"""One row per image: input, then PCA -> RGB and overlay for each backbone.

Columns are: input image, then for every backbone a PCA map and its overlay on
the input. The PCA is fit jointly over all the images of a row set (group
scope) per backbone, so colours are comparable between rows. Each principal
component is sign-oriented so the centre of the image scores high, since PCA
signs are arbitrary. No figure title; only the backbone name over the PCA
column and "Overlay" over the overlay column.

    python visualisations/plot_pca_backbones_row.py \
        --images a.JPEG b.JPEG --out paper/figs/pca_backbones

Features are cached next to the output as an NPZ, so re-running only to change
the styling skips the (slow, escnn-bound) extraction with --from-cache.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.eval.patch_pca import pca_colors

# label -> (hydra experiment, checkpoint, input resolution)
# The three backbones trained with LocaLeJEPA (local BallReg, lambda 0.1, k=4)
# on ImageNet-1K for 100 epochs. All were pretrained at crop 128; ViT position
# embeddings are resampled to the larger grid, and the D4-ResNet (output stride
# 32) needs a larger input to give a usable patch grid at all.
BACKBONES = {
    'Euclidean': ('lejepa/vit_in1k_localballreg_l01_k4',
                  'checkpoints/lejepa_vit_in1k_localballreg_l01_k4/last.ckpt', 256),
    'Hyperbolic': ('lejepa/hvit_in1k_localballreg_l01_k4',
                   'checkpoints/lejepa_hvit_in1k_localballreg_l01_k4/last.ckpt', 256),
    'Equivariant': ('lejepa/d4rn50_in1k_localballreg_l01_k4',
                    'checkpoints/lejepa_d4rn50_in1k_localballreg_l01_k4/last.ckpt', 512),
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
RESAMPLE = {'nearest': Image.Resampling.NEAREST, 'bilinear': Image.Resampling.BILINEAR,
            'bicubic': Image.Resampling.BICUBIC}


def orient_center(rgb, grid):
    """Flip each PC over the whole image set so the central half scores high."""
    maps = rgb.reshape(len(rgb), *grid, 3)
    h, w = grid
    centre = maps[:, h // 4: h - h // 4, w // 4: w - w // 4]
    flip = centre.mean(axis=(0, 1, 2)) < maps.mean(axis=(0, 1, 2))
    return np.where(flip, 1 - rgb, rgb)


def upsample(values, size, interp):
    arr = np.asarray(values, dtype=np.float32)
    mode = 'RGB' if arr.ndim == 3 else 'L'
    im = Image.fromarray(np.uint8(255 * np.clip(arr, 0, 1)), mode).resize(size, RESAMPLE[interp])
    return np.clip(np.asarray(im, dtype=np.float32) / 255, 0, 1)


def extract(paths, labels, device, mask, side):
    """Run each backbone over every image and return {label: (rgb, grid)}."""
    import torch
    from torchvision import transforms
    from src.eval.extract import load_backbone

    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    out = {}
    for label in labels:
        experiment, checkpoint, size = BACKBONES[label]
        crop = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size)])
        extract_fn, _ = load_backbone(experiment, checkpoint, device=torch.device(device),
                                      crop_size=size, feature='patch')
        tensors = []
        for path in paths:
            with Image.open(path) as im:
                im = crop(ImageOps.exif_transpose(im).convert('RGB'))
            tensors.append(extract_fn(normalize(im).unsqueeze(0)))
        features = torch.cat(tensors).numpy()
        grid = int(round(features.shape[1] ** 0.5))
        if grid * grid != features.shape[1]:
            raise SystemExit(f'{label}: non-square patch grid ({features.shape[1]} tokens)')
        chosen = side
        if mask is not None and side == 'auto':
            # Which end of PC1 is the object is arbitrary and differs per backbone:
            # take the side the centre of the image sits on.
            _, pc1, _, _ = pca_colors(features, 'group')
            m = pc1.reshape(len(pc1), grid, grid)
            chosen = 'high' if m[:, grid // 4: -(grid // 4), grid // 4: -(grid // 4)].mean() > m.mean() else 'low'
        elif chosen == 'auto':
            chosen = 'high'
        rgb, _, _, records = pca_colors(features, 'group', mask, chosen)
        out[label] = (orient_center(rgb, (grid, grid)), (grid, grid), records)
        print(f'{label}: {features.shape[0]} images, {grid}x{grid} grid, '
              f'{features.shape[-1]} channels'
              + (f', mask {mask} on the {chosen} side of PC1' if mask is not None else ''), flush=True)
        del extract_fn
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return out


def render(paths, maps, labels, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    crops = []
    for path in paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
        # Same field of view as the network input: resize short side, centre crop.
        crops.append(ImageOps.fit(im, (args.display_size, args.display_size),
                                  Image.Resampling.LANCZOS))
    if args.layout == 'image-rows':
        # One row per image: input, then (PCA, overlay) for each backbone.
        rows, cols = len(paths), 1 + 2 * len(labels)
        headers = [''] + [t for label in labels for t in (label, 'Overlay')]
    else:
        # One row per backbone: for each image (input, PCA, overlay).
        rows, cols = len(labels), 3 * len(paths)
        headers = ['Input', 'PCA', 'Overlay'] * len(paths)
    fig, axes = plt.subplots(rows, cols, figsize=(1.9 * cols, 1.9 * rows), squeeze=False,
                             layout='constrained')
    for i, crop in enumerate(crops):
        for b, label in enumerate(labels):
            rgb, grid, _ = maps[label]
            colour = upsample(rgb[i].reshape(*grid, 3), crop.size, args.interp)
            if args.layout == 'image-rows':
                raw, pca, over = axes[i, 0], axes[i, 1 + 2 * b], axes[i, 2 + 2 * b]
            else:
                raw, pca, over = axes[b, 3 * i], axes[b, 3 * i + 1], axes[b, 3 * i + 2]
            raw.imshow(crop)
            pca.imshow(colour, interpolation='nearest')
            over.imshow(crop); over.imshow(colour, alpha=args.alpha)
    if not args.no_labels:
        for c, name in enumerate(headers):
            axes[0, c].set_title(name, fontsize=11)
        if args.layout == 'backbone-rows':
            for r, label in enumerate(labels):
                axes[r, 0].set_ylabel(label, fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(args.out.with_suffix('.' + ext), dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {args.out}.png and .pdf', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', nargs='+', required=True, help='one image per row')
    ap.add_argument('--backbones', nargs='+', default=list(BACKBONES), choices=list(BACKBONES))
    ap.add_argument('--out', type=Path, required=True, help='output prefix (.png/.pdf/.npz)')
    ap.add_argument('--layout', choices=['image-rows', 'backbone-rows'], default='image-rows',
                    help="'image-rows': one row per image, columns input + (PCA, overlay) per backbone; "
                         "'backbone-rows': one row per backbone, columns (input, PCA, overlay) per image")
    ap.add_argument('--no-labels', action='store_true', help='draw only the images, with no text at all')
    ap.add_argument('--display-size', type=int, default=512)
    ap.add_argument('--interp', choices=list(RESAMPLE), default='bicubic')
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--mask', type=float, help='optional PC1 threshold in [0,1]; background is blacked out and the PCA refit on the foreground')
    ap.add_argument('--foreground', choices=['auto', 'high', 'low'], default='auto',
                    help="which side of PC1 is kept as foreground; 'auto' picks the side the image centre is on, per backbone")
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--from-cache', action='store_true', help='reuse the NPZ written by an earlier run and only redraw')
    ap.add_argument('--cache', type=Path, help='explicit NPZ path (default: alongside --out), so several figures can share one extraction')
    args = ap.parse_args()

    paths = [Path(p) for p in args.images]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit('Missing image files:\n' + '\n'.join(missing))
    cache = args.cache or args.out.with_suffix('.npz')
    if args.from_cache:
        with np.load(cache, allow_pickle=False) as arc:
            maps = {label: (arc[f'{label}_rgb'], tuple(arc[f'{label}_grid']), None)
                    for label in args.backbones}
    else:
        maps = extract(paths, args.backbones, args.device, args.mask, args.foreground)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **{f'{k}_rgb': v[0] for k, v in maps.items()},
                            **{f'{k}_grid': np.array(v[1]) for k, v in maps.items()})
        args.out.with_suffix('.json').write_text(json.dumps({
            'images': [str(p.resolve()) for p in paths],
            'backbones': {label: dict(zip(('experiment', 'checkpoint', 'crop_size'), BACKBONES[label]))
                          for label in args.backbones},
            'normalization': {'mean': MEAN, 'std': STD},
            'scope': 'group', 'mask': args.mask, 'foreground': args.foreground,
            'interp': args.interp, 'pca': {k: v[2] for k, v in maps.items()},
        }, indent=2))
    render(paths, maps, args.backbones, args)


if __name__ == '__main__':
    main()
