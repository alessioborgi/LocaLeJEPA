"""Spatial PCA RGB maps from generic features or supported ViT checkpoints."""
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import yaml
from src.eval.patch_pca import pca_colors, load_spatial_feature_archive


def visualize_feature_archive(args):
    """Render spatial features from any architecture, independently of input data."""
    if args.images or args.image_root or args.classes or args.only:
        raise ValueError('--features cannot be combined with checkpoint/image selection options')
    features, grid = load_spatial_feature_archive(args.features)
    rgb, pc1, mask, records = pca_colors(
        features, args.scope, args.pc1_threshold, args.foreground
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for i in range(len(features)):
        # Preserve the spatial grid without inventing an input-image overlay.
        Image.fromarray(np.uint8(255 * rgb[i].reshape(*grid, 3))).save(
            args.out / f'sample_{i:04d}_pca.png'
        )
        Image.fromarray(np.uint8(255 * pc1[i].reshape(grid))).save(
            args.out / f'sample_{i:04d}_pc1.png'
        )
    np.savez_compressed(args.out / 'pca.npz', rgb=rgb, pc1=pc1, mask=mask, grid=grid)
    with args.features.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    (args.out / 'manifest.json').write_text(json.dumps({
        'source': str(args.features.resolve()), 'sha256': digest,
        'feature_shape': list(features.shape), 'grid': grid, 'scope': args.scope,
        'threshold': args.pc1_threshold, 'foreground': args.foreground, 'pca': records,
    }, indent=2))
    print(f'Wrote PCA maps to {args.out}', flush=True)


def select_images(args, aliases=None):
    """Select arbitrary image files; aliases are optional dataset config data."""
    aliases = aliases or {}
    if args.images:
        groups={'selected':[Path(p).resolve() for p in args.images]}
    else:
        root=Path(args.image_root)
        if not root.is_dir():raise ValueError(f'Image directory missing: {root}')
        classes=sorted(p for p in root.iterdir() if p.is_dir())
        if not classes:
            paths=sorted(p for p in root.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp','.tif','.tiff','.bmp'))
            if args.classes:raise ValueError('--classes requires class subdirectories')
            if len(paths)<args.per_class:raise ValueError(f'Not enough images in {root}')
            rng=np.random.default_rng(args.seed)
            return {'selected':[paths[i].resolve() for i in sorted(rng.choice(len(paths),args.per_class,replace=False))]}
        rng=np.random.default_rng(args.seed)
        if args.classes:
            wanted=set(args.classes)
            resolved={aliases.get(name, name) for name in wanted}
            classes=[p for p in classes if p.name in resolved]
            found={p.name for p in classes}
            if found!=resolved:
                available=', '.join(p.name for p in sorted(Path(args.image_root).iterdir()) if p.is_dir())
                raise ValueError('Requested image classes not found. Use directory names or configured aliases. '
                                 f'Requested: {sorted(wanted)}; available directory names: {available}')
        else:
            classes=[classes[i] for i in sorted(rng.choice(len(classes),min(3,len(classes)),replace=False))]
        groups={}
        for folder in classes:
            paths=sorted(p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.webp'))
            if len(paths)<args.per_class:raise ValueError(f'Not enough images in {folder}')
            groups[folder.name]=[paths[i].resolve() for i in sorted(rng.choice(len(paths),args.per_class,replace=False))]
    if not all(p.is_file() for paths in groups.values() for p in paths):raise ValueError('An image file is missing')
    return groups


def render(originals,colors,pc1,masks,grid,out,title,labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n=len(originals)
    fig,axes=plt.subplots(4,n,figsize=(3*n,10),squeeze=False,layout='constrained')
    for i,im in enumerate(originals):
        axes[0,i].imshow(im);axes[0,i].set_title(labels[i],fontsize=9)
        axes[1,i].imshow(colors[i].reshape(*grid,3),interpolation='nearest')
        axes[2,i].imshow(pc1[i].reshape(grid),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        axes[3,i].imshow(im)
        overlay=colors[i].reshape(*grid,3)
        overlay=np.asarray(Image.fromarray(np.uint8(255*overlay)).resize(im.size,Image.Resampling.NEAREST))/255
        axes[3,i].imshow(overlay,alpha=.55)
        for ax in axes[:,i]:ax.set_xticks([]);ax.set_yticks([])
    for row,label in enumerate(['Input crop','PCA → RGB','Initial PC1','Overlay (55%)']):axes[row,0].set_ylabel(label)
    fig.suptitle(title,fontsize=12)
    fig.savefig(out.with_suffix('.png'),dpi=160)
    plt.close(fig)


def render_comparison(originals, panels, out, rgb_order=(2, 0, 1), render_size=512):
    """One input per row; model columns share a fixed display-only color mapping."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(originals), 1 + len(panels),
                             figsize=(9, 3 * len(originals)), squeeze=False)
    for row, original in enumerate(originals):
        axes[row, 0].imshow(original)
        for col, (label, colors, grid) in enumerate(panels, start=1):
            pixels = colors[row].reshape(*grid, 3)[..., list(rgb_order)]
            # Upsample float channels before quantization to avoid banding.
            pixels = np.stack([
                np.asarray(Image.fromarray(pixels[..., c].astype(np.float32)).resize(
                    (render_size, render_size), Image.Resampling.BILINEAR))
                for c in range(3)
            ], axis=-1)
            axes[row, col].imshow(np.clip(pixels, 0, 1), interpolation='bilinear')
            if row == 0:
                axes[row, col].set_title(label, fontsize=16, pad=10)
        for ax in axes[row]:
            ax.set_axis_off()
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.95,
                        wspace=.025, hspace=.025)
    fig.savefig(out, dpi=300, facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    inputs=ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--config',help='ViT checkpoint and image preprocessing YAML')
    inputs.add_argument('--features',type=Path,help='NPZ with features (N,P,D) and grid (H,W), from any backbone/data')
    ap.add_argument('--only',help='Comma-separated model names')
    ap.add_argument('--images',nargs='+',help='Explicit related images; one shared group')
    ap.add_argument('--image-root',help='Flat image directory or class folders; overrides config image_root')
    ap.add_argument('--classes',nargs='+',help='Optional class folder names')
    ap.add_argument('--per-class',type=int,default=3)
    ap.add_argument('--scope',choices=['image','group'],default='image')
    ap.add_argument('--pc1-threshold',type=float,help='Optional [0,1] initial PC1 foreground threshold; background becomes black')
    ap.add_argument('--foreground',choices=['high','low'],default='high')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out',type=Path,default=Path('logs/patch_pca'))
    ap.add_argument('--overwrite',action='store_true')
    ap.add_argument('--rgb-order', default=None, choices=['123','132','213','231','312','321'],
                    help='PCA components assigned to R,G,B in the comparison figure')
    ap.add_argument('--render-size', type=int, default=512,
                    help='Bilinear PCA display resolution; does not change feature extraction')
    args=ap.parse_args()
    try:
        if args.render_size<1:raise ValueError('--render-size must be positive')
        if args.per_class<1:raise ValueError('--per-class must be positive')
        if args.pc1_threshold is not None and not 0<=args.pc1_threshold<=1:raise ValueError('PC1 threshold must be in [0,1]')
        if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:raise ValueError('Output exists; choose a fresh --out or --overwrite')
        if args.features:
            visualize_feature_archive(args)
            return
        cfg=yaml.safe_load(Path(args.config).read_text());entries=cfg['backbones']
        args.image_root=args.image_root or cfg.get('image_root')
        if not args.images and args.classes is None:
            args.classes = cfg.get('classes')
        args.rgb_order = args.rgb_order or cfg.get('rgb_order', '312')
        if args.rgb_order not in ('123','132','213','231','312','321'):
            raise ValueError('rgb_order must be a permutation of 123')
        comparison = cfg.get('comparison', False)
        comparison_panels = {}
        if not args.images and not args.image_root:raise ValueError('Provide --images, --image-root, or config image_root')
        if args.only:
            names=set(args.only.split(','))
            if names-{e['name'] for e in entries}:raise ValueError('Unknown --only model names')
            entries=[e for e in entries if e['name'] in names]
        missing=[e['checkpoint'] for e in entries if not Path(e['checkpoint']).is_file()]
        if missing:raise ValueError('Checkpoint files are missing:\n'+'\n'.join(missing))
        groups=select_images(args,cfg.get('class_aliases'))
        if comparison and (len(entries) != 2 or any(len(paths) != 3 for paths in groups.values())):
            raise ValueError('Comparison requires two backbones and three images per group')
        from src.eval.extract import load_backbone
        from torchvision import transforms
        size=cfg.get('crop_size',128)
        # Same resize + center crop for displayed images and network tensors.
        crop=transforms.Compose([transforms.Resize(size),transforms.CenterCrop(size)])
        mode=cfg.get('image_mode','RGB')
        if mode not in ('RGB','L'):raise ValueError('image_mode must be RGB or L; use --features for other data modalities')
        norm=cfg.get('normalization')
        channels=3 if mode=='RGB' else 1
        if not isinstance(norm,dict) or len(norm.get('mean',[]))!=channels or len(norm.get('std',[]))!=channels:
            raise ValueError('Specify normalization mean/std for each input channel in the config')
        if not np.isfinite(norm['mean']).all() or not np.isfinite(norm['std']).all() or min(norm['std'])<=0:
            raise ValueError('Normalization must be finite with positive std')
        normalize=transforms.Compose([transforms.ToTensor(),transforms.Normalize(norm['mean'],norm['std'])])
        args.out.mkdir(parents=True,exist_ok=True)
        manifest=dict(comparison=comparison, display_interpolation='bilinear' if comparison else 'nearest', options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},crop_size=size,image_mode=mode,normalization=norm,groups={},models=[],feature='post-final-norm transformer patch tokens; excludes prefix tokens, pooling, output head and SSL projector',sources=['https://arxiv.org/pdf/2304.07193','https://arxiv.org/pdf/2511.08544'])
        for name,paths in groups.items():
            manifest['groups'][name]=[]
            for p in paths:
                with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                manifest['groups'][name].append(dict(path=str(p),sha256=digest))
        for entry in entries:
            extract,dim=load_backbone(entry['experiment'],entry['checkpoint'],device=torch.device(args.device),crop_size=size,feature='patch',overrides=entry.get('overrides'))
            model_dir=args.out/entry['name'];model_dir.mkdir(exist_ok=True)
            for group_index,(name,paths) in enumerate(groups.items()):
                originals=[]
                for p in paths:
                    with Image.open(p) as im:originals.append(crop(ImageOps.exif_transpose(im).convert(mode)))
                features=torch.cat([extract(normalize(im).unsqueeze(0)) for im in originals]).numpy()
                side=int(np.sqrt(features.shape[1]));grid=(side,side)
                if side*side!=features.shape[1]:raise ValueError('Square patch grid required')
                rgb,pc1,mask,records=pca_colors(features,args.scope,args.pc1_threshold,args.foreground)
                prefix=model_dir/f'group_{group_index:02d}'
                np.savez_compressed(prefix.with_suffix('.npz'),features=features,rgb=rgb,pc1=pc1,mask=mask,grid=np.array(grid))
                prefix.with_suffix('.json').write_text(json.dumps(dict(group=name,pca=records),indent=2))
                for i,im in enumerate(originals):im.save(model_dir/f'group_{group_index:02d}_input_{i}.png')
                is_hvit = 'hvit' in entry['experiment'] or any(
                    'model/encoder=hvit' in override for override in entry.get('overrides', [])
                )
                hyper='H-ViT body (before hyperbolic head)' if is_hvit else 'ViT body'
                title=f'{entry["name"]} · {name} · {hyper}\n{args.scope} PCA; {side}×{side} patches; {dim} channels; mask={args.pc1_threshold}, side={args.foreground}'
                if comparison:
                    comparison_panels.setdefault(group_index, []).append(
                        (entry.get('label', entry['name']), rgb, grid))
                    if len(comparison_panels[group_index]) == len(entries):
                        output = args.out / f'comparison_{group_index:02d}.png'
                        render_comparison(originals, comparison_panels[group_index], output,
                                          tuple(int(c)-1 for c in args.rgb_order), args.render_size)
                        print(f'Wrote {output}', flush=True)
                else:
                    render(originals,rgb,pc1,mask,grid,prefix,title,[p.name for p in paths])
                    print(f'Wrote {prefix}.png',flush=True)
            manifest['models'].append(dict(entry,token_dim=dim,checkpoint_size=Path(entry['checkpoint']).stat().st_size))
            del extract
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    except ValueError as exc:ap.error(str(exc))


if __name__=='__main__':main()
