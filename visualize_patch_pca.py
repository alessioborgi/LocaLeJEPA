"""LeJEPA Fig.14 / DINOv2 Fig.1-style PCA maps from trained spatial ViT tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
import yaml
from src.eval.patch_pca import pca_colors


def select_images(args):
    # Imagenette archives normally use WordNet synset IDs as directory names.
    # Accept the human-readable names used in the paper/readme as well.
    aliases = {
        'tench': 'n01440764', 'English springer': 'n02102040',
        'cassette player': 'n02979186', 'chain saw': 'n03000684',
        'church': 'n03028079', 'French horn': 'n03394916',
        'garbage truck': 'n03417042', 'gas pump': 'n03425413',
        'golf ball': 'n03445777', 'parachute': 'n03888257',
    }
    if args.images:
        groups={'selected':[Path(p).resolve() for p in args.images]}
    else:
        root=Path(args.image_root)
        if not root.is_dir():raise ValueError(f'Image directory missing: {root}')
        classes=sorted(p for p in root.iterdir() if p.is_dir())
        if not classes:raise ValueError('Expected class subdirectories, or use --images FILE ...')
        rng=np.random.default_rng(args.seed)
        if args.classes:
            wanted=set(args.classes)
            resolved={aliases.get(name, name) for name in wanted}
            classes=[p for p in classes if p.name in resolved]
            found={p.name for p in classes}
            if found!=resolved:
                available=', '.join(p.name for p in sorted(Path(args.image_root).iterdir()) if p.is_dir())
                raise ValueError('Requested image classes not found. Use human names or synset IDs. '
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


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/patch_pca_imagenette.yaml')
    ap.add_argument('--only',help='Comma-separated model names')
    ap.add_argument('--images',nargs='+',help='Explicit related images; one shared group')
    ap.add_argument('--image-root',default='data/imagenette/imagenette2-160/val')
    ap.add_argument('--classes',nargs='+',help='Optional class folder names')
    ap.add_argument('--per-class',type=int,default=3)
    ap.add_argument('--scope',choices=['image','group'],default='image')
    ap.add_argument('--pc1-threshold',type=float,help='Optional [0,1] initial PC1 foreground threshold; background becomes black')
    ap.add_argument('--foreground',choices=['high','low'],default='high')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out',type=Path,default=Path('logs/patch_pca'))
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    try:
        if args.per_class<1:raise ValueError('--per-class must be positive')
        if args.pc1_threshold is not None and not 0<=args.pc1_threshold<=1:raise ValueError('PC1 threshold must be in [0,1]')
        if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:raise ValueError('Output exists; choose a fresh --out or --overwrite')
        cfg=yaml.safe_load(Path(args.config).read_text());entries=cfg['backbones']
        if args.only:
            names=set(args.only.split(','))
            if names-{e['name'] for e in entries}:raise ValueError('Unknown --only model names')
            entries=[e for e in entries if e['name'] in names]
        missing=[e['checkpoint'] for e in entries if not Path(e['checkpoint']).is_file()]
        if missing:raise ValueError('Run on diufrd with trained checkpoints available:\n'+'\n'.join(missing))
        groups=select_images(args)
        from src.eval.extract import load_backbone
        from torchvision import transforms
        size=cfg.get('crop_size',128)
        # Same resize + center crop for displayed images and network tensors.
        crop=transforms.Compose([transforms.Resize(size),transforms.CenterCrop(size)])
        normalize=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        args.out.mkdir(parents=True,exist_ok=True)
        manifest=dict(options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},crop_size=size,groups={},models=[],feature='post-final-norm transformer patch tokens; excludes prefix tokens, pooling, output head and SSL projector',sources=['https://arxiv.org/pdf/2304.07193','https://arxiv.org/pdf/2511.08544'])
        for name,paths in groups.items():
            manifest['groups'][name]=[]
            for p in paths:
                with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                manifest['groups'][name].append(dict(path=str(p),sha256=digest))
        for entry in entries:
            extract,dim=load_backbone(entry['experiment'],entry['checkpoint'],device=torch.device(args.device),crop_size=size,feature='patch')
            model_dir=args.out/entry['name'];model_dir.mkdir(exist_ok=True)
            for group_index,(name,paths) in enumerate(groups.items()):
                originals=[]
                for p in paths:
                    with Image.open(p) as im:originals.append(crop(ImageOps.exif_transpose(im).convert('RGB')))
                features=torch.cat([extract(normalize(im).unsqueeze(0)) for im in originals]).numpy()
                side=int(np.sqrt(features.shape[1]));grid=(side,side)
                if side*side!=features.shape[1]:raise ValueError('Square patch grid required')
                rgb,pc1,mask,records=pca_colors(features,args.scope,args.pc1_threshold,args.foreground)
                prefix=model_dir/f'group_{group_index:02d}'
                np.savez_compressed(prefix.with_suffix('.npz'),features=features,rgb=rgb,pc1=pc1,mask=mask,grid=np.array(grid))
                prefix.with_suffix('.json').write_text(json.dumps(dict(group=name,pca=records),indent=2))
                for i,im in enumerate(originals):im.save(model_dir/f'group_{group_index:02d}_input_{i}.png')
                hyper='H-ViT body (before hyperbolic head)' if 'hvit' in entry['experiment'] else 'ViT body'
                title=f'{entry["name"]} · {name} · {hyper}\n{args.scope} PCA; {side}×{side} patches; {dim} channels; mask={args.pc1_threshold}, side={args.foreground}'
                render(originals,rgb,pc1,mask,grid,prefix,title,[p.name for p in paths])
                print(f'Wrote {prefix}.png',flush=True)
            manifest['models'].append(dict(entry,token_dim=dim,checkpoint_size=Path(entry['checkpoint']).stat().st_size))
            del extract
            if args.device.startswith('cuda'):torch.cuda.empty_cache()
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    except ValueError as exc:ap.error(str(exc))


if __name__=='__main__':main()
