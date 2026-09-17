"""PCA coloring of spatial ViT features; no labels or learned segmentation head."""
import numpy as np
from sklearn.decomposition import PCA


def pca_colors(features, scope='image', threshold=None, foreground='high'):
    """features: (images, patches, channels).

    image: independent RGB PCA per image (LeJEPA Fig.14-style).
    group: shared PCA across input images (DINOv2 Fig.1-style).
    Optional mask uses normalized PC1 of an initial PCA; refit RGB PCA on
    retained patches. Sign is deterministic by largest loading, NOT semantics.
    Threshold/sign must be reported; no automatic foreground correctness claim.
    """
    f=np.asarray(features,dtype=np.float64)
    if f.ndim!=3 or f.shape[-1]<3 or not np.isfinite(f).all():
        raise ValueError('Expected finite (images,patches,channels>=3) features')
    if scope not in ('image','group') or foreground not in ('high','low'):
        raise ValueError('Invalid PCA scope or foreground side')
    if threshold is not None and not 0<=threshold<=1:
        raise ValueError('PC1 threshold must be in [0,1]')
    rgb=np.zeros((*f.shape[:2],3));pc1=np.zeros(f.shape[:2]);mask=np.ones(f.shape[:2],bool)
    records=[]
    groups=[list(range(len(f)))] if scope=='group' else [[i] for i in range(len(f))]
    def fit(x):
        if len(x)<3 or np.max(np.ptp(x,axis=0))<1e-12:
            raise ValueError('PCA requires at least 3 patches with nonconstant features')
        p=PCA(n_components=3,svd_solver='full').fit(x)
        # Fix otherwise arbitrary component signs for reproducibility.
        signs=np.sign(p.components_[np.arange(3),np.argmax(abs(p.components_),axis=1)])
        p.components_*=np.where(signs==0,1,signs)[:,None]
        return p
    for group in groups:
        x=f[group].reshape(-1,f.shape[-1]);initial=fit(x);s=initial.transform(x)[:,0]
        span=np.ptp(s);normalized=(s-s.min())/span if span>0 else np.zeros_like(s)
        keep=np.ones(len(x),bool) if threshold is None else (normalized>=threshold if foreground=='high' else normalized<=threshold)
        p=initial if threshold is None else fit(x[keep])
        scores=p.transform(x);lo=scores[keep].min(0);hi=scores[keep].max(0)
        color=np.divide(scores-lo,hi-lo,out=np.full_like(scores,.5),where=(hi-lo)>1e-12)
        color=np.clip(color,0,1);color[~keep]=0
        rgb[group]=color.reshape(len(group),f.shape[1],3)
        pc1[group]=normalized.reshape(len(group),f.shape[1]);mask[group]=keep.reshape(len(group),f.shape[1])
        records.append(dict(image_indices=group,explained_variance_ratio=p.explained_variance_ratio_.tolist(),foreground_fraction=float(keep.mean()),rgb_min=lo.tolist(),rgb_max=hi.tolist(),components=p.components_.tolist(),mean=p.mean_.tolist(),mask_pc1_components=initial.components_[0].tolist(),mask_pc1_mean=initial.mean_.tolist()))
    return rgb,pc1,mask,records
