import numpy as np
import pytest
import torch
from src.eval.patch_pca import pca_colors
from src.eval.extract import spatial_patch_tokens


def test_shared_pca_identical_tokens_have_identical_colors():
    rng=np.random.default_rng(7)
    f=rng.normal(size=(2,16,6));f[1,0]=f[0,0]
    rgb,_,mask,records=pca_colors(f,'group')
    np.testing.assert_allclose(rgb[0,0],rgb[1,0])
    assert mask.all() and len(records)==1
    assert np.isfinite(rgb).all() and rgb.min()>=0 and rgb.max()<=1


def test_independent_pca_is_independent_of_other_images():
    rng=np.random.default_rng(2);f=rng.normal(size=(2,16,6))
    a=pca_colors(f,'image')[0][0]
    f[1]*=100
    np.testing.assert_allclose(a,pca_colors(f,'image')[0][0])


def test_mask_refit_and_degenerate_rejection():
    f=np.random.default_rng(3).normal(size=(2,64,8))
    rgb,pc1,mask,_=pca_colors(f,'group',.5,'high')
    assert mask.any() and (~mask).any()
    assert (rgb[~mask]==0).all()
    assert (pc1[mask]>=.5).all()
    with pytest.raises(ValueError,match='nonconstant'):
        pca_colors(np.ones((1,16,4)))


def test_spatial_tokens_exclude_cls_and_head():
    from timm.models.vision_transformer import VisionTransformer
    model=VisionTransformer(img_size=32,patch_size=8,embed_dim=24,depth=1,num_heads=3,num_classes=512).eval()
    x=torch.randn(2,3,32,32)
    with torch.no_grad():
        expected=model.forward_features(x)[:,1:]
        actual=spatial_patch_tokens(model,x)
    torch.testing.assert_close(actual,expected)
    assert actual.shape==(2,16,24)
