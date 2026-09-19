# visualisations/

Visualization tools for frozen-backbone evaluation: Yoda rotation-angle probe
plots and spatial feature maps colored by the first three PCA components.
The Yoda plots use bobblehead evaluation results; spatial PCA can use features
from any dataset and backbone. None of these tools retrains a model.

Install the `experiments` extra for plotting and PCA dependencies. The tools
use two workflows, described below.

## Yoda probe plots

The scripts in this directory read source metrics/feature-caches from the repo's `logs/` tree and
write every figure (PNG **and** vector PDF) into [`figures/`](figures/). Paths are
anchored to the repo root, so scripts run from anywhere:

```bash
python visualisations/plot_yoda_torus.py            # or:  cd visualisations && python plot_yoda_torus.py
```

### Plotting scripts

| script | output(s) in `figures/` | what it shows |
|---|---|---|
| `plot_lowdata_curve.py` | `lowdata_mae`, `lowdata_r2` | learning curves: test MAE / R² vs # training frames, per backbone |
| `build_lowdata_artifact.py` | same PNGs + `yoda_lowdata.html` | polished curves + MAE/R² data tables as one self-contained HTML page |
| `plot_yoda_torus.py [backbone] [out]` | `yoda_torus` | the label space `(angle₀, angle₁)` as a 2-torus: ground truth + one backbone's probe reconstruction |
| `plot_yoda_torus_sweep.py [backbone] [out]` | `yoda_torus_sweep` | one backbone's torus reconstruction sharpening as training size grows (5 → full) |
| `plot_yoda_torus_compare.py [out]` | `yoda_torus_compare` | all backbones × training sizes, tori colored by absolute error |
| `plot_yoda_torus_advantage.py [out]` | `yoda_torus_advantage` | per-region error advantage of `localballreg_l0.1_k2` over every other backbone |

### Evaluation inputs

- `logs/eval_angle_probe_lowdata/results_lowdata.{json,csv}` — learning-curve metrics (produced by `src/eval_angle_probe_lowdata.py` at the repo root).
- `logs/eval_angle_probe_*/cache/*_features.pt` — cached frozen 512-d features + angle labels per backbone.

## Spatial feature PCA

Run `src/visualize_patch_pca.py` from the repository root to produce spatial
PCA maps. Unlike the Yoda plots, this command reads feature archives or trained
checkpoints and writes PNGs and metadata to the directory specified by `--out`
(default: `logs/patch_pca`).

The reusable `pca_colors` function in `src/eval/patch_pca.py` has no dataset or
architecture dependency.
It maps the first three PCA components to RGB for arrays shaped
`(samples, spatial_positions, channels)`. At least three channels and three
nonconstant spatial observations are needed for each PCA fit.

There are two input paths. Neither assumes the model was trained on Imagenette.

### Any backbone or data modality: precomputed features

Export spatial features from the layer you want to inspect. For example, for a
CNN tensor `x` with shape `(N,C,H,W)`:

```python
import numpy as np

N, C, H, W = x.shape
features = x.detach().float().cpu().permute(0, 2, 3, 1).numpy()
np.savez_compressed("features.npz",
                    features=features.reshape(N, H * W, C), grid=[H, W])
```

For a transformer, remove CLS/register tokens and save spatial tokens in raster
order as `(N,H*W,D)`. Supply the actual grid; rectangular grids are supported.
The input can come from any training dataset, checkpoint format or framework.
Feature extraction and the correct input preprocessing remain the exporter's
responsibility. A pooled vector per image has no spatial grid: use an embedding
scatter plot rather than this spatial-map visualization for that representation.

```bash
uv run --extra experiments python src/visualize_patch_pca.py \
  --features features.npz --scope group --out logs/my_feature_pca
```

This path writes one PCA RGB PNG and PC1 grayscale PNG per sample at the native
feature-grid resolution, plus `pca.npz` and a manifest with the source hash,
PCA bases, means, scaling and explained variance. It does not require ordinary
RGB images and does not invent image overlays for other data modalities.

### Direct checkpoint extraction: supported ViT architectures

Use [configs/eval/patch_pca_template.yaml](../configs/eval/patch_pca_template.yaml) as a template and replace the experiment,
checkpoint and normalization with your model's values. This path rebuilds a
backbone from this repo's Hydra config and loads `encoder.*` Lightning weights.
It currently supports timm ViTs and HyperbolicViT, not arbitrary backbone APIs.
The architecture-specific helper is explicitly named `vit_spatial_patch_tokens`.
H-ViT maps come from its Euclidean transformer body, before its hyperbolic head.

```bash
uv run --extra experiments python src/visualize_patch_pca.py \
  --config configs/eval/my_patch_pca.yaml \
  --images /path/to/a.png /path/to/b.png \
  --scope group --out logs/my_patch_pca
```

Alternatively use `--image-root` with either a flat folder or class subfolders.
`--classes` uses literal folder names. Dataset-specific name aliases may be
provided through `class_aliases` in the config. They are never applied implicitly.
`--per-class` sets the sample count per folder (also for a flat folder).

The image pipeline resizes and center-crops to `crop_size`, converts to `RGB` or
`L` as specified by `image_mode`, converts pixels to [0,1], then applies the
explicit `normalization.mean/std`. These must match the trained model. For other
modalities or preprocessing pipelines, use the precomputed-feature path.
ViT positional embeddings are resampled when patch extraction changes resolution;
ordinary pooled evaluation remains strict. Direct ViT maps currently require a
square spatial grid. Outputs include input crops, PCA/PC1/overlay panels, features
and PCA metadata.

### Interpreting PCA colors

`--scope image` fits PCA separately for each sample. `--scope group` shares the
basis and color scale across the group (the entire archive with `--features`).
Only shared PCA makes colors comparable within that group; colors are not
comparable between independently fitted models. Signs are fixed by the largest
component loading, not semantic foreground. Optional `--pc1-threshold 0.5
--foreground high` masks using normalized initial PC1 and refits on retained
features; it is a heuristic, not a segmentation model. Masking is off by default.
