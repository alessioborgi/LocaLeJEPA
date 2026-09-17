# Spatial patch PCA: LeJEPA Fig.14 / DINOv2 Fig.1-style visualizations

Sources: [LeJEPA Figure 14](https://arxiv.org/pdf/2511.08544) and
[DINOv2 Figure 1](https://arxiv.org/pdf/2304.07193).

These are patch-feature RGB maps. They are neither attention weights nor
projections of one pooled feature per image. Our ViT-S/8 at 128px yields a
16x16 spatial grid of 384-D post-final-normalization transformer features.
CLS/prefix tokens, the 512-D output head, and SSL projector are excluded.
For H-ViT this explicitly visualizes its **Euclidean transformer body before
the hyperbolic head**; these are not hyperbolic patch embeddings.

## First run: on diufrd

Copy/update the code on the cluster. Existing Imagenette checkpoints and RGB
validation images are used; Yoda data and its uncertain normalization are not
involved. Nothing is retrained. Confirm the checkpoint paths in
`configs/patch_pca_imagenette.yaml` correspond to your intended runs.

```bash
mkdir -p logs/embedor
sbatch src/slurm_files/slurm_diufrd_patch_pca.sh \
  --only LocaLe_K4 --scope image --out logs/patch_pca_locale_k4
```

Default image root is `data/imagenette/imagenette2-160/val`. The script selects
three class folders and three images per class reproducibly with `--seed 0`.
A shared selection is used across all requested models. The manifest records
exact paths and hashes. No selection based on how attractive the output looks.
Choose other folders explicitly with `--classes CLASS_FOLDER ...`, or supply
related ordinary RGB image files with `--images image1.jpg image2.jpg image3.jpg`.
No image downloads are required.

`--scope image` fits a separate three-component PCA for each image, as stated
in LeJEPA Figure 14. Components are mapped to RGB using per-component min/max
normalization within the PCA fitting set. This color scaling and deterministic
sign convention are our documented implementation choices, not a claim to
reproduce unpublished author code exactly.

## Shared PCA across related images

```bash
sbatch src/slurm_files/slurm_diufrd_patch_pca.sh \
  --only LocaLe_K4 --scope group --out logs/patch_pca_locale_k4_shared
```

Each selected class is a separate group. Explicit `--images` form one group.
A single PCA and RGB scale are fit across that group's patches, independently
for each model. This makes matching colors interpretable across images **within
that group and model**, as in the shared-PCA approach of DINOv2 Figure 1.

Optional foreground masking and foreground-only PCA refit:

```bash
sbatch src/slurm_files/slurm_diufrd_patch_pca.sh \
  --only LocaLe_K4 --scope group --pc1-threshold 0.5 --foreground high \
  --out logs/patch_pca_locale_k4_masked
```

The mask thresholds PC1 after min/max normalization to [0,1] over the PCA group.
PC1 sign is fixed by its largest absolute loading, **not** by semantics. Inspect
the unmasked PC1 row first; `--foreground low` reverses the side. The threshold
and chosen side must be reported. This heuristic can select background instead
of foreground; it is not an evaluated segmentation mask. Do not tune signs or
thresholds per method to make one method look better. All-patch unmasked PCA
should remain the primary comparison. Masking is off by default.

## Compare losses

```bash
sbatch src/slurm_files/slurm_diufrd_patch_pca.sh \
  --only LocaLe_K4,SIGReg,RDMReg --scope image --out logs/patch_pca_comparison
```

The same crops and resolution are used for every model. Each model gets its
own PCA basis; matching literal hues **across models** is not meaningful.
Compare spatial coherence, boundaries and foreground/background differentiation.
Independent per-image PCA likewise cannot establish cross-image color identity.
PCA colors may change under sign flips or nearly degenerate component rotations.
These figures are qualitative evidence of spatial organization, not proof of
segmentation quality, intrinsic geometry preservation, or superiority.

LeJEPA's reference uses a larger model pretrained on ImageNet-1K; DINOv2 uses
a different training regime. This reproduces the visualization method on your
models, not their expected visual performance or exact example images.
At 128px, maps have only 16x16 cells. Display uses nearest-neighbor expansion,
not smoothing that suggests finer spatial detail. Higher-resolution extraction
requires explicit positional-embedding handling and is not silently enabled.

## Outputs

Per model/group: PNG panel (input crop, PCA RGB, initial PC1, RGB overlay),
input crop PNGs, NPZ with raw patch features/RGB/mask/grid, and JSON containing
PCA bases, means, scales and explained variance. The run manifest stores
checkpoint paths, selected images/hashes, settings and feature-layer semantics.
Existing output directories require explicit `--overwrite`.

Copy the output folder back using your usual rsync workflow. The PNGs are
ready to inspect locally. Checkpoints are not needed to view the results.

## Verification

```bash
python -m pytest -q tests/test_patch_pca.py
```

Tests cover shared color correspondence, independent PCA isolation, masking,
degenerate features and exclusion of CLS tokens/output heads. A random-weight
checkpoint can validate the extraction/rendering plumbing but must never be
presented as a trained LocaLeJEPA result.
