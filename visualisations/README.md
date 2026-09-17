# visualisations/

Plotting scripts and rendered figures for the Yoda rotation-angle probe
(frozen-backbone linear/ridge eval on the bobblehead dataset).

The scripts read source metrics/feature-caches from the repo's `logs/` tree and
write every figure (PNG **and** vector PDF) into [`figures/`](figures/). Paths are
anchored to the repo root, so scripts run from anywhere:

```bash
python visualisations/plot_yoda_torus.py            # or:  cd visualisations && python plot_yoda_torus.py
```

## Scripts

| script | output(s) in `figures/` | what it shows |
|---|---|---|
| `plot_lowdata_curve.py` | `lowdata_mae`, `lowdata_r2` | learning curves: test MAE / R² vs # training frames, per backbone |
| `build_lowdata_artifact.py` | same PNGs + `yoda_lowdata.html` | polished curves + MAE/R² data tables as one self-contained HTML page |
| `plot_yoda_torus.py [backbone] [out]` | `yoda_torus` | the label space `(angle₀, angle₁)` as a 2-torus: ground truth + one backbone's probe reconstruction |
| `plot_yoda_torus_sweep.py [backbone] [out]` | `yoda_torus_sweep` | one backbone's torus reconstruction sharpening as training size grows (5 → full) |
| `plot_yoda_torus_compare.py [out]` | `yoda_torus_compare` | all backbones × training sizes, tori colored by absolute error |
| `plot_yoda_torus_advantage.py [out]` | `yoda_torus_advantage` | per-region error advantage of `localballreg_l0.1_k2` over every other backbone |

## Embedding-structure study (Imagenette ViT-S/8)

Evaluates and visualises the clusters behind the SPHERE-JEPA Table-4 numbers for
the LeJEPA / RectLpJEPA / BallJEPA backbones. Three stages, only the first needs
a GPU:

```bash
# 1. cluster (GPU): Table 4 + frozen train/val encoder feature caches
sbatch src/slurm_files/slurm_diufrd_eval_indomain_ksweep.sh
#    -> logs/eval_indomain_imagenette_ksweep/{results.csv,cache/}

# 2. laptop (CPU, minutes): layouts + cluster metrics from those caches
python eval_structure.py

# 3. laptop (seconds): figures + markdown table
python visualisations/plot_embedding_structure.py
```

| script | output(s) in `figures/` | what it shows |
|---|---|---|
| `plot_embedding_structure.py` | `structure_layouts` | rows = backbone, cols = t-SNE / UMAP / EmbedOR / EmbedOR+ORC-ManL, coloured by class. Each panel is titled with the k-NN accuracy measured *inside* that 2-D layout, so a projection that reorganises the data instead of showing it is visible at a glance. Panels are Procrustes-aligned per column. |
| | `structure_metrics` | purity vs neighbour rank, fragmentation vs graph k, k-NN vs k, and bars for Table-4 accuracy / clustering agreement / curvature |
| | `structure_table.md` | the same numbers as a markdown table |

**No silhouette anywhere.** It scores compactness, and BallReg exists to spread
points uniformly inside a ball — it therefore penalises that method by
construction, and on these backbones it ranks RectLpJEPA above models that beat
it on every accuracy measure. `kmeans_ari` (compact convex cells) and
`spectral_ari` (k-NN-graph connectivity) are reported side by side instead; where
they disagree, the disagreement is the finding.

EmbedOR / ORC-ManL implementation: [`src/eval/embedor.py`](../src/eval/embedor.py)
(clean-room, from arXiv:2509.03703 and arXiv:2410.01149).

## Data sources (read-only, under `logs/`)

- `logs/eval_angle_probe_lowdata/results_lowdata.{json,csv}` — learning-curve metrics (produced by `eval_angle_probe_lowdata.py` at the repo root).
- `logs/eval_angle_probe_*/cache/*_features.pt` — cached frozen 512-d features + angle labels per backbone.

## Patch-feature PCA maps

For LeJEPA Figure 14 / DINOv2 Figure 1-style spatial PCA coloring, see
[PATCH_PCA.md](PATCH_PCA.md). `visualize_patch_pca.py` extracts last-layer
spatial tokens from trained ViT checkpoints and renders per-image or shared
RGB PCA maps. A diufrd Slurm wrapper is included.
