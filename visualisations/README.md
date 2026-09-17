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

## Data sources (read-only, under `logs/`)

- `logs/eval_angle_probe_lowdata/results_lowdata.{json,csv}` — learning-curve metrics (produced by `eval_angle_probe_lowdata.py` at the repo root).
- `logs/eval_angle_probe_*/cache/*_features.pt` — cached frozen 512-d features + angle labels per backbone.
