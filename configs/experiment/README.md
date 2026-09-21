# Experiment configs

`uv run python src/train.py experiment=<group>/<name>` runs one of the configs below. Every
config is self-contained: `experiment=<name>` alone reproduces the run (per-step
`data.batch_size: 64` × `trainer.accumulate_grad_batches: 4` = 256 effective on single-GPU runs,
`trainer.gradient_clip_val: 1.0` for the local regularizer, `trainer.max_epochs: 100` for the
ImageNet-1K suite).

## Shared recipe

All LeJEPA-family runs: 4 augmented views per image at 128 px (`data/transform/image/lejepa_train`),
embedding 512-d, projector MLP [2048, 2048, 16] with BatchNorm, online linear probe on the
frozen 512-d embedding (selects `best.ckpt` by `val/probe_acc`), AdamW, weight decay 5e-2,
linear warm-up + cosine. Two recipes appear:

| recipe | lr | warm-up | grad-clip | `lr_eta_min_ratio` | used by |
|---|---|---|---|---|---|
| baseline | 2e-3 | 1 epoch | none | 0.5 | `sigreg`, `susreg`, `subreg`, `rdmreg` (LeJEPA's `MINIMAL.md` recipe; the D4-RN50 baselines were at 1e-3 / flat until 2026-09-20 -- decision 11) |
| stabilized | 1e-3 | 10 epochs | 1.0 | **0.001** | every `local_*` regularizer (λ=0.02 with the baseline recipe collapsed); was **1.0** -- a flat 1e-3 -- until 2026-09-20 (decision 6) |

`lr_eta_min_ratio` is the fourth difference and it is a **schedule-shape** difference, not a
tuning detail. `src/models/lejepa_module.py` computes `eta_min = lr_eta_min_ratio * base_lr`, so:

* baseline — `eta_min = 0.5 * 2e-3 = 1e-3`, a cosine decaying 2e-3 → 1e-3;
* stabilized — `eta_min = 0.001 * 1e-3 = 1e-6`, a real cosine to peak/1000 (LeJEPA's README
  schedule). **Until 2026-09-20 it was 1.0, i.e. a constant 1e-3 after warm-up** — a schedule-
  shape difference between the compared arms that the campaign removes; the ViT-S/8 control
  (`vit_in1k_sigreg_localrecipe`) quantifies what the recipe itself changes.

The ViT-L/14 configs (`*_paper`) put LeJEPA's own ImageNet-1K optimizer and loss settings into
our protocol (decision 13): 2×238 px global + 6×98 px local views, batch 512, lr 5e-4, wd 5e-2,
cosine to peak/1000, λ 0.05, 1024 slices on [-5, 5], projector → 512, 8-GPU DDP. Their
published 74.24 also used an EMA teacher and patch masking this codebase does not have, so it is
cited, not compared. The two arms are identical except `model/reg_loss`, including the statistic
resolution.

## LeJEPA baseline vs. ours, per backbone / dataset

| backbone · dataset | LeJEPA (SIGReg) | **ours: LocaLe-Reg λ0.1 k4** |
|---|---|---|
| ViT-S/8 · ImageNet-1K 100 ep. | `lejepa/vit_in1k_sigreg` | `lejepa/vit_in1k_localballreg_l01_k4` |
| Hyperbolic ViT-S/8 · ImageNet-1K 100 ep. | `lejepa/hvit_in1k_sigreg` | `lejepa/hvit_in1k_localballreg_l01_k4` |
| D4-ResNet-50 · ImageNet-1K 100 ep. | `lejepa/d4rn50_in1k_sigreg` | `lejepa/d4rn50_in1k_localballreg_l01_k4` |
| ViT-L/14 · ImageNet-1K, LeJEPA's settings in our protocol | `lejepa/vitl14_in1k_sigreg_paper` | `lejepa/vitl14_in1k_localballreg_paper` |
| ViT-S/8 · Imagenette 800 ep. | `lejepa/vit_imagenette_sigreg` | `lejepa/vit_imagenette_localballreg_l01_k4` |

What differs between the two columns (compare the YAMLs): `model/reg_loss` (`sigreg` →
`localereg`), `model.lamb` (0.02 → 0.1; 0.05 on both arms of the ViT-L/14 pair),
`model.reg_loss.num_clusters: 4` (= k), and the stabilized recipe — lr 1e-3, warm-up 10,
grad-clip 1.0 **and `lr_eta_min_ratio` 0.5 → 0.001** (in the ViT-L/14 pair nothing but the
regularizer differs, λ included). Everything else is identical. The campaign manifests in
`configs/campaign/` override λ and k per cell; the `_l01_k4` names are the configs' defaults.

## Ablations (ImageNet-1K, 100 epochs; `hvit_*` and `d4rn50_*` exist for both backbones)

| config suffix | regularizer | λ | k |
|---|---|---|---|
| `_localballreg_l01_k2` | LocaLe-Reg | 0.1 | 2 |
| `_localballreg_l01_k4` | LocaLe-Reg | 0.1 | 4 |
| `_localballreg_l03_k2` | LocaLe-Reg | 0.3 | 2 |
| `_localballreg_l01_dynk` | LocaLe-Reg | 0.1 | 1 → 8, +1 every 100 epochs (`callbacks=dynamic_num_clusters`) |
| `_localsigreg_l01_k4` | LocaLeJEPA-SIGReg | 0.1 | 4 |
| `_sigreg` | SIGReg (LeJEPA) | 0.02 | – |
| `rectifiedlpjepa/*_rdmreg` | RDMReg (Rectified LpJEPA, `model=rectifiedlpjepa`: `LeJEPAModule` + ReLU + absolute loss weights, `reg_loss=rdmreg`) | – | – |

Imagenette (800 epochs, ViT-S/8): `vit_imagenette_{sigreg,susreg,subreg,local_sigreg,local_ballreg}`
use the baseline recipe (the two `local_*` ones with the default λ=0.02 / k=8 are the runs that
collapsed and motivated the stabilized recipe), `vit_imagenette_localballreg_l01_k4` and
`vit_imagenette_local_ballreg_dynk` the stabilized one; `hvit_imagenette_sigreg`,
`d4rn50_imagenette_sigreg`, `rectifiedlpjepa/{vit,hvit,d4rn50}_imagenette_rdmreg` are the
per-backbone baselines.

## Other groups (experiments branch)

* `lejepa/{nvit,svit}_{imagenette,in1k}_{sigreg,susreg,subreg}`: the nGPT-style NormalizedViT
  and SphericalViT backbones with SIGReg / SUSReg / SubReg (`model/reg_loss={sigreg,susreg,subreg}`).
* `lejepa/{rn18_stl10,rn50_in100,vits16_in100,svits16_in100}_ep500`: LeJEPA-paper-style
  benchmark settings on STL-10 / ImageNet-100 (multi-crop 2+6 views, 224 px); not part of the
  reported tables.
* `lejepa/vit_imagenette_localballreg_l01_k{1,6,8,16,32}_ep100`: the neighbourhood-count study
  with a short recipe — 100 epochs, batch 256 (no accumulation, so the clustering batch is 256),
  lr 1e-3, warm-up 3, cosine to 1% of the peak. The k x lr grid is one Hydra multirun:
  `uv run python src/train.py -m experiment=lejepa/vit_imagenette_localballreg_l01_k1_ep100,lejepa/vit_imagenette_localballreg_l01_k6_ep100,lejepa/vit_imagenette_localballreg_l01_k16_ep100,lejepa/vit_imagenette_localballreg_l01_k32_ep100 optimizer.lr=3e-4,1e-3,3e-3`.
  On SLURM submit one job per cell instead:
  `for k in 1 6 8 16 32; do for lr in 3e-4 1e-3 3e-3; do sbatch --export=ALL,EXPERIMENT=lejepa/vit_imagenette_localballreg_l01_k${k}_ep100,OVERRIDES="optimizer.lr=$lr" scripts/slurm_submit.sh; done; done`.
* `lejepa/vit_imagenette_local_ballreg_dynk_b_ablation`: batch-size ablation of the dynamic-k run
  (`hydra.sweeper.params` sweeps `data.batch_size` over 32..1024; one GPU, no accumulation).
* `ablations/{batch_k,k_lr,batch_lr}` and the dynamic-k rows `ablations/{batch_dynk,lr_dynk}`:
  the three ablation tables on a common recipe (`ablations/base`: Euclidean ViT-S/8, Imagenette,
  100 epochs, B 256 without accumulation, lr 1e-3 cosine to 1% of the peak, k 4, lambda 0.2);
  each carries its grid in `hydra.sweeper.params`. See `handoff.md`.
* Radius sweep: `uv run python src/train.py -m experiment=lejepa/vit_imagenette_localballreg_l01_k4 model.reg_loss.loss.radius=0.5,1.0,null model.reg_loss.loss.scale_grid=true`
  (`scale_grid` keeps the test's power the same at every radius; it is a no-op at the default R).
