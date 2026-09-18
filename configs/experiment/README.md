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

| recipe | lr | warm-up | grad-clip | used by |
|---|---|---|---|---|
| baseline | 2e-3 (ViT, HViT) / 1e-3 (D4-RN50) | 1 epoch | none | `sigreg`, `susreg`, `subreg`, `rdmreg` |
| stabilized | 1e-3 | 10 epochs | 1.0 | every `local_*` regularizer (λ=0.02 with the baseline recipe collapsed) |

The ViT-L/14 "paper setting" configs (`*_paper`) instead follow LeJEPA §6.1: 2×224 px global +
8×98 px local views, projector → 64-d, lr 5e-4, wd 0.1, 4-GPU DDP, global batch 256.

## LeJEPA baseline vs. ours, per backbone / dataset

| backbone · dataset | LeJEPA (SIGReg) | **ours: LocaLe-Reg λ0.1 k4** |
|---|---|---|
| ViT-S/8 · ImageNet-1K 100 ep. | `lejepa/vit_in1k_sigreg` | `lejepa/vit_in1k_localballreg_l01_k4` |
| Hyperbolic ViT-S/8 · ImageNet-1K 100 ep. | `lejepa/hvit_in1k_sigreg` | `lejepa/hvit_in1k_localballreg_l01_k4` |
| D4-ResNet-50 · ImageNet-1K 100 ep. | `lejepa/d4rn50_in1k_sigreg` | `lejepa/d4rn50_in1k_localballreg_l01_k4` |
| ViT-L/14 · ImageNet-1K paper setting | `lejepa/vitl14_in1k_sigreg_paper` | `lejepa/vitl14_in1k_localballreg_l01_k4_paper` |
| ViT-S/8 · Imagenette 800 ep. | `lejepa/vit_imagenette_sigreg` | `lejepa/vit_imagenette_localballreg_l01_k4` |

What differs between the two columns (compare the YAMLs): `model/reg_loss` (`sigreg` →
`localereg`), `model.lamb` (0.02 → 0.1; 0.05 → 0.1 in the paper setting),
`model.reg_loss.num_clusters: 4` (= k), and the stabilized recipe (lr 1e-3 / warm-up 10 /
grad-clip 1.0, except in the paper setting where only the clip is added). Everything else is
identical.

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
* `lejepa/phc_{blob,gpm}_ep250` + `phc_regression/{blob,gpm}_supervised_ep500`: photonic-crystal
  pretraining and DOS-regression fine-tuning (`scripts/download_phc.py` fetches the data).
