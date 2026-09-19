# Ablations

```bash
uv sync --extra experiments
cp .env.example .env                 # W&B entity / project
```

## What to run

Three tables, lambda 0.2 everywhere; `-m` runs the grid sequentially on one GPU:

```bash
uv run python src/train.py -m experiment=ablations/batch_k       # B {32..1024} x k {2,4,...,16}, lr 1e-3   (48 runs)
uv run python src/train.py -m experiment=ablations/batch_dynk    #   + dynamic-k row: B x (k 2 -> 16)        (6)
uv run python src/train.py -m experiment=ablations/k_lr          # k {2,...,16} x lr {3e-4,1e-3,3e-3}, B 256 (24)
uv run python src/train.py -m experiment=ablations/lr_dynk       #   + dynamic-k row: lr x (k 2 -> 16)       (3)
uv run python src/train.py -m experiment=ablations/batch_lr      # B {32..1024} x lr {3e-4,...}, k 4        (18)
```

Dynamic k: +2 every 12 epochs, k=16 from epoch 84. Everything else is fixed in
`configs/experiment/ablations/base.yaml`: Euclidean ViT-S/8, Imagenette, 100 epochs, no
accumulation (B is what k-means and the optimizer see), 10 warm-up epochs then cosine to 1% of
the peak lr, seed 42.

On SLURM, one job per cell with `scripts/slurm_submit.sh` (it resumes from `last.ckpt` on resubmission):

```bash
for B in 32 64 128 256 512 1024; do for k in 2 4 6 8 10 12 14 16; do
  sbatch --export=ALL,EXPERIMENT=ablations/batch_k,OVERRIDES="data.batch_size=$B model.reg_loss.num_clusters=$k",RUN_NAME=batch_k_B${B}_k$k scripts/slurm_submit.sh
done; done
```

Same for the others with `OVERRIDES="optimizer.lr=$lr ..."`; the `*_dynk` ones take only the B or lr override.

## Where it goes

W&B project from `.env`, group `ablations`, run name = the overrides (`experiment=ablations/k_lr,model.reg_loss.num_clusters=8,optimizer.lr=3e-4`).
Locally `logs/train/runs/<date>/` (config, `train.log`, `checkpoints/{best,last}.ckpt`); `logger=csv` for a CSV instead of W&B.
Metric: `val/probe_acc` (online linear probe).

## Knobs, if something needs changing

| knob | override |
|---|---|
| regularizer | `model/reg_loss=` `sigreg` (LeJEPA), `localereg` (ours), `locale_sigreg`, `susreg`, `subreg`, `uniform_reg` |
| lambda | `model.lamb` |
| k | `model.reg_loss.num_clusters` |
| k schedule | `callbacks=dynamic_num_clusters callbacks.dynamic_num_clusters.{start,end,step_epochs,step}` (`end < start` runs downwards) |
| batch | `data.batch_size` (also what k-means sees), `trainer.accumulate_grad_batches` |
| lr | `optimizer.lr`, `model.warmup_epochs`, `model.lr_eta_min_ratio` (cosine floor / peak) |
| epochs, seed | `trainer.max_epochs`, `seed` |

Resume: `ckpt_path=<run dir>/checkpoints/last.ckpt`. Evaluation of saved checkpoints:
`uv run python src/eval_indomain.py --config configs/eval/indomain_imagenette.yaml` (k-NN + linear probe; see `EXPERIMENTS.md`).
