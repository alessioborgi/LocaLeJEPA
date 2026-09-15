#!/bin/bash
#SBATCH --job-name=blj-geom-imagenette
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/output_%x_%A_%a.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/error_%x_%A_%a.log
#SBATCH --array=0-6%4
# MEMORY: 64GB killed every LocaLeReg arm of job 23032 at epoch ~40-52 --
# OUT_OF_MEMORY, exit 0:125, MaxRSS 66.7-66.9GB against a 64G cgroup limit,
# surfacing as "DataLoader worker killed by signal: Killed". The SIGReg and
# RDMReg arms (tasks 0-3) completed in 8h45-10h53 under the same limit, so the
# local_ballreg path costs ~3GB more host RAM than the global objectives. 96GB
# leaves ~29GB of headroom. This changes no training hyperparameter, so the
# matched-recipe comparison is unaffected.
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=96GB
#SBATCH --partition=GPU
#SBATCH --exclude=diufrd202,diufrd204
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# Does local matching pay off when the backbone already imposes a geometry?
#
# Imagenette, 800 epochs, one seed each. PHASE 1 (the default array) is the
# Hyperbolic ViT-S/8 (Poincare ball, curvature -0.1) crossed with SIGReg
# (LeJEPA), RDMReg (Rectified LpJEPA), and LocaLeReg (local BallReg,
# lambda=0.1) at k in {2,4,8}. The D4-equivariant ResNet-50 arms are wired up
# but held back (tasks 7-11) pending the hyperbolic result.
#
# TWO RECIPE CONDITIONS, because the paper's own limitation (S5.5) is that its
# global-vs-local comparison confounds the objective with the optimizer:
#   matched : lr 1e-3, warmup 10, grad-clip 1.0   -- the paper's "stabilized"
#             recipe, applied to ALL three objectives
#   paper   : lr 2e-3, warmup 1, no clip          -- the paper's global recipe,
#             for SIGReg and RDMReg only
# LocaLeReg's paper recipe IS the stabilized recipe, so it needs no duplicate.
# Phase 1 = 5 matched + 2 paper = 7 runs.
#
# WHAT CANNOT BE MATCHED: the regularizer weight. SIGReg runs lambda=0.02,
# LocaLeReg lambda=0.1, and RDMReg is not a convex lambda at all (25*L_inv +
# 125*R_RDM). Each keeps its own tuned weight; only the optimizer is matched.
#
# Every run keeps weight-only snapshots every 25 epochs
# (callbacks=with_snapshots -> checkpoints/<name>/snapshots/, ~2.7GB per run),
# so eval_snapshots.py can read the 100-epoch comparison -- the horizon at
# which the paper reports its ImageNet-1K gains -- and the 800-epoch one off
# the SAME run. That is the direct test of whether the local advantage is a
# convergence-speed effect or a real asymptote gain.
#
# NODE: any GPU node EXCEPT 202 and 204, which cannot run this at all --
# 202 is a 10GB RTX 3080 (too small for ViT-S/8 @128px x 4 views x bs256) and
# 204 is the V100 (see below). That leaves 203 (4x L40S 48GB, sm_89, bf16) and
# 205 (8x RTX PRO 6000 Blackwell) -- excluding rather than pinning means the
# array picks up 205 automatically if it comes back, instead of queueing behind
# 203 forever. Throttled to 4 concurrent (%4); raise it if 205 returns:
#   sbatch --array=0-6%8 src/slurm_files/slurm_diufrd_pretrain_geometric_imagenette.sh
#
# Why not 204 -- measured 2026-09-10 on that node:
#   Tesla V100-SXM2-32GB, driver 580.159.03, device cap sm_70
#   torch 2.11.0+cu130, arch list: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
# PyTorch's CUDA 13 builds dropped Volta, so any job landing there dies at the
# first CUDA call with "no kernel image is available for execution on the
# device". Not a driver or CUDA-version mismatch: the runtime ships inside the
# wheel and is identical on every node. Using the V100 would need a cu126/cu128
# torch in a separate venv AND fp16 instead of bf16 -- a different numerical
# recipe, which would defeat the point of the matched-recipe design.
#
#   sbatch src/slurm_files/slurm_diufrd_pretrain_geometric_imagenette.sh
#   sbatch --array=4-6%4 ...    # just the LocaLeReg k-sweep (resumes last.ckpt)
#   NUM_WORKERS=2 sbatch ...    # halve the dataloader processes if RAM is still tight
#
# The runs killed as job 23032 logged under the OLD id scheme
# (blj_geom_<name>_23032). To append their resumed epochs to that same wandb run
# instead of opening a fresh one, submit each task with the old id -- one task
# per submission, since the id names a single run:
#   WANDB_ID_OVERRIDE=blj_geom_hvit_locale_k2_23032 sbatch --array=4 <this script>
#   WANDB_ID_OVERRIDE=blj_geom_hvit_locale_k4_23032 sbatch --array=5 <this script>
#   WANDB_ID_OVERRIDE=blj_geom_hvit_locale_k8_23032 sbatch --array=6 <this script>
#   sbatch --array=7-11%4 ...   # the D4 extension, once hyperbolic is in
#
# D4 PRECONDITION (tasks 7-11 only): the equivariant encoder needs escnn, an
# optional extra that builds py3nj and so needs a Fortran compiler:
#   uv sync --extra equivariant
# Those tasks abort early with a clear message if it is missing.

set -euo pipefail

source "$HOME/.local/bin/env" 2>/dev/null || true
source .venv/bin/activate

export RICH_NO_COLOR=1
export NO_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__SERVICE_WAIT=300
export WANDB_START_METHOD=thread

mkdir -p logs/embedor checkpoints
test -d data/imagenette/imagenette2-160/train || { echo "ERROR: Imagenette not staged"; exit 1; }
python -c "import wandb, lightning, timm, hydra" || { echo "ERROR: run 'uv sync'"; exit 1; }

# Tasks 0-6 are the Hyperbolic ViT-S/8 study and run by default. Tasks 7-11 are
# the D4-equivariant ResNet-50 extension, held back until the hyperbolic results
# say whether it is worth the GPU time:
#   sbatch --array=7-11%4 src/slurm_files/slurm_diufrd_pretrain_geometric_imagenette.sh
#
#   0-3  Hyperbolic x {SIGReg, RDMReg} x {matched, paper recipe}
#   4-6  Hyperbolic x LocaLeReg lambda=0.1, k in {2,4,8}   -- the k-sweep,
#        replicating on a hyperbolic backbone the monotone-in-k trend found on
#        the plain ViT (Jonckheere-Terpstra p=0.0058 on frozen k-NN top-1)
#   7-11 D4-equivariant ResNet-50, same shape but LocaLeReg at k=4 only
NAMES=(
  hvit_sigreg_matched     hvit_sigreg_paper     hvit_rdmreg_matched     hvit_rdmreg_paper
  hvit_locale_k2          hvit_locale_k4        hvit_locale_k8
  d4rn50_sigreg_matched   d4rn50_sigreg_paper   d4rn50_rdmreg_matched   d4rn50_rdmreg_paper
  d4rn50_locale_k4
)
EXPERIMENTS=(
  lejepa/hvit_imagenette_sigreg            lejepa/hvit_imagenette_sigreg
  rectifiedlpjepa/hvit_imagenette_rdmreg   rectifiedlpjepa/hvit_imagenette_rdmreg
  lejepa/hvit_imagenette_localballreg_l01_k2
  lejepa/hvit_imagenette_localballreg_l01_k4
  lejepa/hvit_imagenette_localballreg_l01_k8
  lejepa/d4rn50_imagenette_sigreg          lejepa/d4rn50_imagenette_sigreg
  rectifiedlpjepa/d4rn50_imagenette_rdmreg rectifiedlpjepa/d4rn50_imagenette_rdmreg
  lejepa/d4rn50_imagenette_localballreg_l01_k4
)
# matched = lr 1e-3 / warmup 10 / clip 1.0 ; paper = lr 2e-3 / warmup 1 / no clip
RECIPES=(matched paper matched paper
         matched matched matched
         matched paper matched paper
         matched)

i="${SLURM_ARRAY_TASK_ID:?run as an array job}"
NAME="${NAMES[$i]}"
EXP="${EXPERIMENTS[$i]}"
RECIPE="${RECIPES[$i]}"
SEED="${SEED:-0}"
CKPT_DIR="checkpoints/${NAME}"

if [[ "$RECIPE" == "matched" ]]; then
  RECIPE_ARGS=(optimizer.lr=1e-3 model.warmup_epochs=10 +trainer.gradient_clip_val=1.0)
else
  RECIPE_ARGS=(optimizer.lr=2e-3 model.warmup_epochs=1)
fi

# escnn is an optional extra; without it the D4 arms fail deep inside Hydra
# instantiation with an import error that is easy to misread as a config bug.
if [[ "$NAME" == d4rn50_* ]]; then
  python -c "import escnn" 2>/dev/null || {
    echo "ERROR: ${NAME} needs escnn. On the login node: uv sync --extra equivariant"
    echo "       (builds py3nj -- needs a Fortran compiler, e.g. gfortran)"; exit 1; }
fi

# Fail in seconds, not after datamodule + wandb setup, when the node's GPU arch
# is not in this torch build (the sm_70 / V100 case above surfaces as an opaque
# "no kernel image is available" from deep inside the fit loop).
python - <<'PY' || exit 1
import sys, torch
cap = torch.cuda.get_device_capability()
print(f"GPU {torch.cuda.get_device_name()} (sm_{cap[0]}{cap[1]}); "
      f"torch builds: {' '.join(torch.cuda.get_arch_list())}")
# Launch a real kernel rather than string-matching the arch list: CUDA is
# forward-compatible within a major version, so an sm_89 L40S runs sm_86
# kernels quite happily and an exact-match test would reject it.
try:
    (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum().item()
except Exception as exc:
    sys.exit(f"ERROR: this GPU cannot run kernels from this torch build -- {exc}\n"
             f"       Exclude this node and resubmit.")
PY

RESUME=""
[[ -f "${CKPT_DIR}/last.ckpt" ]] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

# Keyed on the run NAME, not the array job id: a resubmitted task resumes from
# last.ckpt, and with an id that changed between submissions `resume=allow`
# would have nothing to resume and would open a SECOND run -- splitting the
# learning curve exactly where the 100-epoch comparison reads it.
WANDB_ID="${WANDB_ID_OVERRIDE:-blj_geom_${NAME}}"

echo "=== [$i] ${NAME}  (${EXP}, ${RECIPE} recipe, seed ${SEED})  start $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/train.py \
  experiment="${EXP}" \
  seed="${SEED}" \
  callbacks=with_snapshots \
  data.num_workers="${NUM_WORKERS:-4}" \
  "${RECIPE_ARGS[@]}" \
  logger=many_loggers \
  "logger.wandb.name=blj_geom_${NAME}" \
  "logger.wandb.id=${WANDB_ID}" \
  "logger.wandb.group=geometric_imagenette" \
  "logger.wandb.entity=${WANDB_ENTITY:-null}" \
  "+logger.wandb.resume=allow" \
  "logger.wandb.tags=[geometric,imagenette,${NAME},${RECIPE}]" \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}" \
  ${RESUME}

echo "=== [$i] ${NAME}  end $(date) ==="
