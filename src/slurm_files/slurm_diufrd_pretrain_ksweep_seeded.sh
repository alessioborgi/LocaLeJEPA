#!/bin/bash
#SBATCH --job-name=blj-ksweep-seeded
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/output_%x_%A_%a.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/error_%x_%A_%a.log
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem=96GB
#SBATCH --partition=GPU
#SBATCH --exclude=diufrd202,diufrd204
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# Seeded k-sweep for the "probe learns faster as k grows" finding.
#   tasks 0-8   : BallJEPA local BallReg, lambda=0.1, k in {2,4,8} x seed {0,1,2}
#   tasks 9-11  : LeJEPA (SIGReg) x seed {0,1,2}   -- run with --array=0-11 to
#                 include the cross-method speed baseline
#   tasks 12-17 : k in {16,32} x seed {0,1,2}  -- the extension PAST the number
#                 of Imagenette classes (10). Through k=8 the frozen k-NN top-1
#                 rises monotonically (Jonckheere-Terpstra p=0.0058); these arms
#                 decide whether it keeps rising or turns over near #classes,
#                 which is the test that separates "locality is the lever" from
#                 "k~#classes is doing weak supervised clustering".
#                 3 seeds because the JT test needs every k at the same n --
#                 mixing n=1 with n=3 makes the trend statistic unusable.
#
# CLUSTER SIZE at the new k (batch 256, 16-d projector): 16 points/cluster at
# k=16, 8 at k=32, against min_cluster_size 4. See the config headers -- the
# per-cluster covariance is rank-deficient at k=32 and some clusters are skipped,
# so the effective k is a little below the nominal one. Kept as-is deliberately:
# every banked arm uses min_cluster_size 4, and changing it would unmatch them.
#
# RESOURCES changed after job 23032 killed three local_ballreg arms:
#   --mem 64GB -> 96GB. Those arms hit MaxRSS 66.7-66.9GB against a 64G cgroup
#     limit (OUT_OF_MEMORY, exit 0:125), surfacing as "DataLoader worker killed
#     by signal: Killed". Same objective, so the same headroom applies here.
#   --time 24h -> 36h. The per-step Python loop in local_ballreg_loss runs once
#     per cluster, so k=32 does 4x the iterations of k=8; runs measured ~10h at
#     k<=8 and the wall is not worth risking. (last.ckpt resumes, but a wall hit
#     costs a queue round-trip.)
#   --nodelist=diufrd205 -> --exclude=202,204. 205 has been unavailable for
#     stretches, and 203 measured FASTER for this workload under load (8h45 vs
#     10h31 for the same H-ViT RDMReg job). SAFE because the primary endpoints
#     are per-EPOCH (AULC of frozen k-NN over epochs 0-200). NOT safe for the
#     wall-clock claims in the Confound-04 analysis: those compared k at fixed
#     hardware, and an arm that lands on a different node cannot join that
#     comparison. Pin with --nodelist on the CLI if you need wall-clock parity.
#
# Each run keeps periodic weight-only snapshots (callbacks=with_snapshots,
# every 25 epochs -> checkpoints/<name>/snapshots/) for eval_snapshots.py, plus
# the usual best.ckpt / last.ckpt. ~2.7 GB of snapshots per run.
#
#   sbatch src/slurm_files/slurm_diufrd_pretrain_ksweep_seeded.sh              # k 2/4/8 only
#   sbatch --array=0-11 src/slurm_files/slurm_diufrd_pretrain_ksweep_seeded.sh
#   sbatch --array=6-8  src/slurm_files/slurm_diufrd_pretrain_ksweep_seeded.sh  # just k=8
#   sbatch --array=12-17%4 src/slurm_files/slurm_diufrd_pretrain_ksweep_seeded.sh
#                                                     # the k=16/32 extension
#   NUM_WORKERS=2 sbatch ...     # halve the dataloader processes if RAM is tight

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

NAMES=(
  ball_l01_k2_s0 ball_l01_k2_s1 ball_l01_k2_s2
  ball_l01_k4_s0 ball_l01_k4_s1 ball_l01_k4_s2
  ball_l01_k8_s0 ball_l01_k8_s1 ball_l01_k8_s2
  lejepa_s0      lejepa_s1      lejepa_s2
  ball_l01_k16_s0 ball_l01_k16_s1 ball_l01_k16_s2
  ball_l01_k32_s0 ball_l01_k32_s1 ball_l01_k32_s2
)
EXPERIMENTS=(
  lejepa/vit_imagenette_localballreg_l01_k2 lejepa/vit_imagenette_localballreg_l01_k2 lejepa/vit_imagenette_localballreg_l01_k2
  lejepa/vit_imagenette_localballreg_l01_k4 lejepa/vit_imagenette_localballreg_l01_k4 lejepa/vit_imagenette_localballreg_l01_k4
  lejepa/vit_imagenette_localballreg_l01_k8 lejepa/vit_imagenette_localballreg_l01_k8 lejepa/vit_imagenette_localballreg_l01_k8
  lejepa/vit_imagenette_sigreg              lejepa/vit_imagenette_sigreg              lejepa/vit_imagenette_sigreg
  lejepa/vit_imagenette_localballreg_l01_k16 lejepa/vit_imagenette_localballreg_l01_k16 lejepa/vit_imagenette_localballreg_l01_k16
  lejepa/vit_imagenette_localballreg_l01_k32 lejepa/vit_imagenette_localballreg_l01_k32 lejepa/vit_imagenette_localballreg_l01_k32
)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)

i="${SLURM_ARRAY_TASK_ID:?run as an array job}"
NAME="${NAMES[$i]}"
EXP="${EXPERIMENTS[$i]}"
SEED="${SEEDS[$i]}"
CKPT_DIR="checkpoints/${NAME}"

RESUME=""
[[ -f "${CKPT_DIR}/last.ckpt" ]] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

WANDB_ENTITY="${WANDB_ENTITY:-}"
# Keyed on the run NAME, not the array job id: a resubmitted task resumes from
# last.ckpt, and an id that changed between submissions would leave
# `resume=allow` nothing to resume and open a SECOND run -- splitting the
# learning curve that eval_snapshots.py and the AULC endpoint read.
# Set WANDB_ID_OVERRIDE to append to a run logged under the old scheme.
WANDB_ID="${WANDB_ID_OVERRIDE:-blj_lc_${NAME}}"

echo "=== [$i] ${NAME}  (${EXP}, seed ${SEED})  start $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/train.py \
  experiment="${EXP}" \
  seed="${SEED}" \
  callbacks=with_snapshots \
  data.num_workers="${NUM_WORKERS:-4}" \
  +trainer.gradient_clip_val=1.0 \
  logger=many_loggers \
  "logger.wandb.name=blj_lc_${NAME}" \
  "logger.wandb.id=${WANDB_ID}" \
  "logger.wandb.group=embedor_lc_ksweep" \
  "logger.wandb.entity=${WANDB_ENTITY:-null}" \
  "+logger.wandb.resume=allow" \
  "logger.wandb.tags=[embedor,imagenette,vit-s8,learning-curve,${NAME}]" \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}" \
  ${RESUME}

echo "=== [$i] ${NAME}  end $(date) ==="
