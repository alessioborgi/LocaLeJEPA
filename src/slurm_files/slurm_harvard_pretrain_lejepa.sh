#!/bin/bash
#SBATCH -J BundLeJEPA
#SBATCH -p gpu,gpu_h200,gpu_requeue,hsph_gpu,hsph_gpu
#SBATCH -c 8
#SBATCH -t 1-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -o logs/BundLeJEPA_%j.out
#SBATCH -e logs/BundLeJEPA_%j.err

##################### Experiment configuration ##################

##### LeJEPA Image Pre-training #####
# Selects the experiment config via DEFAULT_PRETRAIN_RUN_MODE.
# Each mode maps to a config under configs/experiment/lejepa/.
#
# Available run modes and their dataset / backbone:
#   rn18_stl10_ep500         : ResNet-18      on STL-10         (500 epochs)
#   rn50_in100_ep500         : ResNet-50      on ImageNet-100   (500 epochs)
#   vits16_in100_ep500       : ViT-S/16       on ImageNet-100   (500 epochs)
#   vits8_imagenette_ep800   : ViT-S/8        on Imagenette     (800 epochs)
#   vits8_in1k_ep800         : ViT-S/8        on ImageNet-1K    (800 epochs)
#   svits16_in100_ep500      : SphericalViT-S/16 on ImageNet-100   (500 epochs)
#   svits8_imagenette_ep800  : SphericalViT-S/8  on Imagenette     (800 epochs)
#   phc_blob_ep250           : PHC encoder    on PHC-Blob       (250 epochs)
#   phc_gpm_ep250            : PHC encoder    on PHC-GPM        (250 epochs)
#
# Override at submission time with:
#   sbatch --export=ALL,RUN_MODE=vits8_imagenette_ep800 slurm_files/slurm_harvard_pretrain_lejepa.sh

DEFAULT_PRETRAIN_RUN_MODE="phc_gpm_ep250"
DEFAULT_PRETRAIN_LABEL="Pretrained_LeJEPA_phc_gpm_ep250"
DEFAULT_PRETRAIN_WANDB_TAGS="pretraining;lejepa;phc;gpm"
DEFAULT_PRETRAIN_WANDB_RUN_NAME=""  # Leave empty to auto-name as: PreTrain_LeJEPA_${RUN_MODE}

# Path to a class-subset file passed as data.class_subset_file (for in100 modes).
# Leave empty to skip the override. Override at submission time with:
#   sbatch --export=ALL,CLASS_SUBSET_FILE=configs/data/class_subsets/imagenet100.txt ...
# Example values:
#   configs/data/class_subsets/imagenet100.txt  (for rn50_in100_ep500, vits16_in100_ep500)
DEFAULT_CLASS_SUBSET_FILE=""

# Lambda (model.lamb) override. Leave empty to use the config default.
# Override at submission time with:
#   sbatch --export=ALL,LAMB=0.02 slurm_files/slurm_harvard_pretrain_lejepa.sh
DEFAULT_PRETRAIN_LAMB=""

DEFAULT_PRETRAIN_HYDRA_OVERRIDES=(
  # "trainer.max_epochs=300"
)

###############################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

module purge

# Activate the project's Conda environment.
# Hardcode the Mambaforge base so module loads don't influence conda info --base.
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

wandb login
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Launching W&B Agent..."
echo "Python: $(which python)"
python -c "import sys; print('Python version:', sys.version)"

# Fail early with a clear message if the environment is missing core training deps.
python -c "import hydra, lightning, omegaconf, wandb" || {
  echo "Missing required Python packages in the active Conda environment."
  echo "Install the project dependencies first, for example:"
  echo "  pip install -r requirements.txt"
  exit 1
}

RUN_MODE="${RUN_MODE:-${DEFAULT_PRETRAIN_RUN_MODE:-}}"

if [[ -z "$RUN_MODE" ]]; then
  echo "RUN_MODE is not set. Set DEFAULT_PRETRAIN_RUN_MODE in the config section"
  echo "or pass it at submission: sbatch --export=ALL,RUN_MODE=rn18_stl10_ep500 slurm_files/slurm_harvard_pretrain_lejepa.sh"
  echo ""
  echo "Available run modes:"
  echo "  rn18_stl10_ep500          ResNet-18         on STL-10        (500 ep)"
  echo "  rn50_in100_ep500          ResNet-50         on ImageNet-100  (500 ep)"
  echo "  vits16_in100_ep500        ViT-S/16          on ImageNet-100  (500 ep)"
  echo "  vits8_imagenette_ep800    ViT-S/8           on Imagenette    (800 ep)"
  echo "  vits8_in1k_ep800          ViT-S/8           on ImageNet-1K   (800 ep)"
  echo "  svits16_in100_ep500       SphericalViT-S/16 on ImageNet-100  (500 ep)"
  echo "  svits8_imagenette_ep800   SphericalViT-S/8  on Imagenette    (800 ep)"
  echo "  phc_blob_ep250            PHC               on PHC-Blob      (250 ep)"
  echo "  phc_gpm_ep250             PHC               on PHC-GPM       (250 ep)"
  exit 1
fi

# Validate RUN_MODE against known experiment configs.
case "$RUN_MODE" in
  rn18_stl10_ep500|\
  rn50_in100_ep500|\
  vits16_in100_ep500|\
  vits8_imagenette_ep800|\
  vits8_in1k_ep800|\
  svits16_in100_ep500|\
  svits8_imagenette_ep800|\
  phc_blob_ep250|\
  phc_gpm_ep250)
    ;;
  *)
    echo "Unsupported RUN_MODE: $RUN_MODE"
    echo "Valid options: rn18_stl10_ep500 | rn50_in100_ep500 | vits16_in100_ep500 | vits8_imagenette_ep800 | vits8_in1k_ep800 | svits16_in100_ep500 | svits8_imagenette_ep800 | phc_blob_ep250 | phc_gpm_ep250"
    exit 1
    ;;
esac

# Optional W&B tags passed at submission time.
# Prefer ';' as the separator because Slurm uses ',' inside --export itself.
# Example:
# sbatch --export=ALL,WANDB_TAGS='lejepa;stl10;pretraining' slurm_files/slurm_harvard_pretrain_lejepa.sh
WANDB_TAGS="${WANDB_TAGS:-${DEFAULT_PRETRAIN_WANDB_TAGS:-}}"
WANDB_TAGS_OVERRIDE=()
if [[ -n "$WANDB_TAGS" ]]; then
  if [[ "$WANDB_TAGS" == *";"* ]]; then
    IFS=';' read -r -a TAG_ARRAY <<< "$WANDB_TAGS"
  else
    IFS=',' read -r -a TAG_ARRAY <<< "$WANDB_TAGS"
  fi
  TAGS_JOINED=$(printf '"%s",' "${TAG_ARRAY[@]}")
  TAGS_JOINED="[${TAGS_JOINED%,}]"
  WANDB_TAGS_OVERRIDE=("logger.wandb.tags=${TAGS_JOINED}")
  echo "Using W&B tags: ${WANDB_TAGS}"
fi

WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DEFAULT_PRETRAIN_WANDB_RUN_NAME:-}}"
WANDB_RUN_NAME_OVERRIDE=()
if [[ -n "$WANDB_RUN_NAME" ]]; then
  WANDB_RUN_NAME_OVERRIDE=("logger.wandb.name=${WANDB_RUN_NAME}")
  echo "Using W&B run name: ${WANDB_RUN_NAME}"
fi

CLASS_SUBSET_FILE="${CLASS_SUBSET_FILE:-${DEFAULT_CLASS_SUBSET_FILE:-}}"
CLASS_SUBSET_FILE_OVERRIDE=()
if [[ -n "$CLASS_SUBSET_FILE" ]]; then
  CLASS_SUBSET_FILE_OVERRIDE=("data.class_subset_file=${CLASS_SUBSET_FILE}")
  echo "Using class subset file: ${CLASS_SUBSET_FILE}"
fi

LAMB="${LAMB:-${DEFAULT_PRETRAIN_LAMB:-}}"
LAMB_OVERRIDE=()
if [[ -n "$LAMB" ]]; then
  LAMB_OVERRIDE=("model.lamb=${LAMB}")
  echo "Using lambda: ${LAMB}"
fi

RUN_CMD=(python src/train.py experiment="lejepa/${RUN_MODE}" logger=wandb)
HYDRA_OVERRIDES=("${DEFAULT_PRETRAIN_HYDRA_OVERRIDES[@]}")

# On this cluster ImageNet lives directly under imagenet/{train,val} — no imagenet_full_size/ subfolder.
# encode() flattens all views into one tensor (batch * num_views images at once), so batch_size=256
# with 8 crops = 2048 images through ResNet50 simultaneously overflows an 80 GB A100.
# Use batch_size=64 + accumulate_grad_batches=4 to keep the effective batch at 256.
case "$RUN_MODE" in
  rn50_in100_ep500|vits16_in100_ep500|vits8_in1k_ep800|vits8_imagenette_ep800|\
  svits16_in100_ep500)
    HYDRA_OVERRIDES+=("data.image_folder=.")
    HYDRA_OVERRIDES+=("data.batch_size=64")
    HYDRA_OVERRIDES+=("+trainer.accumulate_grad_batches=4")
    ;;
esac

if [[ -z "$WANDB_RUN_NAME" ]]; then
  _AUTO_RUN_NAME="PreTrain_LeJEPA_${RUN_MODE}"
  WANDB_RUN_NAME_OVERRIDE=("logger.wandb.name=${_AUTO_RUN_NAME}")
  echo "Using auto W&B run name: ${_AUTO_RUN_NAME}"
fi

RUN_CMD+=("${WANDB_TAGS_OVERRIDE[@]}")
RUN_CMD+=("${WANDB_RUN_NAME_OVERRIDE[@]}")
RUN_CMD+=("${CLASS_SUBSET_FILE_OVERRIDE[@]}")
RUN_CMD+=("${LAMB_OVERRIDE[@]}")
RUN_CMD+=("${HYDRA_OVERRIDES[@]}")

echo "Run mode  : ${RUN_MODE}"
echo "Experiment: lejepa/${RUN_MODE}"
echo "Command   : ${RUN_CMD[*]}"
"${RUN_CMD[@]}"

echo "=== Job End: $(date) ==="
