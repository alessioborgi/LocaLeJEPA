#!/usr/bin/env bash
set -euo pipefail

# Run the four LeJEPA ViT/Imagenette regularizer comparisons:
#   1. sigreg
#   2. susreg
#   3. local_sigreg
#   4. local_ballreg
#
# Extra Hydra overrides can be appended to this script. Examples:
#   scripts/run_lejepa_vit_imagenette_regularizers.sh trainer=mps data.batch_size=64
#   scripts/run_lejepa_vit_imagenette_regularizers.sh trainer.max_epochs=100 logger=csv
#   scripts/run_lejepa_vit_imagenette_regularizers.sh model.local_sigreg_num_clusters=16

EXPERIMENTS="lejepa/vit_imagenette_sigreg,lejepa/vit_imagenette_susreg,lejepa/vit_imagenette_local_sigreg,lejepa/vit_imagenette_local_ballreg"

uv run python src/train.py -m \
  "experiment=${EXPERIMENTS}" \
  "$@"
