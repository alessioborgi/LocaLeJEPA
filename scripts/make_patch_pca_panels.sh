#!/bin/bash
# Build the paper patch-PCA panels (variants + methods figures) for every
# backbone family, class group and image from the group-scope extraction runs,
# plus the "showcase" group (LeJEPA Fig.1 golden retriever + 4 val retrievers).
#   bash scripts/make_patch_pca_panels.sh [256|128] [out_prefix]
set -euo pipefail
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"
RES="${1:-256}"; OUT="${2:-paper/figs/patch_pca/patch_pca_${RES}}"
if [ "$RES" = 128 ]; then VIT=logs/patch_pca_in1k; D4=logs/patch_pca_in1k_d4rn50_128
else VIT=logs/patch_pca_in1k_256; D4=logs/patch_pca_in1k_d4rn50_256; fi
panel() { python visualisations/plot_patch_pca_panels.py "$@" 2>&1 | grep -v "^$" || true; }
for fam in vit hvit d4rn50; do
  base=$VIT; [ "$fam" = d4rn50 ] && base=$D4
  for fig in variants methods; do
    [ "$fam" = vit ] && [ "$fig" = variants ] && continue   # plain ViT only has SIGReg + BallReg k4
    if [ -f "$base/group/manifest.json" ]; then
      for g in 00 01 02; do for i in 0 1 2 3 4; do
        panel --run-dir $base/group --family $fam --figure $fig --group $g --image $i --out "$OUT"
      done; done
    else echo "skip $fam ($base/group missing)"; fi
    if [ -f "$base/showcase/manifest.json" ]; then
      for i in 0 1 2 3 4; do
        panel --run-dir $base/showcase --family $fam --figure $fig --group 00 --image $i \
          --label "golden retriever (LeJEPA Fig. 1 image + 4 val)" --out "${OUT}_showcase"
      done
    else echo "skip $fam showcase ($base/showcase missing)"; fi
  done
done
