#!/usr/bin/env bash
# Downloads the released weights of the paper's rows and checks their SHA-256.
#   bash scripts/download_weights.sh [row ...]      # default: all rows, into ./weights
# Then: python -m scoreclr.evaluation --run runs/<row> --checkpoint weights/<row>.pt
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${URL:-https://github.com/chris-mrn/ScoreCLR_code/releases/download/weights-v1}
ROWS=("$@")
[ ${#ROWS[@]} -eq 0 ] && ROWS=(cifar10_scoreclr cifar10_simclr_adamw cifar10_lejepa cifar10_simclr_lars
                               imagenet100_scoreclr imagenet100_simclr_adamw imagenet100_simclr_lars
                               imagenet100_simclr_lars_proj4096)
mkdir -p weights
curl -fsSL "$URL/SHA256SUMS" -o weights/SHA256SUMS
for row in "${ROWS[@]}"; do
  [ -f "weights/$row.pt" ] || curl -fL "$URL/$row.pt" -o "weights/$row.pt"
  (cd weights && grep " $row.pt\$" SHA256SUMS | shasum -a 256 -c -)
done
