#!/usr/bin/env bash
# Reproduces every table and figure of the paper: trains each row (config/<row>.yaml), evaluates it
# with the protocol shared by all methods (python -m scoreclr.evaluation), prints the tables next to
# the published numbers and draws the cost figure.
#
#   DATA=./datasets IMAGENET100=/path/to/imagenet100 bash scripts/reproduce_paper.sh [row ...]
#
# rows: the names of config/*.yaml (default: all eight). CIFAR-10 rows use 1 GPU; ImageNet-100 rows
# use $NGPU GPUs (default 2, 128 images per GPU as in the paper). Training resumes from <run>/last.pt
# (a finished run is not retrained); a run with evaluation.json is not re-evaluated.
# DATA holds (or receives, with downloads) cifar-10-batches-py/, stl10_binary/ and dtd/;
# IMAGENET100 holds train/ and val/ (python scripts/make_imagenet100.py).
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-./datasets}
IMAGENET100=${IMAGENET100:-./datasets/imagenet100}
NGPU=${NGPU:-2}
WORKERS=${WORKERS:-8}
OUT=${OUT:-runs}
ROWS=("$@")
[ ${#ROWS[@]} -eq 0 ] && ROWS=(cifar10_scoreclr cifar10_simclr_adamw cifar10_lejepa cifar10_simclr_lars
                               imagenet100_scoreclr imagenet100_simclr_adamw imagenet100_simclr_lars
                               imagenet100_simclr_lars_proj4096)
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH}

for row in "${ROWS[@]}"; do
  cfg=config/$row.yaml
  [ -f "$cfg" ] || { echo "unknown row $row (no $cfg)"; exit 1; }
  run=$OUT/$row
  if [[ $row == cifar10_* ]]; then
    data=(data.root="$DATA" data.num_workers="$WORKERS")
    launch=(python -m scoreclr.train)
    roots=(--root stl10="$DATA" --root dtd="$DATA")
  else
    data=(data.train_path="$IMAGENET100/train" data.val_path="$IMAGENET100/val" data.num_workers="$WORKERS")
    launch=(torchrun --standalone --nproc_per_node="$NGPU" -m scoreclr.train)
    roots=(--root cifar10="$DATA" --root stl10="$DATA" --root dtd="$DATA")
  fi
  echo "### train $row"
  "${launch[@]}" --config "$cfg" output_dir="$run" "${data[@]}"
  if [ ! -f "$run/evaluation.json" ]; then
    echo "### evaluate $row"
    python -m scoreclr.evaluation --run "$run" "${roots[@]}" --download --num_workers "$WORKERS" "${data[@]}"
  fi
done

python -m scoreclr.evaluation.report "$OUT"
python benchmarks/cost_model.py "$OUT/cost_vs_batch.pdf"
