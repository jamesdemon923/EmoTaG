#!/usr/bin/env bash
set -euo pipefail

scene_root=${1:?Usage: bash tools/run_sapiens_priors.sh <scene_root>}

if [[ ! -d "$scene_root/gt_imgs" ]]; then
  echo "Missing image directory: $scene_root/gt_imgs" >&2
  exit 1
fi

# Run this inside a Sapiens-compatible environment.
# Configure checkpoint paths, GPU ids, and frame count in tools/sapiens/lite/scripts.
bash tools/sapiens/run.sh "$scene_root"
