#!/usr/bin/env bash
set -euo pipefail

scene_root=${1:?Usage: bash tools/sapiens/run.sh <scene_root>}

bash tools/sapiens/lite/scripts/depth.sh "$scene_root"
bash tools/sapiens/lite/scripts/normal.sh "$scene_root"
