#!/usr/bin/env bash
# Extract OpenFace Action Unit features for an EmoTaG scene.
#
# EmoTaG uses six AU channels (AU01, AU04, AU05, AU06, AU07, AU45) as the
# upper-face expression cue that audio cannot carry. This wrapper runs OpenFace
# FeatureExtraction over the scene frames and writes `au_features.csv` at the
# scene root, where the dataset readers expect it (utils/au_utils.py reads the
# AU**_r columns and clips AU45 to [0, 2]).
#
# Prerequisite: build OpenFace (https://github.com/TadasBaltrusaitis/OpenFace)
# and either put FeatureExtraction on PATH or set OPENFACE_BIN.
#
# Usage:
#   bash tools/extract_openface_au.sh <scene_root> [images_subdir]
set -euo pipefail

scene_root=${1:?Usage: bash tools/extract_openface_au.sh <scene_root> [images_subdir]}
images_subdir=${2:-gt_imgs}
openface_bin=${OPENFACE_BIN:-FeatureExtraction}

img_dir="$scene_root/$images_subdir"
if [[ ! -d "$img_dir" ]]; then
  echo "Missing image directory: $img_dir" >&2
  exit 1
fi
if ! command -v "$openface_bin" >/dev/null 2>&1; then
  echo "OpenFace '$openface_bin' not found. Build OpenFace and set OPENFACE_BIN to its FeatureExtraction binary." >&2
  exit 1
fi

work_dir="$scene_root/openface_out"
mkdir -p "$work_dir"

# -fdir processes a directory of frames; -aus emits the AU intensity/presence columns.
"$openface_bin" -fdir "$img_dir" -aus -out_dir "$work_dir" -of openface

csv_out="$work_dir/openface.csv"
if [[ ! -f "$csv_out" ]]; then
  # Some OpenFace builds name the output after the input directory.
  csv_out=$(ls -1 "$work_dir"/*.csv | head -n 1)
fi
if [[ ! -f "$csv_out" ]]; then
  echo "OpenFace did not produce a CSV under $work_dir" >&2
  exit 1
fi

cp "$csv_out" "$scene_root/au_features.csv"
echo "Saved AU features to $scene_root/au_features.csv (source: $csv_out)"
echo "Columns used by EmoTaG: AU01_r, AU04_r, AU05_r, AU06_r, AU07_r, AU45_r (AU45 clipped to [0,2] at load time)."
