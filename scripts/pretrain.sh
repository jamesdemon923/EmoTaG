#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?Usage: bash scripts/pretrain.sh <dataset_root> <output_root> <gpu_id> [scene_names]}
workspace=${2:?Usage: bash scripts/pretrain.sh <dataset_root> <output_root> <gpu_id> [scene_names]}
gpu_id=${3:?Usage: bash scripts/pretrain.sh <dataset_root> <output_root> <gpu_id> [scene_names]}
scene_names=${4:-}

audio_extractor=${AUDIO_EXTRACTOR:-wav2vec2}
iterations=${ITERATIONS:-30000}
batch_size=${BATCH_SIZE:-1}
warmup=${WARMUP_PER_SCENE:-500}

export CUDA_VISIBLE_DEVICES="$gpu_id"

extra_args=()
if [[ -n "$scene_names" ]]; then
  extra_args+=(--scene_names "$scene_names")
fi

python pretrain_emotag.py \
  -s "$dataset" \
  -m "$workspace" \
  --audio_extractor "$audio_extractor" \
  --iterations "$iterations" \
  --warm_up_iter_per_scene "$warmup" \
  --w_flame_reg 1.0 \
  --batch_size "$batch_size" \
  "${extra_args[@]}"
