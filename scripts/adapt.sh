#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?Usage: bash scripts/adapt.sh <scene_root> <output_root> <n_views> <gpu_id> <pretrain_checkpoint>}
workspace=${2:?Usage: bash scripts/adapt.sh <scene_root> <output_root> <n_views> <gpu_id> <pretrain_checkpoint>}
n_views=${3:-125}
gpu_id=${4:?Usage: bash scripts/adapt.sh <scene_root> <output_root> <n_views> <gpu_id> <pretrain_checkpoint>}
pretrain_checkpoint=${5:?Usage: bash scripts/adapt.sh <scene_root> <output_root> <n_views> <gpu_id> <pretrain_checkpoint>}

audio_extractor=${AUDIO_EXTRACTOR:-wav2vec2}
iterations=${ITERATIONS:-20000}
warmup=${WARMUP_ITER:-1000}

export CUDA_VISIBLE_DEVICES="$gpu_id"

python adapt_emotag.py \
  --type face \
  -s "$dataset" \
  -m "$workspace" \
  --audio_extractor "$audio_extractor" \
  --pretrain_path "$pretrain_checkpoint" \
  --iterations "$iterations" \
  --warm_up_iter "$warmup" \
  --sh_degree 1 \
  --N_views "$n_views"
