#!/usr/bin/env bash
# Batch feature-preprocessing for every HDTF identity in a list, sharded across
# the available GPUs. The parallel counterpart of tools/vhap_track_all.sh:
# vhap_track_all.sh produces the VHAP exports, this turns each export into a
# fully processed EmoTaG scene (import + landmarks + wav2vec2 + AU + emotion +
# identity) by calling tools/preprocess_scene.sh per id.
#
# Each GPU drains its own queue sequentially. preprocess_scene.sh is re-run only
# for ids whose scene is not already complete, so this is safe to re-run.
#
# Usage:
#   bash tools/preprocess_all.sh [GPU_LIST] [IDS_FILE]
#   GPU_LIST defaults to "0 1 2 3 4 5 6 7"
#   IDS_FILE defaults to data/hdtf_pretrain_ids.txt
#
# Logs: $WORK/logs/pre_<ID>.log    Scenes: data/<ID>/
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS=(${1:-0 1 2 3 4 5 6 7})
IDS_FILE=${2:-data/hdtf_pretrain_ids.txt}
WORK=${WORK:-./vhap_work}
DATA_DIR=${DATA_DIR:-data}

# Shared resources for preprocess_scene.sh.
export EMOTAG_FLAME_MODEL=${EMOTAG_FLAME_MODEL:-/path/to/FLAME/generic_model.pkl}
export CVLFACE_ROOT=${CVLFACE_ROOT:-/path/to/CVLface}
export LINK_MODE=${LINK_MODE:-symlink}
export RUN_SAPIENS=${RUN_SAPIENS:-0}            # pre-training does not need geometry priors
export TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch}
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}

mkdir -p "$WORK/logs"
mapfile -t IDS < <(grep -vE '^\s*#|^\s*$' "$IDS_FILE")
echo "Preprocessing ${#IDS[@]} ids across GPUs: ${GPUS[*]}"

# A scene is considered complete when the final-stage products all exist.
scene_done() {
  local d="$1"
  [[ -f "$d/flame_params.npz" && -f "$d/aud_w2v.npy" && -f "$d/au_features.csv" \
     && -f "$d/emotion_features.npy" && -f "$d/identity_feature.npy" ]]
}

# Clear any stale queues from a previous run.
for gpu in "${GPUS[@]}"; do rm -f "$WORK/logs/.prequeue_gpu${gpu}"; done

ngpu=${#GPUS[@]}
qi=0
for id in "${IDS[@]}"; do
  scene="$DATA_DIR/$id"
  export_dir="$WORK/export/$id"
  if scene_done "$scene"; then
    echo "[done]  $id (scene already complete)"; continue
  fi
  if [[ ! -f "$export_dir/transforms.json" ]]; then
    echo "[wait]  $id (no VHAP export yet at $export_dir)"; continue
  fi
  gpu=${GPUS[$(( qi % ngpu ))]}
  echo "$id" >> "$WORK/logs/.prequeue_gpu${gpu}"
  qi=$((qi+1))
done

if [[ "$qi" -eq 0 ]]; then echo "Nothing to do (all complete or no exports ready)."; exit 0; fi

pids=()
for gpu in "${GPUS[@]}"; do
  q="$WORK/logs/.prequeue_gpu${gpu}"
  [[ -f "$q" ]] || continue
  (
    while read -r id; do
      scene="$DATA_DIR/$id"
      echo "[gpu$gpu] >>> $id $(date +%T)"
      CUDA_VISIBLE_DEVICES=$gpu DEVICE=cuda \
      VHAP_EXPORT="$WORK/export/$id" \
      WAV="$WORK/wav/${id}.wav" \
        bash tools/preprocess_scene.sh "$scene" \
        < /dev/null > "$WORK/logs/pre_${id}.log" 2>&1 \
        && echo "[gpu$gpu] OK   $id $(date +%T)" \
        || echo "[gpu$gpu] FAIL $id $(date +%T)"
    done < "$q"
    rm -f "$q"
  ) &
  pids+=($!)
done

wait "${pids[@]}"
echo "All preprocessing workers finished."
