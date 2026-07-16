#!/usr/bin/env bash
# Track every HDTF identity in data/hdtf_pretrain_ids.txt with VHAP, sharded
# across the available GPUs. Each GPU processes its ids sequentially; vhap_track_one.sh
# is idempotent (skips finished stages), so this is safe to re-run.
#
# Usage:
#   bash tools/vhap_track_all.sh [GPU_LIST] [IDS_FILE]
#   GPU_LIST defaults to "0 1 2 3 4 5 6 7"
#
# Logs: $WORK/logs/<ID>.log   Done markers: $WORK/export/<ID>/transforms.json
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPUS=(${1:-0 1 2 3 4 5 6 7})
IDS_FILE=${2:-data/hdtf_pretrain_ids.txt}
WORK=${WORK:-./vhap_work}
VIDEO_DIR=${VIDEO_DIR:-/path/to/HDTF/processed_videos}

mkdir -p "$WORK/logs"
# Clear any stale per-GPU queues from a previous (possibly interrupted) run.
# These are append-only worklists; leftovers would double-queue ids or inject
# malformed/empty entries (the origin of earlier truncated export names).
rm -f "$WORK/logs/.queue_gpu"*
mapfile -t IDS < <(grep -vE '^\s*#|^\s*$' "$IDS_FILE")
echo "Tracking ${#IDS[@]} ids across GPUs: ${GPUS[*]}"

ngpu=${#GPUS[@]}
for i in "${!IDS[@]}"; do
  id=${IDS[$i]}
  [[ -z "$id" ]] && continue   # never queue an empty id
  gpu=${GPUS[$(( i % ngpu ))]}
  video="$VIDEO_DIR/${id}.mp4"
  if [[ ! -f "$video" ]]; then
    echo "[skip] no video for $id ($video)"
    continue
  fi
  # Append to a per-GPU worklist file.
  echo "$id $video" >> "$WORK/logs/.queue_gpu${gpu}"
done

# Launch one worker per GPU that drains its queue sequentially.
pids=()
for gpu in "${GPUS[@]}"; do
  q="$WORK/logs/.queue_gpu${gpu}"
  [[ -f "$q" ]] || continue
  (
    while read -r id video; do
      echo "[gpu$gpu] >>> $id $(date +%T)"
      CUDA_VISIBLE_DEVICES=$gpu \
      TORCH_HOME=${HOME}/.cache/torch \
      HF_HOME=${HOME}/.cache/huggingface \
      MPLCONFIGDIR=/tmp/mpl \
        bash tools/vhap_track_one.sh "$id" "$video" "$WORK" \
        < /dev/null > "$WORK/logs/${id}.log" 2>&1 \
        && echo "[gpu$gpu] OK  $id $(date +%T)" \
        || echo "[gpu$gpu] FAIL $id $(date +%T)"
    done < "$q"
    rm -f "$q"
  ) &
  pids+=($!)
done

wait "${pids[@]}"
echo "All VHAP tracking workers finished."
