#!/usr/bin/env bash
# Track one HDTF identity with VHAP and export a NeRF-style dataset that
# tools/import_vhap_export.py can consume.
#
# Steps (all in the VHAP conda env):
#   1. preprocess_video.py  : frames @25fps + RobustVideoMatting fg masks
#   2. track.py             : monocular FLAME tracking
#   3. export_as_nerf_dataset.py : transforms.json + images + flame_param
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash tools/vhap_track_one.sh <ID> [<video.mp4>] [<work_root>]
#
# Defaults: video = HDTF/data_90/processed_videos/<ID>.mp4
#           work_root = ./vhap_work
# Output export folder is printed on the last line as: EXPORT=<path> (the frames with background: FRAMES=<path>)
set -euo pipefail

ID=${1:?Usage: bash tools/vhap_track_one.sh <ID> [video] [work_root]}
VIDEO=${2:-/path/to/HDTF/processed_videos/${ID}.mp4}
WORK=$(realpath -m "${3:-./vhap_work}")   # absolute: the VHAP steps run from VHAP_DIR
VHAP_DIR=${VHAP_DIR:-/path/to/VHAP}
VHAP_ENV=${VHAP_ENV:-/path/to/conda/envs/VHAP}
VHAP_PY=${VHAP_PY:-$VHAP_ENV/bin/python}
# ffprobe/ffmpeg live in the env bin; verify_ninja and ffmpeg-python need them on PATH.
export PATH="$VHAP_ENV/bin:$PATH"
# Keep JIT-compiled CUDA plugins (nvdiffrast) and model caches off the home quota.
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}
export TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch}
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}
mkdir -p "$TORCH_EXTENSIONS_DIR"

[[ -f "$VIDEO" ]] || { echo "Missing video: $VIDEO" >&2; exit 1; }

DATA_ROOT="$WORK/data/monocular"
TRACK_OUT="$WORK/output/track/${ID}"
EXPORT_OUT="$WORK/export/${ID}"
SEQ_DIR="$DATA_ROOT/${ID}"

mkdir -p "$DATA_ROOT"
# Cap clip length to speed up tracking: long HDTF clips (120-240s) dominate VHAP
# time, but pretraining only needs identity diversity, not long per-id clips.
# Trim anything over MAX_VIDEO_SECONDS (default 160s). Stage the (possibly trimmed)
# video under the root so preprocess_video writes frames to <root>/<seq>/images,
# exactly where track.py (root_folder + sequence) looks.
MAX_SECONDS=${MAX_VIDEO_SECONDS:-160}
STAGED="$DATA_ROOT/${ID}.mp4"
need_stage=0
if [[ ! -f "$STAGED" ]]; then
  need_stage=1
else
  sdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$STAGED" 2>/dev/null | cut -d. -f1)
  if [[ -n "$sdur" && "$sdur" -gt "$((MAX_SECONDS + 2))" ]]; then
    echo "[$ID] staged clip ${sdur}s > cap ${MAX_SECONDS}s; re-staging trimmed (clearing cached frames/track)"
    rm -rf "$SEQ_DIR/images" "$SEQ_DIR"/*.npz "$TRACK_OUT"
    need_stage=1
  fi
fi
if [[ "$need_stage" == "1" ]]; then
  vdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VIDEO" 2>/dev/null | cut -d. -f1)
  if [[ -n "$vdur" && "$vdur" -gt "$MAX_SECONDS" ]]; then
    echo "[$ID] trimming source ${vdur}s -> ${MAX_SECONDS}s"
    ffmpeg -nostdin -y -i "$VIDEO" -t "$MAX_SECONDS" -c copy "$STAGED" 2>/dev/null \
      || ffmpeg -nostdin -y -i "$VIDEO" -t "$MAX_SECONDS" "$STAGED" 2>/dev/null
  else
    cp "$VIDEO" "$STAGED"
  fi
fi

cd "$VHAP_DIR"

# VHAP tracks with FLAME 2023 by default; EmoTaG decodes the tracking with FLAME 2020 (generic_model.pkl),
# so track.py / export run with VHAP_FLAME_MODEL (path relative to the VHAP root) as FlameHead's model.
VHAP_FLAME_MODEL=${VHAP_FLAME_MODEL:-asset/flame/generic_model.pkl}
run_vhap() {
  "$VHAP_PY" -c 'import runpy, sys
import vhap.model.flame as F
d = list(F.FlameHead.__init__.__defaults__); d[0] = F.FLAME_MODEL_PATH = sys.argv[1]
F.FlameHead.__init__.__defaults__ = tuple(d)
sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], run_name="__main__")' "$VHAP_FLAME_MODEL" "$@"
}

if [[ ! -d "$SEQ_DIR/images" || -z "$(ls -A "$SEQ_DIR/images" 2>/dev/null)" ]]; then
  echo "[$ID] (1/3) preprocess_video"
  "$VHAP_PY" vhap/preprocess_video.py \
    --input "$DATA_ROOT/${ID}.mp4" \
    --matting_method robust_video_matting
fi

# track.py writes into a time-stamped sub-folder of TRACK_OUT; use the latest one
track_dir() { ls -d "$TRACK_OUT"/*/ 2>/dev/null | sort | tail -1; }
if ! ls "$(track_dir)"/tracked_flame_params*.npz >/dev/null 2>&1; then
  echo "[$ID] (2/3) track"
  run_vhap vhap/track.py \
    --data.root_folder "$DATA_ROOT" \
    --data.sequence "$ID" \
    --exp.output_folder "$TRACK_OUT"
fi

if [[ ! -f "$EXPORT_OUT/transforms.json" ]]; then
  echo "[$ID] (3/3) export"
  run_vhap vhap/export_as_nerf_dataset.py \
    --src_folder "$(track_dir)" \
    --tgt_folder "$EXPORT_OUT" \
    --background-color white
fi

echo "[$ID] done"
echo "FRAMES=$SEQ_DIR/images"
echo "EXPORT=$EXPORT_OUT"
