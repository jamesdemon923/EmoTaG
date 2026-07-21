#!/usr/bin/env bash
# End-to-end preprocessing for a single EmoTaG scene.
#
# Chains every step needed to turn an HDTF/monocular video into a processed
# scene that pretrain_emotag.py / adapt_emotag.py can consume:
#
#   0. vhap_track_one.sh          (optional) VHAP track + export from a video
#   1. import_vhap_export.py      VHAP export -> EmoTaG scene + FLAME/Gaussian init
#   2. extract_landmarks.py       real FAN 68-pt landmarks (ori_imgs/<id>.lms)
#   3. extract_wav2vec2_features  audio features (aud_w2v.npy)
#   4. extract_pyfeat_au.py       py-feat Action Units (au_features.csv)
#   5. extract_deepface_emotion   DeepFace emotion teacher (emotion_features.npy)
#   6. extract_adaface_identity   AdaFace/CVLface identity descriptor (identity_feature.npy)
#   7. run_sapiens_priors.sh      depth/normal priors (adaptation only; optional)
#   8. repro_check.py             validate the scene
#
# Each feature step runs in its own conda env via *_PY overrides (defaults below
# point at the envs on this machine). Any step can be skipped with SKIP_<STEP>=1.
#
# Usage (features only; VHAP export already exists):
#   VHAP_EXPORT=/path/to/EmoTaG_vhap_work/export/<ID> \
#   WAV=/path/to/<ID>.wav \
#   EMOTAG_FLAME_MODEL=/path/to/generic_model.pkl \
#   bash tools/preprocess_scene.sh data/<ID>
#
# Usage (run VHAP first from a raw video):
#   RUN_VHAP=1 VIDEO=/path/to/<ID>.mp4 ... bash tools/preprocess_scene.sh data/<ID>
set -euo pipefail

scene=${1:?Usage: bash tools/preprocess_scene.sh <scene_out_dir>}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Keep model caches off the home quota (py-feat / HF / torch hub all download).
export TORCH_HOME=${TORCH_HOME:-${HOME}/.cache/torch}
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}

# Cap CPU threads per scene. The py-feat AU step (HOG+XGBoost) and the BLAS/torch
# CPU ops otherwise each spawn one thread per core; running several scenes in
# parallel then oversubscribes the machine. Capping keeps N parallel scenes within
# the core count. Override with EMOTAG_CPU_THREADS.
export OMP_NUM_THREADS=${EMOTAG_CPU_THREADS:-36}
export OPENBLAS_NUM_THREADS=${EMOTAG_CPU_THREADS:-36}
export MKL_NUM_THREADS=${EMOTAG_CPU_THREADS:-36}
export NUMEXPR_NUM_THREADS=${EMOTAG_CPU_THREADS:-36}
export VECLIB_MAXIMUM_THREADS=${EMOTAG_CPU_THREADS:-36}

# import_vhap_export.py builds the FLAME/Gaussian init via VHAP's FlameHead, which
# is imported from the VHAP source tree (not pip-installed in the EmoTaG env).
# Without this, geometry generation is silently skipped and the scene has no
# points3D.ply / flame_params.npz / bindings. Override EMOTAG_VHAP_ROOT to point
# at your VHAP checkout.
export EMOTAG_VHAP_ROOT=${EMOTAG_VHAP_ROOT:-/path/to/VHAP}

# Per-step interpreters. EMOTAG_PY defaults to the python on PATH (the emotag
# env); the feature extractors live in their own envs (see README Step 4-6) —
# point AU_PY / DEEPFACE_PY / ADAFACE_PY at them, or leave unset to reuse
# EMOTAG_PY if you installed those packages into one env.
EMOTAG_PY=${EMOTAG_PY:-python}
AU_PY=${AU_PY:-$EMOTAG_PY}
DEEPFACE_PY=${DEEPFACE_PY:-$EMOTAG_PY}
ADAFACE_PY=${ADAFACE_PY:-$EMOTAG_PY}

# ---- 0. (optional) VHAP track + export ------------------------------------
if [[ "${RUN_VHAP:-0}" == "1" ]]; then
  : "${VIDEO:?Set VIDEO to the source mp4 when RUN_VHAP=1.}"
  ID=$(basename "$scene")
  echo "[0] VHAP tracking $ID from $VIDEO"
  export_line=$(bash tools/vhap_track_one.sh "$ID" "$VIDEO" | tee /dev/stderr | grep '^EXPORT=' | tail -1)
  VHAP_EXPORT=${export_line#EXPORT=}
fi

# ---- 1. Import VHAP export -------------------------------------------------
if [[ "${SKIP_IMPORT:-0}" != "1" ]]; then
  : "${VHAP_EXPORT:?Set VHAP_EXPORT to the VHAP export directory (or SKIP_IMPORT=1).}"
  : "${EMOTAG_FLAME_MODEL:?Set EMOTAG_FLAME_MODEL to generic_model.pkl.}"
  echo "[1/8] Importing VHAP export -> $scene"
  "$EMOTAG_PY" tools/import_vhap_export.py \
    --vhap_export "$VHAP_EXPORT" \
    --output "$scene" \
    --flame_model_path "$EMOTAG_FLAME_MODEL" \
    --num_points "${NUM_POINTS:-60000}" \
    --link_mode "${LINK_MODE:-copy}"
fi

# ---- 2. Real landmarks -----------------------------------------------------
if [[ "${SKIP_LANDMARKS:-0}" != "1" ]]; then
  echo "[2/8] Extracting FAN landmarks"
  "$EMOTAG_PY" tools/extract_landmarks.py --scene "$scene" --device "${DEVICE:-cuda}"
fi

# ---- 3. Wav2Vec2 audio features -------------------------------------------
if [[ "${SKIP_AUDIO:-0}" != "1" ]]; then
  : "${WAV:?Set WAV to the scene audio .wav (or SKIP_AUDIO=1).}"
  echo "[3/8] Extracting Wav2Vec2 features"
  "$EMOTAG_PY" tools/extract_wav2vec2_features.py \
    --wav "$WAV" \
    --output "$scene/aud_w2v.npy" \
    --transforms "$scene/transforms.json"
fi

# ---- 4. py-feat Action Unit features --------------------------------------
if [[ "${SKIP_AU:-0}" != "1" ]]; then
  echo "[4/8] Extracting py-feat AU features"
  "$AU_PY" tools/extract_pyfeat_au.py --scene "$scene"
fi

# ---- 5. DeepFace emotion teacher ------------------------------------------
if [[ "${SKIP_EMOTION:-0}" != "1" ]]; then
  echo "[5/8] Extracting DeepFace emotion teacher"
  "$DEEPFACE_PY" tools/extract_deepface_emotion.py --scene "$scene"
fi

# ---- 6. AdaFace identity descriptor ---------------------------------------
if [[ "${SKIP_IDENTITY:-0}" != "1" ]]; then
  echo "[6/8] Extracting AdaFace identity descriptor (CVLface)"
  "$ADAFACE_PY" tools/extract_adaface_identity.py \
    --scene "$scene" \
    ${CVLFACE_ROOT:+--cvlface_root "$CVLFACE_ROOT"}
fi

# ---- 7. Sapiens geometry priors (adaptation only) -------------------------
if [[ "${RUN_SAPIENS:-0}" == "1" ]]; then
  echo "[7/8] Running Sapiens depth/normal priors"
  bash tools/run_sapiens_priors.sh "$scene"
else
  echo "[7/8] Skipping Sapiens priors (set RUN_SAPIENS=1 for adaptation scenes)."
fi

# ---- 8. Validate -----------------------------------------------------------
echo "[8/8] Validating scene"
"$EMOTAG_PY" tools/repro_check.py --root "$scene" --imports || true

echo "Done. Processed scene: $scene"
