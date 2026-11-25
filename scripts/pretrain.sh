#!/usr/bin/env bash
# Pre-training of the released checkpoint (run from the repository root):
#   1. multi-identity pre-training on the HDTF scenes in data/ (Wav2Vec2)  -> output/pretrain/chkpnt_ema_face_latest.pth
#   2. audio-to-motion part on public FLAME tracks (Whisper)               -> output/prior/chkpnt_ema_face_latest.pth
#
#   bash scripts/pretrain.sh <D5_unitalker_faceforensics++> <HDTF_TFHP-lmdb> <FLAME2020/generic_model.pkl>
#
# Data: the HDTF scenes of data/hdtf_pretrain_ids.txt (processed as in the README), and public FLAME tracks:
#   D5_unitalker_faceforensics++  https://github.com/X-niper/UniTalker#download-data
#   HDTF_TFHP-lmdb                https://github.com/DiffPoseTalk/DiffPoseTalk/tree/main/datasets/HDTF_TFHP
#   FLAME 2020 generic_model.pkl  https://flame.is.tue.mpg.de (basis of the tracks)
# data/audio_prior_exclude.txt keeps the evaluation speakers and recordings out of training.
# For multi-GPU pre-training, run pretrain_emotag_ddp.py with torchrun (same arguments).
#
# PRETRAIN=<checkpoint> skips step 1 and starts step 2 from that checkpoint.
# DATA, OUTPUT and PRIOR_DATA change the scene, output and prepared-track directories.
set -euo pipefail

usage='Usage: bash scripts/pretrain.sh <D5_unitalker_faceforensics++> <HDTF_TFHP-lmdb> <FLAME2020/generic_model.pkl>'
d5=${1:?$usage}
hdtf_tfhp=${2:?$usage}
flame2020=${3:?$usage}
data=${DATA:-data}
output=${OUTPUT:-output}
prior_data=${PRIOR_DATA:-data_prior}
pretrain=${PRETRAIN:-}

if [[ -z "$pretrain" ]]; then
  # --iterations is per scene: 3600 over the 70 HDTF scenes is the paper's ~250K steps
  python pretrain_emotag.py -s "$data" -m "$output/pretrain" --scene_names_file data/hdtf_pretrain_ids.txt \
    --audio_extractor wav2vec2 --iterations 3600 --warm_up_iter_per_scene 1000 --w_flame_reg 1.0 --batch_size 1
  pretrain=$output/pretrain/chkpnt_ema_face_latest.pth
fi

python tools/prepare_audio_prior_data.py --source unitalker --root "$d5" --flame2020 "$flame2020" --out "$prior_data/d5"
python tools/prepare_audio_prior_data.py --source diffposetalk --root "$hdtf_tfhp" --flame2020 "$flame2020" \
  --out "$prior_data/hdtf_tfhp" --shards 4
python tools/train_audio_prior.py --stage 1 --init "$pretrain" --data "$prior_data/d5" "$prior_data/hdtf_tfhp" \
  --out "$output/prior/stage1.pth"
python tools/train_audio_prior.py --stage 2 --init "$output/prior/stage1.pth" --data "$prior_data/hdtf_tfhp" \
  --out "$output/prior/chkpnt_ema_face_latest.pth"
