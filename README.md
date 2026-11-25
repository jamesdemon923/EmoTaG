<div align="center">

# EmoTaG: Emotion-Aware Talking Head Synthesis<br>with Gaussian Splatting

<p>
  <a href="https://arxiv.org/abs/2603.21332"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv" alt="Paper"></a>
  <a href="https://emotag26.github.io/"><img src="https://img.shields.io/badge/Project-Page-1f6feb.svg?logo=githubpages" alt="Project Page"></a>
  <a href="https://www.youtube.com/watch?v=kEPHIu8I7HE"><img src="https://img.shields.io/badge/YouTube-Video-FF0000.svg?logo=youtube&logoColor=white" alt="Video"></a>
</p>

Official implementation of **EmoTaG** (CVPR 2026).

<img src="./img/pipeline.png" alt="EmoTaG overview" width="92%" />

</div>

---

## Installation

> Tested on **Ubuntu 22.04** with **CUDA 12.1** and **PyTorch**.

### 1. Clone the repository and create the environment

```bash
git clone https://github.com/jamesdemon923/EmoTaG.git
cd EmoTaG
git submodule update --init --recursive

conda env create --file environment_cu121.yml
conda activate emotag
```

### 2. Install PyTorch3D and project submodules

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -e submodules/simple-knn
pip install -e gridencoder
pip install -e shencoder
```

The renderer uses the depth/normal/alpha-capable rasterizer
[`diff_gauss`](https://github.com/slothfulxtx/diff-gaussian-rasterization):

```bash
git clone --recursive https://github.com/slothfulxtx/diff-gaussian-rasterization.git
pip install ./diff-gaussian-rasterization
```

### 3. Third-party resources

| Resource | Purpose |
| :--- | :--- |
| [FLAME](https://flame.is.tue.mpg.de/index.html) | FLAME parametric head model |
| [VHAP](https://github.com/ShenhanQian/VHAP) | FLAME tracking environment |
| [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) | Audio feature extraction backbone |
| [Whisper](https://huggingface.co/openai/whisper-small) | Multilingual audio feature extraction |
| [py-feat](https://py-feat.org/) | Action Unit (AU) feature extraction |
| [DeepFace](https://github.com/serengil/deepface) | Per-frame emotion features |
| [CVLface](https://github.com/mk-minchul/CVLface) | Identity embedding |
| [Sapiens](https://github.com/facebookresearch/sapiens/blob/main/lite/README.md) | Depth and normal priors for adaptation |
| [SyncNet](https://github.com/joonson/syncnet_python) | Audio/video offset and lip-sync metrics (`SYNCNET_ROOT`) |

Export the FLAME / VHAP paths used by the preprocessing tools:

```bash
export EMOTAG_FLAME_MODEL=/path/to/flame/generic_model.pkl
export EMOTAG_VHAP_ROOT=/path/to/VHAP
```

---

## Datasets

EmoTaG is trained and evaluated on two public talking-head corpora:

| Dataset | Role | Subset used by EmoTaG |
| :--- | :--- | :--- |
| [HDTF](https://github.com/MRzzm/HDTF) | Multi-identity pre-training | 70 videos (one identity each, 90–240 s) sampled from HDTF, to learn the identity-agnostic audio-motion prior. |
| [MEAD](https://wywu.github.io/projects/MEAD/MEAD.html) | Emotional evaluation | A subset of 5 emotions × 3 intensity levels (2 identities per emotion) from MEAD, for the emotional / emotion-intensity tests. |

Please obtain both datasets from their official sources and follow their
respective licenses. All clips must be face-centered cropped and resized to
**512 × 512 at 25 FPS** before processing. HDTF ships official crop metadata;
for MEAD / custom videos use the bundled cropper (a square, uniformly scaled
crop — no aspect distortion):

```bash
python tools/crop_video.py --input raw.mp4 --output data/<ID>/<ID>.mp4
# preview the detected crop first with --dry_run
```

---

### Evaluation data

The processed evaluation scenes (10 neutral identities + MEAD emotional sets,
ready for adaptation/evaluation without any preprocessing) are available here:

### 📦 [**Download test data (Google Drive)**](https://drive.google.com/drive/folders/1zVB6NwWjVlXnZozIXUJxcjLVEMku9kNW?usp=drive_link)

```bash
tar xzf neutral_Obama.tar.gz -C data/
# regenerate the Sapiens depth/normal priors of adaptation scenes (excluded from the
# archives for size; deterministic, see Data Preparation Step 7):
bash tools/run_sapiens_priors.sh data/neutral_Obama
```

See `README.txt` in the folder for its contents.

---

## Data Preparation

These steps turn your own video into a processed scene (the released evaluation scenes
above are already processed). A **processed scene** contains calibrated images, FLAME tracking, Gaussian
initialization, audio features, AU features, the emotion teacher, and the
identity descriptor. `tools/preprocess_scene.sh` runs all steps below for one
scene (add `RUN_REFINE=1` for Step 4b, and `RUN_SAPIENS=1` for adaptation scenes); then fix
the audio offset as in Step 3:

```bash
EMOTAG_FLAME_MODEL=/path/to/FLAME/generic_model.pkl \
VHAP_EXPORT=/path/to/VHAP/export/<ID> \
VHAP_FRAMES=/path/to/VHAP/data/<ID>/images \
WAV=data/<ID>/aud.wav \
CVLFACE_ROOT=/path/to/CVLface \
bash tools/preprocess_scene.sh data/<ID>
```

The steps below document what each stage produces if you prefer to run them
individually.

<details>
<summary><b>Expected directory layout</b></summary>

```text
data/<ID>/
├── transforms.json
├── flame_params.npz
├── model_center.npy
├── points3D.ply
├── face_indices.npy
├── bary_coords.npy
├── vertices.npy
├── mouth_point_indices.npy
├── aud.wav
├── aud_whisper.npy
├── aud_w2v.npy
├── au_features.csv
├── emotion_features.npy      
├── identity_feature.npy     
├── ori_imgs/
├── gt_imgs/
├── parsing/
├── torso_imgs/
└── sapiens/                 # for adaptation
```

</details>

### Step 1 — FLAME tracking with VHAP

Run the VHAP monocular pipeline, then convert the export to the EmoTaG scene layout:

```bash
python tools/import_vhap_export.py \
  --vhap_export /path/to/VHAP/export/monocular/<ID>_whiteBg_staticOffset \
  --output      data/<ID> \
  --flame_model_path "$EMOTAG_FLAME_MODEL" \
  --frames_dir  /path/to/VHAP/data/<ID>/images \
  --num_points 60000 \
  --link_mode  symlink
```

### Step 2 — Facial landmarks (FAN)

Per-frame 68-point landmarks (`ori_imgs/<id>.lms`) define the lip/mouth crops
that supervise the mouth Gaussians:

```bash
python tools/extract_landmarks.py --scene data/<ID>
# -> data/<ID>/ori_imgs/<id>.lms   shape [68, 2]
```

### Step 3 — Audio features (Whisper / Wav2Vec2)

EmoTaG supports two audio encoders (`--audio_extractor whisper` or `wav2vec2`); the released
checkpoint uses Whisper. The audio of a recording is often a few frames off the video; since the
mouth motion is learned from the audio, `check_av_sync.py` measures this offset with SyncNet and
shifts the audio to match.

```bash
python tools/extract_whisper_features.py \
  --wav        data/<ID>/aud.wav \
  --output     data/<ID>/aud_whisper.npy \
  --transforms data/<ID>/transforms.json
python tools/extract_wav2vec2_features.py \
  --wav        data/<ID>/aud.wav \
  --output     data/<ID>/aud_w2v.npy \
  --transforms data/<ID>/transforms.json

# fix the audio-video offset (needs SYNCNET_ROOT=/path/to/syncnet_python);
# add --split_at 125 when the first 125 frames come from another take
python tools/check_av_sync.py data/<ID> --fix
```

### Step 4 — AU features (py-feat)

The six AU channels (`AU01, AU04, AU05, AU06, AU07, AU45`) supply the upper-face
expression cues that audio cannot carry (`AU45` is clipped to `[0, 2]` at load
time, following Supp. Sec. B.2). Extract them with [py-feat](https://py-feat.org/):

```bash
python tools/extract_pyfeat_au.py --scene data/<ID>
# -> data/<ID>/au_features.csv
```

### Step 4b — Lip and eyelid re-fit

Monocular tracking often misses lip closures and blinks, which supervise the mouth and eyelid motion.
This re-fits the lip and eyelid state to the image (the tracking is kept as `flame_params_tracked.npz`):

```bash
python tools/refine_landmarks.py data/<ID>
```

### Step 5 — Emotion teacher (DeepFace)

Run [DeepFace](https://github.com/serengil/deepface) per frame to obtain the
7-way emotion distribution used to supervise the emotion-aware branches:

```bash
python tools/extract_deepface_emotion.py --scene data/<ID>
# -> data/<ID>/emotion_features.npy   shape [N, 7]
#    columns = [angry, disgust, fear, happy, sad, surprise, neutral]
```

### Step 6 — Identity descriptor (CVLface)

Compute the identity descriptor `s` as the average face-recognition embedding
over the top-50 neutral frames (run Step 5 first so neutral frames can be
ranked). The extractor uses a [CVLface](https://github.com/mk-minchul/CVLface)
model; clone CVLface and point `--cvlface_root` at it (its pretrained models are
fetched under `<cvlface_root>/cvlface/pretrained_models`):

```bash
python tools/extract_adaface_identity.py \
  --scene         data/<ID> \
  --cvlface_root  /path/to/CVLface
# -> data/<ID>/identity_feature.npy   shape [512]
```

### Step 7 — Geometry priors (Sapiens)

> Not required for pre-training.

Put the Sapiens 0.3b TorchScript checkpoints ([depth](https://huggingface.co/facebook/sapiens-depth-0.3b-torchscript),
[normal](https://huggingface.co/facebook/sapiens-normal-0.3b-torchscript)) into
`tools/sapiens/checkpoint/sapiens-depth-0.3b-torchscript/` and `.../sapiens-normal-0.3b-torchscript/`, then:

```bash
bash tools/run_sapiens_priors.sh data/<ID>
```

The script writes depth and normal maps under:

```text
data/<ID>/sapiens/
├── depth/sapiens_*/<frame_id>.npy
└── normal/sapiens_*/<frame_id>.npy
```

### Step 8 — Validate the scene

```bash
python tools/repro_check.py --root data/<ID> --imports
```

The checker reports whether each required scene file is present and valid.

---

## Training

### Pre-training (multi-identity)

```bash
python pretrain_emotag.py \
  -s data \
  -m output/pretrain \
  --scene_names_file data/hdtf_pretrain_ids.txt \
  --audio_extractor wav2vec2 \
  --iterations 3600 \
  --warm_up_iter_per_scene 1000 \
  --w_flame_reg 1.0 \
  --batch_size 1
```

> `--iterations` is **per scene** (multiplied by the number of scenes
> internally), so `3600` over the 70 HDTF ids is the paper's ~250K-step budget.
> Full hyper-parameters are in
> [`configs/pretrain_hdtf70.yaml`](configs/pretrain_hdtf70.yaml); the EMA
> checkpoint is written to `output/pretrain/chkpnt_ema_face_latest.pth`.
> For multi-GPU, run `pretrain_emotag_ddp.py` with `torchrun` (same arguments).

EmoTaG supports both Wav2Vec2 and Whisper features (`--audio_extractor`); the released checkpoint uses Whisper.
To obtain it, train the audio-to-motion part of the pre-trained model on public FLAME tracks of talking-head videos:
- FaceForensics++: `D5_unitalker_faceforensics++` from the
  [UniTalker data release V1](https://github.com/X-niper/UniTalker#download-data).
- HDTF + TFHP: `HDTF_TFHP-lmdb` from
  [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk/tree/main/datasets/HDTF_TFHP) (`pip install lmdb`).
- [FLAME 2020](https://flame.is.tue.mpg.de) `generic_model.pkl`, to map their expressions to FLAME 2019.

```bash
F2020=/path/to/FLAME2020/generic_model.pkl
python tools/prepare_audio_prior_data.py --source unitalker --root /path/to/D5_unitalker_faceforensics++ \
  --flame2020 $F2020 --out data_prior/d5
for i in 0 1 2 3; do
  python tools/prepare_audio_prior_data.py --source diffposetalk --root /path/to/HDTF_TFHP-lmdb --flame2020 $F2020 \
    --exclude data/audio_prior_exclude.txt --splits train --shard $i/4 --out data_prior/hdtf_tfhp
done
python tools/prepare_audio_prior_data.py --source diffposetalk --root /path/to/HDTF_TFHP-lmdb --flame2020 $F2020 \
  --exclude data/audio_prior_exclude.txt --splits val --out data_prior/hdtf_tfhp

python tools/train_audio_prior.py --init output/pretrain/chkpnt_ema_face_latest.pth \
  --data data_prior/d5 data_prior/hdtf_tfhp --out output/prior/stage1.pth --steps 40000
python tools/train_audio_prior.py --init output/prior/stage1.pth \
  --data data_prior/hdtf_tfhp --out output/prior/chkpnt_ema_face_latest.pth --steps 30000 --w_vertex 1 --align --select aperture
```

`output/prior/chkpnt_ema_face_latest.pth` is the released checkpoint;
[`data/audio_prior_exclude.txt`](data/audio_prior_exclude.txt) keeps the evaluation speakers and recordings out of training.

### Few-shot adaptation (target identity)

```bash
python adapt_emotag.py \
  -s data/<ID> \
  -m output/<ID> \
  --audio_extractor whisper \
  --pretrain_path /path/to/chkpnt_ema_face_latest.pth \
  --iterations 20000 \
  --warm_up_iter 1000 \
  --N_views 125 \
  --adapt_adain_only \
  --adapt_unfreeze_heads \
  --intraoral rigid --intraoral_contain --intraoral_budget 6000 \
  --motion_steps 1000
```

> `--adapt_adain_only` freezes the pretrained GRMN and tunes only the AdaIN
> modulation; `--adapt_unfreeze_heads` additionally fine-tunes the small
> FLAME/mouth prediction heads. Full hyper-parameters are in
> [`configs/adapt_5s.yaml`](configs/adapt_5s.yaml). `--pretrain_path` takes
> `output/prior/chkpnt_ema_face_latest.pth` or the released checkpoint below; for Wav2Vec2, use
> `output/pretrain/chkpnt_ema_face_latest.pth` with `--audio_extractor wav2vec2`.

### Pretrained checkpoint

To skip pre-training, download the multi-identity motion prior and adapt directly:

### 📦 [**Download checkpoint (Google Drive)**](https://drive.google.com/drive/folders/17wgFUGF7pxbW3QlBkiHQzo3nYTzF-fiN?usp=drive_link)

---

## Inference

Render the composited video (the synthesized head over the source video's background,
torso from the source frames, the standard talking-head protocol):

```bash
python tools/render_composited.py -s data/<ID> -m output/<ID>
```

This writes `output/<ID>/test/rendered_comp_audio.mp4` (with audio), `rendered_comp.mp4` and
the matching `gt_comp.mp4`.

---

## Evaluation

Render the held-out segment (frames after the 5s/125-frame adaptation clip)
and compute the video metrics on it (commands for each test set: `README.txt` of the test data):

```bash
python tools/render_composited.py -s data/<ID> -m output/<ID> --eval_from 125

python evaluate_metrics.py video \
  output/<ID>/test/rendered_comp.mp4 \
  output/<ID>/test/gt_comp.mp4 \
  output/<ID>/test --masks output/<ID>/test/masks_comp.npz
```

All test settings on the released scenes (Sync-C/E need `SYNCNET_ROOT`):

```bash
python tools/run_benchmark.py all --data data --output output       # models in output/<scene>
python tools/run_benchmark.py summary --output output
```

### Action-Unit metrics (optional)

The AU error (AUE) compares AU intensities between the rendered and the
ground-truth videos. First run an AU extractor (see Step 4) on the **frames of
each video** to produce two CSVs, then:

```bash
python evaluate_metrics.py au \
  output/<ID>/test/rendered_au.csv \
  output/<ID>/test/gt_au.csv \
  --output_dir output/<ID>/test
```

## Acknowledgements

This code is developed upon [InsTaG](https://github.com/Fictionarry/InsTaG) and
[GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars). FLAME tracking
is from [VHAP](https://github.com/ShenhanQian/VHAP). Emotion and identity features
are obtained from [DeepFace](https://github.com/serengil/deepface) and
[CVLface](https://github.com/mk-minchul/CVLface). Geometry priors are from
[Sapiens](https://github.com/facebookresearch/sapiens). Thanks for these great projects!

---

## Responsible Use

EmoTaG is provided strictly for **research purposes**. Users are responsible for obtaining proper consent, complying with dataset licenses, and avoiding the creation of misleading or harmful synthetic media.

---

## Citation

If you find EmoTaG useful for your research, please consider citing:

```bibtex
@inproceedings{xu2026emotag,
  title     = {EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization},
  author    = {Xu, Haolan and Cheng, Keli and Wang, Lei and Bi, Ning and Liu, Xiaoming},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```
