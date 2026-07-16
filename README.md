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

EmoTaG's renderer requires the depth/normal/alpha-capable rasterizer
[`slothfulxtx/diff-gaussian-rasterization`](https://github.com/slothfulxtx/diff-gaussian-rasterization)
(installed as the **`diff_gauss`** package). The bundled
`submodules/diff-gaussian-rasterization` is the vanilla INRIA rasterizer and is
**not** used by EmoTaG. Install `diff_gauss` with:

```bash
git clone --recurse-submodules https://github.com/slothfulxtx/diff-gaussian-rasterization.git
cd diff-gaussian-rasterization
python setup.py install
cd ..
python -c "import diff_gauss; print('diff_gauss OK')"
```

### 3. Third-party resources

| Resource | Purpose |
| :--- | :--- |
| [FLAME](https://flame.is.tue.mpg.de/index.html) | FLAME parametric head model |
| [VHAP](https://github.com/ShenhanQian/VHAP) | FLAME tracking environment |
| [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) | Audio feature extraction backbone |
| [py-feat](https://py-feat.org/) | Action Unit (AU) feature extraction |
| [DeepFace](https://github.com/serengil/deepface) | Per-frame emotion features |
| [CVLface](https://github.com/mk-minchul/CVLface) | Identity embedding |
| [Sapiens](https://github.com/facebookresearch/sapiens/blob/main/lite/README.md) | Depth and normal priors for adaptation |

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
respective licenses. All clips are face-centered cropped and resized to
**512 × 512 at 25 FPS** before processing.

---

## Data Preparation

EmoTaG operates on a **processed scene** containing calibrated images, FLAME tracking, Gaussian initialization, audio features, AU features, the DeepFace emotion teacher, and the identity descriptor.

> **One command:** after running VHAP tracking, `tools/preprocess_scene.sh`
> chains all of the steps below for a single scene. The per-step instructions
> that follow document what each stage produces.
>
> ```bash
> EMOTAG_FLAME_MODEL=/path/to/FLAME/generic_model.pkl \
> VHAP_EXPORT=/path/to/VHAP/export/monocular/<ID>_whiteBg_staticOffset \
> WAV=data/<ID>/aud.wav \
> CVLFACE_ROOT=/path/to/CVLface \
> bash tools/preprocess_scene.sh data/<ID>
> # add RUN_SAPIENS=1 for adaptation (target) scenes
> # per-tool conda interpreters can be overridden with AU_PY / DEEPFACE_PY / ADAFACE_PY
> ```

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
  --num_points 60000 \
  --link_mode  symlink
```

### Step 1b — Facial landmarks (FAN)

The dataset readers use per-frame 68-point landmarks (`ori_imgs/<id>.lms`) to
derive the lip / mouth crops that supervise the mouth Gaussians. Extract real
landmarks with the FAN detector:

```bash
python tools/extract_landmarks.py --scene data/<ID>
# -> data/<ID>/ori_imgs/<id>.lms   shape [68, 2]
```

> `import_vhap_export.py` no longer writes landmarks by default. Its
> `--use_fallback_landmarks` flag only emits coarse synthetic landmarks for
> smoke tests and must not be used for reproduction.

### Step 2 — Audio features (Wav2Vec2)

```bash
python tools/extract_wav2vec2_features.py \
  --wav        data/<ID>/aud.wav \
  --output     data/<ID>/aud_w2v.npy \
  --transforms data/<ID>/transforms.json
```

### Step 3 — AU features (py-feat)

The six AU channels (`AU01, AU04, AU05, AU06, AU07, AU45`) supply the upper-face
expression cues that audio cannot carry (`AU45` is clipped to `[0, 2]` at load
time, following Supp. Sec. B.2). Extract them with [py-feat](https://py-feat.org/):

```bash
python tools/extract_pyfeat_au.py --scene data/<ID>
# -> data/<ID>/au_features.csv
```

> An OpenFace-based alternative wrapper (`tools/extract_openface_au.sh`, needs a
> local OpenFace build via `OPENFACE_BIN`) is also provided; the released
> checkpoint was trained with the py-feat features above.

### Step 4 — Emotion teacher (DeepFace)

Run [DeepFace](https://github.com/serengil/deepface) per frame to obtain the
7-way emotion distribution used to supervise the emotion-aware branches:

```bash
python tools/extract_deepface_emotion.py --scene data/<ID>
# -> data/<ID>/emotion_features.npy   shape [N, 7]
#    columns = [angry, disgust, fear, happy, sad, surprise, neutral]
```

### Step 5 — Identity descriptor (CVLface)

Compute the identity descriptor `s` as the average face-recognition embedding
over the top-50 neutral frames (run Step 4 first so neutral frames can be
ranked). The extractor uses a [CVLface](https://github.com/mk-minchul/CVLface)
model; clone CVLface and point `--cvlface_root` at it (its pretrained models are
fetched under `<cvlface_root>/cvlface/pretrained_models`):

```bash
python tools/extract_adaface_identity.py \
  --scene         data/<ID> \
  --cvlface_root  /path/to/CVLface
# -> data/<ID>/identity_feature.npy   shape [512]
```

### Step 6 — Geometry priors (Sapiens)

> Not required for pre-training.

```bash
conda activate sapiens_lite
bash tools/run_sapiens_priors.sh data/<ID>
```

The script writes depth and normal maps under:

```text
data/<ID>/sapiens/
├── depth/sapiens_*/<frame_id>.npy
└── normal/sapiens_*/<frame_id>.npy
```

### Step 7 — Validate the scene

```bash
python tools/repro_check.py --root data/<ID> --imports
```

The checker reports whether each required scene file is present and valid.

---

## Training

### Pre-training (multi-identity)

```bash
python pretrain_emotag.py \
  -s data/pretrain \
  -m output/pretrain \
  --scene_names_file data/hdtf_pretrain_ids.txt \
  --audio_extractor wav2vec2 \
  --iterations 3600 \
  --warm_up_iter_per_scene 1000 \
  --w_flame_reg 1.0 \
  --batch_size 1
```

> `data/hdtf_pretrain_ids.txt` lists the 70 HDTF identities (one clip each,
> 90–240 s) used for the universal motion prior; point `-s` at the directory
> that contains those processed scenes. **`--iterations` is PER SCENE**:
> `pretrain_emotag.py` multiplies it by the number of scenes internally, so
> `--iterations 3600` over the 70 HDTF ids gives ~250K total optimiser steps
> (the paper budget). For a quick smoke run use a small value (e.g.
> `--iterations 60`). These canonical hyper-parameters are documented in
> [`configs/pretrain_hdtf70.yaml`](configs/pretrain_hdtf70.yaml). The
> multi-identity EMA checkpoint is written to
> `output/pretrain/chkpnt_ema_face_latest.pth` for adaptation.
>
> **Multi-GPU:** `pretrain_emotag_ddp.py` runs the same pre-training across
> several GPUs (scenes are sharded across ranks), e.g.
> `torchrun --nproc_per_node=<N> pretrain_emotag_ddp.py <same args>`.

### Few-shot adaptation (target identity)

```bash
python adapt_emotag.py \
  -s data/<ID> \
  -m output/<ID> \
  --audio_extractor wav2vec2 \
  --pretrain_path output/pretrain/chkpnt_ema_face_latest.pth \
  --iterations 20000 \
  --N_views 125 \
  --adapt_adain_only          
```

> `--adapt_adain_only` freezes the pretrained GRMN and tunes only the AdaIN
> modulation parameters. Optionally add `--adapt_unfreeze_heads` to also
> fine-tune the small FLAME/mouth prediction heads for identity-specific
> articulation. The canonical adaptation hyper-parameters are in
> [`configs/adapt_5s.yaml`](configs/adapt_5s.yaml).
>
> **Checkpoint to use:** few-shot adaptation expects the multi-identity
> **EMA / motion-only** checkpoint saved at the pre-training output **root**,
> i.e. `output/pretrain/chkpnt_ema_face_latest.pth` (a 3-tuple
> `(motion_state, optimizer_state, iter)`). The per-identity
> `output/pretrain/<ID>/chkpnt_face_latest.pth` files are 4-tuples that also
> bundle the Gaussian model and are **not** the right input here (the loader now
> handles both, but the EMA checkpoint is recommended).

### Pretrained checkpoint

We release the multi-identity EMA motion-prior checkpoint so you can skip the
compute-heavy pre-training and go straight to few-shot adaptation.

| File | Contents | Use |
| :--- | :--- | :--- |
| `chkpnt_ema_face_latest.pth` | 3-tuple `(motion_state, optimizer_state, iter)` — the GRMN motion network (EMA weights), no per-identity Gaussians | `--pretrain_path` for `adapt_emotag.py` |

```bash
# download the released checkpoint, then adapt to your identity
python adapt_emotag.py -s data/<ID> -m output/<ID> \
  --audio_extractor wav2vec2 \
  --pretrain_path /path/to/chkpnt_ema_face_latest.pth \
  --iterations 20000 --N_views 125 --adapt_adain_only
```

> **Download:** [Google Drive](https://drive.google.com/drive/folders/17wgFUGF7pxbW3QlBkiHQzo3nYTzF-fiN?usp=drive_link)
>
> **Provenance:** the multi-identity GRMN motion prior is trained on HDTF
> identities (see [`data/hdtf_pretrain_ids.txt`](data/hdtf_pretrain_ids.txt)) with
> the recipe in [`configs/pretrain_hdtf70.yaml`](configs/pretrain_hdtf70.yaml).

---

## Inference

Render a trained scene:

```bash
python synthesize.py \
  -s data/<ID> \
  -m output/<ID> \
  --use_train
```

---

## Evaluation

### Video-level metrics

```bash
python evaluate_metrics.py video \
  output/<ID>/train/rendered_video.mp4 \
  output/<ID>/train/gt_video.mp4 \
  output/<ID>/train
```

### Action-Unit metrics (optional)

The AU error (AUE) compares AU intensities between the rendered and the
ground-truth videos. First run an AU extractor (see Step 3) on the **frames of
each video** to produce two CSVs, then:

```bash
python evaluate_metrics.py au \
  output/<ID>/train/rendered_au.csv \
  output/<ID>/train/gt_au.csv \
  --output_dir output/<ID>/train
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
