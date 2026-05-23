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
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e gridencoder
pip install -e shencoder
```

### 3. Third-party resources

| Resource | Purpose |
| :--- | :--- |
| [FLAME](https://flame.is.tue.mpg.de/index.html) | FLAME parametric head model |
| [VHAP](https://github.com/ShenhanQian/VHAP) | FLAME tracking environment |
| [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h) | Audio feature extraction backbone |
| [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) | Action Unit (AU) feature extraction |
| [Sapiens](https://github.com/facebookresearch/sapiens/blob/main/lite/README.md) | Depth and normal priors for adaptation |

Export the FLAME / VHAP paths used by the preprocessing tools:

```bash
export EMOTAG_FLAME_MODEL=/path/to/flame/generic_model.pkl
export EMOTAG_VHAP_ROOT=/path/to/VHAP
```

---

## Data Preparation

EmoTaG operates on a **processed scene** containing calibrated images, FLAME tracking, Gaussian initialization, audio features, and AU features.

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

### Step 2 — Audio features (Wav2Vec2)

```bash
python tools/extract_wav2vec2_features.py \
  --wav        data/<ID>/aud.wav \
  --output     data/<ID>/aud_w2v.npy \
  --transforms data/<ID>/transforms.json
```

### Step 3 — AU features (OpenFace)

Run `FeatureExtraction` from OpenFace and place the CSV under the scene folder:

```bash
mv /path/to/openface_output.csv data/<ID>/au_features.csv
```

### Step 4 — Geometry priors (Sapiens, *adaptation only*)

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

### Step 5 — Validate the scene

```bash
python tools/repro_check.py --root data/<ID> --imports
```

---

## Training

### Pre-training (multi-identity)

```bash
python pretrain_emotag.py \
  -s data/pretrain \
  -m output/pretrain \
  --scene_names <ID_1>,<ID_2> \
  --audio_extractor wav2vec2 \
  --iterations 30000 \
  --batch_size 1
```

### Few-shot adaptation (target identity)

```bash
python adapt_emotag.py \
  -s data/<ID> \
  -m output/<ID> \
  --audio_extractor wav2vec2 \
  --pretrain_path output/pretrain/<ID_1>/chkpnt_ema_face_latest.pth \
  --iterations 20000 \
  --N_views 125
```

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

### Action-Unit metrics

```bash
python evaluate_metrics.py au \
  output/<ID>/train/rendered_openface.csv \
  output/<ID>/train/gt_openface.csv \
  --output_dir output/<ID>/train
```

---

## To-Do List

- [ ] Release demo
- [ ] Release checkpoints
- [ ] Release data processing scripts
- [ ] Release evaluation configuration files

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
