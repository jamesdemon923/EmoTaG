<div align="center">

# EmoTaG: Emotion-Aware Talking Head Synthesis with Gaussian Splatting

Official repository for **EmoTaG**.

[`Paper`](https://arxiv.org/abs/2603.21332) | [`Project`](https://emotag26.github.io/)

<img src="./img/pipeline.png" alt="EmoTaG overview" style="zoom:80%;" />

</div>

## Installation

Tested on Ubuntu Linux 22.04 with CUDA 12.1 and PyTorch.

```bash
git submodule update --init --recursive
conda env create --file environment_cu121.yml
conda activate emotag
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
pip install -e gridencoder
pip install -e shencoder
```

Prepare the following third-party resources:

- [FLAME](https://flame.is.tue.mpg.de/index.html): FLAME model.
- [VHAP](https://github.com/ShenhanQian/VHAP): FLAME tracking environment.
- [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h): audio feature extraction backbone.
- [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace): AU feature extraction environment.
- [Sapiens](https://github.com/facebookresearch/sapiens/blob/main/lite/README.md): depth and normal prior environment for adaptation.

Set the FLAME / VHAP paths for the preprocessing tools:

```bash
export EMOTAG_FLAME_MODEL=/path/to/flame/generic_model.pkl
export EMOTAG_VHAP_ROOT=/path/to/VHAP
```

## Usage

### Data Processing

EmoTaG uses processed scenes as input. A scene contains calibrated images, FLAME tracking, Gaussian initialization, audio features, and AU features:

```text
data/<ID>/
|-- transforms.json
|-- flame_params.npz
|-- model_center.npy
|-- points3D.ply
|-- face_indices.npy
|-- bary_coords.npy
|-- vertices.npy
|-- mouth_point_indices.npy
|-- aud_w2v.npy
|-- au_features.csv
|-- ori_imgs/
|-- gt_imgs/
|-- parsing/
|-- torso_imgs/
`-- sapiens/                 # for adaptation
```

Run the VHAP monocular pipeline to obtain FLAME tracking. Then convert the VHAP export into the EmoTaG scene layout:

```bash
python tools/import_vhap_export.py \
  --vhap_export /path/to/VHAP/export/monocular/<ID>_whiteBg_staticOffset \
  --output data/<ID> \
  --flame_model_path "$EMOTAG_FLAME_MODEL" \
  --num_points 60000 \
  --link_mode symlink
```

Extract frame-aligned Wav2Vec2 features:

```bash
python tools/extract_wav2vec2_features.py \
  --wav data/<ID>/aud.wav \
  --output data/<ID>/aud_w2v.npy \
  --transforms data/<ID>/transforms.json
```

Run `FeatureExtraction` in OpenFace, then rename and move the output CSV file:

```bash
mv /path/to/openface_output.csv data/<ID>/au_features.csv
```

Generate Sapiens geometry priors for adaptation:

```bash
conda activate sapiens_lite
bash tools/run_sapiens_priors.sh data/<ID>
```

This step is not required for pre-training. The script writes depth and normal maps under:

```text
data/<ID>/sapiens/
  depth/sapiens_*/<frame_id>.npy
  normal/sapiens_*/<frame_id>.npy
```

Validate the scene before training:

```bash
python tools/repro_check.py --root data/<ID> --imports
```

### Training

Pretrain on multiple processed scenes:

```bash
CUDA_VISIBLE_DEVICES=0 python pretrain_emotag.py \
  -s data/pretrain \
  -m output/pretrain \
  --scene_names <ID_1>,<ID_2> \
  --audio_extractor wav2vec2 \
  --iterations 30000 \
  --batch_size 1
```

Adapt to a target identity:

```bash
CUDA_VISIBLE_DEVICES=0 python adapt_emotag.py \
  -s data/<ID> \
  -m output/<ID> \
  --audio_extractor wav2vec2 \
  --pretrain_path output/pretrain/<ID_1>/chkpnt_ema_face_latest.pth \
  --iterations 20000 \
  --N_views 125
```

### Inference

Render a trained scene:

```bash
CUDA_VISIBLE_DEVICES=0 python synthesize.py \
  -s data/<ID> \
  -m output/<ID> \
  --use_train
```

### Metrics Evaluation

Evaluate rendered and ground-truth videos:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_metrics.py video \
  output/<ID>/train/rendered_video.mp4 \
  output/<ID>/train/gt_video.mp4 \
  output/<ID>/train
```

Evaluate AU metrics from OpenFace CSV outputs:

```bash
python evaluate_metrics.py au \
  output/<ID>/train/rendered_openface.csv \
  output/<ID>/train/gt_openface.csv \
  --output_dir output/<ID>/train
```

## To-Do List

- [ ] Release demo.
- [ ] Release checkpoints.
- [ ] Release data processing scripts.
- [ ] Release evaluation configuration files.

## Responsible Use

EmoTaG is provided for research use. Users are responsible for consent, dataset-license compliance, and avoiding misleading or harmful synthetic media.

## Citation

```bibtex
@inproceedings{xu2026emotag,
  title={EmoTaG: Emotion-Aware Talking Head Synthesis on Gaussian Splatting with Few-Shot Personalization},
  author={Xu, Haolan and Cheng, Keli and Wang, Lei and Bi, Ning and Liu, Xiaoming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
