#!/usr/bin/env python3
"""Compute the AdaFace identity descriptor for a scene.

The identity feature `s` is the L2-normalised average AdaFace embedding over the
top-K neutral frames (ranked by the DeepFace neutral probability from
emotion_features.npy, falling back to the first K frames). It is written to:

    identity_feature.npy   shape = [512]

We use the CVLface release of AdaFace (ViT-base + KPRPE, WebFace4M), which ships
its own face aligner (DFA mobilenet). The weights are loaded from a local clone
of https://github.com/mk-minchul/CVLface with the two HF model folders already
downloaded under ``cvlface/pretrained_models``. Run this with the ``cvlface``
conda env (it needs that repo's transformers/safetensors stack).

Example:
    CUDA_VISIBLE_DEVICES=0 \
    /path/to/conda/envs/cvlface/bin/python \
        tools/extract_adaface_identity.py --scene data/<ID>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

EMOTAG_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOTAG_ROOT))

from utils.emotion_utils import NEUTRAL_INDEX  # noqa: E402

DEFAULT_CVLFACE_ROOT = Path("/path/to/CVLface")


def select_neutral_frames(scene: Path, images: str, extension: str, top_k: int) -> list[Path]:
    image_dir = scene / images
    all_frames = sorted(image_dir.glob(f"*{extension}"), key=lambda p: int(p.stem))
    emotion_path = scene / "emotion_features.npy"
    if emotion_path.exists():
        p_emo = np.load(emotion_path).astype(np.float32)
        neutral = p_emo[:, NEUTRAL_INDEX]
        ranked = np.argsort(-neutral)
        selected = []
        for frame_id in ranked:
            candidate = image_dir / f"{int(frame_id)}{extension}"
            if candidate.exists():
                selected.append(candidate)
            if len(selected) >= top_k:
                break
        if selected:
            return selected
    return all_frames[:top_k]


def load_cvlface_models(cvlface_root: Path, device: str):
    """Load the AdaFace recogniser and its DFA aligner from a local CVLface clone."""
    if str(cvlface_root) not in sys.path:
        sys.path.insert(0, str(cvlface_root))
    from model import load_model_from_local_path  # provided by CVLface

    pretrained = cvlface_root / "cvlface" / "pretrained_models"
    fr_model = load_model_from_local_path(
        str(pretrained / "cvlface_adaface_vit_base_kprpe_webface4m")
    ).to(device).eval()
    aligner = load_model_from_local_path(
        str(pretrained / "cvlface_DFA_mobilenet")
    ).to(device).eval()
    return fr_model, aligner


def to_input(image_path: Path, device: str):
    from PIL import Image
    from torchvision.transforms import Compose, Normalize, ToTensor

    img = Image.open(str(image_path)).convert("RGB")
    trans = Compose([ToTensor(), Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
    return trans(img).unsqueeze(0).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute the AdaFace identity descriptor for a scene.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--cvlface_root", type=Path, default=DEFAULT_CVLFACE_ROOT,
                        help="Local clone of github.com/mk-minchul/CVLface with weights downloaded.")
    parser.add_argument("--images", type=str, default="gt_imgs")
    parser.add_argument("--extension", type=str, default=".jpg")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <scene>/identity_feature.npy")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"

    frames = select_neutral_frames(args.scene, args.images, args.extension, args.top_k)
    if not frames:
        raise SystemExit(f"No frames found for scene {args.scene}.")

    fr_model, aligner = load_cvlface_models(args.cvlface_root, device)

    embeddings = []
    with torch.no_grad():
        for frame in frames:
            x = to_input(frame, device)
            aligned_x, _, aligned_ldmks, _, _, _ = aligner(x)
            feature = fr_model(aligned_x, aligned_ldmks)
            embeddings.append(feature.squeeze(0).cpu())

    stacked = torch.stack(embeddings, dim=0)
    mean_feature = stacked.mean(dim=0, keepdim=True)
    identity = F.normalize(mean_feature, p=2, dim=1).squeeze(0).numpy().astype(np.float32)

    output = args.output or (args.scene / "identity_feature.npy")
    np.save(output, identity)
    print(f"Saved AdaFace identity descriptor to {output} with shape {identity.shape} from {len(frames)} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
