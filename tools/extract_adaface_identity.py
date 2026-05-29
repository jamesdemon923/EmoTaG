#!/usr/bin/env python3
"""Compute the AdaFace identity descriptor for a scene.

The identity feature `s` is the average AdaFace embedding over the top-K neutral
frames (ranked by the DeepFace neutral probability from emotion_features.npy,
falling back to the first K frames). It is written to:

    identity_feature.npy   shape = [512]

AdaFace is not pip-installable, so point --adaface_repo at a local clone of
https://github.com/mk-minchul/adaface and --adaface_ckpt at one of its released
checkpoints (e.g. adaface_ir101_webface12m.ckpt).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EMOTAG_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOTAG_ROOT))

from utils.emotion_utils import NEUTRAL_INDEX  # noqa: E402


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


def build_adaface(adaface_repo: Path, ckpt: Path, architecture: str):
    if str(adaface_repo) not in sys.path:
        sys.path.insert(0, str(adaface_repo))
    import net  # provided by the AdaFace repository
    import torch

    model = net.build_model(architecture)
    state = torch.load(ckpt, map_location="cpu")["state_dict"]
    weights = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(weights)
    model.eval()
    return model


def to_adaface_input(image_path: Path):
    import cv2
    import torch

    bgr = cv2.imread(str(image_path))
    bgr = cv2.resize(bgr, (112, 112))
    tensor = ((bgr[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.tensor(tensor.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute the AdaFace identity descriptor for a scene.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--adaface_repo", type=Path, required=True, help="Local clone of github.com/mk-minchul/adaface")
    parser.add_argument("--adaface_ckpt", type=Path, required=True)
    parser.add_argument("--architecture", type=str, default="ir_101")
    parser.add_argument("--images", type=str, default="gt_imgs")
    parser.add_argument("--extension", type=str, default=".jpg")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <scene>/identity_feature.npy")
    args = parser.parse_args()

    import torch

    frames = select_neutral_frames(args.scene, args.images, args.extension, args.top_k)
    if not frames:
        raise SystemExit(f"No frames found for scene {args.scene}.")

    model = build_adaface(args.adaface_repo, args.adaface_ckpt, args.architecture)
    embeddings = []
    with torch.no_grad():
        for frame in frames:
            feature, _ = model(to_adaface_input(frame))
            embeddings.append(feature.squeeze(0).cpu().numpy())
    identity = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)

    output = args.output or (args.scene / "identity_feature.npy")
    np.save(output, identity)
    print(f"Saved AdaFace identity descriptor to {output} with shape {identity.shape} from {len(frames)} frames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
