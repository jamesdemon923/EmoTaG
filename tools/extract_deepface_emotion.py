#!/usr/bin/env python3
"""Extract per-frame DeepFace emotion features for a scene.

Runs the DeepFace recognizer (https://github.com/serengil/deepface) on every
frame and stores a categorical distribution over the seven basic emotions:

    emotion_features.npy   shape = [num_frames, 7]
    columns = [angry, disgust, fear, happy, sad, surprise, neutral]

Install with:  pip install deepface
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EMOTAG_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOTAG_ROOT))

from utils.emotion_utils import EMOTION_LABELS  # noqa: E402


def frame_ids_from_transforms(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    frames = data.get("frames", [])
    return [int(frame.get("timestep_index", idx)) for idx, frame in enumerate(frames)]


def distribution_from_deepface(result: dict) -> np.ndarray:
    """Map a DeepFace emotion result to the canonical EMOTION_LABELS ordering."""
    emotions = result["emotion"] if "emotion" in result else result
    values = np.array([float(emotions[label]) for label in EMOTION_LABELS], dtype=np.float32)
    total = values.sum()
    if total <= 0:
        values = np.ones_like(values)
        total = values.sum()
    return values / total


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DeepFace emotion teacher over a scene.")
    parser.add_argument("--scene", type=Path, required=True, help="Processed scene root (contains gt_imgs/ and transforms.json).")
    parser.add_argument("--images", type=str, default="gt_imgs", help="Sub-directory of frames relative to the scene root.")
    parser.add_argument("--extension", type=str, default=".jpg")
    parser.add_argument("--detector_backend", type=str, default="retinaface")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <scene>/emotion_features.npy")
    args = parser.parse_args()

    from deepface import DeepFace

    transforms = args.scene / "transforms.json"
    if transforms.exists():
        frame_ids = frame_ids_from_transforms(transforms)
    else:
        frame_ids = sorted(int(p.stem) for p in (args.scene / args.images).glob(f"*{args.extension}"))
    if not frame_ids:
        raise SystemExit(f"No frames found for scene {args.scene}.")

    num_frames = max(frame_ids) + 1
    distributions = np.tile(np.eye(len(EMOTION_LABELS))[EMOTION_LABELS.index("neutral")], (num_frames, 1)).astype(np.float32)

    for frame_id in frame_ids:
        image_path = args.scene / args.images / f"{frame_id}{args.extension}"
        if not image_path.exists():
            continue
        try:
            results = DeepFace.analyze(img_path=str(image_path), actions=["emotion"], detector_backend=args.detector_backend, enforce_detection=False)
            if isinstance(results, list):
                results = results[0]
            distributions[frame_id] = distribution_from_deepface(results)
        except Exception as exc:  # pragma: no cover - depends on detector
            print(f"[warn] DeepFace failed on {image_path.name}: {exc}; keeping neutral prior.")

    output = args.output or (args.scene / "emotion_features.npy")
    np.save(output, distributions)
    print(f"Saved DeepFace emotion teacher to {output} with shape {distributions.shape}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
