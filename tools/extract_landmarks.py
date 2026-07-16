#!/usr/bin/env python3
"""Extract real 68-point facial landmarks for an EmoTaG scene.

The dataset readers (`scene/dataset_readers.py`, `scene_pretrain/dataset_readers.py`)
read one landmark file per frame from `ori_imgs/<frame_id>.lms` and use the mouth
landmarks (points 48:68) to derive the lip / mouth-interior crops and bounds that
supervise the mouth Gaussians. Accurate, per-frame landmarks are therefore
required for faithful reproduction.

This tool runs the FAN detector from `face_alignment` (already an EmoTaG
dependency) over each frame and writes:

    ori_imgs/<frame_id>.lms      shape = [68, 2], columns = [x (col), y (row)]

saved with `np.savetxt`, matching `np.loadtxt` in the readers. Frames where no
face is detected fall back to the most recent successful detection (and, if the
very first frames fail, to the next successful one), so every frame gets a file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

EMOTAG_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOTAG_ROOT))


def frame_ids_for_scene(scene: Path, images: str, extension: str) -> list[int]:
    transforms = scene / "transforms.json"
    if transforms.exists():
        with transforms.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        frames = data.get("frames", [])
        ids = [int(frame.get("timestep_index", idx)) for idx, frame in enumerate(frames)]
        if ids:
            return ids
    image_dir = scene / images
    return sorted(int(p.stem) for p in image_dir.glob(f"*{extension}"))


def build_detector(device: str):
    import face_alignment

    try:
        landmark_type = face_alignment.LandmarksType.TWO_D
    except AttributeError:  # older face_alignment releases
        landmark_type = face_alignment.LandmarksType._2D
    return face_alignment.FaceAlignment(landmark_type, flip_input=False, device=device)


def largest_face(landmark_sets: list[np.ndarray]) -> np.ndarray:
    """Pick the detection with the largest bounding box (most likely the subject)."""
    best = None
    best_area = -1.0
    for lms in landmark_sets:
        width = lms[:, 0].max() - lms[:, 0].min()
        height = lms[:, 1].max() - lms[:, 1].min()
        area = float(width * height)
        if area > best_area:
            best_area = area
            best = lms
    return best


def read_image_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return bgr[:, :, ::-1].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract FAN 68-point landmarks for an EmoTaG scene.")
    parser.add_argument("--scene", type=Path, required=True, help="Processed scene root (contains gt_imgs/ and transforms.json).")
    parser.add_argument("--images", type=str, default="gt_imgs", help="Sub-directory of frames relative to the scene root.")
    parser.add_argument("--extension", type=str, default=".jpg")
    parser.add_argument("--output_subdir", type=str, default="ori_imgs", help="Where <frame_id>.lms files are written.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (passed to face_alignment).")
    parser.add_argument("--overwrite", action="store_true", help="Re-detect even if a .lms already exists.")
    args = parser.parse_args()

    frame_ids = frame_ids_for_scene(args.scene, args.images, args.extension)
    if not frame_ids:
        raise SystemExit(f"No frames found for scene {args.scene}.")

    output_dir = args.scene / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(args.device)

    results: dict[int, np.ndarray] = {}
    failed: list[int] = []
    last_good: np.ndarray | None = None

    for position, frame_id in enumerate(frame_ids):
        out_path = output_dir / f"{frame_id}.lms"
        if out_path.exists() and not args.overwrite:
            results[frame_id] = np.loadtxt(out_path).astype(np.float32)
            last_good = results[frame_id]
            continue
        image_path = args.scene / args.images / f"{frame_id}{args.extension}"
        landmarks = None
        if image_path.exists():
            try:
                detections = detector.get_landmarks(read_image_rgb(image_path))
                if detections:
                    landmarks = largest_face(detections).astype(np.float32)
            except Exception as exc:  # pragma: no cover - detector dependent
                print(f"[warn] landmark detection failed on {image_path.name}: {exc}")
        if landmarks is None:
            failed.append(frame_id)
            if last_good is not None:
                landmarks = last_good.copy()
            else:
                results[frame_id] = None  # resolve later with the first good detection
                continue
        results[frame_id] = landmarks
        last_good = landmarks
        if (position + 1) % 200 == 0 or position == len(frame_ids) - 1:
            print(f"Processed {position + 1}/{len(frame_ids)} frames (failures so far: {len(failed)}).")

    first_good = next((lms for lms in results.values() if lms is not None), None)
    if first_good is None:
        raise SystemExit(f"No face detected in any frame of scene {args.scene}.")
    for frame_id in frame_ids:
        lms = results.get(frame_id)
        if lms is None:
            lms = first_good
        np.savetxt(output_dir / f"{frame_id}.lms", lms, fmt="%.6f")

    print(f"Saved {len(frame_ids)} landmark files to {output_dir}.")
    if failed:
        print(f"[warn] {len(failed)} frame(s) had no detection and reuse a neighbouring frame: "
              f"{failed[:10]}{' ...' if len(failed) > 10 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
