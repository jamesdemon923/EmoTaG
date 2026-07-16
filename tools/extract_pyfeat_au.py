#!/usr/bin/env python3
"""Extract Action Unit features for an EmoTaG scene using py-feat.

This is the OpenFace-free replacement for tools/extract_openface_au.sh. It writes
an `au_features.csv` whose columns match what the rest of EmoTaG expects:

  * training (utils/au_utils.py): AU01_r, AU04_r, AU05_r, AU06_r, AU07_r, AU45_r
  * evaluation (evaluate_metrics.py): 17 AU**_r intensity columns

py-feat's AU detector emits per-frame AU activations. We expose them with an
`_r` suffix so both consumers can read them unchanged. py-feat reports eye
closure as AU43 (rather than OpenFace's AU45 blink); we alias AU43 -> AU45 so
the blink channel is populated. Any AU the detector does not provide is written
as a zero column so downstream column checks pass.

Note on scale: py-feat AU values are ~[0, 1] activations, not OpenFace's [0, 5]
intensities. This is internally consistent because every consumer (preprocessing
of pretrain/adapt scenes, conditioning at inference, and the AUE metric on both
rendered and GT videos) uses this same extractor.

Run with the `pyfeat` conda env:
    CUDA_VISIBLE_DEVICES=0 \
    /path/to/conda/envs/pyfeat/bin/python \
        tools/extract_pyfeat_au.py --scene data/<ID>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Columns required downstream (without the _r suffix). AU45 is filled from py-feat AU43.
REQUIRED_AUS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
    "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45",
]


def list_frames(image_dir: Path, extension: str) -> list[Path]:
    return sorted(image_dir.glob(f"*{extension}"), key=lambda p: int(p.stem))


def run_detector(frames: list[Path], batch_size: int):
    """Run ONLY face -> landmark -> AU.

    py-feat's Detector.detect_image runs a full waterfall that also computes
    head pose (img2pose), emotion (resmasknet) and identity (facenet) on every
    frame. Those three models dominate runtime (~3-4 s/frame total) and are
    unused by EmoTaG, which needs AUs only. We reproduce detect_image's own
    preprocessing (ImageDataset with the same options) but call just the three
    sub-detectors we need, which leaves the AU values identical while cutting
    the per-frame cost by roughly 3-5x. Inputs are native 512x512, so no
    output_size rescale is applied and AUs match the full pipeline exactly.
    """
    import pandas as pd
    from torch.utils.data import DataLoader
    from feat import Detector
    from feat.data import ImageDataset

    detector = Detector(device="cuda" if _cuda_available() else "cpu")
    au_columns = list(detector.info["au_presence_columns"])
    paths = [str(p) for p in frames]

    loader = DataLoader(
        ImageDataset(paths, output_size=None, preserve_aspect_ratio=True, padding=True),
        batch_size=batch_size,
        num_workers=min(8, max(1, (os.cpu_count() or 8) // 4)),
        shuffle=False,
    )

    rows: list[np.ndarray] = []
    nan_row = np.full(len(au_columns), np.nan, dtype=np.float32)
    for batch in loader:
        images = batch["Image"]
        faces = detector.detect_faces(images, threshold=0.5)
        landmarks = detector.detect_landmarks(images, detected_faces=faces)
        aus = detector.detect_aus(images, landmarks)
        for per_image in aus:  # one entry per image; each is [n_faces, n_au]
            arr = np.asarray(per_image, dtype=np.float32)
            if arr.ndim == 2 and len(arr) > 0:
                rows.append(arr[0])           # first (largest) face
            elif arr.ndim == 1 and arr.size == len(au_columns):
                rows.append(arr)
            else:
                rows.append(nan_row)          # no detection -> filled downstream
    return pd.DataFrame(rows, columns=au_columns)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def fex_to_au_frame(fex, n_frames: int) -> pd.DataFrame:
    """Turn a py-feat Fex result into a frame-indexed AU**_r DataFrame."""
    aus = fex.aus if hasattr(fex, "aus") else fex
    aus = pd.DataFrame(aus).reset_index(drop=True)
    # Normalise column names like "AU01" / "AU01_r" / "au01" -> "AU01".
    col_map = {}
    for col in aus.columns:
        key = str(col).strip().upper().replace("_R", "")
        col_map[key] = col

    out = pd.DataFrame()
    out["frame"] = np.arange(len(aus))
    for au in REQUIRED_AUS:
        source = col_map.get(au)
        if source is None and au == "AU45":
            source = col_map.get("AU43")  # py-feat eye-closure aliases OpenFace blink
        if source is None:
            out[f"{au}_r"] = 0.0
        else:
            out[f"{au}_r"] = aus[source].to_numpy(dtype=np.float32)

    # Fill frames with no detection (NaN) from temporal neighbours, then zero.
    au_cols = [c for c in out.columns if c.endswith("_r")]
    out[au_cols] = out[au_cols].ffill().bfill().fillna(0.0)

    # Pad/truncate to the frame count (py-feat can drop frames with no detection).
    if len(out) < n_frames:
        last = out.iloc[[-1]].copy()
        pad = pd.concat([last] * (n_frames - len(out)), ignore_index=True)
        out = pd.concat([out, pad], ignore_index=True)
    elif len(out) > n_frames:
        out = out.iloc[:n_frames].reset_index(drop=True)
    out["frame"] = np.arange(len(out))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract AU features for a scene with py-feat.")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--images", type=str, default="gt_imgs")
    parser.add_argument("--extension", type=str, default=".jpg")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <scene>/au_features.csv")
    # py-feat AU (HOG+xgb) is CPU-bound; shard a scene's frames across processes
    # (different CUDA_VISIBLE_DEVICES) to cut wall-clock, then --merge the parts.
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--merge", action="store_true", help="Merge <output>.part*.csv into the final CSV.")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    output = args.output or (args.scene / "au_features.csv")

    image_dir = args.scene / args.images
    frames = list_frames(image_dir, args.extension)
    n_frames = len(frames)
    if not frames and not args.merge:
        raise SystemExit(f"No frames found under {image_dir}")

    if args.merge:
        parts = sorted(Path(output).parent.glob(Path(output).name + ".part*.csv"))
        if not parts:
            raise SystemExit(f"No part files to merge for {output}")
        merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        merged = merged.sort_values("frame").reset_index(drop=True)
        au_cols = [c for c in merged.columns if c.endswith("_r")]
        merged[au_cols] = merged[au_cols].ffill().bfill().fillna(0.0)
        merged = fex_to_au_frame(merged, n_frames)  # final pad/truncate to frame count
        merged.to_csv(output, index=False)
        for p in parts:
            p.unlink()
        print(f"Merged {len(parts)} shards -> {output} ({len(merged)} rows).")
        return 0

    # Contiguous block for this shard (keeps global frame order simple).
    if args.num_shards > 1:
        per = -(-n_frames // args.num_shards)  # ceil
        lo, hi = args.shard_id * per, min((args.shard_id + 1) * per, n_frames)
        block = frames[lo:hi]
        au_frame = fex_to_au_frame(run_detector(block, args.batch_size), len(block))
        au_frame["frame"] = np.arange(lo, lo + len(au_frame))
        part = Path(str(output) + f".part{args.shard_id:03d}.csv")
        au_frame.to_csv(part, index=False)
        print(f"Shard {args.shard_id}/{args.num_shards}: frames [{lo},{hi}) -> {part}")
        return 0

    au_frame = fex_to_au_frame(run_detector(frames, args.batch_size), n_frames)
    au_frame.to_csv(output, index=False)
    print(f"Saved {len(au_frame)} rows of AU features to {output} "
          f"({len([c for c in au_frame.columns if c.endswith('_r')])} AU columns).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
