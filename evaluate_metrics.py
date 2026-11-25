#!/usr/bin/env python3
"""Evaluate EmoTaG video and AU metrics."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


AU_COLUMNS = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
]
LOWER_AU_COLUMNS = ["AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r"]
UPPER_AU_COLUMNS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU45_r"]


class LMDMeter:
    def __init__(self, backend: str = "fan", region: str = "mouth"):
        self.backend = backend
        self.region = region
        if backend == "dlib":
            import dlib

            self.predictor_path = "./shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(self.predictor_path):
                raise FileNotFoundError("Download shape_predictor_68_face_landmarks.dat from dlib before using --backend dlib.")
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.predictor_path)
        else:
            import face_alignment

            try:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
            except AttributeError:
                self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        self.value = 0.0
        self.count = 0

    def get_landmarks(self, image: np.ndarray) -> np.ndarray:
        if self.backend == "dlib":
            detections = self.detector(image, 1)
            if not detections:
                raise ValueError("No face detected")
            shape = self.predictor(image, detections[0])
            landmarks = np.zeros((68, 2), dtype=np.float32)
            for idx in range(68):
                landmarks[idx, 0] = shape.part(idx).x
                landmarks[idx, 1] = shape.part(idx).y
            return landmarks
        landmarks = self.predictor.get_landmarks(image)
        if not landmarks:
            raise ValueError("No face landmarks detected")
        return landmarks[-1].astype(np.float32)

    def update(self, preds: torch.Tensor, truths: torch.Tensor) -> None:
        pred = (preds[0].detach().cpu().numpy() * 255).astype(np.uint8)
        truth = (truths[0].detach().cpu().numpy() * 255).astype(np.uint8)
        pred_lms = self.get_landmarks(pred)
        truth_lms = self.get_landmarks(truth)
        if self.region == "mouth":
            pred_lms = pred_lms[48:68]
            truth_lms = truth_lms[48:68]
        pred_lms = pred_lms - pred_lms.mean(0)
        truth_lms = truth_lms - truth_lms.mean(0)
        self.value += np.sqrt(((pred_lms - truth_lms) ** 2).sum(1)).mean()
        self.count += 1

    def measure(self) -> float:
        return float("nan") if self.count == 0 else self.value / self.count

    def report(self) -> str:
        value = self.measure()
        return "LMD: unavailable" if np.isnan(value) else f"LMD: {value:.6f}"


class PSNRMeter:
    def __init__(self):
        self.value = 0.0
        self.count = 0

    def update(self, preds: torch.Tensor, truths: torch.Tensor) -> None:
        pred = preds.detach().cpu().numpy()
        truth = truths.detach().cpu().numpy()
        self.value += -10 * np.log10(np.mean((pred - truth) ** 2))
        self.count += 1

    def measure(self) -> float:
        return self.value / self.count

    def report(self) -> str:
        return f"PSNR: {self.measure():.6f}"


class SSIMMeter:
    def __init__(self, device: torch.device | None = None):
        from utils.loss_utils import ssim as ssim_fn

        self.value = 0.0
        self.count = 0
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fn = ssim_fn

    def update(self, preds: torch.Tensor, truths: torch.Tensor) -> None:
        pred = preds.permute(0, 3, 1, 2).contiguous().to(self.device)
        truth = truths.permute(0, 3, 1, 2).contiguous().to(self.device)
        self.value += self.fn(pred, truth).item()
        self.count += 1

    def measure(self) -> float:
        return self.value / self.count

    def report(self) -> str:
        return f"SSIM: {self.measure():.6f}"


class LPIPSMeter:
    def __init__(self, net: str = "alex", device: torch.device | None = None):
        import lpips

        self.value = 0.0
        self.count = 0
        self.net = net
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def update(self, preds: torch.Tensor, truths: torch.Tensor) -> None:
        pred = preds.permute(0, 3, 1, 2).contiguous().to(self.device)
        truth = truths.permute(0, 3, 1, 2).contiguous().to(self.device)
        self.value += self.fn(truth, pred, normalize=True).item()
        self.count += 1

    def measure(self) -> float:
        return self.value / self.count

    def report(self) -> str:
        return f"LPIPS ({self.net}): {self.measure():.6f}"


def load_masks(path: str | None):
    """Per-frame head-region masks written by tools/render_composited.py."""
    if not path:
        return None
    data = np.load(path)
    shape = tuple(int(x) for x in data["shape"])
    return np.unpackbits(data["masks"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)


class RegionMeter:
    """PSNR inside the head mask, SSIM / LPIPS on the mask's bounding box."""

    def __init__(self, lpips_fn, device):
        from utils.loss_utils import ssim as ssim_fn
        self.lpips_fn, self.ssim_fn, self.device = lpips_fn, ssim_fn, device
        self.psnr, self.ssim, self.lpips, self.count = 0.0, 0.0, 0.0, 0

    def update(self, pred: torch.Tensor, truth: torch.Tensor, mask: np.ndarray) -> None:
        if mask is None or not mask.any():
            return
        m = torch.from_numpy(mask).to(pred.device)
        diff = ((pred[0] - truth[0]) ** 2)[m]
        self.psnr += float(-10 * torch.log10(diff.mean().clamp_min(1e-10)))
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pc = pred[:, y0:y1, x0:x1].permute(0, 3, 1, 2).contiguous()
        tc = truth[:, y0:y1, x0:x1].permute(0, 3, 1, 2).contiguous()
        self.ssim += float(self.ssim_fn(pc, tc))
        self.lpips += float(self.lpips_fn(tc, pc, normalize=True))
        self.count += 1

    def measure(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {"PSNR_head": self.psnr / self.count, "SSIM_head": self.ssim / self.count,
                "LPIPS_head": self.lpips / self.count}


def evaluate_videos(rendered_video_path: str, gt_video_path: str, output_dir: str, backend: str = "fan",
                    masks_path: str | None = None, skip_lmd: bool = False) -> dict[str, float]:
    if not os.path.exists(rendered_video_path):
        raise FileNotFoundError(f"Rendered video not found: {rendered_video_path}")
    if not os.path.exists(gt_video_path):
        raise FileNotFoundError(f"Ground-truth video not found: {gt_video_path}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lmd_meter = None if skip_lmd else LMDMeter(backend=backend)
    psnr_meter = PSNRMeter()
    ssim_meter = SSIMMeter(device=device)
    lpips_meter = LPIPSMeter(device=device)
    masks = load_masks(masks_path)
    region_meter = RegionMeter(lpips_meter.fn, device) if masks is not None else None

    rendered_capture = cv2.VideoCapture(rendered_video_path)
    gt_capture = cv2.VideoCapture(gt_video_path)
    if not rendered_capture.isOpened():
        raise RuntimeError(f"Failed to open rendered video: {rendered_video_path}")
    if not gt_capture.isOpened():
        raise RuntimeError(f"Failed to open ground-truth video: {gt_video_path}")

    rendered_frames = int(rendered_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    gt_frames = int(gt_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Rendered frames: {rendered_frames}")
    print(f"Ground-truth frames: {gt_frames}")
    if masks is not None and len(masks) != min(rendered_frames, gt_frames):
        print(f"Warning: {len(masks)} head masks for {min(rendered_frames, gt_frames)} frames; the masks must come from "
              f"the same render (same --eval_from/--eval_to). Head metrics cover only the overlap.")
    if rendered_frames != gt_frames:
        print(f"Warning: frame count mismatch; evaluating {min(rendered_frames, gt_frames)} aligned frames.")

    processed = 0
    failed_frames = 0
    failed_lmd_frames = 0
    while True:
        ret_rendered, frame_rendered = rendered_capture.read()
        ret_gt, frame_gt = gt_capture.read()
        if not (ret_rendered and ret_gt):
            break
        try:
            rendered = torch.FloatTensor(frame_rendered[..., ::-1] / 255.0)[None, ...].to(device)
            gt = torch.FloatTensor(frame_gt[..., ::-1] / 255.0)[None, ...].to(device)
            psnr_meter.update(rendered, gt)
            ssim_meter.update(rendered, gt)
            lpips_meter.update(rendered, gt)
            if region_meter is not None and processed < len(masks):
                if masks[processed].shape != tuple(frame_rendered.shape[:2]):
                    raise ValueError(f"mask size {masks[processed].shape} != frame size {frame_rendered.shape[:2]}")
                region_meter.update(rendered, gt, masks[processed])
            if lmd_meter is not None:
                try:
                    lmd_meter.update(rendered, gt)
                except Exception as exc:
                    failed_lmd_frames += 1
                    print(f"LMD unavailable for frame {processed}: {exc}")
        except Exception as exc:
            failed_frames += 1
            print(f"Failed to evaluate frame {processed}: {exc}")
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed} frames...")

    rendered_capture.release()
    gt_capture.release()
    if processed == 0:
        raise RuntimeError("No frames were processed.")

    results = {
        "LMD": lmd_meter.measure() if lmd_meter is not None else float("nan"),
        "PSNR": psnr_meter.measure(),
        "SSIM": ssim_meter.measure(),
        "LPIPS": lpips_meter.measure(),
        "total_frames": float(processed),
        "failed_frames": float(failed_frames),
        "failed_lmd_frames": float(failed_lmd_frames),
    }

    if region_meter is not None:
        results.update(region_meter.measure())
    if lmd_meter is not None:
        print(lmd_meter.report())
    print(psnr_meter.report())
    print(ssim_meter.report())
    print(lpips_meter.report())

    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("EmoTaG Video Metrics\n")
        file.write(f"Rendered video: {rendered_video_path}\n")
        file.write(f"Ground-truth video: {gt_video_path}\n")
        file.write(f"Total frames: {processed}\n")
        file.write(f"Failed frames: {failed_frames}\n")
        file.write(f"Frames without LMD: {failed_lmd_frames}\n")
        file.write(f"LMD backend: {backend}\n")
        file.write(f"LMD: {results['LMD']:.6f}\n" if not np.isnan(results["LMD"]) else "LMD: unavailable\n")
        file.write(f"PSNR: {results['PSNR']:.6f}\n")
        file.write(f"SSIM: {results['SSIM']:.6f}\n")
        file.write(f"LPIPS: {results['LPIPS']:.6f}\n")
        for key in ("PSNR_head", "SSIM_head", "LPIPS_head"):
            if key in results:
                file.write(f"{key}: {results[key]:.6f}\n")
                print(f"{key}: {results[key]:.6f}")
    import json
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    return results


def load_openface_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    missing = [column for column in AU_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing AU columns in {path}: {', '.join(missing)}")
    return frame[AU_COLUMNS]


def compute_au_error(rendered_csv: Path, gt_csv: Path) -> dict[str, float]:
    rendered = load_openface_csv(rendered_csv)
    gt = load_openface_csv(gt_csv)
    length = min(len(rendered), len(gt))
    if length == 0:
        raise ValueError("CSV files contain no frames.")
    rendered = rendered.iloc[:length]
    gt = gt.iloc[:length]
    error = (rendered - gt) ** 2
    return {
        "AUE": float(error.mean().sum()),
        "AUE_lower": float(error[LOWER_AU_COLUMNS].mean().sum()),
        "AUE_upper": float(error[UPPER_AU_COLUMNS].mean().sum()),
        "frames": float(length),
    }


def evaluate_au(rendered_csv: Path, gt_csv: Path, output_dir: Path | None = None) -> dict[str, float]:
    results = compute_au_error(rendered_csv, gt_csv)
    print(f"AUE:       {results['AUE']:.6f}")
    print(f"AUE lower: {results['AUE_lower']:.6f}")
    print(f"AUE upper: {results['AUE_upper']:.6f}")
    print(f"Frames:    {int(results['frames'])}")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "au_metrics.txt"
        with metrics_path.open("w", encoding="utf-8") as file:
            file.write("EmoTaG AU Metrics\n")
            file.write(f"Rendered CSV: {rendered_csv}\n")
            file.write(f"Ground-truth CSV: {gt_csv}\n")
            file.write(f"AUE: {results['AUE']:.6f}\n")
            file.write(f"AUE lower: {results['AUE_lower']:.6f}\n")
            file.write(f"AUE upper: {results['AUE_upper']:.6f}\n")
            file.write(f"Frames: {int(results['frames'])}\n")
        print(f"AU metrics saved to: {metrics_path}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate EmoTaG metrics.")
    subparsers = parser.add_subparsers(dest="metric", required=True)

    video_parser = subparsers.add_parser("video", help="Evaluate rendered video metrics.")
    video_parser.add_argument("rendered_video", help="Path to rendered video.")
    video_parser.add_argument("gt_video", help="Path to ground-truth video.")
    video_parser.add_argument("output_dir", help="Directory to save metrics.txt.")
    video_parser.add_argument("--backend", choices=["fan", "dlib"], default="fan", help="Landmark backend for LMD.")

    video_parser.add_argument("--masks", default=None, help="masks_<name>.npz from render_composited.py: adds head-region PSNR/SSIM/LPIPS.")
    video_parser.add_argument("--skip_lmd", action="store_true", help="Skip the (slow) FAN landmark distance.")

    sync_parser = subparsers.add_parser("sync", help="Sync-C / Sync-E (LSE-C / LSE-D) with SyncNet.")
    sync_parser.add_argument("video", help="Rendered video (frames).")
    sync_parser.add_argument("audio", help="Driving audio (wav, or a video with an audio track).")
    sync_parser.add_argument("--syncnet_root", default=None, help="joonson/syncnet_python checkout with weights (or $SYNCNET_ROOT).")
    sync_parser.add_argument("--output", default=None, help="Optional JSON output path.")
    sync_parser.add_argument("--audio_start_frame", type=int, default=0,
                             help="Frame of the audio that matches the first video frame (e.g. 125 with --eval_from 125 and the full aud.wav).")

    au_parser = subparsers.add_parser("au", help="Evaluate AU metrics from OpenFace CSV files.")
    au_parser.add_argument("rendered_csv", type=Path, help="OpenFace CSV for rendered frames.")
    au_parser.add_argument("gt_csv", type=Path, help="OpenFace CSV for ground-truth frames.")
    au_parser.add_argument("--output_dir", type=Path, default=None, help="Optional directory for au_metrics.txt.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.metric == "video":
            evaluate_videos(args.rendered_video, args.gt_video, args.output_dir, args.backend, args.masks, args.skip_lmd)
        elif args.metric == "sync":
            import json
            from utils.syncnet_lse import SyncNetScorer
            res = SyncNetScorer(args.syncnet_root).score(args.video, args.audio, audio_start_frame=args.audio_start_frame)
            print(json.dumps(res, indent=2))
            if args.output:
                with open(args.output, "w", encoding="utf-8") as file:
                    json.dump(res, file, indent=2)
        elif args.metric == "au":
            evaluate_au(args.rendered_csv, args.gt_csv, args.output_dir)
    except Exception as exc:
        print(f"Error during evaluation: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
