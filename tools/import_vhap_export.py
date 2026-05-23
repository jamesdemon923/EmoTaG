#!/usr/bin/env python3
"""Import a VHAP monocular export into the EmoTaG scene layout."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
_mpl_config_dir = os.environ.get('MPLCONFIGDIR')
if not _mpl_config_dir or not os.access(_mpl_config_dir, os.W_OK):
    os.environ['MPLCONFIGDIR'] = os.path.join('/tmp', 'emotag_mpl')
import cv2
import numpy as np
from PIL import Image
EMOTAG_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOTAG_ROOT))

def load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def save_json(path: Path, data: dict) -> None:
    with path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=4)

def ensure_empty_or_new(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f'Output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(f'Destination already exists: {dst}')
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'symlink':
        dst.symlink_to(src)
    else:
        raise ValueError(f'Unsupported link mode: {mode}')

def normalize_frame_id(frame: dict, fallback_idx: int) -> int:
    value = frame.get('timestep_index', fallback_idx)
    return int(value)

def frame_image_path(vhap_root: Path, frame: dict) -> Path:
    file_path = frame.get('file_path')
    if file_path:
        return vhap_root / file_path
    frame_id = normalize_frame_id(frame, 0)
    for suffix in ('.jpg', '.png', '.jpeg'):
        candidate = vhap_root / 'images' / f'{frame_id}{suffix}'
        if candidate.exists():
            return candidate
        candidate = vhap_root / 'images' / f'{frame_id:05d}{suffix}'
        if candidate.exists():
            return candidate
    return vhap_root / 'images' / f'{frame_id}.jpg'

def frame_mask_path(vhap_root: Path, frame: dict) -> Path | None:
    mask_path = frame.get('fg_mask_path')
    if mask_path:
        candidate = vhap_root / mask_path
        if candidate.exists():
            return candidate
    frame_id = normalize_frame_id(frame, 0)
    for suffix in ('.png', '.jpg', '.jpeg'):
        for stem in (str(frame_id), f'{frame_id:05d}'):
            candidate = vhap_root / 'fg_masks' / f'{stem}{suffix}'
            if candidate.exists():
                return candidate
    return None

def make_fallback_landmarks(width: int, height: int) -> np.ndarray:
    """Create coarse 68-point landmarks for dataset-reader bookkeeping."""
    landmarks = np.zeros((68, 2), dtype=np.float32)
    cx = width * 0.5
    cy = height * 0.52
    rx = width * 0.28
    ry = height * 0.34
    for idx in range(68):
        angle = 2.0 * np.pi * idx / 68.0
        landmarks[idx, 0] = cx + rx * np.cos(angle)
        landmarks[idx, 1] = cy + ry * np.sin(angle)
    mouth_outer = [(0.38, 0.6), (0.43, 0.57), (0.48, 0.56), (0.52, 0.56), (0.57, 0.57), (0.62, 0.6), (0.57, 0.63), (0.52, 0.64), (0.48, 0.64), (0.43, 0.63), (0.4, 0.61), (0.6, 0.61)]
    mouth_inner = [(0.44, 0.6), (0.48, 0.59), (0.52, 0.59), (0.56, 0.6), (0.52, 0.62), (0.48, 0.62), (0.45, 0.61), (0.55, 0.61)]
    for offset, (x, y) in enumerate(mouth_outer + mouth_inner):
        landmarks[48 + offset, 0] = width * x
        landmarks[48 + offset, 1] = height * y
    nose_points = [(0.42, 0.5), (0.45, 0.51), (0.48, 0.52), (0.52, 0.52), (0.55, 0.51)]
    for offset, (x, y) in enumerate(nose_points):
        landmarks[31 + offset, 0] = width * x
        landmarks[31 + offset, 1] = height * y
    return landmarks

def make_parsing_mask(image_path: Path, fg_mask_path: Path | None, output_path: Path) -> None:
    image = Image.open(image_path).convert('RGB')
    width, height = image.size
    parsing = np.full((height, width, 3), 255, dtype=np.uint8)
    if fg_mask_path is not None:
        fg = np.array(Image.open(fg_mask_path).convert('L'))
        fg_mask = fg > 10
    else:
        fg_mask = np.ones((height, width), dtype=bool)
    parsing[fg_mask] = np.array([0, 0, 255], dtype=np.uint8)
    mouth_center = (int(width * 0.5), int(height * 0.61))
    mouth_axes = (max(8, int(width * 0.1)), max(4, int(height * 0.035)))
    cv2.ellipse(parsing, mouth_center, mouth_axes, 0, 0, 360, (100, 100, 100), -1)
    Image.fromarray(parsing).save(output_path)

def make_transparent_torso(width: int, height: int, output_path: Path) -> None:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    Image.fromarray(rgba, mode='RGBA').save(output_path)

def make_background(image_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert('RGB')
    bg = np.full((image.height, image.width, 3), 255, dtype=np.uint8)
    Image.fromarray(bg).save(output_path)

def selected_frames(transform_data: dict, max_frames: int | None) -> list[dict]:
    frames = list(transform_data.get('frames', []))
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames

def load_vhap_transforms(vhap_root: Path, max_frames: int | None) -> dict:
    transforms_path = vhap_root / 'transforms.json'
    if not transforms_path.exists():
        raise FileNotFoundError(f'Expected transforms.json under {vhap_root}')
    data = load_json(transforms_path)
    frames = selected_frames(data, max_frames)
    if not frames:
        raise RuntimeError(f'No frames found in {vhap_root}')
    data = dict(data)
    data['frames'] = frames
    return data

def normalize_transform_data(transform_data: dict, frames: list[dict]) -> dict:
    normalized = dict(transform_data)
    normalized['frames'] = []
    if frames:
        first = frames[0]
        for key in ('fl_x', 'fl_y', 'cx', 'cy', 'h', 'w', 'camera_angle_x', 'camera_angle_y'):
            if key in first:
                normalized[key] = first[key]
    for new_idx, frame in enumerate(frames):
        new_frame = dict(frame)
        original_idx = normalize_frame_id(frame, new_idx)
        new_frame['timestep_index_original'] = frame.get('timestep_index_original', original_idx)
        new_frame['timestep_index'] = original_idx
        new_frame['file_path'] = f'gt_imgs/{original_idx}.jpg'
        normalized['frames'].append(new_frame)
    return normalized

def write_scene_files(args: argparse.Namespace) -> None:
    vhap_root = args.vhap_export.resolve()
    output_root = args.output.resolve()
    ensure_empty_or_new(output_root)
    transform_data = load_vhap_transforms(vhap_root, args.max_frames)
    frames = selected_frames(transform_data, args.max_frames)
    all_frames = sorted(frames, key=lambda frame: normalize_frame_id(frame, 0))
    for dirname in ('ori_imgs', 'gt_imgs', 'parsing', 'torso_imgs'):
        (output_root / dirname).mkdir(parents=True, exist_ok=True)
    first_image_path = None
    for fallback_idx, frame in enumerate(all_frames):
        frame_id = normalize_frame_id(frame, fallback_idx)
        image_path = frame_image_path(vhap_root, frame)
        if not image_path.exists():
            raise FileNotFoundError(f'Frame image not found for frame {frame_id}: {image_path}')
        if first_image_path is None:
            first_image_path = image_path
        with Image.open(image_path) as image:
            width, height = image.size
        ori_path = output_root / 'ori_imgs' / f'{frame_id}.jpg'
        gt_path = output_root / 'gt_imgs' / f'{frame_id}.jpg'
        link_or_copy(image_path, ori_path, args.link_mode)
        link_or_copy(image_path, gt_path, args.link_mode)
        landmarks = make_fallback_landmarks(width, height)
        np.savetxt(output_root / 'ori_imgs' / f'{frame_id}.lms', landmarks, fmt='%.6f')
        make_parsing_mask(image_path=image_path, fg_mask_path=frame_mask_path(vhap_root, frame), output_path=output_root / 'parsing' / f'{frame_id}.png')
        make_transparent_torso(width, height, output_root / 'torso_imgs' / f'{frame_id}.png')
    if first_image_path is None:
        raise RuntimeError('No frames were found in the VHAP export.')
    make_background(first_image_path, output_root / 'bc.jpg')
    save_json(output_root / 'transforms.json', normalize_transform_data(transform_data, frames))
    if args.audio_features:
        shutil.copy2(args.audio_features, output_root / args.audio_features.name)
    else:
        frame_ids = [normalize_frame_id(frame, idx) for idx, frame in enumerate(all_frames)]
        num_frames = max(frame_ids) + 1 if frame_ids else 1
        zeros = np.zeros((num_frames, 16, args.audio_dim), dtype=np.float32)
        np.save(output_root / args.audio_output, zeros)
    from assets.flame_loader import generate_and_save_initial_mesh_data
    generate_and_save_initial_mesh_data(flame_root=str(vhap_root / 'flame_param'), output_dir=str(output_root), flame_model=str(args.flame_model_path.resolve()), num_points=args.num_points)

def main() -> int:
    parser = argparse.ArgumentParser(description='Import a VHAP export as an EmoTaG processed scene.')
    parser.add_argument('--vhap_export', type=Path, required=True, help='Path to a VHAP export_as_nerf_dataset output folder.')
    parser.add_argument('--output', type=Path, required=True, help='New EmoTaG scene output directory.')
    parser.add_argument('--flame_model_path', type=Path, default=Path(os.environ['EMOTAG_FLAME_MODEL']) if os.environ.get('EMOTAG_FLAME_MODEL') else None, help='Path to generic_model.pkl. Can also be provided through EMOTAG_FLAME_MODEL.')
    parser.add_argument('--num_points', type=int, default=50000)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--link_mode', choices=('symlink', 'copy'), default='symlink')
    parser.add_argument('--audio_features', type=Path, default=None)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--audio_output', type=str, default='aud_w2v.npy')
    args = parser.parse_args()
    if args.flame_model_path is None:
        parser.error('--flame_model_path is required unless EMOTAG_FLAME_MODEL is set.')
    write_scene_files(args)
    print(f'Imported VHAP scene to: {args.output}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
