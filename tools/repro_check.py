#!/usr/bin/env python3
"""Non-destructive checks for the EmoTaG research pipeline.

The script is intentionally lightweight: it reports environment health and
dataset readiness without creating, deleting, or modifying training assets.
"""
from __future__ import annotations
import argparse
import importlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
EMOTAG_ROOT = Path(__file__).resolve().parents[1]
if str(EMOTAG_ROOT) not in sys.path:
    sys.path.insert(0, str(EMOTAG_ROOT))
REQUIRED_SCENE_FILES = ('transforms.json', 'flame_params.npz', 'model_center.npy', 'mouth_point_indices.npy', 'points3D.ply', 'face_indices.npy', 'bary_coords.npy', 'vertices.npy')
AUDIO_FEATURE_PATTERNS = ('aud_w2v.npy',)

@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str

    @property
    def status(self) -> str:
        return 'OK' if self.ok else 'FAIL'

def print_section(title: str) -> None:
    print(f'\n[{title}]')

def print_result(result: CheckResult) -> None:
    print(f'{result.status:>4}  {result.name}: {result.detail}')

def try_import(module_name: str) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return CheckResult(module_name, False, repr(exc))
    version = getattr(module, '__version__', None)
    location = getattr(module, '__file__', None)
    detail_parts = []
    if version is not None:
        detail_parts.append(f'version={version}')
    if location is not None:
        detail_parts.append(f'path={location}')
    return CheckResult(module_name, True, ', '.join(detail_parts) or 'imported')

def check_python() -> list[CheckResult]:
    return [CheckResult('python', True, sys.executable), CheckResult('python_version', True, sys.version.replace('\n', ' ')), CheckResult('platform', True, platform.platform()), CheckResult('cwd', True, os.getcwd())]

def check_torch(require_cuda: bool) -> list[CheckResult]:
    result = try_import('torch')
    results = [result]
    if not result.ok:
        return results
    import torch
    cuda_available = torch.cuda.is_available()
    results.append(CheckResult('torch.cuda.is_available', cuda_available or not require_cuda, str(cuda_available)))
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        results.append(CheckResult('torch.cuda.device_count', device_count > 0, str(device_count)))
        for idx in range(device_count):
            results.append(CheckResult(f'cuda:{idx}', True, torch.cuda.get_device_name(idx)))
    return results

def check_imports(include_heavy: bool) -> list[CheckResult]:
    modules = ['numpy', 'cv2', 'PIL', 'torch', 'diff_gauss', 'gridencoder', 'scene.motion_net', 'scene.flame_wrapper', 'gaussian_renderer']
    if include_heavy:
        modules.extend(['transformers', 'face_alignment', 'lpips'])
    return [try_import(module_name) for module_name in modules]

def load_json(path: Path) -> tuple[bool, str]:
    try:
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
    except Exception as exc:
        return (False, repr(exc))
    frames = data.get('frames')
    if isinstance(frames, list):
        return (True, f'frames={len(frames)}')
    return (True, 'loaded, frames key missing or not a list')

def shape_for_npy(path: Path) -> str:
    try:
        import numpy as np
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        return f'shape={arr.shape}, dtype={arr.dtype}'
    except Exception as exc:
        return repr(exc)

def shape_for_au_csv(path: Path) -> tuple[bool, str]:
    try:
        from utils.au_utils import AU_BASE_COLUMNS, load_openface_au_csv
        arr = load_openface_au_csv(path)
        return (arr.shape[1] == len(AU_BASE_COLUMNS), f'shape={arr.shape}, columns={list(AU_BASE_COLUMNS)}')
    except Exception as exc:
        return (False, repr(exc))

def find_audio_features(scene_root: Path) -> list[Path]:
    found = []
    for pattern in AUDIO_FEATURE_PATTERNS:
        candidate = scene_root / pattern
        if candidate.exists():
            found.append(candidate)
    found.extend(sorted(scene_root.glob('aud*.npy')))
    unique = []
    seen = set()
    for item in found:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique

def looks_like_processed_scene(path: Path) -> bool:
    return any(((path / name).exists() for name in REQUIRED_SCENE_FILES)) or any(path.glob('aud*.npy'))

def discover_scenes(root: Path, max_depth: int) -> list[Path]:
    root = root.resolve()
    if looks_like_processed_scene(root):
        return [root]
    scenes = []
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        depth = len(current.relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if looks_like_processed_scene(current):
            scenes.append(current)
            dirnames[:] = []
    return sorted(scenes)

def check_scene(scene_root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(CheckResult('scene_path', scene_root.exists(), str(scene_root)))
    if not scene_root.exists():
        return results
    for name in REQUIRED_SCENE_FILES:
        path = scene_root / name
        if path.exists():
            detail = str(path)
            if path.suffix == '.json':
                ok, json_detail = load_json(path)
                results.append(CheckResult(name, ok, json_detail))
            elif path.suffix == '.npy':
                results.append(CheckResult(name, True, shape_for_npy(path)))
            else:
                results.append(CheckResult(name, True, str(path)))
        else:
            results.append(CheckResult(name, False, 'missing'))
    audio_features = find_audio_features(scene_root)
    if audio_features:
        for audio_path in audio_features:
            results.append(CheckResult(audio_path.name, True, shape_for_npy(audio_path)))
    else:
        results.append(CheckResult('audio_features', False, 'missing aud*.npy'))
    au_path = scene_root / 'au_features.csv'
    if au_path.exists():
        ok, detail = shape_for_au_csv(au_path)
        results.append(CheckResult('au_features.csv', ok, detail))
    else:
        results.append(CheckResult('au_features.csv', False, 'missing; expected OpenFace columns AU01_r, AU04_r, AU05_r, AU06_r, AU07_r, AU45_r'))
    emotion_path = scene_root / 'emotion_features.npy'
    if emotion_path.exists():
        results.append(CheckResult('emotion_features.npy', True, shape_for_npy(emotion_path)))
    else:
        results.append(CheckResult('emotion_features.npy', False, 'missing; run tools/extract_deepface_emotion.py'))
    identity_path = scene_root / 'identity_feature.npy'
    if identity_path.exists():
        results.append(CheckResult('identity_feature.npy', True, shape_for_npy(identity_path)))
    else:
        results.append(CheckResult('identity_feature.npy', False, 'missing; run tools/extract_adaface_identity.py'))
    raw_videos = sorted(scene_root.glob('*.mp4'))
    results.append(CheckResult('raw_mp4_count', True, str(len(raw_videos))))
    expected_dirs = ('ori_imgs', 'gt_imgs', 'parsing', 'torso_imgs')
    for dirname in expected_dirs:
        path = scene_root / dirname
        if path.exists():
            file_count = sum((1 for item in path.iterdir() if item.is_file()))
            results.append(CheckResult(dirname, True, f'files={file_count}'))
        else:
            results.append(CheckResult(dirname, False, 'missing'))
    return results

def summarize(results: Iterable[CheckResult]) -> tuple[int, int]:
    result_list = list(results)
    failures = sum((1 for item in result_list if not item.ok))
    return (len(result_list) - failures, failures)

def main() -> int:
    parser = argparse.ArgumentParser(description='Check EmoTaG environment and dataset readiness.')
    parser.add_argument('--root', type=Path, default=Path('data/pretrain'), help='Dataset root or one processed scene root.')
    parser.add_argument('--scene', type=Path, default=None, help='Optional scene path relative to --root or absolute.')
    parser.add_argument('--max-depth', type=int, default=4, help='Maximum depth for processed scene discovery.')
    parser.add_argument('--imports', action='store_true', help='Run project import checks.')
    parser.add_argument('--heavy-imports', action='store_true', help='Also check optional heavier dependencies.')
    parser.add_argument('--require-cuda', action='store_true', help='Fail when CUDA is not visible.')
    args = parser.parse_args()
    all_results: list[CheckResult] = []
    print_section('Python')
    python_results = check_python()
    for result in python_results:
        print_result(result)
    all_results.extend(python_results)
    print_section('Torch')
    torch_results = check_torch(args.require_cuda)
    for result in torch_results:
        print_result(result)
    all_results.extend(torch_results)
    if args.imports or args.heavy_imports:
        print_section('Imports')
        import_results = check_imports(args.heavy_imports)
        for result in import_results:
            print_result(result)
        all_results.extend(import_results)
    root = args.root.resolve()
    if args.scene is not None:
        scene_root = args.scene if args.scene.is_absolute() else root / args.scene
        scenes = [scene_root.resolve()]
    else:
        scenes = discover_scenes(root, args.max_depth)
        if not scenes:
            scenes = [root]
    print_section('Dataset')
    print(f'Root: {root}')
    print(f'Scenes discovered: {len(scenes)}')
    dataset_results: list[CheckResult] = []
    for scene_root in scenes:
        print_section(str(scene_root))
        scene_results = check_scene(scene_root)
        for result in scene_results:
            print_result(result)
        dataset_results.extend(scene_results)
    all_results.extend(dataset_results)
    ok_count, fail_count = summarize(all_results)
    print_section('Summary')
    print(f'Passed: {ok_count}, failed: {fail_count}')
    return 1 if fail_count else 0
if __name__ == '__main__':
    raise SystemExit(main())
