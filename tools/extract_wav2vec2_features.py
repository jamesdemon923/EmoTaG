#!/usr/bin/env python3
"""Extract frame-aligned Wav2Vec2 hidden features for EmoTaG scenes.

Output shape is [num_video_frames, window_size, 768]. Existing dataset readers
then permute this to [num_video_frames, 768, window_size], matching the current
MotionNetwork audio-window convention.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2Model, Wav2Vec2Processor

def load_wav_mono(path: Path, sample_rate: int) -> np.ndarray:
    wav, sr = sf.read(path)
    wav = wav.astype(np.float32)
    if wav.ndim > 1:
        wav = wav[:, 0]
    if sr != sample_rate:
        import resampy
        wav = resampy.resample(wav, sr_orig=sr, sr_new=sample_rate).astype(np.float32)
    return wav

def num_frames_from_transforms(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
    frame_ids = [int(frame.get('timestep_index', idx)) for idx, frame in enumerate(data.get('frames', []))]
    if not frame_ids:
        raise ValueError(f'No frames found in transforms file: {path}')
    return max(frame_ids) + 1

def window_features(features: torch.Tensor, num_video_frames: int, window_size: int) -> torch.Tensor:
    """Build per-video-frame windows from 50Hz Wav2Vec2 tokens."""
    if features.dim() != 2:
        raise ValueError(f'Expected 2D Wav2Vec2 features, got shape {tuple(features.shape)}')
    pad = window_size // 2
    left_pad = features[:1].repeat(pad, 1)
    right_pad = features[-1:].repeat(pad, 1)
    padded = torch.cat([left_pad, features, right_pad], dim=0)
    windows = []
    for frame_idx in range(num_video_frames):
        center = frame_idx * 2 + pad
        start = max(0, center - pad)
        end = start + window_size
        if end > padded.shape[0]:
            end = padded.shape[0]
            start = end - window_size
        windows.append(padded[start:end])
    return torch.stack(windows, dim=0)

def extract_features(args: argparse.Namespace) -> np.ndarray:
    device = torch.device(args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu')
    wav = load_wav_mono(args.wav, args.sample_rate)
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2Model.from_pretrained(args.model).to(device).eval()
    inputs = processor(wav, sampling_rate=args.sample_rate, return_tensors='pt', padding=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        outputs = model(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state.squeeze(0).float().cpu()
    if args.num_frames is not None:
        num_video_frames = args.num_frames
    elif args.transforms is not None:
        num_video_frames = num_frames_from_transforms(args.transforms)
    else:
        duration = len(wav) / float(args.sample_rate)
        num_video_frames = int(round(duration * args.video_fps))
    windows = window_features(hidden, num_video_frames, args.window_size)
    return windows.numpy().astype(np.float32)

def main() -> int:
    parser = argparse.ArgumentParser(description='Extract frame-aligned Wav2Vec2 features for EmoTaG.')
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transforms', type=Path, default=None)
    parser.add_argument('--num_frames', type=int, default=None)
    parser.add_argument('--model', type=str, default='facebook/wav2vec2-base-960h')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--video_fps', type=float, default=25.0)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='')
    args = parser.parse_args()
    features = extract_features(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, features)
    print(f'Saved Wav2Vec2 features to {args.output} with shape {features.shape}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
