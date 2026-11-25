#!/usr/bin/env python3
"""Frame-aligned Whisper encoder features (aud_whisper.npy, shape [frames, 16, 768]).

    python tools/extract_whisper_features.py --wav data/<ID>/aud.wav --output data/<ID>/aud_whisper.npy --transforms data/<ID>/transforms.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.extract_wav2vec2_features import load_wav_mono, num_frames_from_transforms, window_features

MODEL = 'openai/whisper-small'
SR, CHUNK_S = 16000, 30


class WhisperEncoder:
    def __init__(self, model=MODEL, device=None):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.fe = WhisperFeatureExtractor.from_pretrained(model)
        self.enc = WhisperModel.from_pretrained(model).get_encoder().to(self.device).eval()

    @torch.no_grad()
    def hidden(self, wav: np.ndarray) -> np.ndarray:
        """50 Hz encoder states for a mono 16 kHz waveform of any length."""
        out = []
        step = CHUNK_S * SR
        for s in range(0, max(len(wav), 1), step):
            chunk = wav[s:s + step]
            n50 = int(np.ceil(len(chunk) / SR * 50))
            feats = self.fe(chunk, sampling_rate=SR, return_tensors='pt').input_features.to(self.device)
            h = self.enc(feats).last_hidden_state[0, :n50].float().cpu().numpy()
            out.append(h)
        return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--transforms', type=Path, default=None)
    ap.add_argument('--num_frames', type=int, default=None)
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--window_size', type=int, default=16)
    args = ap.parse_args()
    wav = load_wav_mono(args.wav, SR)
    h = torch.from_numpy(WhisperEncoder(args.model).hidden(wav))
    if args.num_frames is not None:
        n = args.num_frames
    elif args.transforms is not None:
        n = num_frames_from_transforms(args.transforms)
    else:
        n = int(round(len(wav) / SR * 25))
    w = window_features(h, n, args.window_size).numpy().astype(np.float32)
    np.save(args.output, w)
    print(f'Saved Whisper features to {args.output} with shape {w.shape}')


if __name__ == '__main__':
    main()
