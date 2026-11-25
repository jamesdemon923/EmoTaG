#!/usr/bin/env python3
"""Measure (and fix) the audio-video offset of processed scenes with SyncNet.

    python tools/check_av_sync.py data/<ID>                  # report the offset in frames
    python tools/check_av_sync.py data/<ID> --fix            # shift aud.wav and re-extract the audio features
    python tools/check_av_sync.py data/<ID> --split_at 125   # adaptation frames and the rest measured separately

Offsets follow SyncNet's convention (offset > 0: audio leads). Needs SYNCNET_ROOT.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_features import AUDIO_FEATURE_FILES

EXTRACTORS = {'wav2vec2': 'extract_wav2vec2_features.py', 'whisper': 'extract_whisper_features.py'}
TOOLS = os.path.dirname(os.path.abspath(__file__))


MIN_PART_FRAMES = 50


def frames_video(scene: str, out: str, sub: str = 'bg_imgs', first: int = 0, last: int = None) -> list:
    """Write the frames [first, last) of the scene (positions in transforms.json) as a 25 fps video."""
    import imageio
    d = os.path.join(scene, sub)
    if not os.path.isdir(d):
        d = os.path.join(scene, 'gt_imgs')
    tf = json.load(open(os.path.join(scene, 'transforms.json')))
    ids = [int(f['timestep_index']) for f in tf['frames']][first:last]
    imgs = [imageio.imread(os.path.join(d, f'{i}.jpg')) for i in ids]
    imageio.mimwrite(out, imgs, fps=25, quality=9, macro_block_size=1)
    return ids


def score_parts(scene: str, wav: str, scorer, tmp: str, split_at: int, gap: int, tol: int) -> dict:
    """Offsets of the first `split_at` frames and of the frames after them (skipping `gap` frames, e.g."""
    n = len(json.load(open(os.path.join(scene, 'transforms.json')))['frames'])
    parts = {}
    for name, first, last in (('first', 0, min(split_at, n)), ('rest', split_at + gap, n)):
        if last - first < MIN_PART_FRAMES:
            parts[name] = {'status': 'too_short', 'frames': [first, max(first, last)]}
            continue
        vid = os.path.join(tmp, f'part_{name}.mp4')
        ids = frames_video(scene, vid, first=first, last=last)
        parts[name] = dict(scorer.score(vid, wav, audio_start_frame=ids[0]), frames=[first, last])
    out = {'parts': parts}
    if all(p.get('status') == 'ok' for p in parts.values()):
        out['parts_offset_diff'] = int(parts['first']['offset'] - parts['rest']['offset'])
        out['parts_differ'] = abs(out['parts_offset_diff']) >= tol
    return out


def shift_wav(src: str, dst: str, frames: int, fps: int = 25):
    """Delay (frames > 0) or advance (frames < 0) the audio, keeping its length."""
    import soundfile as sf
    wav, sr = sf.read(src)
    n = int(round(abs(frames) / fps * sr))
    pad = np.zeros((n,) + wav.shape[1:], dtype=wav.dtype)
    out = np.concatenate([pad, wav])[:len(wav)] if frames > 0 else np.concatenate([wav[n:], pad])
    sf.write(dst, out, sr)


def extract_features(scene: str, wav: str, out_dir: str) -> list:
    """Re-extract every audio feature file of the scene (aud_w2v.npy always, the others when the scene has them)."""
    staged = []
    for ext, fname in AUDIO_FEATURE_FILES.items():
        feat = os.path.join(scene, fname)
        if ext != 'wav2vec2' and not os.path.exists(feat):
            continue
        out = os.path.join(out_dir, fname)
        subprocess.run([sys.executable, os.path.join(TOOLS, EXTRACTORS[ext]), '--wav', wav, '--output', out,
                        '--transforms', os.path.join(scene, 'transforms.json')], check=True)
        staged.append((out, feat))
    return staged


def commit(staged: list):
    for out, feat in staged:
        bak = feat.replace('.npy', '_orig_unsynced.npy')
        if os.path.exists(feat) and not os.path.exists(bak):
            shutil.copy2(feat, bak)
        os.replace(out, feat)


def check_scene(scene: str, wav: str, scorer, args) -> dict:
    shift_file = os.path.join(scene, 'aud_sync_shift.txt')
    backup = os.path.join(scene, 'aud_orig_unsynced.wav')
    res_extra = {}
    if args.fix and os.path.exists(shift_file):
        stale = [f for f in AUDIO_FEATURE_FILES.values() if os.path.exists(os.path.join(scene, f))
                 and os.path.getmtime(os.path.join(scene, f)) < os.path.getmtime(wav)]
        if stale:
            print(f'{os.path.basename(scene)}: WARNING {stale} older than the shifted aud.wav: re-extracting')
            stage = tempfile.mkdtemp(prefix='.avsync_', dir=scene)
            try:
                staged = [x for x in extract_features(scene, wav, stage) if os.path.basename(x[1]) in stale]
                commit(staged)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
            res_extra['reextracted_stale'] = stale
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, 'gt.mp4')
        frames_video(scene, vid)
        res = scorer.score(vid, wav)
        if args.split_at > 0:
            res.update(score_parts(scene, wav, scorer, tmp, args.split_at, args.split_gap, args.split_tol))
            if res.get('parts_differ'):
                first, rest = res['parts']['first'], res['parts']['rest']
                print(f'{os.path.basename(scene)}: WARNING the first {args.split_at} frames and the rest of the clip have '
                      f'different audio-video offsets ({first["offset"]:+d} vs {rest["offset"]:+d} frames; whole clip '
                      f'{res.get("offset", "n/a")}). An avatar adapted on the first part is driven out of sync on the rest. '
                      f'Cut the clip into its takes and synchronise each on its own.')
                if args.fix:
                    res['status'] = 'parts_differ'
        if args.fix and res.get('status') == 'ok' and abs(res['offset']) >= args.min_offset:
            if not os.path.exists(backup):
                shutil.copy2(wav, backup)
            prev = int(open(shift_file).read().strip()) if os.path.exists(shift_file) else 0
            total = prev + int(res['offset'])
            stage = tempfile.mkdtemp(prefix='.avsync_', dir=scene)
            try:
                tmp_wav = os.path.join(stage, 'aud.wav')
                shift_wav(backup, tmp_wav, total)
                staged = extract_features(scene, tmp_wav, stage)
                commit(staged)
                os.replace(tmp_wav, wav)
                with open(shift_file + '.tmp', 'w') as f:
                    f.write(str(total))
                os.replace(shift_file + '.tmp', shift_file)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
            res2 = scorer.score(vid, wav)
            res = {'before': res, 'after': res2, 'shift_frames': total,
                   'reextracted': [os.path.basename(f) for _, f in staged]}
    res.update(res_extra)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('scenes', nargs='+')
    ap.add_argument('--syncnet_root', default=None)
    ap.add_argument('--fix', action='store_true',
                    help='shift aud.wav by the measured offset and re-extract the audio features '
                         '(aud_w2v.npy, and aud_whisper.npy if present)')
    ap.add_argument('--min_offset', type=int, default=1, help='only fix |offset| >= this many frames')
    ap.add_argument('--split_at', type=int, default=0,
                    help='also measure the offset of the first N frames (the adaptation frames, e.g. 125) and of the '
                         'remaining frames separately (0 = off)')
    ap.add_argument('--split_gap', type=int, default=0,
                    help='with --split_at: frames skipped after the first part (e.g. a cross-dissolve between two takes)')
    ap.add_argument('--split_tol', type=int, default=2,
                    help='with --split_at: report the clip when the two offsets differ by this many frames or more')
    ap.add_argument('--out', default=None, help='JSON report')
    args = ap.parse_args()
    from utils.syncnet_lse import SyncNetScorer
    scorer = SyncNetScorer(args.syncnet_root)
    report = {}
    for scene in args.scenes:
        scene = scene.rstrip('/')
        wav = os.path.join(scene, 'aud.wav')
        if not os.path.exists(wav):
            print(f'{scene}: no aud.wav, skipped')
            continue
        try:
            res = check_scene(scene, wav, scorer, args)
        except Exception as e:
            res = {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
        report[os.path.basename(scene)] = res
        print(os.path.basename(scene), json.dumps(res))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
    bad = [k for k, r in report.items() if r.get('status') == 'error'
           or (args.fix and 'status' in r and r['status'] != 'ok')]
    if bad:
        print(f'ERROR: {len(bad)} scene(s) could not be checked/fixed and were left unchanged: {", ".join(bad)}',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
