#!/usr/bin/env python3
"""Measure (and fix) the audio-video offset of processed scenes with SyncNet.

    python tools/check_av_sync.py data/<ID>                  # report the offset in frames
    python tools/check_av_sync.py data/<ID> --fix            # shift aud.wav and re-extract the audio features
    python tools/check_av_sync.py data/<ID> --split_at 125   # adaptation frames and the rest measured separately
    python tools/check_av_sync.py data/<ID> --split_at 125 --fix   # also re-synchronise the first 125 frames on their own

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


def shifted(wav: np.ndarray, frames: int, sr: int, fps: int = 25) -> np.ndarray:
    """Delay (frames > 0) or advance (frames < 0) the samples, keeping their length."""
    n = int(round(abs(frames) / fps * sr))
    pad = np.zeros((n,) + wav.shape[1:], dtype=wav.dtype)
    return np.concatenate([pad, wav])[:len(wav)] if frames > 0 else np.concatenate([wav[n:], pad])


def shift_wav(src: str, dst: str, frames: int, fps: int = 25):
    """Delay (frames > 0) or advance (frames < 0) the audio, keeping its length."""
    import soundfile as sf
    wav, sr = sf.read(src)
    sf.write(dst, shifted(wav, frames, sr, fps), sr)


def write_first_part(raw: str, dst: str, clip_shift: int, first_shift: int, n_first: int, fps: int = 25):
    """Audio shifted by clip_shift, except the first n_first frames, which are shifted by clip_shift + first_shift."""
    import soundfile as sf
    wav, sr = sf.read(raw)
    out, seg = shifted(wav, clip_shift, sr, fps), int(round(n_first / fps * sr))
    out[:seg] = shifted(wav, clip_shift + first_shift, sr, fps)[:seg]
    sf.write(dst, out, sr)


def resync_first_part(scene: str, wav: str, scorer, args, tmp: str, offset: int) -> dict:
    """Shift the audio of the first --split_at frames by `offset`; later frames keep their audio and features."""
    backup = os.path.join(scene, 'aud_orig_unsynced.wav')
    shift_file = os.path.join(scene, 'aud_sync_shift.txt')
    first_file = os.path.join(scene, 'resync.json')
    if not os.path.exists(backup):
        shutil.copy2(wav, backup)
    clip = int(open(shift_file).read().strip()) if os.path.exists(shift_file) else 0
    prev = json.load(open(first_file)) if os.path.exists(first_file) else {}
    if prev and prev.get('n_adapt') != args.split_at:
        raise ValueError(f'{first_file} was made with --split_at {prev.get("n_adapt")}')
    total = prev.get('adaptation_take_audio_shift_frames', 0) + offset
    stage = tempfile.mkdtemp(prefix='.avsync_', dir=scene)
    try:
        tmp_wav = os.path.join(stage, 'aud.wav')
        write_first_part(backup, tmp_wav, clip, total, args.split_at)
        staged = extract_features(scene, tmp_wav, stage)
        for out, feat in staged:
            new, old = np.load(out), np.load(feat)
            assert new.shape == old.shape, (new.shape, old.shape)
            old[:args.split_at] = new[:args.split_at]
            np.save(out, old)
            os.replace(out, feat)
        os.replace(tmp_wav, wav)
        with open(first_file + '.tmp', 'w') as f:
            json.dump({'adaptation_take_audio_shift_frames': total, 'n_adapt': args.split_at}, f, indent=1)
        os.replace(first_file + '.tmp', first_file)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    after = score_parts(scene, wav, scorer, tmp, args.split_at, args.split_gap, args.split_tol)
    return {'first_part_shift_frames': total, 'after': after, 'reextracted': [os.path.basename(f) for _, f in staged]}


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
            if res.get('parts_differ') and not args.fix:
                first, rest = res['parts']['first'], res['parts']['rest']
                print(f'{os.path.basename(scene)}: WARNING the first {args.split_at} frames and the rest of the clip have '
                      f'different audio-video offsets ({first["offset"]:+d} vs {rest["offset"]:+d} frames; whole clip '
                      f'{res.get("offset", "n/a")}). An avatar adapted on the first part is driven out of sync on the rest. '
                      f'Run with --fix to re-synchronise the first part on its own.')
        if args.fix and res.get('status') == 'ok' and abs(res['offset']) >= args.min_offset:
            if not os.path.exists(backup):
                shutil.copy2(wav, backup)
            prev = int(open(shift_file).read().strip()) if os.path.exists(shift_file) else 0
            total = prev + int(res['offset'])
            stage = tempfile.mkdtemp(prefix='.avsync_', dir=scene)
            try:
                tmp_wav = os.path.join(stage, 'aud.wav')
                first_file = os.path.join(scene, 'resync.json')
                if os.path.exists(first_file):  # keep an earlier re-synchronisation of the first frames
                    first = json.load(open(first_file))
                    write_first_part(backup, tmp_wav, total, first['adaptation_take_audio_shift_frames'], first['n_adapt'])
                else:
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
        if args.fix and args.split_at > 0 and res.get('status', 'ok') == 'ok':
            first = score_parts(scene, wav, scorer, tmp, args.split_at, args.split_gap, args.split_tol)['parts']['first']
            if first.get('status') != 'ok':
                res['first_part'] = first
            elif abs(first['offset']) >= args.min_offset:
                res['first_part'] = dict(resync_first_part(scene, wav, scorer, args, tmp, int(first['offset'])),
                                         before=first)
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
                         'remaining frames separately (0 = off); with --fix, the first N frames are then '
                         're-synchronised on their own (resync.json)')
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
