#!/usr/bin/env python3
"""Prepare (audio, FLAME track) pairs for tools/train_audio_prior.py.

    python tools/prepare_audio_prior_data.py --source unitalker --root <D5_unitalker_faceforensics++> --flame2020 <FLAME2020/generic_model.pkl> --out data_prior/d5
    python tools/prepare_audio_prior_data.py --source diffposetalk --root <HDTF_TFHP-lmdb> --flame2020 <FLAME2020/generic_model.pkl> --out data_prior/hdtf_tfhp --shards 4

Writes <out>/<split>[_<shard>].npz: hidden [sum T50, 768] fp16 (Whisper), h_off [n+1], exp [sum T25, 50] (FLAME 2019 basis,
centred per clip), jaw [sum T25, 3], f_off [n+1], names.
"""
import argparse
import glob
import io
import os
import pickle
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.flame_regions import N_SHAPE, _load

SR = 16000


def expression_map(flame2020, n_src, n_out=50, lower_weight=3.0):
    """M [n_out, n_src] with exp_2019 = M @ exp_2020 (vertex least squares, lower face weighted)."""
    src = np.asarray(pickle.load(open(flame2020, 'rb'), encoding='latin1')['shapedirs'])[:, :, N_SHAPE:N_SHAPE + n_src]
    dst, lower, _ = _load(n_out)
    w = np.ones(src.shape[0])
    w[lower] = lower_weight
    w = np.repeat(np.sqrt(w), 3)[:, None]
    return np.linalg.lstsq(dst.reshape(-1, n_out) * w, src.reshape(-1, n_src) * w, rcond=None)[0].astype(np.float32)


def resample(x, src_fps, n_out, dst_fps=25.0):
    t = np.arange(n_out) / dst_fps * src_fps
    i0 = np.clip(np.floor(t).astype(int), 0, len(x) - 1)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    w = (t - np.floor(t))[:, None]
    return (1 - w) * x[i0] + w * x[i1]


def clips_unitalker(root, split, names):
    """UniTalker release: <root>/<split>/<clip>.npy (30 fps; shape 300, expression 100, jaw 3, ...) + <clip>.wav."""
    for name in names:
        npy = os.path.join(root, split, name + '.npy')
        if not os.path.exists(npy[:-4] + '.wav'):
            continue
        p = np.load(npy).astype(np.float32)
        if p.ndim != 2 or p.shape[1] != 413 or len(p) < 30:
            continue
        wav, sr = sf.read(npy[:-4] + '.wav', dtype='float32')
        wav = wav if wav.ndim == 1 else wav.mean(1)
        assert sr == SR, npy
        yield name, wav, p[:, 300:400], p[:, 400:403], 30.0, int(np.floor(min(len(wav) / SR, len(p) / 30.0) * 25))


def clips_diffposetalk(root, split, names):
    """DiffPoseTalk lmdb (25 fps; coef exp 50, pose [global 3, jaw 3]; audio stored per segment)."""
    import lmdb
    with lmdb.open(root, readonly=True, lock=False, readahead=False).begin() as txn:
        seg = pickle.loads(txn.get(b'metadata'))['seg_len']
        for name in names:
            n = pickle.loads(txn.get(f'{name}/metadata'.encode()))['n_frames']
            exp, pose, wav = [], [], []
            for c in range((n - 1) // seg + 1):
                e = pickle.loads(txn.get(f'{name}/{c:03d}'.encode()))
                exp.append(e['coef']['exp'])
                pose.append(e['coef']['pose'])
                a, sr = sf.read(io.BytesIO(e['audio']), dtype='float32')
                assert sr == SR, name
                wav.append(a if a.ndim == 1 else a.mean(1))
            wav = np.concatenate(wav)
            n = min(n, int(len(wav) / SR * 25))
            yield name, wav[:int(n / 25 * SR)], np.concatenate(exp)[:n, :50], np.concatenate(pose)[:n, 3:6], 25.0, n


def split_names(source, root, split):
    if source == 'unitalker':
        return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(root, split, '*.npy')))
    return [x.strip() for x in open(os.path.join(root, f'{split}.txt')) if x.strip()]


def excluded(name, exclude):
    return any(name == x or name.startswith(x + '/') for x in exclude)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['unitalker', 'diffposetalk'], required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--flame2020', required=True, help='FLAME 2020 generic_model.pkl (basis of the tracks)')
    ap.add_argument('--exclude', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'audio_prior_exclude.txt'),
                    help='text file of clip names or speaker prefixes to leave out (evaluation speakers and recordings)')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--shards', type=int, default=1, help='write the train split as train_<i>.npz, every n-th clip starting at i')
    args = ap.parse_args()
    from tools.extract_whisper_features import WhisperEncoder
    os.makedirs(args.out, exist_ok=True)
    exclude = [x.strip() for x in open(args.exclude) if x.strip()] if args.exclude else []
    encoder, maps = WhisperEncoder(), {}
    gen = clips_unitalker if args.source == 'unitalker' else clips_diffposetalk
    for split, si, sn in [(sp, i, args.shards if sp == 'train' else 1) for sp in args.splits
                          for i in range(args.shards if sp == 'train' else 1)]:
        names = [n for n in split_names(args.source, args.root, split) if not excluded(n, exclude)][si::sn]
        hid, exp, jaw, kept, h_off, f_off = [], [], [], [], [0], [0]
        for name, wav, e, j, fps, n in gen(args.root, split, names):
            if n < 25:
                continue
            if e.shape[1] not in maps:
                maps[e.shape[1]] = expression_map(args.flame2020, e.shape[1])
            e = e @ maps[e.shape[1]].T
            e = e - e.mean(0, keepdims=True)
            h = encoder.hidden(wav)
            hid.append(h.astype(np.float16))
            exp.append((resample(e, fps, n) if fps != 25 else e[:n]).astype(np.float32))
            jaw.append((resample(j, fps, n) if fps != 25 else j[:n]).astype(np.float32))
            kept.append(name)
            h_off.append(h_off[-1] + len(h))
            f_off.append(f_off[-1] + n)
        tag = f'{split}_{si}' if sn > 1 else split
        if not kept:
            raise SystemExit(f'[prep] {tag}: no clips')
        np.savez(os.path.join(args.out, f'{tag}.npz'), hidden=np.concatenate(hid), h_off=np.array(h_off),
                 exp=np.concatenate(exp), jaw=np.concatenate(jaw), f_off=np.array(f_off), names=np.array(kept))
        print(f'[prep] {tag}: {len(kept)} clips, {f_off[-1] / 25 / 3600:.2f} h', flush=True)


if __name__ == '__main__':
    main()
