#!/usr/bin/env python3
"""Copy frames [start, end) of a processed scene into a new scene, re-indexed from 0 (e.g. a test segment).

    python tools/split_scene.py data/<ID> --start 150 --out data/<ID>_test
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_features import AUDIO_FEATURE_FILES

FRAME_DIRS = ('gt_imgs', 'bg_imgs', 'matte', 'parsing', 'torso_imgs')
FRAME_ARRAYS = tuple(AUDIO_FEATURE_FILES.values()) + ('emotion_features.npy',)
FLAME_FILES = ('flame_params.npz', 'flame_params_tracked.npz')
STATIC_FILES = ('identity_feature.npy', 'bc.jpg', 'vertices.npy', 'model_center.npy', 'face_indices.npy',
                'bary_coords.npy', 'points3D.ply', 'mouth_point_indices.npy')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('scene')
    ap.add_argument('--start', type=int, required=True, help='first frame (timestep index) of the new scene')
    ap.add_argument('--end', type=int, default=None, help='frame after the last one (default: end of the clip)')
    ap.add_argument('--out', required=True, help='new scene directory (must not exist)')
    args = ap.parse_args()
    src, dst, start = args.scene.rstrip('/'), args.out.rstrip('/'), args.start
    tf = json.load(open(os.path.join(src, 'transforms.json')))
    end = args.end if args.end is not None else max(f['timestep_index'] for f in tf['frames']) + 1
    frames = [f for f in tf['frames'] if start <= f['timestep_index'] < end]
    n = len(frames)
    assert n and [f['timestep_index'] for f in frames] == list(range(start, start + n)), 'frames must be contiguous'
    os.makedirs(dst)
    for d in FRAME_DIRS:
        if not os.path.isdir(os.path.join(src, d)):
            continue
        os.makedirs(os.path.join(dst, d))
        ext = os.path.splitext(sorted(os.listdir(os.path.join(src, d)))[0])[1]
        for i in range(n):
            shutil.copy(os.path.join(src, d, f'{start + i}{ext}'), os.path.join(dst, d, f'{i}{ext}'))
    os.makedirs(os.path.join(dst, 'ori_imgs'))
    for i in range(n):
        for ext in ('.jpg', '.lms'):
            shutil.copy(os.path.join(src, 'ori_imgs', f'{start + i}{ext}'), os.path.join(dst, 'ori_imgs', f'{i}{ext}'))
    out = []
    for i, f in enumerate(frames):
        g = dict(f)
        g.update(timestep_index=i, timestep_index_original=i, timestep_id=f'{i:06d}', file_path=f'gt_imgs/{i}.jpg',
                 fg_mask_path=f'fg_masks/{i:05d}_00.png', flame_param_path=f'flame_param/{i:05d}.npz')
        out.append(g)
    tf2 = dict(tf, frames=out, timestep_indices=list(range(n)))
    json.dump(tf2, open(os.path.join(dst, 'transforms.json'), 'w'), indent=2)
    for name in FLAME_FILES:
        if os.path.exists(os.path.join(src, name)):
            z = np.load(os.path.join(src, name), allow_pickle=True)
            d = {str(i): z[str(start + i)] for i in range(n)}
            if '_meta' in z.files:
                d['_meta'] = z['_meta']
            np.savez(os.path.join(dst, name), **d)
    for name in FRAME_ARRAYS:
        if os.path.exists(os.path.join(src, name)):
            a = np.load(os.path.join(src, name))
            assert len(a) >= start + n, (name, a.shape)
            np.save(os.path.join(dst, name), a[start:start + n])
    if os.path.exists(os.path.join(src, 'aud.wav')):
        a, sr = sf.read(os.path.join(src, 'aud.wav'))
        s0 = int(round(start * sr / 25))
        sf.write(os.path.join(dst, 'aud.wav'), a[s0:s0 + int(round(n * sr / 25))], sr)
    if os.path.exists(os.path.join(src, 'au_features.csv')):
        lines = open(os.path.join(src, 'au_features.csv')).read().strip().split('\n')
        assert lines[0].startswith('frame,')
        body = [f'{i},' + lines[1 + start + i].split(',', 1)[1] for i in range(n)]
        open(os.path.join(dst, 'au_features.csv'), 'w').write('\n'.join(lines[:1] + body) + '\n')
    for name in STATIC_FILES:
        if os.path.exists(os.path.join(src, name)):
            shutil.copy(os.path.join(src, name), os.path.join(dst, name))
    print(dst, n, 'frames')


if __name__ == '__main__':
    main()
