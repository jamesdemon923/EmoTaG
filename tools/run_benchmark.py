#!/usr/bin/env python3
"""EmoTaG benchmark on the released evaluation scenes.

  neutral_<ID>  adapt on frames [0, 125), evaluate frames [125, 375)
  mead1_<e>, mead2_<e>  adapt on frames [0, 125), evaluate the test recording <scene>_test
  novel audio   neutral clips driven by another speaker (same / other language), Sync-C/E only
  --test_scene  evaluate on any separately tracked recording of the same identity

    python tools/run_benchmark.py eval --data data --model output/neutral_Obama --scene neutral_Obama
    python tools/run_benchmark.py eval --data data --model output/neutral_Obama --scene neutral_Obama --novel_audio xid
    python tools/run_benchmark.py eval --data data --model output/mead2_happy --scene mead2_happy --test_scene mead2_happy_lvl1 --tag intensity_lvl1
    python tools/run_benchmark.py all --data data --output output
    python tools/run_benchmark.py summary --output output
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSS_IDENTITY = {
    'neutral_Obama': 'neutral_english_m', 'neutral_english_m': 'neutral_Obama',
    'neutral_cnn': 'neutral_may', 'neutral_may': 'neutral_cnn',
    'neutral_english_w': 'neutral_Lieu', 'neutral_Lieu': 'neutral_Shaheen', 'neutral_Shaheen': 'neutral_english_w',
}
CROSS_LANGUAGE = {
    'neutral_chinese': 'neutral_english_w', 'neutral_english_w': 'neutral_chinese',
    'neutral_Shaheen': 'neutral_macron',
    'neutral_Jaein': 'neutral_Lieu', 'neutral_Lieu': 'neutral_Jaein',
    'neutral_macron': 'neutral_english_w',
}
NOVEL = {'xid': CROSS_IDENTITY, 'xlang': CROSS_LANGUAGE}
COLS = [('PSNR', 'PSNR', 2), ('SSIM', 'SSIM', 3), ('LPIPS', 'LPIPS', 3), ('PSNR_head', 'PSNR head', 2),
        ('SSIM_head', 'SSIM head', 3), ('LPIPS_head', 'LPIPS head', 3), ('LMD', 'LMD', 3),
        ('sync_Sync-C', 'Sync-C', 2), ('sync_Sync-E', 'Sync-E', 2), ('sync_offset', 'AV offset', 1)]


def _run(cmd, log):
    with open(log, 'a') as f:
        f.write('\n$ ' + ' '.join(cmd) + '\n')
        f.flush()
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f'failed ({r.returncode}): {" ".join(cmd)} (see {log})')


def model_extractor(model):
    cfg = os.path.join(model, 'cfg_args')
    if os.path.exists(cfg):
        from argparse import Namespace
        return getattr(eval(open(cfg).read()), 'audio_extractor', 'wav2vec2') or 'wav2vec2'
    return 'wav2vec2'


def protocol(scene, data, novel_audio=None, sync_data=None, extractor='wav2vec2', test_scene=None):
    """(render scene dir, audio wav, eval_from, extra render args, audio start frame)."""
    sync_data = sync_data or data
    if test_scene:
        if novel_audio:
            raise ValueError('--test_scene and --novel_audio are exclusive')
        return (os.path.join(data, test_scene), os.path.join(sync_data, test_scene, 'aud.wav'), 0,
                ['--N_views', '-1'], 0)
    if novel_audio:
        src = NOVEL[novel_audio][scene]
        return (os.path.join(data, scene), os.path.join(sync_data, src, 'aud.wav'), 0,
                ['--audio_file', os.path.join(sync_data, src, 'aud_whisper.npy' if extractor == 'whisper' else 'aud_w2v.npy'),
                 '--N_views', '-1'], 0)
    if os.path.isdir(os.path.join(data, scene + '_test')):
        return os.path.join(data, scene + '_test'), os.path.join(sync_data, scene + '_test', 'aud.wav'), 0, ['--N_views', '-1'], 0
    start = 150 if scene.startswith('mead2_') else 125
    return os.path.join(data, scene), os.path.join(sync_data, scene, 'aud.wav'), start, [], start


def tag_suffix(render_args):
    """Non-default renderer settings never overwrite the default results."""
    ra, sfx = list(render_args), ''
    if '--no_aux_motion' in ra:
        sfx += '_single'
    if '--articulation_gain' in ra:
        sfx += '_g' + ra[ra.index('--articulation_gain') + 1].replace(',', '-')
    if '--smooth_sigma' in ra:
        sfx += '_s' + ra[ra.index('--smooth_sigma') + 1]
    if '--motion_norm' in ra:
        sfx += '_' + ra[ra.index('--motion_norm') + 1]
    return sfx


def evaluate(model, scene, data, novel_audio=None, tag=None, skip_lmd=False, sync_data=None, render_args=(),
             test_scene=None):
    tag = tag or ((f'ood_{novel_audio}' if novel_audio else 'heldout') + (f'_{test_scene}' if test_scene else '')
                  + tag_suffix(render_args))
    py = sys.executable
    os.makedirs(os.path.join(model, 'eval'), exist_ok=True)
    log = os.path.join(model, 'eval', f'{tag}.log')
    render_scene, wav, eval_from, extra, audio_start = protocol(scene, data, novel_audio, sync_data, model_extractor(model),
                                                                test_scene)
    if not os.path.exists(wav):
        raise FileNotFoundError(f'audio for SyncNet missing: {wav}')
    _run([py, 'tools/render_composited.py', '-s', render_scene, '-m', model, '--out_name', tag,
          '--eval_from', str(eval_from), '--wav', wav] + extra + list(render_args), log)
    t, mdir = os.path.join(model, 'test'), os.path.join(model, 'eval', tag)
    os.makedirs(mdir, exist_ok=True)
    res = {'scene': scene, 'tag': tag, 'render_scene': os.path.basename(render_scene), 'audio': '/'.join(wav.split(os.sep)[-2:]),
           'render_args': list(render_args)}
    if not novel_audio:
        cmd = [py, 'evaluate_metrics.py', 'video', f'{t}/rendered_{tag}.mp4', f'{t}/gt_{tag}.mp4', mdir,
               '--masks', f'{t}/masks_{tag}.npz'] + (['--skip_lmd'] if skip_lmd else [])
        _run(cmd, log)
        res.update(json.load(open(os.path.join(mdir, 'metrics.json'))))
    _run([py, 'evaluate_metrics.py', 'sync', f'{t}/rendered_{tag}.mp4', wav, '--audio_start_frame', str(audio_start),
          '--output', os.path.join(mdir, 'sync.json')], log)
    res.update({f'sync_{k}': v for k, v in json.load(open(os.path.join(mdir, 'sync.json'))).items()})
    if scene.startswith('neutral_') and not novel_audio and not test_scene:
        out = os.path.join(mdir, 'drivers.json')
        _run([py, 'tools/analyze_motion_drivers.py', '-s', render_scene, '-m', model, '--eval_from', '125',
              '--flame_regions', '--out', out] + (['--no_aux_motion'] if '--no_aux_motion' in render_args else []), log)
        res['drivers'] = json.load(open(out))['rows']
    with open(os.path.join(model, 'eval', f'{tag}.json'), 'w') as f:
        json.dump(res, f, indent=2)
    return res


def summary(output, tag):
    lines = []
    for group in ('neutral_', 'mead1_', 'mead2_'):
        rows = []
        for p in sorted(glob.glob(os.path.join(output, f'{group}*', 'eval', f'{tag}.json'))):
            rows.append(json.load(open(p)))
        if not rows:
            continue
        lines.append(f'\n### {group.rstrip("_")} ({tag}, n={len(rows)})\n')
        lines.append('| scene | ' + ' | '.join(c[1] for c in COLS) + ' |')
        lines.append('|' + '---|' * (len(COLS) + 1))
        for r in rows + [None]:
            if r is None:
                vals = []
                for k, _, d in COLS:
                    v = [x[k] for x in rows if isinstance(x.get(k), (int, float))]
                    vals.append(f'{np.mean(v):.{d}f}' if v else '-')
                lines.append('| **mean** | ' + ' | '.join(vals) + ' |')
            else:
                lines.append(f"| {r['scene']} | " + ' | '.join(
                    f"{r[k]:.{d}f}" if isinstance(r.get(k), (int, float)) else '-' for k, _, d in COLS) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('eval')
    e.add_argument('--data', required=True, help='directory holding the scenes (data/)')
    e.add_argument('--sync_data', default=None, help='scenes whose aud.wav/aud_w2v.npy are used for SyncNet and novel audio (default: --data)')
    e.add_argument('--model', required=True)
    e.add_argument('--scene', required=True)
    e.add_argument('--novel_audio', choices=['xid', 'xlang'], default=None,
                   help='drive with another speaker: same language (xid) or another language (xlang)')
    e.add_argument('--skip_lmd', action='store_true')
    e.add_argument('--test_scene', default=None,
                   help='evaluate on this scene of --data instead: another recording of the same identity, tracked '
                        'separately (whole clip, driven by its own pose/AUs and audio; its aud.wav in --sync_data)')
    e.add_argument('--tag', default=None, help='result name (default: heldout / ood_<kind>, + _single / _g<gain> '
                                                 'with --no_aux_motion / --articulation_gain)')
    a = sub.add_parser('all')
    a.add_argument('--data', required=True)
    a.add_argument('--sync_data', default=None)
    a.add_argument('--output', required=True, help='models are <output>/<scene>')
    a.add_argument('--scenes', nargs='*', default=None)
    a.add_argument('--skip_lmd', action='store_true')
    for p_ in (e, a):
        p_.add_argument('--no_aux_motion', action='store_true', help='render without the motion ensemble')
        p_.add_argument('--smooth_sigma', default=None, help='passed to render_composited.py')
        p_.add_argument('--motion_norm', default=None, choices=['auto', 'identity', 'gain', 'none'],
                        help='passed to render_composited.py')
        p_.add_argument('--articulation_gain', default=None,
                        help="passed to render_composited.py (default: the renderer's; released models get 1,1)")
    s = sub.add_parser('summary')
    s.add_argument('--output', required=True)
    s.add_argument('--tag', default='heldout')
    args = ap.parse_args()
    rargs = (['--no_aux_motion'] if getattr(args, 'no_aux_motion', False) else []) + \
        (['--articulation_gain', args.articulation_gain] if getattr(args, 'articulation_gain', None) else []) + \
        (['--smooth_sigma', args.smooth_sigma] if getattr(args, 'smooth_sigma', None) else []) + \
        (['--motion_norm', args.motion_norm] if getattr(args, 'motion_norm', None) else [])
    if args.cmd == 'eval':
        print(json.dumps({k: v for k, v in evaluate(args.model, args.scene, args.data, args.novel_audio, tag=args.tag,
                                                      skip_lmd=args.skip_lmd, sync_data=args.sync_data,
                                                      render_args=rargs, test_scene=args.test_scene).items()
                          if k != 'drivers'}, indent=1))
    elif args.cmd == 'all':
        scenes = args.scenes or sorted(os.path.basename(os.path.dirname(p))
                                       for p in glob.glob(os.path.join(args.output, '*', 'chkpnt_face_latest.pth')))
        for sc in scenes:
            m = os.path.join(args.output, sc)
            evaluate(m, sc, args.data, skip_lmd=args.skip_lmd, sync_data=args.sync_data, render_args=rargs)
            for kind, pairs in NOVEL.items():
                if sc in pairs:
                    evaluate(m, sc, args.data, novel_audio=kind, sync_data=args.sync_data, render_args=rargs)
            print('done', sc, flush=True)
        for tag in ('heldout', 'ood_xid', 'ood_xlang'):
            print(summary(args.output, tag + tag_suffix(rargs)))
    else:
        print(summary(args.output, args.tag))


if __name__ == '__main__':
    main()
