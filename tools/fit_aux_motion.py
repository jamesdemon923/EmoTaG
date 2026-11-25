#!/usr/bin/env python3
"""Fit an auxiliary audio -> motion network for an adapted avatar (optional two-encoder ensemble).

    python tools/fit_aux_motion.py -m output/<ID> --audio_extractor wav2vec2 --pretrain_path <Wav2Vec2 prior>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.motion_net import MotionNetwork
from utils.au_utils import load_openface_au_csv
from utils.audio_features import feature_path
from utils.audio_lag import calibrate_audio_lag
from utils.audio_utils import get_audio_features
from utils.drive_sanitize import load_drive_params
from utils.emotion_utils import load_emotion_features, emotion_score_from_distribution
from utils.loss_utils import semantic_emotion_guidance_loss

LEGACY_SETUP = {'adapt_adain_only': True, 'adapt_unfreeze_heads': True, 'adapt_unfreeze_backbone': False, 'window_supervision': True,
                'w_vel': 1.0, 'iterations': 20000, 'warm_up_iter': 1000, 'w_jaw_reg': 100.0,
                'w_emotion_kl': 0.01, 'w_emotion_score': 0.1}


def adaptation_setup(model: str, extra_scenes=None, extra_views=None) -> dict:
    """Clips (source_path + frame ids) and motion-training flags of an adapted model."""
    p = os.path.join(model, 'adaptation_setup.json')
    if os.path.exists(p):
        return json.load(open(p))
    cfg = eval(open(os.path.join(model, 'cfg_args')).read())
    setup = dict(LEGACY_SETUP)
    ids = [int(x) for x in open(os.path.join(model, 'adaptation_frames.txt')).read().split()]
    setup['clips'] = [{'source_path': cfg.source_path, 'frame_ids': ids}]
    extras = list(extra_scenes or [])
    cc = os.path.join(model, 'clip_colors.pth')
    if not extras and os.path.exists(cc):
        extras = torch.load(cc, map_location='cpu', weights_only=False)['clips'][1:]
    if extras and extra_views is None:
        sys.exit('[fit_aux] multi-clip model without adaptation_setup.json: pass --extra_views (the value of '
                 'adapt_emotag.py --extra_views, or its --N_views if --extra_views was not given)')
    for extra in extras:
        tf = json.load(open(os.path.join(extra, 'transforms.json')))['frames']
        n = int(extra_views)
        setup['clips'].append({'source_path': extra,
                               'frame_ids': sorted(int(f['timestep_index']) for f in (tf[:n] if n > 0 else tf))})
    if extras:
        print(f'[fit_aux] no adaptation_setup.json: {len(extras)} extra clip(s), first {extra_views} frames each')
    return setup


def load_clip(src: str, ext: str, ids: list) -> dict:
    aud = torch.from_numpy(np.load(feature_path(src, ext))).float().permute(0, 2, 1)
    fp = load_drive_params(os.path.join(src, 'flame_params.npz'), isolate=ids)[0]
    n = max(fp) + 1
    ge = torch.zeros(n, 50, device='cuda')
    gj = torch.zeros(n, 3, device='cuda')
    for k, p in fp.items():
        ge[int(k)] = torch.as_tensor(np.asarray(p['expression']).reshape(-1)[:50]).float()
        gj[int(k)] = torch.as_tensor(np.asarray(p['jaw_pose']).reshape(-1)[:3]).float()
    au = torch.from_numpy(load_openface_au_csv(os.path.join(src, 'au_features.csv'), required_len=n)).float().cuda()
    emo = load_emotion_features(os.path.join(src, 'emotion_features.npy'), required_len=n)
    mask = torch.zeros(n, dtype=torch.bool, device='cuda')
    mask[ids] = True
    return {'src': src, 'aud': aud, 'ge': ge, 'gj': gj, 'au': au, 'emo': emo, 'mask': mask, 'n': n, 'ids': list(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', required=True, help='adapted model dir (cfg_args, adaptation_setup.json)')
    ap.add_argument('--audio_extractor', required=True, choices=['wav2vec2', 'whisper'])
    ap.add_argument('--pretrain_path', required=True, help='prior for this extractor')
    ap.add_argument('--iterations', type=int, default=None, help='default: as the adaptation')
    ap.add_argument('--warm_up_iter', type=int, default=None, help='default: as the adaptation')
    ap.add_argument('--w_vel', type=float, default=None, help='default: as the adaptation')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--extra_scenes', nargs='*', default=None,
                    help='only for multi-clip models without adaptation_setup.json: the extra clips of the '
                         'adaptation (default: from clip_colors.pth)')
    ap.add_argument('--extra_views', type=int, default=None,
                    help='only for multi-clip models without adaptation_setup.json: frames used per extra clip')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    setup = adaptation_setup(args.model, args.extra_scenes, args.extra_views)
    for key in ('iterations', 'warm_up_iter', 'w_vel'):
        if getattr(args, key) is not None:
            setup[key] = getattr(args, key)
    ext = args.audio_extractor

    ck = torch.load(args.pretrain_path, map_location='cpu', weights_only=False)
    state = ck[0] if isinstance(ck, (list, tuple)) and len(ck) == 3 else (ck[1] if isinstance(ck, (list, tuple)) else ck)
    from utils.audio_features import check_prior_extractor
    state = check_prior_extractor(state, ext, args.pretrain_path,
                                  untagged=None if isinstance(ck, (list, tuple)) and len(ck) == 4 else 'wav2vec2')
    primary = setup['clips'][0]
    lag, _ = calibrate_audio_lag(primary['source_path'], state, len(primary['frame_ids']), audio_extractor=ext)
    net = MotionNetwork(args=SimpleNamespace(audio_extractor=ext)).cuda()
    net.load_state_dict(state)

    clips = [load_clip(c['source_path'], ext, c['frame_ids']) for c in setup['clips']]
    ident = torch.from_numpy(np.load(os.path.join(primary['source_path'], 'identity_feature.npy')).reshape(-1)).float().cuda()

    main_state = torch.load(os.path.join(args.model, 'chkpnt_face_latest.pth'), map_location='cpu', weights_only=False)[1]
    decoupled = 'au_decoupled' in main_state and float(main_state['au_decoupled'].reshape(-1)[0]) > 0.5
    if decoupled:
        net.configure_au_decoupling(main_state['au_reference'].float(), main_state['upper_face_proj'].float(), True)
    else:
        print('[fit_aux] main model is not AU-decoupled: fitting with the legacy AU fusion')
    net.init_novel_code()
    opt = torch.optim.AdamW(net.get_params(0.005, 0.0005, include_novel_code=True), betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01)
    if setup.get('adapt_adain_only', True):
        net.freeze_for_adaptation(unfreeze_heads=bool(setup['adapt_unfreeze_heads']),
                                  unfreeze_backbone=bool(setup['adapt_unfreeze_backbone']))
    else:
        print('[fit_aux] main model was adapted without --adapt_adain_only: fine-tuning the whole motion network')
    iters, warm = int(setup['iterations']), int(setup['warm_up_iter'])
    motion_steps = int(setup.get('motion_steps', 0))
    n_steps = motion_steps if motion_steps > 0 else iters - warm
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: 0.1 ** (min(it, n_steps) / n_steps))
    windows = {(ci, t): get_audio_features(c['aud'], 2, t - lag).cuda() for ci, c in enumerate(clips) for t in c['ids']}
    pairs = [(ci, t) for ci, c in enumerate(clips) for t in c['ids']]
    w_vel, w_jaw = float(setup['w_vel']), float(setup['w_jaw_reg'])
    print(f'[fit_aux] {len(clips)} clip(s), {len(pairs)} frames, audio lag {lag}, '
          f"{'decoupled' if decoupled else 'legacy'} AUs, {n_steps} steps")
    net.train()
    for _ in range(n_steps):
        ci, t = pairs[rng.integers(len(pairs))]
        c = clips[ci]
        n = c['n']
        w = torch.arange(t - 4, t + 4, device='cuda')
        wc = w.clamp(0, n - 1)
        out = net(windows[(ci, t)], c=net.identity_code(0), au=c['au'][wc], identity=ident)
        pe, pj = out['flame_exp'], out['flame_jaw']
        ge, gj = c['ge'][wc], c['gj'][wc]
        if setup['window_supervision']:
            inside = (w >= 0) & (w < n)
            valid = torch.zeros_like(inside)
            valid[inside] = c['mask'][w[inside]]
            valid &= ((w - lag) >= 0) & ((w - lag) < n)
            v = valid.float().unsqueeze(-1)
            nv = v.sum().clamp_min(1.0)
            pair = (valid[1:] & valid[:-1]).float().unsqueeze(-1)
            npair = pair.sum().clamp_min(1.0)
            le = (((pe - ge) ** 2) * v).sum() / (nv * 50) + w_vel * \
                ((((pe[1:] - pe[:-1]) - (ge[1:] - ge[:-1])) ** 2) * pair).sum() / (npair * 50)
            lj = (((pj - gj) ** 2) * v).sum() / (nv * 3) + w_vel * \
                ((((pj[1:] - pj[:-1]) - (gj[1:] - gj[:-1])) ** 2) * pair).sum() / (npair * 3)
        else:
            le = torch.nn.functional.mse_loss(pe[pe.shape[0] // 2], c['ge'][t])
            lj = torch.nn.functional.mse_loss(pj[pj.shape[0] // 2], c['gj'][t])
        loss = 0.1 * (le + w_jaw * lj)
        seg, _, _ = semantic_emotion_guidance_loss(out['emotion_logits'], out['gate'], torch.from_numpy(c['emo'][t]).float(),
                                                   float(emotion_score_from_distribution(c['emo'][t])),
                                                   w_kl=float(setup['w_emotion_kl']), w_score=float(setup['w_emotion_score']))
        loss = loss + seg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    out_path = os.path.join(args.model, f'aux_motion_{ext}.pth')
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()}, 'audio_extractor': ext,
                'audio_lag': int(lag), 'prior': os.path.basename(args.pretrain_path),
                'clips': [c['source_path'] for c in setup['clips']]}, out_path)
    print(f'saved {out_path} (audio lag {lag}, clips: {[os.path.basename(c["src"]) for c in clips]})')


if __name__ == '__main__':
    main()
