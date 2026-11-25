#!/usr/bin/env python3
"""Attribute the predicted FLAME motion to its inputs (audio vs. action units).

    python tools/analyze_motion_drivers.py -s data/<ID> -m output/<ID> --eval_from 125
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.motion_net import MotionNetwork
from utils.au_utils import load_openface_au_csv
from utils.audio_utils import get_audio_features


def _atomic_save(path: str, arr: np.ndarray):
    """np.save via a temporary file + rename (parallel jobs may share the cache)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path[:-4]}.{os.getpid()}.tmp.npy'
    np.save(tmp, arr)
    os.replace(tmp, path)


def silence_features(num_frames: int, cache: str | None, extractor: str = 'wav2vec2') -> np.ndarray:
    """Audio features of digital silence with the same frame count."""
    if cache and extractor != 'wav2vec2':
        cache = cache.replace('.npy', {'whisper': '_whisper_tiled_v2.npy'}.get(extractor, f'_{extractor}.npy'))
    if cache and os.path.exists(cache):
        feats = np.load(cache)
        constant_ok = extractor != 'whisper' or float(np.abs(feats - feats[:1]).max()) == 0.0
        if feats.shape[0] >= num_frames and constant_ok:
            return feats[:num_frames]
    from tools.extract_wav2vec2_features import window_features
    sr = 16000
    wav = np.zeros(int(round(num_frames / 25.0 * sr)) + sr // 5, dtype=np.float32)
    if extractor == 'whisper':
        from tools.extract_whisper_features import WhisperEncoder
        h = WhisperEncoder().hidden(np.zeros(sr, dtype=np.float32))
        c = len(h) // 2
        win = np.repeat(h[None, c - 8:c + 8], num_frames, axis=0).astype(np.float32)
        if cache:
            _atomic_save(cache, win)
        return win
    else:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        proc = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').cuda().eval()
        with torch.no_grad():
            iv = proc(wav, sampling_rate=sr, return_tensors='pt').input_values.cuda()
            hidden = model(iv).last_hidden_state.squeeze(0).float().cpu()
    feats = window_features(hidden, num_frames, 16).numpy().astype(np.float32)
    if cache:
        _atomic_save(cache, feats)
    return feats


def pearson_per_dim(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p = pred - pred.mean(0)
    g = gt - gt.mean(0)
    den = np.sqrt((p ** 2).sum(0) * (g ** 2).sum(0))
    out = np.full(pred.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = (p * g).sum(0)[ok] / den[ok]
    flat = pred.std(0) < 1e-3 * (gt.std(0) + 1e-12)
    out[flat & np.isfinite(gt.std(0))] = 0.0
    return out


def region_corr(pred_exp, gt_exp, basis):
    """Correlation of the expression-induced displacement of a vertex region (flattened over vertices and xyz), i.e."""
    if basis is None:
        return float('nan')
    p = pred_exp @ basis.T
    g = gt_exp @ basis.T
    return float(np.nanmean(pearson_per_dim(p, g)))


def load_motion_net(ckpt_path: str, extractor: str = 'wav2vec2'):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ck, (list, tuple)) and len(ck) == 4:
        state = ck[1]
    elif isinstance(ck, (list, tuple)) and len(ck) == 3:
        state = ck[0]
    else:
        state = ck
    from utils.audio_features import EXTRACTOR_KEY, check_prior_extractor
    if EXTRACTOR_KEY in state:
        state = check_prior_extractor(state, extractor, ckpt_path)
    net = MotionNetwork(args=SimpleNamespace(audio_extractor=extractor)).cuda()
    net.load_state_dict(state, strict=False)
    net.eval()
    return net


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('-s', '--source_path', required=True)
    ap.add_argument('-m', '--model_path', default=None, help='adapted model dir (chkpnt_face_latest.pth)')
    ap.add_argument('--ckpt', default=None, help='explicit checkpoint (e.g. the released prior)')
    ap.add_argument('--eval_from', type=int, default=125)
    ap.add_argument('--audio_extractor', default='wav2vec2', help='used with --ckpt (otherwise read from cfg_args)')
    ap.add_argument('--no_aux_motion', action='store_true', help='analyse only the main motion network')
    ap.add_argument('--eval_to', type=int, default=-1)
    ap.add_argument('--silence_cache', default=os.path.join(os.path.expanduser('~'), '.cache', 'emotag', 'w2v_silence.npy'))
    ap.add_argument('--flame_regions', action='store_true', help='also report lower/upper-face region correlations (needs FLAME)')
    ap.add_argument('--out', default=None, help='write the table as JSON here')
    args = ap.parse_args()

    ckpt = args.ckpt or os.path.join(args.model_path, 'chkpnt_face_latest.pth')
    src = args.source_path
    audio_lag, extractor = 0, args.audio_extractor
    cfg = os.path.join(args.model_path, 'cfg_args') if args.model_path else None
    if cfg and os.path.exists(cfg):
        from argparse import Namespace
        _cfg = eval(open(cfg).read())
        audio_lag = int(getattr(_cfg, 'audio_lag', 0) or 0)
        extractor = getattr(_cfg, 'audio_extractor', extractor) or extractor
    net = load_motion_net(ckpt, extractor)
    members = [(net, extractor, audio_lag)]
    if args.model_path and not args.ckpt and not args.no_aux_motion:
        import glob as _glob
        for ap_ in sorted(_glob.glob(os.path.join(args.model_path, 'aux_motion_*.pth'))):
            aux = torch.load(ap_, map_location='cpu', weights_only=False)
            anet = MotionNetwork(args=SimpleNamespace(audio_extractor=aux['audio_extractor'])).cuda()
            anet.load_state_dict(aux['state_dict'], strict=False)
            anet.eval()
            members.append((anet, aux['audio_extractor'], int(aux['audio_lag'])))

    tf = json.load(open(os.path.join(src, 'transforms.json')))
    frame_ids = [int(f['timestep_index']) for f in tf['frames']]
    n = max(frame_ids) + 1
    from utils.audio_features import feature_path
    feats = {}
    for _, ext, _ in members:
        if ext not in feats:
            a_ = torch.from_numpy(np.load(feature_path(src, ext))).float().permute(0, 2, 1)
            s_ = torch.from_numpy(silence_features(a_.shape[0], args.silence_cache, ext)).float().permute(0, 2, 1)
            feats[ext] = (a_, s_)
    aud = feats[extractor][0]
    au = load_openface_au_csv(os.path.join(src, 'au_features.csv'), required_len=n)
    ident = None
    idp = os.path.join(src, 'identity_feature.npy')
    if os.path.exists(idp):
        ident = torch.from_numpy(np.load(idp).reshape(-1)).float().cuda()
    from utils.drive_sanitize import load_drive_params
    fp = load_drive_params(os.path.join(src, 'flame_params.npz'))[0]
    gt_exp = np.stack([np.asarray(fp[i]['expression']).reshape(-1)[:50] for i in frame_ids])
    gt_jaw = np.stack([np.asarray(fp[i]['jaw_pose']).reshape(-1)[:3] for i in frame_ids])

    au_ref = net.get_au_reference() if hasattr(net, 'get_au_reference') else None
    if au_ref is None:
        au_ref = au[frame_ids].mean(0)
    else:
        au_ref = au_ref.cpu().numpy()

    code = None
    if net.individual_dim > 0 and hasattr(net, 'identity_code'):
        code = net.identity_code()

    def run(audio_kind, au_mode):
        all_e, all_j = [], []
        with torch.no_grad():
            for m_net, m_ext, m_lag in members:
                src_feats = feats[m_ext][0 if audio_kind == 'own' else 1]
                m_code = m_net.identity_code(0) if m_net.individual_dim > 0 else None
                exps, jaws = [], []
                for t in frame_ids:
                    a = get_audio_features(src_feats, 2, t - m_lag).cuda()
                    u = au[min(t, au.shape[0] - 1)] if au_mode == 'native' else au_ref
                    out = m_net(a, c=m_code, au=torch.from_numpy(np.asarray(u)).float().cuda(), identity=ident)
                    e, j = out['flame_exp'], out['flame_jaw']
                    exps.append(e[e.shape[0] // 2].cpu().numpy())
                    jaws.append(j[j.shape[0] // 2].cpu().numpy())
                all_e.append(np.stack(exps)); all_j.append(np.stack(jaws))
        return np.mean(all_e, 0), np.mean(all_j, 0)

    basis_lower = basis_upper = None
    if args.flame_regions:
        from utils.flame_regions import expression_region_bases
        basis_lower, basis_upper = expression_region_bases()

    lo = args.eval_from
    hi = len(frame_ids) if args.eval_to < 0 else args.eval_to
    rows = []
    for name, a_src, au_mode in [('own audio + native AU (standard)', 'own', 'native'),
                                 ('own audio + AU frozen to mean (audio alone)', 'own', 'mean'),
                                 ('silence + native AU (AU alone)', 'silence', 'native'),
                                 ('silence + AU frozen to mean (baseline)', 'silence', 'mean')]:
        pe, pj = run(a_src, au_mode)
        pe, pj, ge, gj = pe[lo:hi], pj[lo:hi], gt_exp[lo:hi], gt_jaw[lo:hi]
        row = {
            'inputs': name,
            'jaw_corr': float(np.nanmean(pearson_per_dim(pj, gj))),
            'jaw_open_corr': float(pearson_per_dim(pj[:, :1], gj[:, :1])[0]),
            'exp_corr': float(np.nanmean(pearson_per_dim(pe, ge))),
            'pred_jaw_open_std': float(pj[:, 0].std()),
        }
        if args.flame_regions:
            row['exp_lower_corr'] = region_corr(pe, ge, basis_lower)
            row['exp_upper_corr'] = region_corr(pe, ge, basis_upper)
        rows.append(row)
    print(f'checkpoint: {ckpt} (audio lag {audio_lag}; ensemble members: {len(members)})')
    print(f'held-out frames [{lo}, {hi}), GT jaw-open std = {gt_jaw[lo:hi, 0].std():.4f}')
    keys = [k for k in rows[0] if k != 'inputs']
    print(f"{'inputs':46s} " + ' '.join(f'{k:>16s}' for k in keys))
    for r in rows:
        print(f"{r['inputs']:46s} " + ' '.join(f'{r[k]:16.3f}' for k in keys))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'checkpoint': ckpt, 'eval_from': lo, 'eval_to': hi, 'rows': rows}, f, indent=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
