"""Audio-lag calibration for few-shot adaptation."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import torch


MAX_LAG = 8
PRIOR_LEAD_MAX = 3


def _lagcorr(a, b, lag):
    n = len(a)
    if lag >= 0:
        x, y = a[:n - lag], b[lag:]
    else:
        x, y = a[-lag:], b[:n + lag]
    if len(x) < 10 or x.std() < 1e-8 or y.std() < 1e-8:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def calibrate_audio_lag(scene_path: str, motion_state: dict, n_views: int, lags=range(0, MAX_LAG + 1), min_gain: float = 0.0,
                        min_corr: float = 0.3, audio_extractor: str = 'wav2vec2'):
    """(lag, {lag: correlation})."""
    from scene.motion_net import MotionNetwork
    from utils.au_utils import load_openface_au_csv
    from utils.audio_utils import get_audio_features

    tf = json.load(open(os.path.join(scene_path, 'transforms.json')))
    frames = tf['frames'] if n_views is None or n_views <= 0 else tf['frames'][:n_views]
    ids = [int(f['timestep_index']) for f in frames]
    from utils.audio_features import feature_path
    aud = torch.from_numpy(np.load(feature_path(scene_path, audio_extractor))).float().permute(0, 2, 1)
    from utils.drive_sanitize import load_drive_params
    fp = load_drive_params(os.path.join(scene_path, 'flame_params.npz'), isolate=ids)[0]
    gt = np.array([np.asarray(fp[i]['jaw_pose']).reshape(-1)[0] for i in ids])
    au = load_openface_au_csv(os.path.join(scene_path, 'au_features.csv'), required_len=max(ids) + 1)
    idp = os.path.join(scene_path, 'identity_feature.npy')
    ident = torch.from_numpy(np.load(idp).reshape(-1)).float().cuda() if os.path.exists(idp) else None
    net = MotionNetwork(args=SimpleNamespace(audio_extractor=audio_extractor)).cuda().eval()
    net.load_state_dict(motion_state, strict=False)
    net.init_novel_code()
    au_mean = torch.from_numpy(au[ids].mean(0)).float().cuda()
    preds = []
    with torch.no_grad():
        for t in ids:
            out = net(get_audio_features(aud, 2, t).cuda(), c=net.identity_code(0), au=au_mean, identity=ident)
            preds.append(out['flame_jaw'][out['flame_jaw'].shape[0] // 2, 0].item())
    preds = np.array(preds)
    corr = {int(L): _lagcorr(preds, gt, L) for L in lags}
    finite = {k: v for k, v in corr.items() if np.isfinite(v)}
    if not finite:
        return 0, corr
    best = max(finite, key=finite.get)
    if best != 0 and (finite[best] - finite.get(0, -1.0) < min_gain or finite[best] < min_corr):
        best = 0
    if best != 0 and best == max(finite):
        print(f'[audio lag] WARNING: the best lag ({best} frames) is at the end of the search range '
              f'{min(finite)}..{max(finite)}; the true lag may be larger.')
    if best > PRIOR_LEAD_MAX:
        print(f'[audio lag] WARNING: a lag of {best} frames is more than the prior\'s own lead (0-{PRIOR_LEAD_MAX} frames): '
              f'the audio of the adaptation frames is probably offset against their video. Frames of another take '
              f'with another offset will be driven out of sync; check the clip with '
              f'tools/check_av_sync.py --split_at <number of adaptation frames>.')
    del net
    torch.cuda.empty_cache()
    return best, corr
