"""Sanitising of the tracked FLAME drive (head pose, gaze) and of the driving-frame order."""
import json
import os

import numpy as np

META_KEY = '_meta'
TRACKED_CONVENTION = 'tracked'

GLITCH_RATIO = 30.0
GLITCH_MAX_RUN = 5
GAZE_MEDIAN_FRAMES = 7
GAZE_PERCENTILES = (1.0, 99.0)
GAZE_MARGIN_DEG = 2.0


def raw_drive():
    """True when the raw (unsanitised) drive was requested via EMOTAG_RAW_DRIVE=1."""
    return os.environ.get('EMOTAG_RAW_DRIVE', '0') == '1'


def _stack(all_params, ids, key):
    return np.stack([np.asarray(all_params[i][key], dtype=np.float32).reshape(-1) for i in ids])


def _write(all_params, ids, key, values):
    for i, v in zip(ids, values):
        ref = np.asarray(all_params[i][key])
        all_params[i][key] = v.astype(ref.dtype).reshape(ref.shape)


def _jumps(translation, ratio=GLITCH_RATIO):
    """Indices j where frames j and j+1 are further apart than `ratio` x the median step."""
    step = np.linalg.norm(np.diff(translation, axis=0), axis=1)
    typical = float(np.median(step)) if len(step) else 0.0
    if not typical > 0:
        return step, np.zeros(0, dtype=int)
    return step, np.flatnonzero(step > ratio * typical)


def find_glitched_frames(translation, ratio=GLITCH_RATIO, max_run=GLITCH_MAX_RUN):
    """Boolean mask of frames whose tracked head position is an outlier."""
    translation = np.asarray(translation, dtype=np.float64).reshape(len(translation), -1)
    n = len(translation)
    bad = np.zeros(n, dtype=bool)
    if n < 2 * max_run + 2:
        return bad
    step, jumps = _jumps(translation, ratio)
    if len(jumps) == 0:
        return bad
    bounds = [0] + [int(j) + 1 for j in jumps] + [n]
    trusted = np.zeros(n, dtype=bool)
    for a, b in zip(bounds[:-1], bounds[1:]):
        trusted[a:b] = (b - a) > max_run
    if not trusted.any():
        return bad
    i = 0
    while i < n:
        if trusted[i]:
            i += 1
            continue
        a = i
        while i < n and not trusted[i]:
            i += 1
        b = i
        if b - a > max_run:
            continue
        if a == 0 or b == n:
            bad[a:b] = True
        elif np.linalg.norm(translation[b] - translation[a - 1]) < 0.5 * min(step[a - 1], step[b - 1]):
            bad[a:b] = True
    return bad


def repair_glitched_frames(all_params, bad=None):
    """Replace the whole FLAME state of glitched frames by the nearest valid frame's (in place)."""
    ids = sorted(all_params)
    if len(ids) < 2 or 'translation' not in all_params[ids[0]]:
        return []
    if bad is None:
        bad = find_glitched_frames(_stack(all_params, ids, 'translation'))
    if not bad.any() or bad.all():
        return []
    good = np.flatnonzero(~bad)
    for k in np.flatnonzero(bad):
        src = good[np.argmin(np.abs(good - k))]
        all_params[ids[k]] = {key: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                              for key, v in all_params[ids[src]].items()}
    return [ids[k] for k in np.flatnonzero(bad)]


def median_filter_gaze(all_params, size=GAZE_MEDIAN_FRAMES):
    """Temporal median filter of eyes_pose over the clip (in place)."""
    ids = sorted(all_params)
    if size <= 1 or len(ids) < size or 'eyes_pose' not in all_params[ids[0]]:
        return
    from scipy.ndimage import median_filter
    eyes = _stack(all_params, ids, 'eyes_pose')
    _write(all_params, ids, 'eyes_pose', median_filter(eyes, size=(size, 1), mode='nearest'))


def gaze_range(clips, percentiles=GAZE_PERCENTILES, margin_deg=GAZE_MARGIN_DEG):
    """Per-component eyes_pose range covered by the adaptation frames."""
    eyes = [_stack(p, [i for i in frame_ids if i in p], 'eyes_pose') for p, frame_ids in clips
            if any(i in p for i in frame_ids)]
    if not eyes:
        return None
    eyes = np.concatenate(eyes)
    margin = np.deg2rad(margin_deg)
    lo = np.percentile(eyes, percentiles[0], axis=0) - margin
    hi = np.percentile(eyes, percentiles[1], axis=0) + margin
    return lo.astype(np.float32), hi.astype(np.float32)


def clamp_gaze(all_params, lo, hi):
    """Clamp eyes_pose of every frame to [lo, hi] (in place)."""
    ids = sorted(all_params)
    if not ids or 'eyes_pose' not in all_params[ids[0]]:
        return
    _write(all_params, ids, 'eyes_pose', np.clip(_stack(all_params, ids, 'eyes_pose'), lo, hi))


def sanitize_params(all_params, isolate=None):
    """Repair glitched frames and median filter the gaze of one clip (in place)."""
    isolate = set(int(i) for i in isolate) if isolate is not None else set()
    ids = sorted(all_params)
    own = [i for i in ids if i in isolate]
    rest = [i for i in ids if i not in isolate]
    rest_bad = None
    if own and len(rest) >= 2 and 'translation' in all_params[ids[0]]:
        whole = find_glitched_frames(_stack(all_params, ids, 'translation'))
        rest_bad = whole[[k for k, i in enumerate(ids) if i not in isolate]]
    repaired = []
    for part, bad in ((own, None), (rest, rest_bad)):
        if not part:
            continue
        sub = {i: all_params[i] for i in part}
        repaired += repair_glitched_frames(sub, bad)
        median_filter_gaze(sub)
        all_params.update(sub)
    return sorted(repaired)


def drive_meta(flame_params_file):
    """The `_meta` entry of a flame_params.npz as a dict ({} when the file has none, i.e."""
    with np.load(flame_params_file, allow_pickle=True) as data:
        if META_KEY not in data.files:
            return {}
        meta = data[META_KEY].item()
    return json.loads(meta) if isinstance(meta, str) else dict(meta)


def drive_convention(flame_params_file):
    """Convention tag of a flame_params.npz."""
    if not os.path.exists(flame_params_file):
        return None
    return drive_meta(flame_params_file).get('convention', TRACKED_CONVENTION)


def has_refit_eyelids(convention):
    """True when the convention's eyelid state follows the image (blinks close the lids)."""
    return bool(convention) and convention != TRACKED_CONVENTION and 'eyelids' in convention.split(':')[-1].split('+')


def load_drive_params(flame_params_file, isolate=None):
    """{frame id: FLAME state} of one clip, glitch-repaired and gaze-filtered unless the raw drive is requested."""
    data = np.load(flame_params_file, allow_pickle=True)
    all_params = {}
    for key in data.files:
        if key == META_KEY:
            continue
        try:
            all_params[int(key)] = data[key].item()
        except (ValueError, AttributeError) as e:
            print(f'Skipping invalid FLAME parameter entry {key}: {e}')
    repaired = []
    if not raw_drive():
        repaired = sanitize_params(all_params, isolate)
    return all_params, repaired


def playback_order(n_frames, n_out):
    """Indices of the driving frames for `n_out` output frames."""
    t = np.arange(n_out)
    if n_frames <= 1:
        return np.zeros(n_out, dtype=int)
    if raw_drive():
        return t % n_frames
    period = 2 * (n_frames - 1)
    t = t % period
    return np.where(t < n_frames, t, period - t)
