#!/usr/bin/env python3
"""Re-fit the lip and eyelid state of a tracked scene to the image.

Runs once per scene after tracking, landmark and action-unit extraction, before adaptation.
The tracked parameters are kept as flame_params_tracked.npz; the re-fit is written to flame_params.npz.

    python tools/refine_landmarks.py data/<ID>                     # in place
    python tools/refine_landmarks.py data/<ID> --out_scene DIR     # linked copy, data/<ID> untouched
    python tools/refine_landmarks.py data/<ID> --restore           # back to the tracked parameters
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.drive_sanitize import META_KEY, TRACKED_CONVENTION, drive_convention, drive_meta

PARAMS, TRACKED = 'flame_params.npz', 'flame_params_tracked.npz'
REFIT_VERSION = 1
UP = [60, 61, 62, 63, 64]; LO = [60, 67, 66, 65, 64]
PARAM_KEYS = ('shape', 'expression', 'rotation', 'jaw_pose', 'neck_pose', 'eyes_pose', 'translation')


def load_scene(scene_dir, params_file=None):
    """Raw (unsanitised) per-frame FLAME state of `params_file` (default <scene>/flame_params.npz), cameras and."""
    d = np.load(params_file or os.path.join(scene_dir, PARAMS), allow_pickle=True)
    ids = sorted(int(k) for k in d.files if k != META_KEY)
    frames = {i: d[str(i)].item() for i in ids}
    t = json.load(open(os.path.join(scene_dir, 'transforms.json')))
    by = {f['timestep_index']: f for f in t['frames']}
    w2c, intr = [], []
    for i in ids:
        f = by[i]
        c2w = np.array(f['transform_matrix'], np.float64); c2w[:3, 1:3] *= -1
        w2c.append(np.linalg.inv(c2w))
        intr.append([f.get('fl_x', t['fl_x']), f.get('fl_y', t['fl_y']), t['w'] / 2, t['h'] / 2])
    lms = np.stack([np.loadtxt(os.path.join(scene_dir, 'ori_imgs', f'{i}.lms'))[:, :2] for i in ids])
    return dict(ids=np.array(ids), frames=frames, w2c=np.stack(w2c), intr=np.array(intr), lms=lms, dir=scene_dir)


def build_flame(params_file, device):
    """The FLAME model of scene/flame_wrapper.py (plain head mesh)."""
    from scene.flame_wrapper import SimpleFlameWrapper
    return SimpleFlameWrapper(flame_params_file=params_file, device=device).flame_model.eval()


def stack_params(scene, device):
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    return {k: T(np.concatenate([np.asarray(scene['frames'][i][k]) for i in scene['ids']], 0)) for k in PARAM_KEYS}


class LipGeometry:
    """Static (model-only) quantities."""

    def __init__(self, flame, sheet_mm=45.0, corner_mm=2.0):
        mk = flame.mask.v; dev = flame.v_template.device
        vt = flame.v_template.detach().cpu().numpy().astype(np.float64); F = flame.faces.cpu().numpy()
        lips = torch.unique(torch.cat([mk.lips, mk.lips_tight, mk.lip_inside])).cpu().numpy()
        ru, rl = mk.lip_inside_ring_upper.cpu().numpy(), mk.lip_inside_ring_lower.cpu().numpy()
        E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        oc = np.concatenate([mk.lip_outside_ring_upper.cpu().numpy()[[0, -1]], mk.lip_outside_ring_lower.cpu().numpy()[[0, -1]], ru[[0, -1]], rl[[0, -1]]])
        x_cut = np.abs(vt[oc, 0]).min() - corner_mm * 1e-3
        E = E[(np.abs(vt[E[:, 0], 0]) < x_cut) & (np.abs(vt[E[:, 1], 0]) < x_cut)]
        w = np.linalg.norm(vt[E[:, 0]] - vt[E[:, 1]], axis=1)
        G = coo_matrix((w, (E[:, 0], E[:, 1])), shape=(len(vt), len(vt)))
        mid = len(ru) // 2
        du = dijkstra(G, directed=False, indices=ru[mid - 3:mid + 4]).min(0); dl = dijkstra(G, directed=False, indices=rl[mid - 3:mid + 4]).min(0)
        up = np.where((du < sheet_mm * 1e-3) & (du < dl))[0]; lo = np.where((dl < sheet_mm * 1e-3) & (dl < du))[0]
        self.up, self.lo = torch.as_tensor(up, device=dev), torch.as_tensor(lo, device=dev)
        self.e_up = torch.as_tensor(E[np.isin(E[:, 0], up) & np.isin(E[:, 1], up)], device=dev)
        self.e_lo = torch.as_tensor(E[np.isin(E[:, 0], lo) & np.isin(E[:, 1], lo)], device=dev)
        self.lips = torch.as_tensor(lips, device=dev)
        n_s = flame.n_shape_params
        Ex = flame.shapedirs[:, :, n_s:n_s + 50].double() * 1e3
        upper = torch.unique(torch.cat([mk.eye_region, mk.forehead, mk.left_eyeball, mk.right_eyeball]))
        self.A_lip = (torch.einsum('vci,vcj->ij', Ex[self.lips], Ex[self.lips]) / len(self.lips)).float()
        self.A_up = (torch.einsum('vci,vcj->ij', Ex[upper], Ex[upper]) / len(upper)).float()
        self.upper = upper


def flame_forward(flame, P, expr, jaw):
    return flame(shape=P['shape'], expr=expr, rotation=P['rotation'], neck=P['neck_pose'], jaw=jaw, eyes=P['eyes_pose'],
                 translation=P['translation'], return_landmarks=True)


def project(X, w2c, intr):
    Xc = torch.einsum('nij,nvj->nvi', w2c[:, :3, :3], X) + w2c[:, None, :3, 3]
    return torch.stack([intr[:, None, 0] * Xc[..., 0] / Xc[..., 2] + intr[:, None, 2],
                        intr[:, None, 1] * Xc[..., 1] / Xc[..., 2] + intr[:, None, 3]], -1)


def mouth_axes(p48, p54):
    """Unit vectors along the mouth (corner to corner) and across it (pointing to the chin in the image)."""
    a = p54 - p48; a = a / a.norm(dim=-1, keepdim=True)
    n = torch.stack([-a[..., 1], a[..., 0]], -1)
    return a, n * torch.sign(n[..., 1:2])


def _silhouette(x, y, edges, xk, upper):
    """Extreme of a lip sheet at the columns xk."""
    x0, x1 = x[:, edges[:, 0], None], x[:, edges[:, 1], None]; y0, y1 = y[:, edges[:, 0], None], y[:, edges[:, 1], None]
    dx = x1 - x0; dx = torch.where(dx.abs() < 1e-4, torch.full_like(dx, 1e-4), dx)
    t = (xk[:, None] - x0) / dx
    ok = (t >= 0) & (t <= 1)
    yk = y0 + t.clamp(0, 1) * (y1 - y0)
    if upper:
        return torch.where(ok, yk, torch.full_like(yk, -1e4)).amax(1), ok.any(1)
    return torch.where(ok, yk, torch.full_like(yk, 1e4)).amin(1), ok.any(1)


def mesh_gap(pv, pl, geo, s, axes=None):
    """Visible inner-lip opening of the mesh at the normalised positions s [K] between the mesh inner corners (FLAME."""
    a, n = mouth_axes(pl[:, 48], pl[:, 54]) if axes is None else axes
    x = (pv * a[:, None]).sum(-1); y = (pv * n[:, None]).sum(-1)
    xl = (pl[:, 60] * a).sum(-1, keepdim=True); xr = (pl[:, 64] * a).sum(-1, keepdim=True)
    xk = (xl + s[None] * (xr - xl)).detach()
    top, ok_u = _silhouette(x, y, geo.e_up, xk, True); bot, ok_l = _silhouette(x, y, geo.e_lo, xk, False)
    ok = ok_u & ok_l
    return torch.where(ok, top, torch.zeros_like(top)), torch.where(ok, bot, torch.zeros_like(bot)), xk, ok


def aperture_from_gap(gap, xk, width):
    """Mean visible gap (px)."""
    g = gap.clamp(min=0)
    return (0.5 * (g[:, 1:] + g[:, :-1]) * (xk[:, 1:] - xk[:, :-1]).abs()).sum(1) / width.clamp(min=1.0)


def fan_profile(lms, s, axes, origin):
    """FAN inner-lip upper / lower edge at normalised positions s between FAN landmarks 60 and 64 ([N,K] each."""
    a, n = axes
    w = ((lms[:, 64] - lms[:, 60]) * a).sum(-1).clamp(min=1.0)
    def prof(idx):
        x = ((lms[:, idx] - lms[:, 60:61]) * a[:, None]).sum(-1) / w[:, None]
        y = ((lms[:, idx] - origin[:, None]) * n[:, None]).sum(-1)
        out = []
        for k in range(len(s)):
            j = (x[:, 1:-1] < s[k]).sum(1, keepdim=True)
            x0, x1 = x.gather(1, j)[:, 0], x.gather(1, j + 1)[:, 0]; y0, y1 = y.gather(1, j)[:, 0], y.gather(1, j + 1)[:, 0]
            t = ((s[k] - x0) / (x1 - x0).clamp(min=1e-3)).clamp(0, 1)
            out.append(y0 + t * (y1 - y0))
        return torch.stack(out, 1)
    p = lms[:, 60:68]
    area = 0.5 * (p[..., 0] * torch.roll(p[..., 1], -1, -1) - torch.roll(p[..., 0], -1, -1) * p[..., 1]).sum(-1).abs()
    return prof(UP), prof(LO), area / (lms[:, 60] - lms[:, 64]).norm(dim=-1).clamp(min=1.0)


def dead_zone(g, mean_gap, closed_px, seal_px):
    """FAN landmarks are integer-quantised and report up to ~2 px of gap on sealed lips."""
    gate = torch.sigmoid((mean_gap - closed_px) / (0.15 * closed_px))[:, None]
    return gate * g - (1 - gate) * seal_px, gate


def smooth1d(x, sigma):
    """Gaussian filter along time (dim 0) with edge replication."""
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma)); k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) / sigma) ** 2); k = k / k.sum()
    sh = x.shape; y = x.reshape(sh[0], -1).T[:, None]
    y = torch.nn.functional.pad(y, (r, r), mode='replicate')
    return torch.nn.functional.conv1d(y, k[None, None])[:, 0].T.reshape(sh)


DEFAULTS = dict(n_col=11, iters=500, lr=0.03, w_gap=1.0, w_mid=0.5, w_lmk=0.4, w_width=0.3, w_chin=0.05,
                w_prior_exp=0.3, w_prior_jaw=0.02, w_jaw_neg=20.0, w_upper=3.0, w_temp=1.0, closed_px=2.2, seal_px=0.5,
                fan_sigma=0.6)


def refine(scene, flame, device, verbose=True, lower_proj=None, **kw):
    """Returns expr[N,100] (first 50 re-fitted), jaw[N,3] and a diagnostics dict."""
    c = dict(DEFAULTS); c.update(kw)
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    ids = scene['ids']; N = len(ids)
    P = stack_params(scene, device)
    w2c, intr, lms = T(scene['w2c']), T(scene['intr']), T(scene['lms'])
    geo = LipGeometry(flame)
    s = torch.linspace(0.1, 0.9, c['n_col'], device=device)
    exp0, jaw0 = P['expression'][:, :50].clone(), P['jaw_pose'].clone()
    valid = torch.ones(N, dtype=torch.bool, device=device); valid[-1] = False
    vm = valid.float(); nv = vm.sum()
    mean_v = lambda e: (e * vm).sum() / nv

    with torch.no_grad():
        V0, L0 = flame_forward(flame, P, exp0, jaw0)
        pv0 = project(V0, w2c, intr); pl0 = project(L0[:, :68], w2c, intr)
        axes = mouth_axes(pl0[:, 48], pl0[:, 54])
        a, n = axes
        o_fan = smooth1d(lms[:, 27:36].mean(1) - pl0[:, 27:36].mean(1), c['fan_sigma']) + pl0[:, 27:36].mean(1)
        o_mesh = pl0[:, 27:36].mean(1)
        lms_s = smooth1d(lms - o_fan[:, None], c['fan_sigma']) + o_fan[:, None]
        up_f, lo_f, ap_fan_s = fan_profile(lms_s, s, axes, o_fan)
        _, _, ap_fan = fan_profile(lms, s, axes, o_fan)
        g_fan = (lo_f - up_f).clamp(min=0)
        g_tgt, gate = dead_zone(g_fan, ap_fan_s, c['closed_px'], c['seal_px'])
        mid_f = 0.5 * (lo_f + up_f)
        tgt_lmk = lms_s[:, 48:68] - o_fan[:, None]
        top0, bot0, xk0, ok0 = mesh_gap(pv0, pl0, geo, s, axes)
        wo_m0 = (pl0[:, 54] - pl0[:, 48]).norm(dim=-1); wi_m0 = (pl0[:, 64] - pl0[:, 60]).norm(dim=-1)
        wo_f = (lms_s[:, 54] - lms_s[:, 48]).norm(dim=-1); wi_f = (lms_s[:, 64] - lms_s[:, 60]).norm(dim=-1)
        wo_t = wo_m0[valid].median() + wo_f - wo_f[valid].median(); wi_t = wi_m0[valid].median() + wi_f - wi_f[valid].median()
        off0 = ((pl0[:, 48:68] - o_mesh[:, None]) - tgt_lmk)[valid].median(0).values
        mid0 = 0.5 * (top0 + bot0) - (o_mesh * n).sum(-1, keepdim=True)
        cmid0 = (mid0 - mid_f)[valid].median()
        chin_f = ((lms_s[:, 7:10] - o_fan[:, None]) * n[:, None]).sum(-1)
        chin_m0 = ((pl0[:, 7:10] - o_mesh[:, None]) * n[:, None]).sum(-1)
        chin_off = (chin_m0 - chin_f)[valid].median(0).values

    z = torch.zeros(N, 50, device=device, requires_grad=True)
    Lo = None if lower_proj is None else T(lower_proj)
    corr = (lambda: z) if Lo is None else (lambda: z @ Lo.T)
    jopen0 = torch.rad2deg(jaw0[:, 0])
    dj = torch.zeros(N, 3, device=device, requires_grad=True)
    off = off0.clone().requires_grad_(True); cmid = cmid0.clone().requires_grad_(True)
    opt = torch.optim.Adam([{'params': [z, dj], 'lr': c['lr']}, {'params': [off, cmid], 'lr': c['lr']}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['iters'], eta_min=c['lr'] * 0.05)
    jscale = torch.tensor([1.0, 0.3, 0.3], device=device)
    hist = []
    for it in range(c['iters']):
        d = corr(); expr = exp0 + d
        jaw = jaw0 + torch.deg2rad(dj * jscale)
        Vv, Ll = flame_forward(flame, P, expr, jaw)
        pv = project(Vv, w2c, intr); pl = project(Ll[:, :68], w2c, intr)
        top, bot, _, ok = mesh_gap(pv, pl, geo, s, axes)
        okf = ok.float(); nk = okf.sum(1).clamp(min=1)
        e = bot - top - g_tgt
        l_gap = mean_v(((torch.relu(e) ** 2 + gate * torch.relu(-e) ** 2) * okf).sum(1) / nk)
        mid = 0.5 * (top + bot) - (o_mesh * n).sum(-1, keepdim=True)
        l_mid = mean_v((((mid - cmid - mid_f) ** 2) * okf).sum(1) / nk)
        l_lmk = mean_v(((pl[:, 48:68] - o_mesh[:, None] - off[None] - tgt_lmk) ** 2).sum(-1).mean(1))
        l_wid = mean_v(((pl[:, 54] - pl[:, 48]).norm(dim=-1) - wo_t) ** 2 + ((pl[:, 64] - pl[:, 60]).norm(dim=-1) - wi_t) ** 2)
        chin = ((pl[:, 7:10] - o_mesh[:, None]) * n[:, None]).sum(-1)
        l_chin = mean_v(((chin - chin_off - chin_f) ** 2).mean(1))
        l_pe = (d ** 2).sum(1).mean(); l_pj = (dj ** 2).sum(1).mean()
        l_jn = (torch.relu(-(jopen0 + dj[:, 0])) ** 2).mean()
        l_up = ((d @ geo.A_up) * d).sum(1).mean()
        dz = d[1:] - d[:-1]
        l_t = ((dz @ geo.A_lip) * dz).sum(1).mean() + ((dj[1:] - dj[:-1]) ** 2).sum(1).mean()
        loss = (c['w_gap'] * l_gap + c['w_mid'] * l_mid + c['w_lmk'] * l_lmk + c['w_width'] * l_wid + c['w_chin'] * l_chin
                + c['w_prior_exp'] * l_pe + c['w_prior_jaw'] * l_pj + c['w_jaw_neg'] * l_jn + c['w_upper'] * l_up + c['w_temp'] * l_t)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        hist.append(loss.item())
        if verbose and (it % 100 == 0 or it == c['iters'] - 1):
            print(f'   it {it:4d} loss {loss.item():8.3f} | gap {l_gap.item():6.3f} mid {l_mid.item():6.3f} lmk {l_lmk.item():6.3f} '
                  f'width {l_wid.item():6.3f} chin {l_chin.item():6.3f} | prior exp {l_pe.item():6.3f} jaw {l_pj.item():6.3f} '
                  f'upper {l_up.item():6.4f} temp {l_t.item():6.4f}', flush=True)

    with torch.no_grad():
        expr = exp0 + corr(); jaw = jaw0 + torch.deg2rad(dj * jscale)
        expr[~valid] = exp0[~valid]; jaw[~valid] = jaw0[~valid]
        Vv, Ll = flame_forward(flame, P, expr, jaw)
        pv = project(Vv, w2c, intr); pl = project(Ll[:, :68], w2c, intr)
        top, bot, xk, ok = mesh_gap(pv, pl, geo, s, axes)
        np_ = lambda t: t.detach().cpu().numpy()
        diag = dict(ids=ids, s=np_(s), ap_fan=np_(ap_fan), ap_fan_smooth=np_(ap_fan_s),
                    ap_mesh_tracked=np_(aperture_from_gap(bot0 - top0, xk0, wo_m0)),
                    ap_mesh_refit=np_(aperture_from_gap(bot - top, xk, (pl[:, 54] - pl[:, 48]).norm(dim=-1))),
                    gap_fan=np_(g_fan), gap_target=np_(g_tgt), gap_tracked=np_(bot0 - top0), gap_refit=np_(bot - top),
                    lmk_tracked=np_(pl0), lmk_refit=np_(pl), lmk_offset=np_(off), mid_offset=np_(cmid),
                    upper_face_move_mm=np_((Vv[:, geo.upper] - V0[:, geo.upper]).norm(dim=-1).mean(1) * 1e3),
                    lip_move_mm=np_((Vv[:, geo.lips] - V0[:, geo.lips]).norm(dim=-1).mean(1) * 1e3),
                    d_exp=np_(expr - exp0), dj_deg=np_(torch.rad2deg(jaw - jaw0)),
                    loss=np.array(hist, np.float32))
        e100 = P['expression'].clone(); e100[:, :50] = expr
    return np_(e100), np_(jaw), diag


EYE_DEFAULTS = dict(eye_cols=7, eye_iters=300, eye_lr=0.03, w_eye_prior=0.01, w_eye_temp=0.02, au_lo=0.15, au_hi=0.70,
                    ear_open=0.88, ear_closed=0.68, ear_open2=0.72, ear_closed2=0.52, eye_seal_px=0.4, eye_level_win=25, eye_level_min=0.75, eye_level_max=1.3)
AU_BLINK = 'AU45'


def load_blink_au(scene_dir, n):
    """Blink action unit per frame ([n], py-feat scale 0..1) from <scene>/au_features.csv, or None."""
    p = os.path.join(scene_dir, 'au_features.csv')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        head = [h.strip() for h in f.readline().split(',')]
    col = next((i for i, h in enumerate(head) if h in (AU_BLINK + '_r', AU_BLINK)), None)
    if col is None:
        return None
    a = np.nan_to_num(np.loadtxt(p, delimiter=',', skiprows=1, usecols=[col], dtype=np.float32, ndmin=1))
    if a.max() > 1.5:
        a = np.clip(a / 5.0, 0.0, 1.0)
    if len(a) < n:
        a = np.concatenate([a, np.repeat(a[-1:], n - len(a))])
    return a[:n]


def fan_ear(lms):
    """Eye aspect ratio of both eyes from the 68-point landmarks."""
    out = []
    for o in (36, 42):
        p = lms[:, o:o + 6]
        w = np.linalg.norm(p[:, 3] - p[:, 0], axis=-1)
        h = 0.5 * (np.linalg.norm(p[:, 1] - p[:, 5], axis=-1) + np.linalg.norm(p[:, 2] - p[:, 4], axis=-1))
        out.append(h / np.maximum(w, 1.0))
    return np.stack(out, 1)


def running_level(x, keep, win):
    """Running median of x over the frames with keep=True (window `win` frames, centred), then lightly smoothed."""
    n = len(x); g = np.median(x[keep]) if keep.any() else np.median(x); out = np.full(n, g, np.float64); r = win // 2
    for i in range(n):
        a, b = max(0, i - r), min(n, i + r + 1)
        k = keep[a:b]
        if k.sum() >= 5:
            out[i] = np.median(x[a:b][k])
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(out, 3.0, mode='nearest'), g


def refine_eyes(scene, flame, device, expr, jaw, signal='auto', level='fan', verbose=True, **kw):
    """Eyelid stage.  expr [N,100], jaw [N,3]."""
    c = dict(EYE_DEFAULTS); c.update(kw)
    from utils.flame_regions import upper_face_projector
    from utils.flame_eyelids import LidGeometry, blink_direction, lid_gap, eye_axes
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    ids = scene['ids']; N = len(ids)
    P = stack_params(scene, device)
    w2c, intr = T(scene['w2c']), T(scene['intr'])
    geo = LidGeometry(flame)
    u, wdual = blink_direction(flame, geo, upper_face_projector(50)[0])
    e0, jw = T(expr[:, :50]), T(jaw)
    valid = np.ones(N, bool); valid[-1] = False
    ear = fan_ear(scene['lms']).mean(1)
    au = load_blink_au(scene.get('dir', ''), N) if signal in ('auto', 'au', 'both') else None
    if signal in ('au', 'both') and au is None:
        raise FileNotFoundError(f'--eye_signal {signal}: no blink action unit in au_features.csv')
    if au is not None:
        clo = np.clip((au - c['au_lo']) / (c['au_hi'] - c['au_lo']), 0.0, 1.0); used = 'au'
        lvl, ear_med = running_level(ear, valid & (clo < 0.05), c['eye_level_win'])
        if signal == 'both':
            r = ear / lvl
            clo = np.maximum(clo, np.clip((c['ear_open2'] - r) / (c['ear_open2'] - c['ear_closed2']), 0.0, 1.0)); used = 'au+fan'
            lvl, ear_med = running_level(ear, valid & (clo < 0.05), c['eye_level_win'])
    else:
        from scipy.ndimage import median_filter
        lvl, ear_med = running_level(ear, valid, c['eye_level_win'])
        r = median_filter(ear / lvl, size=3, mode='nearest')
        clo = np.clip((c['ear_open'] - r) / (c['ear_open'] - c['ear_closed']), 0.0, 1.0); used = 'fan'
        lvl, ear_med = running_level(ear, valid & (clo < 0.05), c['eye_level_win'])
    rel = np.clip(lvl / ear_med, c['eye_level_min'], c['eye_level_max']) if level == 'fan' else np.ones(N)
    opened = valid & (clo < 0.05)
    s = torch.linspace(0.25, 0.75, c['eye_cols'], device=device)
    vm = T(valid.astype(np.float32)); nv = vm.sum()

    def gaps(e50, axes=None):
        V, L = flame_forward(flame, P, e50, jw)
        pv = project(V, w2c, intr); pl = project(L[:, :68], w2c, intr)
        out = [lid_gap(pv, pl, sh, s, None if axes is None else axes[k]) for k, sh in enumerate(geo.sheets)]
        return out, pl, V

    with torch.no_grad():
        out0, pl0, V0 = gaps(e0)
        axes = [eye_axes(pl0[:, sh['a']], pl0[:, sh['b']]) for sh in geo.sheets]
        g0 = [o[0] for o in out0]; ok0 = [o[1].float() for o in out0]; wid = [o[2].clamp(min=1.0) for o in out0]
        om = T(opened.astype(np.float32))[:, None]
        tgt, gate, prof = [], T((clo < 0.98).astype(np.float32))[:, None], []
        for e in range(2):
            gn = g0[e] / wid[e][:, None]
            p = torch.stack([gn[opened][:, k].median() for k in range(gn.shape[1])]) if opened.sum() >= 5 else gn.median(0).values
            prof.append(p)
            t_open = wid[e][:, None] * p[None] * T(rel)[:, None] * (1.0 - T(clo))[:, None]
            tgt.append(gate * t_open - (1 - gate) * c['eye_seal_px'])
        travel = float(np.mean([(wid[e][:, None] * prof[e][None]).mean().item() for e in range(2)]))

    cc = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([cc], lr=c['eye_lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['eye_iters'], eta_min=c['eye_lr'] * 0.05)
    for it in range(c['eye_iters']):
        out, _, _ = gaps(e0 + cc[:, None] * u[None], axes)
        l_gap = 0.0
        for e in range(2):
            err = out[e][0] - tgt[e]; okf = out[e][1].float()
            l_gap = l_gap + ((((torch.relu(err) ** 2 + gate * torch.relu(-err) ** 2) * okf).sum(1) / okf.sum(1).clamp(min=1)) * vm).sum() / nv
        l_p = ((cc * travel) ** 2 * vm).sum() / nv
        l_t = (((cc[1:] - cc[:-1]) * travel) ** 2).mean()
        loss = l_gap + c['w_eye_prior'] * l_p + c['w_eye_temp'] * l_t
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if verbose and (it % 100 == 0 or it == c['eye_iters'] - 1):
            print(f'   eye it {it:4d} loss {loss.item():8.3f} | gap {l_gap.item():6.3f} prior {l_p.item():6.3f} temp {l_t.item():6.4f}', flush=True)

    with torch.no_grad():
        cv = cc.detach() * vm
        e1 = e0 + cv[:, None] * u[None]
        out1, pl1, V1 = gaps(e1, axes)
        np_ = lambda t: t.detach().cpu().numpy()
        ap = lambda o: np_(torch.stack([(o[e][0] * o[e][1].float()).sum(1) / o[e][1].float().sum(1).clamp(min=1) / wid[e] for e in range(2)], 1))
        mk = flame.mask.v; vt = flame.v_template
        lower = torch.unique(torch.cat([mk.lips, mk.face[vt[mk.face, 1] < vt[mk.nose, 1].min() - 0.004]]))
        lips = torch.unique(torch.cat([mk.lips, mk.lips_tight, mk.lip_inside]))
        mv = lambda ix: np_((V1[:, ix] - V0[:, ix]).norm(dim=-1).mean(1) * 1e3)
        lg = LipGeometry(flame); sl = torch.linspace(0.1, 0.9, 11, device=device)
        ax_m = mouth_axes(pl0[:, 48], pl0[:, 54])
        def lipgap(e50):
            V, L = flame_forward(flame, P, e50, jw); pv = project(V, w2c, intr); pl = project(L[:, :68], w2c, intr)
            top, bot, xk, ok = mesh_gap(pv, pl, lg, sl, ax_m)
            return aperture_from_gap(bot - top, xk, (pl[:, 54] - pl[:, 48]).norm(dim=-1))
        diag = dict(eye_signal=used, eye_closure_signal=clo.astype(np.float32), eye_level=rel.astype(np.float32), eye_c=np_(cv),
                    ear_fan=fan_ear(scene['lms']).astype(np.float32), blink_au=(au if au is not None else np.zeros(0, np.float32)),
                    lid_ap_tracked=ap(out0), lid_ap_refit=ap(out1),
                    lid_ap_target=np_(torch.stack([tgt[e].mean(1) / wid[e] for e in range(2)], 1)),
                    lid_gap_refit_max_px=np_(torch.stack([torch.where(out1[e][1], out1[e][0], torch.full_like(out1[e][0], -1e3)).amax(1) for e in range(2)], 1)),
                    eye_width_px=np_(torch.stack(wid, 1)), blink_dir=np_(u), blink_dual=np_(wdual),
                    eye_lower_face_move_mm=mv(lower), eye_lip_move_mm=mv(lips), eye_lid_move_mm=mv(geo.lids),
                    eye_lip_gap_change_px=np_(lipgap(e1) - lipgap(e0)), eye_loss=np.float32(loss.item()))
        ex = np.array(expr, np.float32, copy=True); ex[:, :50] = np_(e1)
    return ex, diag


def lower_projector():
    """I - P_upper with P_upper = utils.flame_regions.upper_face_projector(50)."""
    from utils.flame_regions import upper_face_projector
    return np.eye(50, dtype=np.float32) - upper_face_projector(50)[0]


def convention_tag(lips, eyes):
    return f'refit{REFIT_VERSION}:' + '+'.join(n for n, on in (('lips', lips), ('eyelids', eyes)) if on)


def write_params(scene, expr, jaw, out_npz, meta):
    """Per-frame dicts as in the tracked file with expression / jaw_pose / pose replaced, plus the `_meta` entry."""
    out = {}
    for k, i in enumerate(scene['ids']):
        f = dict(scene['frames'][i])
        f['expression'] = expr[k:k + 1].astype(np.float32)
        f['jaw_pose'] = jaw[k:k + 1].astype(np.float32)
        if 'pose' in f:
            f['pose'] = np.concatenate([np.asarray(f['rotation'], np.float32), f['jaw_pose']], -1)
        out[str(i)] = f
    out[META_KEY] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    tmp = f'{out_npz}.{os.getpid()}.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, out_npz)


def _same_file(a, b):
    import filecmp
    return os.path.realpath(a) == os.path.realpath(b) or filecmp.cmp(a, b, shallow=False)


def _keep(src, dst):
    """Make `dst` hold what `src` holds now without copying the data when possible."""
    if os.path.islink(src):
        os.symlink(os.path.realpath(src), dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_scene(scene_dir, out_dir):
    """Scene directory copy."""
    scene_dir = os.path.abspath(scene_dir)
    if os.path.realpath(out_dir) == os.path.realpath(scene_dir):
        raise SystemExit('[refine_landmarks] --out_scene is the scene itself: omit it to re-fit in place')
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(scene_dir)):
        dst = os.path.join(out_dir, name)
        if name in (PARAMS, TRACKED) or os.path.lexists(dst):
            continue
        os.symlink(os.path.join(scene_dir, name), dst)


def resolve(scene_dir, out_scene, force):
    """Decide what to do for one scene."""
    scene_dir = os.path.abspath(scene_dir)
    src_params, src_tracked = os.path.join(scene_dir, PARAMS), os.path.join(scene_dir, TRACKED)
    name = os.path.basename(scene_dir)

    def tracked_of(d):
        """The tracked parameters of scene directory d, or an error."""
        p, t = os.path.join(d, PARAMS), os.path.join(d, TRACKED)
        tagged = os.path.exists(p) and drive_convention(p) != TRACKED_CONVENTION
        if os.path.exists(t):
            if drive_convention(t) != TRACKED_CONVENTION:
                raise SystemExit(f'[refine_landmarks] {t} is itself a re-fit: restore the tracker import of this scene')
            if os.path.exists(p) and not tagged and not _same_file(p, t):
                raise SystemExit(f'[refine_landmarks] {d}: {PARAMS} is untagged but differs from {TRACKED} (was the '
                                 f'scene re-imported from the tracker?). Delete {TRACKED} to re-fit the new tracking.')
            return t, tagged
        if tagged:
            raise SystemExit(f'[refine_landmarks] {d}: {PARAMS} is a re-fit but {TRACKED} is missing: '
                             f're-import the scene from the tracker')
        if not os.path.exists(p):
            raise SystemExit(f'[refine_landmarks] {d}: no {PARAMS}')
        return p, False

    source, src_done = tracked_of(scene_dir)
    if out_scene:
        out_scene = os.path.abspath(out_scene)
        out_params = os.path.join(out_scene, PARAMS)
        done = os.path.exists(out_params) and drive_convention(out_params) != TRACKED_CONVENTION
        target = out_scene
    else:
        done, target = src_done, scene_dir
    if done and not force:
        print(f'[refine_landmarks] {name}: {os.path.join(target, PARAMS)} is already re-fitted '
              f'({drive_convention(os.path.join(target, PARAMS))}); nothing to do (--force repeats the fit from the '
              f'tracked parameters)')
        return target, None
    return target, source


def restore(scene_dir):
    """Put the tracked parameters back as flame_params.npz (the backup is kept)."""
    scene_dir = os.path.abspath(scene_dir)
    p, t = os.path.join(scene_dir, PARAMS), os.path.join(scene_dir, TRACKED)
    if not os.path.exists(t):
        raise SystemExit(f'[refine_landmarks] {scene_dir}: no {TRACKED} to restore')
    tmp = f'{p}.{os.getpid()}.tmp'
    _keep(t, tmp)
    os.replace(tmp, p)
    print(f'[refine_landmarks] {os.path.basename(scene_dir)}: {PARAMS} restored from {TRACKED} '
          f'(avatars adapted on the re-fit must be re-adapted)')


def process(scene_dir, a):
    target, source = resolve(scene_dir, a.out_scene, a.force)
    if source is None:
        return
    t0 = time.time()
    scene = load_scene(scene_dir, source)
    flame = build_flame(source, a.device)
    t1 = time.time()
    if a.lips == 'on':
        expr, jaw, diag = refine(scene, flame, a.device, verbose=a.verbose,
                                 lower_proj=None if a.all_expression else lower_projector(),
                                 **{k: getattr(a, k) for k in DEFAULTS})
    else:
        P0 = stack_params(scene, 'cpu'); expr, jaw = P0['expression'].numpy().copy(), P0['jaw_pose'].numpy().copy()
        diag = dict(ids=scene['ids'])
    t2 = time.time()
    meta = dict(convention=convention_tag(a.lips == 'on', a.eyes == 'on'), tool='tools/refine_landmarks.py',
                version=REFIT_VERSION, lips=a.lips == 'on', eyelids=a.eyes == 'on', lower_only=not a.all_expression,
                frames=int(len(scene['ids'])),
                settings={k: getattr(a, k) for k in list(DEFAULTS) + list(EYE_DEFAULTS) + ['eye_signal', 'eye_level']})
    name = os.path.basename(os.path.abspath(scene_dir))
    if a.eyes == 'on':
        expr, ediag = refine_eyes(scene, flame, a.device, expr, jaw, signal=a.eye_signal, level=a.eye_level,
                                  verbose=a.verbose, **{k: getattr(a, k) for k in EYE_DEFAULTS})
        diag.update(ediag)
        meta['eye_signal'] = ediag['eye_signal']
        cl = ediag['eye_closure_signal'] >= 0.98; op = ediag['eye_closure_signal'] < 0.05
        at = lambda x, m: float(x[m].mean()) if m.any() else float('nan')
        deep = ediag['eye_c'] > 0.5
        print(f'[refine_landmarks] {name} eyelids ({ediag["eye_signal"]} signal): {int(cl.sum())} sealed-target frames, '
              f'lid aperture / eye width open {at(ediag["lid_ap_tracked"], op):.3f} -> {at(ediag["lid_ap_refit"], op):.3f}, '
              f'at sealed-target frames {at(ediag["lid_ap_tracked"], cl):.3f} -> {at(ediag["lid_ap_refit"], cl):.3f} '
              f'(largest remaining column gap {ediag["lid_gap_refit_max_px"][cl].max() if cl.any() else float("nan"):.2f}px) '
              f'| closure c max {ediag["eye_c"].max():.2f} | at c>0.5: lower face '
              f'{at(ediag["eye_lower_face_move_mm"], deep) if deep.any() else 0:.3f}mm, lips '
              f'{at(ediag["eye_lip_move_mm"], deep) if deep.any() else 0:.3f}mm, lip aperture change max '
              f'{np.abs(ediag["eye_lip_gap_change_px"]).max():.3f}px | {time.time() - t2:.1f}s', flush=True)
    t3 = time.time()
    if a.out_scene:
        link_scene(scene_dir, target)
    tracked = os.path.join(target, TRACKED)
    if not os.path.lexists(tracked):
        _keep(source, tracked)
    write_params(scene, expr, jaw, os.path.join(target, PARAMS), meta)
    if a.drive_npz:
        np.savez(a.drive_npz, exp=expr[:, :50].astype(np.float32), jaw=jaw.astype(np.float32))
    if a.diag:
        np.savez(a.diag, **diag)
    msg = f'[refine_landmarks] {name}: {len(scene["ids"])} frames -> {os.path.join(target, PARAMS)} ({meta["convention"]})'
    if a.lips == 'on':
        msg += (f' | lips: landmark mean gap {diag["ap_fan"].mean():.2f}px, mesh tracked {diag["ap_mesh_tracked"].mean():.2f}px, '
                f're-fit {diag["ap_mesh_refit"].mean():.2f}px; correction |d_exp| rms '
                f'{np.sqrt((diag["d_exp"] ** 2).sum(1).mean()):.2f}, jaw rms {np.sqrt((diag["dj_deg"] ** 2).sum(1).mean()):.2f}deg, '
                f'lips {diag["lip_move_mm"].mean():.2f}mm, upper face {diag["upper_face_move_mm"].mean():.3f}mm')
    print(msg + f' | load {t1 - t0:.1f}s fit {t3 - t1:.1f}s write {time.time() - t3:.1f}s on {a.device}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('scenes', nargs='+', help='processed scene directories (re-fitted in place)')
    ap.add_argument('--out_scene', default=None, help='one scene only: write a linked copy of the scene here instead of '
                    'changing the scene (entries are symlinks, the FLAME parameter files are its own)')
    ap.add_argument('--force', action='store_true', help='repeat the fit of an already re-fitted scene (always from the '
                    'tracked parameters)')
    ap.add_argument('--restore', action='store_true', help='undo: make flame_params.npz the tracked parameters again')
    ap.add_argument('--lips', choices=['on', 'off'], default='on', help='lip stage')
    ap.add_argument('--eyes', choices=['on', 'off'], default='on', help='eyelid stage')
    ap.add_argument('--all_expression', action='store_true', help='lip stage: let the correction use the whole expression '
                    'space instead of the lower-face sub-space of utils/flame_regions.py')
    ap.add_argument('--eye_signal', choices=['auto', 'au', 'fan', 'both'], default='auto', help='blink signal: the blink '
                    'action unit of au_features.csv (auto: when present), the eye aspect ratio of the image landmarks '
                    '(noisy: closed eyes are often not reported), or both (the action unit plus deep drops of the ratio)')
    ap.add_argument('--eye_level', choices=['fan', 'const'], default='fan', help='slow opening level: running median of '
                    'the landmarks\' eye aspect ratio, or constant')
    ap.add_argument('--diag', default=None, help='one scene only: write per-frame diagnostics of the fit to this npz')
    ap.add_argument('--drive_npz', default=None, help='one scene only: also write exp [N,50] / jaw [N,3] of the re-fit '
                    '(tools/render_composited.py --drive_file)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=0, help='CPU threads (0 = torch default)')
    ap.add_argument('--verbose', action='store_true', help='print the loss terms during the fit')
    g = ap.add_argument_group('fit settings', 'weights and thresholds of the two stages (see refine / refine_eyes); '
                              'the defaults are the validated recipe and are recorded in the tag')
    for k, v in list(DEFAULTS.items()) + list(EYE_DEFAULTS.items()):
        g.add_argument(f'--{k}', type=type(v), default=v)
    a = ap.parse_args()
    if len(a.scenes) > 1 and (a.out_scene or a.diag or a.drive_npz):
        ap.error('--out_scene / --diag / --drive_npz take exactly one scene')
    if a.lips == 'off' and a.eyes == 'off' and not a.restore:
        ap.error('nothing to do: --lips off --eyes off')
    if a.threads:
        torch.set_num_threads(a.threads)
    for scene_dir in a.scenes:
        scene_dir = scene_dir.rstrip('/')
        if a.restore:
            restore(scene_dir)
        else:
            process(scene_dir, a)


if __name__ == '__main__':
    main()
