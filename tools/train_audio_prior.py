#!/usr/bin/env python3
"""Train the audio -> FLAME motion prior on (audio, FLAME track) pairs from tools/prepare_audio_prior_data.py.

    python tools/train_audio_prior.py --init output/pretrain/chkpnt_ema_face_latest.pth --data data_prior/d5 data_prior/hdtf_tfhp --out output/prior/stage1.pth --steps 40000
    python tools/train_audio_prior.py --init output/prior/stage1.pth --data data_prior/hdtf_tfhp --out output/prior/chkpnt_ema_face_latest.pth --steps 30000 --w_vertex 1 --align --select aperture

Only the audio -> motion path is trained (no images); the output (state_dict, None, step) is used as --pretrain_path.
--align first measures each clip's audio-video offset with the --init prior and removes it (offsets saved next to --out).
"""
import argparse
import copy
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene.motion_net import MotionNetwork
from tools.refine_landmarks import LipGeometry, aperture_from_gap, build_flame, mesh_gap
from utils.audio_features import EXTRACTOR_KEY
from utils.audio_lag import _lagcorr
from utils.flame_regions import expression_region_bases

EXTRACTOR, DEV = 'whisper', 'cuda'
BATCH, SEQ, LR, W_JAW, W_VEL, EMA, EVAL_EVERY, AU_CONST = 64, 8, 1e-4, 100.0, 1.0, 0.999, 2000, 0.35
LAGS = list(range(-8, 9))


class PriorData:
    """Clips of <dir>/<split>[_<i>].npz; frame t of clip c reads the 16 audio tokens around 2 t + shift[c]."""

    def __init__(self, dirs, split):
        files = [f for d in dirs for f in sorted(glob.glob(os.path.join(d, f'{split}.npz')) + glob.glob(os.path.join(d, f'{split}_*.npz')))]
        hid, exp, jaw, h_off, f_off, self.names = [], [], [], [0], [0], []
        for f in files:
            d = np.load(f)
            hid.append(d['hidden']); exp.append(d['exp']); jaw.append(d['jaw']); self.names += [str(x) for x in d['names']]
            h_off += list(h_off[-1] + d['h_off'][1:]); f_off += list(f_off[-1] + d['f_off'][1:])
        self.hidden = torch.from_numpy(np.concatenate(hid)).to(DEV)
        self.exp = torch.from_numpy(np.concatenate(exp)).float().to(DEV)
        self.jaw = torch.from_numpy(np.concatenate(jaw)).float().to(DEV)
        self.h_off, self.f_off, self.n = torch.tensor(h_off), torch.tensor(f_off), len(f_off) - 1
        self.shift = torch.zeros(self.n, dtype=torch.long)
        print(f'[prior] {split}: {len(files)} file(s), {self.n} clips, {len(self.exp) / 90000:.2f} h', flush=True)

    def windows(self, clips, frames):
        T = (self.h_off[clips + 1] - self.h_off[clips])[:, None, None]
        idx = torch.minimum((2 * frames[..., None] + torch.arange(16) - 8 + self.shift[clips][:, None, None]).clamp(min=0), T - 1)
        fi = (self.f_off[clips][:, None] + frames).to(DEV)
        return self.hidden[(self.h_off[clips][:, None, None] + idx).to(DEV)].float(), self.exp[fi], self.jaw[fi]

    def batch(self, gen):
        lens = self.f_off[1:] - self.f_off[:-1]
        ok = torch.nonzero(lens >= SEQ).squeeze(1)
        clips = ok[torch.randint(len(ok), (BATCH,), generator=gen)]
        starts = (torch.rand(BATCH, generator=gen) * (lens[clips] - SEQ + 1).float()).long()
        return self.windows(clips, starts[:, None] + torch.arange(SEQ)[None])

    def clip(self, c):
        win, e, j = self.windows(torch.tensor([c]), torch.arange(int(self.f_off[c + 1] - self.f_off[c]))[None])
        return win[0], e[0], j[0]


class Lips:
    """Inner-lip aperture (mm) of FLAME expression/jaw on the mean shape, canonical pose, frontal view."""

    def __init__(self):
        self.flame = build_flame(None, DEV)
        self.geo, self.s = LipGeometry(self.flame), torch.linspace(0.1, 0.9, 11, device=DEV)

    @torch.no_grad()
    def aperture(self, exp, jaw):
        out = []
        for e, j in zip(exp.split(2048), jaw.split(2048)):
            z = torch.zeros(len(e), 3, device=DEV)
            V, L = self.flame(shape=torch.zeros(len(e), 300, device=DEV), expr=e, rotation=z, neck=z, jaw=j,
                              eyes=torch.zeros(len(e), 6, device=DEV), translation=z, return_landmarks=True)
            pv, pl = [torch.stack([X[..., 0], -X[..., 1]], -1) * 1e3 for X in (V, L[:, :68])]
            top, bot, xk, _ = mesh_gap(pv, pl, self.geo, self.s)
            out.append(aperture_from_gap(bot - top, xk, (pl[:, 54] - pl[:, 48]).norm(dim=-1)))
        return torch.cat(out).cpu().numpy()


def load_prior(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = MotionNetwork(args=SimpleNamespace(audio_extractor=EXTRACTOR)).to(DEV)
    net.load_state_dict({k: v for k, v in ck[0].items() if k != EXTRACTOR_KEY})
    net.init_novel_code()
    return net


def predict(net, win):
    B, L = win.shape[:2]
    out = net(win.reshape(B * L, 16, 768), c=net.identity_code(0), au=torch.full((B * L, 6), AU_CONST, device=DEV),
              identity=torch.zeros(512, device=DEV))
    return out['flame_exp'].reshape(B, L, -1), out['flame_jaw'].reshape(B, L, -1)


@torch.no_grad()
def estimate_offsets(net, data, lips):
    """Per-clip shift (audio tokens) maximising the correlation of predicted and tracked lip aperture; 0 unless the peak is clear."""
    net.eval()
    shift = []
    for c in range(data.n):
        win, e, j = data.clip(c)
        if len(e) < 60:
            shift.append(0)
            continue
        pe, pj = predict(net, win[None])
        ag, ap = lips.aperture(e, j), lips.aperture(pe[0], pj[0])
        cc = np.nan_to_num(np.array([_lagcorr(ag, ap, L) for L in LAGS]), nan=-1.0)
        k = int(cc.argmax())
        lag, inner = float(LAGS[k]), 0 < k < len(LAGS) - 1
        den = cc[k - 1] - 2 * cc[k] + cc[k + 1] if inner else 0.0
        if den < -1e-6:
            lag += float(np.clip(0.5 * (cc[k - 1] - cc[k + 1]) / den, -0.5, 0.5))
        clear = inner and cc[k] > 0.3 and cc[k] - cc[LAGS.index(0)] > 0.2 / np.sqrt(len(ag) / 25.0)
        shift.append(int(round(2 * lag)) if clear else 0)
    data.shift = torch.tensor(shift, dtype=torch.long)
    net.train()
    return dict(zip(data.names, shift))


def corr(a, b):
    a, b = a - a.mean(0), b - b.mean(0)
    return (a * b).sum(0) / (a.norm(dim=0) * b.norm(dim=0) + 1e-9)


@torch.no_grad()
def evaluate(net, data, lips, bl):
    """Per clip: aperture / jaw-opening / lower-face correlation, open-vs-closed AUC, amplitude ratio, aperture on closed frames (mm)."""
    net.eval()
    rows = []
    for c in range(data.n):
        win, e, j = data.clip(c)
        pe, pj = predict(net, win[None])
        pe, pj = pe[0], pj[0]
        ag, ap = lips.aperture(e, j), lips.aperture(pe, pj)
        lo, hi = ag <= np.percentile(ag, 15), ag >= np.percentile(ag, 50)
        with np.errstate(invalid='ignore', divide='ignore'):
            ap_r = np.corrcoef(ag, ap)[0, 1]
        rows.append(dict(ap_r=ap_r, jaw_r=float(corr(pj[:, :1], j[:, :1])[0]), lower_r=float(corr(pe @ bl.T, e @ bl.T).nanmean()),
                         auc=(ap[hi][:, None] > ap[lo][None]).mean(), amp=ap.std() / max(ag.std(), 1e-9), closed_mm=ap[lo].mean()))
    net.train()
    return {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0]}


SCORES = {'corr': lambda m: m['jaw_r'].mean() + m['lower_r'].mean(),
          'aperture': lambda m: np.nanmedian(m['ap_r']) + np.nanmedian(m['auc']) + 0.5 * (np.nanmedian(m['jaw_r']) + np.nanmedian(m['lower_r']))}


def summary(m):
    return json.dumps({k: round(float(np.nanmedian(v)), 3) for k, v in m.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', required=True, help='motion prior to start from (3-tuple EMA checkpoint)')
    ap.add_argument('--data', nargs='+', required=True, help='tools/prepare_audio_prior_data.py output dirs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--w_vertex', type=float, default=0.0, help='weight of the lower-face vertex-displacement loss')
    ap.add_argument('--align', action='store_true', help='remove per-clip audio-video offsets measured with the --init prior')
    ap.add_argument('--select', choices=sorted(SCORES), default='corr',
                    help='validation score of the kept checkpoint: mean jaw + lower-face correlation, or lip-aperture medians')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    train, val = PriorData(args.data, 'train'), PriorData(args.data, 'val')
    net, lips = load_prior(args.init), Lips()
    bl = torch.from_numpy(expression_region_bases(50)[0]).to(DEV)
    with torch.no_grad():
        sub = train.exp[torch.randperm(len(train.exp), device=DEV)[:20000]]
        v_var = float(((sub - sub.mean(0)) @ bl.T).pow(2).mean())
    if args.align:
        offsets = {s: estimate_offsets(net, d, lips) for s, d in (('train', train), ('val', val))}
        json.dump(offsets, open(os.path.splitext(args.out)[0] + '.offsets.json', 'w'))
        print('[prior] clips with an offset:', {s: round(float(np.mean([v != 0 for v in o.values()])), 2) for s, o in offsets.items()}, flush=True)
    mods = [net.audio_proj, net.audio_transformer, net.backbone_net, net.flame_exp_head, net.flame_jaw_head,
            net.residual_net, net.residual_motion_head, net.gate_net, net.audio_adain]
    net.requires_grad_(False)
    params = [p for m in mods for p in m.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.99), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 1000) * 0.1 ** (s / args.steps))
    ema = copy.deepcopy(net).eval()
    print('[prior] init val (medians)', summary(evaluate(net, val, lips, bl)), flush=True)
    d1 = lambda x: x[:, 1:] - x[:, :-1]
    mse = lambda a, b: ((a - b) ** 2).mean()
    best = -1e9
    for step in range(1, args.steps + 1):
        win, ge, gj = train.batch(gen)
        pe, pj = predict(net, win)
        le = mse(pe, ge) + W_VEL * mse(d1(pe), d1(ge))
        lj = mse(pj, gj) + W_VEL * mse(d1(pj), d1(gj))
        lv = ((((pe - ge) @ bl.T) ** 2).mean() + W_VEL * (((d1(pe) - d1(ge)) @ bl.T) ** 2).mean()) / v_var
        loss = le + W_JAW * lj + args.w_vertex * lv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe_, pn_ in zip(ema.parameters(), net.parameters()):
                pe_.mul_(EMA).add_(pn_.detach(), alpha=1 - EMA)
        if step % 500 == 0:
            print(f'step {step} loss {loss.item():.4f} exp {le.item():.4f} jaw {lj.item():.6f} vertex {lv.item():.4f}', flush=True)
        if step % EVAL_EVERY == 0 or step == args.steps:
            m = evaluate(ema, val, lips, bl)
            score = float(SCORES[args.select](m))
            print(f'step {step} EMA val (medians)', summary(m), f'{args.select} score {score:.4f}', flush=True)
            if score > best:
                best = score
                torch.save(({**{k: v.detach().cpu() for k, v in ema.state_dict().items()}, EXTRACTOR_KEY: EXTRACTOR}, None, step), args.out)
    print('[prior] saved the best EMA prior to', args.out, f'(score {best:.4f})', flush=True)


if __name__ == '__main__':
    main()
