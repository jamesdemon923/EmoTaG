"""Eyelid geometry of the FLAME mesh and the blink direction of its expression basis."""
from __future__ import annotations

import numpy as np
import torch

EYES = ((36, 39, 'right'), (42, 45, 'left'))


def _zeros_forward(flame, expr):
    z = lambda n: torch.zeros(expr.shape[0], n, device=expr.device)
    return flame(shape=z(flame.n_shape_params), expr=expr, rotation=z(3), neck=z(3), jaw=z(3), eyes=z(6),
                 translation=z(3), return_landmarks=True)


class LidGeometry:
    """Upper / lower eyelid sheets of both eyes (static, model only)."""

    def __init__(self, flame, corner_frac=0.12, cone_mm=5.0):
        mk = flame.mask.v
        dev = flame.v_template.device
        vt = flame.v_template.detach().cpu().numpy().astype(np.float64)
        F = flame.faces.cpu().numpy()
        E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0)
        with torch.no_grad():
            L = _zeros_forward(flame, torch.zeros(1, flame.n_expr_params, device=dev))[1][0].cpu().numpy().astype(np.float64)
        T = lambda x: torch.as_tensor(x, device=dev)
        self.sheets, lids = [], []
        for a, b, name in EYES:
            lid = getattr(mk, f'{name}_eyelid').cpu().numpy()
            c = vt[getattr(mk, f'{name}_eyeball').cpu().numpy()].mean(0)
            ax = (L[b] - L[a])[:2]
            w = np.linalg.norm(ax)
            ax = ax / w
            n = np.array([-ax[1], ax[0]])
            n = n if n[1] > 0 else -n
            t = ((vt[lid, :2] - L[a, :2]) @ ax) / w
            h = (vt[lid, :2] - L[a, :2]) @ n
            ok = (t > corner_frac) & (t < 1 - corner_frac) & (vt[lid, 2] > c[2] + cone_mm * 1e-3)
            up, lo = lid[ok & (h > 0)], lid[ok & (h <= 0)]
            self.sheets.append(dict(a=a, b=b, up=T(up), lo=T(lo),
                                    e_up=T(E[np.isin(E[:, 0], up) & np.isin(E[:, 1], up)]),
                                    e_lo=T(E[np.isin(E[:, 0], lo) & np.isin(E[:, 1], lo)])))
            lids += [up, lo]
        self.lids = T(np.unique(np.concatenate(lids)))


def eye_axes(pa, pb):
    """Unit vectors along the eye (corner to corner) and across it (pointing down in the image)."""
    a = pb - pa
    a = a / a.norm(dim=-1, keepdim=True)
    n = torch.stack([-a[..., 1], a[..., 0]], -1)
    return a, n * torch.sign(n[..., 1:2])


def silhouette(x, y, edges, xk, upper):
    """Extreme of a sheet at the columns xk."""
    x0, x1 = x[:, edges[:, 0], None], x[:, edges[:, 1], None]
    y0, y1 = y[:, edges[:, 0], None], y[:, edges[:, 1], None]
    dx = x1 - x0
    dx = torch.where(dx.abs() < 1e-4, torch.full_like(dx, 1e-4), dx)
    t = (xk[:, None] - x0) / dx
    ok = (t >= 0) & (t <= 1)
    yk = y0 + t.clamp(0, 1) * (y1 - y0)
    if upper:
        return torch.where(ok, yk, torch.full_like(yk, -1e4)).amax(1), ok.any(1)
    return torch.where(ok, yk, torch.full_like(yk, 1e4)).amin(1), ok.any(1)


def lid_gap(pv, pl, sheet, s, axes=None):
    """Visible lid opening of one eye at the normalised positions s [K] between the eye-corner landmarks."""
    a, n = eye_axes(pl[:, sheet['a']], pl[:, sheet['b']]) if axes is None else axes
    x = (pv * a[:, None]).sum(-1)
    y = (pv * n[:, None]).sum(-1)
    xa = (pl[:, sheet['a']] * a).sum(-1, keepdim=True)
    xb = (pl[:, sheet['b']] * a).sum(-1, keepdim=True)
    xk = (xa + s[None] * (xb - xa)).detach()
    top, ok_u = silhouette(x, y, sheet['e_up'], xk, True)
    bot, ok_l = silhouette(x, y, sheet['e_lo'], xk, False)
    ok = ok_u & ok_l
    return torch.where(ok, bot - top, torch.zeros_like(top)), ok, (xb - xa).abs()[:, 0]


def blink_direction(flame, geo=None, upper_proj=None, w_prior=0.003, w_lower=10.0, w_forehead=0.3, seal_mm=0.3,
                    iters=600, n_col=13):
    """(u, w): expression offset u [n_exp] that seals both eyes of the template (frontal view) with the least."""
    dev = flame.v_template.device
    n_exp = flame.n_expr_params
    geo = geo or LidGeometry(flame)
    mk = flame.mask.v
    n_s = flame.n_shape_params
    Ex = flame.shapedirs[:, :, n_s:n_s + n_exp].detach().double() * 1e3
    vt = flame.v_template.detach()
    quad = lambda idx: (torch.einsum('vci,vcj->ij', Ex[idx], Ex[idx]) / len(idx)).float()
    nose_bottom = vt[mk.nose, 1].min()
    lower = torch.unique(torch.cat([mk.lips, mk.face[vt[mk.face, 1] < nose_bottom - 0.004]]))
    A_low, A_fore, A_lid = quad(lower), quad(mk.forehead), quad(geo.lids)
    if upper_proj is None:
        basis = torch.eye(n_exp, device=dev)
    else:
        P = torch.as_tensor(np.asarray(upper_proj), dtype=torch.float32, device=dev)
        U, S, _ = torch.linalg.svd(P)
        basis = U[:, :int((S > 1e-4 * S[0]).sum())]
    s = torch.linspace(0.2, 0.8, n_col, device=dev)
    flip = torch.tensor([1.0, -1.0], device=dev) * 1e3
    with torch.no_grad():
        _, L0 = _zeros_forward(flame, torch.zeros(1, n_exp, device=dev))
        pl0 = L0[:, :68, :2] * flip
        axes = [eye_axes(pl0[:, sh['a']], pl0[:, sh['b']]) for sh in geo.sheets]
    with torch.enable_grad():
        q = torch.zeros(basis.shape[1], device=dev, requires_grad=True)
        opt = torch.optim.Adam([q], lr=0.1)
        for _ in range(iters):
            d = basis @ q
            V, L = _zeros_forward(flame, d[None])
            g = torch.cat([lid_gap(V[:, :, :2] * flip, L[:, :68, :2] * flip, sh, s, ax)[0] for sh, ax in zip(geo.sheets, axes)], 1)
            loss = (torch.relu(g + seal_mm) ** 2).mean() + w_prior * (d ** 2).sum() + w_lower * (d @ A_low) @ d + w_forehead * (d @ A_fore) @ d
            opt.zero_grad()
            loss.backward()
            opt.step()
    u = (basis @ q).detach()
    w = (A_lid @ u) / ((u @ A_lid) @ u)
    return u, w
