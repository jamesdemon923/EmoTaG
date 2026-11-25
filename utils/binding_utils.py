"""Geometry helpers for gaussians bound to the FLAME surface."""

import torch


def closest_on_triangles(p, tri):
    """Closest points on triangles."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = p - b
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = p - c
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    eps = 1e-20
    den = (va + vb + vc).clamp_min(eps)
    v, w = vb / den, vc / den
    bary = torch.stack([1 - v - w, v, w], -1)
    z, o = torch.zeros_like(d1), torch.ones_like(d1)

    def put(mask, val):
        return torch.where(mask[..., None], val, bary)
    t = (d4 - d3) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
    bary = put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), torch.stack([z, 1 - t, t], -1))
    t = d2 / (d2 - d6).clamp_min(eps)
    bary = put((vb <= 0) & (d2 >= 0) & (d6 <= 0), torch.stack([1 - t, z, t], -1))
    t = d1 / (d1 - d3).clamp_min(eps)
    bary = put((vc <= 0) & (d1 >= 0) & (d3 <= 0), torch.stack([1 - t, t, z], -1))
    bary = put((d6 >= 0) & (d5 <= d6), torch.stack([z, z, o], -1))
    bary = put((d3 >= 0) & (d4 <= d3), torch.stack([z, o, z], -1))
    bary = put((d1 <= 0) & (d2 <= 0), torch.stack([o, z, z], -1))
    q = (bary[..., None] * tri).sum(-2)
    return bary, ((q - p) ** 2).sum(-1)
