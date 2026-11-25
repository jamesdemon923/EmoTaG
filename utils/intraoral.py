"""Ownership of the mouth cavity (rigid intra-oral set, hybrid binding)."""
import torch

from scene.flame_wrapper import N_FLAME_VERTS
from utils.binding_utils import closest_on_triangles

SIDE_MARGIN = 0.003
TOP_MARGIN = 0.003
BOTTOM_MARGIN = 0.010
FRONT_MARGIN = 0.001
DEPTH = 0.030
MAX_SCALE = 0.004


def cavity_frame(gaussians):
    """Teeth-aligned frame of the canonical mesh."""
    cached = getattr(gaussians, '_cavity_frame', None)
    if cached is not None:
        return cached
    V = gaussians.flame_binding.canonical_mesh_vertices
    if V.shape[0] <= N_FLAME_VERTS:
        raise RuntimeError('the FLAME mesh has no teeth geometry')
    T = V[N_FLAME_VERTS:].reshape(8, -1, 3)
    down = torch.nn.functional.normalize((T[2] - T[0]).mean(0), dim=0)
    back = (T[4] - T[0]).mean(0)
    back = torch.nn.functional.normalize(back - (back @ down) * down, dim=0)
    R = torch.stack([torch.linalg.cross(down, back), down, back], 0)
    org = 0.5 * (T[2].mean(0) + T[3].mean(0))
    tl = (T - org) @ R.T
    xs, zs = tl[2, :, 0], tl[2, :, 2]
    order = torch.argsort(xs)
    frame = {'R': R, 'org': org, 'xs': xs[order].contiguous(), 'zs': zs[order].contiguous(),
             'y_top': float(tl[0, :, 1].mean()), 'y_bottom': float(tl[1, :, 1].mean())}
    gaussians._cavity_frame = frame
    return frame


def _front(frame, x):
    xs, zs = frame['xs'], frame['zs']
    i = torch.searchsorted(xs, x.contiguous()).clamp(1, len(xs) - 1)
    t = ((x - xs[i - 1]) / (xs[i] - xs[i - 1])).clamp(0, 1)
    return zs[i - 1] + t * (zs[i] - zs[i - 1])


def _teeth_coords(gaussians):
    """Canonical positions in the teeth frame [N, 3] and the depth behind the teeth front surface [N]."""
    frame = cavity_frame(gaussians)
    b = gaussians.flame_binding
    loc = (b.get_canonical_points() + gaussians._xyz.detach() - frame['org']) @ frame['R'].T
    return frame, loc, loc[:, 2] - _front(frame, loc[:, 0])


def _hybrid():
    from scene.flame_gaussian_model import binding_mode
    return binding_mode() == 'hybrid'


@torch.no_grad()
def contain(gaussians):
    """Project the intra-oral set back into the cavity volume and cap its size."""
    m = gaussians.mouth_mask
    if m is None or not bool(m.any()) or not _hybrid():
        return 0
    frame, loc, _ = _teeth_coords(gaussians)
    loc = loc[m]
    x = loc[:, 0].clamp(float(frame['xs'][0]) - SIDE_MARGIN, float(frame['xs'][-1]) + SIDE_MARGIN)
    y = loc[:, 1].clamp(frame['y_top'] - TOP_MARGIN, frame['y_bottom'] + BOTTOM_MARGIN)
    f = _front(frame, x)
    z = torch.minimum(torch.maximum(loc[:, 2], f - FRONT_MARGIN), f + DEPTH)
    new = torch.stack([x, y, z], 1)
    moved = (new - loc).abs().amax(1) > 1e-7
    if bool(moved.any()):
        world = new @ frame['R'] + frame['org']
        gaussians._xyz.data[m] = world - gaussians.flame_binding.get_canonical_points()[m]
    cap = float(torch.log(torch.tensor(MAX_SCALE)))
    s = gaussians._scaling.data
    s[m] = s[m].clamp(max=cap)
    return int(moved.sum())


@torch.no_grad()
def _to_set(gaussians, take, loc):
    """Re-anchor the gaussians `take` on the nearest teeth triangle (upper teeth above the biting plane, lower teeth."""
    n = int(take.sum())
    if n == 0:
        return 0
    b = gaussians.flame_binding
    verts, faces = b.canonical_mesh_vertices, b.faces.long()
    groups = gaussians.flame_wrapper.teeth_face_groups()
    upper = torch.cat([groups[k] for k in ('upper_front', 'upper_back', 'upper_edge')])
    lower = torch.cat([groups[k] for k in ('lower_front', 'lower_back', 'lower_edge')])
    canon = (b.get_canonical_points() + gaussians._xyz.detach())[take]
    is_upper = loc[take, 1] < 0
    fi = torch.empty(n, dtype=torch.long, device=canon.device)
    bary = torch.empty(n, 3, device=canon.device, dtype=b.bary_coords.dtype)
    for sel, fids in ((is_upper, upper), (~is_upper, lower)):
        if not bool(sel.any()):
            continue
        p = canon[sel]
        bc, d2 = closest_on_triangles(p[:, None].expand(-1, len(fids), -1), verts[faces[fids]][None].expand(len(p), -1, -1, -1))
        j = d2.argmin(1)
        ar = torch.arange(len(p), device=p.device)
        fi[sel] = fids[j]
        bary[sel] = bc[ar, j].to(bary.dtype)
    b.face_indices[take] = fi.to(b.face_indices.dtype)
    b.bary_coords[take] = bary
    b._update_canonical_points()
    gaussians._xyz.data[take] = canon - b.get_canonical_points()[take]
    gaussians.mouth_mask = gaussians.mouth_mask | take
    gaussians.mouth_indices = gaussians._recompute_mouth_indices_from_mask()
    return n


@torch.no_grad()
def adopt(gaussians, min_offset=0.003):
    """Hand surface gaussians that sit inside the cavity volume to the intra-oral set."""
    m = gaussians.mouth_mask
    if m is None or not _hybrid():
        return 0
    frame, loc, dz = _teeth_coords(gaussians)
    inside = (loc[:, 0] > frame['xs'][0]) & (loc[:, 0] < frame['xs'][-1]) & (loc[:, 1] > frame['y_top']) & \
             (loc[:, 1] < frame['y_bottom']) & (dz > 0) & (dz < DEPTH)
    return _to_set(gaussians, inside & ~m & (gaussians._xyz.detach().norm(dim=1) > min_offset), loc)
