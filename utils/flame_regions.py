"""FLAME expression-space geometry used to keep AUs off the mouth."""
from __future__ import annotations

import os
import pickle
from functools import lru_cache

import numpy as np

N_SHAPE = 300


def _flame_paths():
    model = os.environ.get('EMOTAG_FLAME_MODEL')
    vhap_root = os.environ.get('EMOTAG_VHAP_ROOT')
    if not model and vhap_root:
        model = os.path.join(vhap_root, 'asset', 'flame', 'generic_model.pkl')
    if not model:
        raise FileNotFoundError('Set EMOTAG_FLAME_MODEL (and EMOTAG_VHAP_ROOT) to locate FLAME.')
    masks = os.path.join(os.path.dirname(model), 'FLAME_masks.pkl')
    if not os.path.exists(masks) and vhap_root:
        masks = os.path.join(vhap_root, 'asset', 'flame', 'FLAME_masks.pkl')
    return model, masks


@lru_cache(maxsize=4)
def _load(n_exp: int = 50):
    model_path, masks_path = _flame_paths()
    with open(model_path, 'rb') as f:
        flame = pickle.load(f, encoding='latin1')
    with open(masks_path, 'rb') as f:
        masks = pickle.load(f, encoding='latin1')
    shapedirs = np.asarray(flame['shapedirs'], dtype=np.float64)
    exp_basis = shapedirs[:, :, N_SHAPE:N_SHAPE + n_exp]
    v_template = np.asarray(flame['v_template'], dtype=np.float64)
    face = np.asarray(masks['face'])
    lips = np.asarray(masks['lips'])
    nose = np.asarray(masks['nose'])
    nose_bottom = v_template[nose, 1].min()
    lower = np.union1d(lips, face[v_template[face, 1] < nose_bottom - 0.004])
    upper = np.union1d(np.asarray(masks['forehead']), np.asarray(masks['eye_region']))
    return exp_basis, lower, upper


def expression_region_bases(n_exp: int = 50):
    """(B_lower [3*V_lower, n_exp], B_upper [3*V_upper, n_exp]) displacement bases."""
    exp_basis, lower, upper = _load(n_exp)
    b_lower = exp_basis[lower].reshape(-1, n_exp)
    b_upper = exp_basis[upper].reshape(-1, n_exp)
    return b_lower.astype(np.float32), b_upper.astype(np.float32)


def upper_face_projector(n_exp: int = 50, max_lower_ratio: float = 0.5):
    """Projector P [n_exp, n_exp] onto expression directions that move the upper face but leave the lower face."""
    b_lower, b_upper = expression_region_bases(n_exp)
    b_lower = b_lower.astype(np.float64)
    b_upper = b_upper.astype(np.float64)
    a_l = b_lower.T @ b_lower / (b_lower.shape[0] / 3)
    a_u = b_upper.T @ b_upper / (b_upper.shape[0] / 3)
    w, U = np.linalg.eigh(a_l + 1e-9 * np.trace(a_l) / n_exp * np.eye(n_exp))
    wh = U @ np.diag(w ** -0.5) @ U.T
    mu, Q = np.linalg.eigh(wh @ a_u @ wh)
    V = wh @ Q
    ratio = 1.0 / np.sqrt(np.maximum(mu, 1e-12))
    keep = ratio < max_lower_ratio
    Vk = V[:, keep]
    if Vk.shape[1] == 0:
        return np.zeros((n_exp, n_exp), dtype=np.float32), {'kept': 0, 'ratios': ratio.tolist()}
    P = Vk @ np.linalg.solve(Vk.T @ a_u @ Vk, Vk.T @ a_u)
    info = {'kept': int(keep.sum()), 'ratios_kept': ratio[keep].round(4).tolist()}
    return P.astype(np.float32), info
