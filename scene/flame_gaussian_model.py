"""FLAME-driven gaussian deformation (strict triangle-local parameterization).

Implements Eq.7 of the EmoTaG paper (the GaussianAvatars rigging): every gaussian
lives in its parent triangle's local frame, with learnable local position mu_l
(_xyz), local rotation q_l (_rotation) and local log-scale (_scaling). World
attributes for a mesh state (canonical or deformed) are

    mu    = k^j * R^j @ mu_l + C^j
    r     = R^j * q_l
    s     = k^j * exp(_scaling)

where C^j / R^j / k^j are the parent triangle's center / orientation / size.
Canonical and deformed states use the SAME mapping (only the mesh differs), so
the avatar generalizes to unseen expressions/poses by construction. A previous
revision stored world-space offsets instead; those offsets grew to ~25% of the
head width during training and only rendered correctly on training frames.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Internal helper."""
    return torch.sum(x * y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float=1e-20) -> torch.Tensor:
    """Internal helper."""
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x: torch.Tensor, eps: float=1e-20) -> torch.Tensor:
    """Internal helper."""
    return x / length(x, eps)

def compute_face_properties(verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-triangle center [F,3], orientation [F,3,3] and size [F,3] (GaussianAvatars)."""
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()
    v0 = verts[i0]
    v1 = verts[i1]
    v2 = verts[i2]
    face_center = (v0 + v1 + v2) / 3.0
    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))
    face_orien_mat = torch.stack([a0, a1, a2], dim=-1)
    s0 = length(v1 - v0).squeeze(-1)
    s1 = dot(a2, v2 - v0).abs().squeeze(-1)
    scale_factor = (s0 + s1) / 2
    face_scaling = scale_factor.unsqueeze(-1).repeat(1, 3)
    return (face_center, face_orien_mat, face_scaling)

def degenerate_faces(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Sliver/collinear triangles whose orientation frame is numerically unstable."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    twice_area = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)
    mean_edge = ((v1 - v0).norm(dim=-1) + (v2 - v1).norm(dim=-1) + (v0 - v2).norm(dim=-1)) / 3.0
    return twice_area / (mean_edge * mean_edge + 1e-12) < 0.05

def matrix_to_quaternion(rotation_matrices: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [N,3,3] -> unnormalized wxyz quaternions [N,4]."""
    trace = rotation_matrices[:, 0, 0] + rotation_matrices[:, 1, 1] + rotation_matrices[:, 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-06)) / 2.0
    x = (rotation_matrices[:, 2, 1] - rotation_matrices[:, 1, 2]) / (4.0 * w + 1e-06)
    y = (rotation_matrices[:, 0, 2] - rotation_matrices[:, 2, 0]) / (4.0 * w + 1e-06)
    z = (rotation_matrices[:, 1, 0] - rotation_matrices[:, 0, 1]) / (4.0 * w + 1e-06)
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of wxyz quaternion batches."""
    w1, x1, y1, z1 = (q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3])
    w2, x2, y2, z2 = (q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3])
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

import os

def binding_mode() -> str:
    """'hybrid' (default): _xyz is a world-space offset around the barycentric anchor,
    deformed by the relative per-triangle rotation; empirically the more robust choice
    under monocular VHAP tracking noise. 'local': strict Eq.7 triangle-local
    parameterization (per-frame triangle frames also drive scale/rotation), which is
    theoretically cleaner but amplifies per-triangle tracking jitter into the render.
    Selected via EMOTAG_BINDING_MODE."""
    return os.environ.get('EMOTAG_BINDING_MODE', 'hybrid')

class FLAMEGaussianModel:
    """Maps triangle-local gaussian parameters to world space for a given mesh."""

    def __init__(self, base_gaussian_model):
        """Internal helper."""
        self.base_model = base_gaussian_model
        self.face_center = None
        self.face_orien_mat = None
        self.face_scaling = None

    def get_gaussians_for_mesh(self, verts: torch.Tensor, degenerate_fallback=None) -> Dict[str, torch.Tensor]:
        """World-space gaussians for an arbitrary mesh state (Eq.7)."""
        faces = self.base_model.flame_binding.faces
        if verts.dim() == 3:
            verts = verts.squeeze(0)
        face_center, face_orien_mat, face_scaling = compute_face_properties(verts, faces)
        if degenerate_fallback is not None:
            fallback_center, fallback_orien, fallback_scaling = degenerate_fallback
            bad = degenerate_faces(verts, faces)
            face_orien_mat = torch.where(bad.view(-1, 1, 1), fallback_orien, face_orien_mat)
            face_scaling = torch.where(bad.view(-1, 1), fallback_scaling, face_scaling)
        self.face_center = face_center
        self.face_orien_mat = face_orien_mat
        self.face_scaling = face_scaling
        fi = self.base_model.flame_binding.face_indices
        local_xyz = self.base_model._xyz
        rotated = torch.bmm(face_orien_mat[fi], local_xyz.unsqueeze(-1)).squeeze(-1)
        final_xyz = rotated * face_scaling[fi] + face_center[fi]
        face_quat = self.base_model.rotation_activation(matrix_to_quaternion(face_orien_mat[fi]))
        local_rot = self.base_model.rotation_activation(self.base_model._rotation)
        final_rotation = self.base_model.rotation_activation(quaternion_multiply(face_quat, local_rot))
        local_scale = self.base_model.scaling_activation(self.base_model._scaling)
        final_scale = local_scale * face_scaling[fi]
        raw_scale = self.base_model.scaling_inverse_activation(final_scale)
        return {'xyz': final_xyz, 'rotation': final_rotation, 'scale': final_scale, 'raw_scale': raw_scale}

    def get_deformed_gaussians(self, flame_verts: torch.Tensor, flame_pose: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """World-space gaussians on the deformed mesh (degenerate faces keep the canonical frame)."""
        if binding_mode() == 'hybrid':
            return self._get_deformed_gaussians_hybrid(flame_verts)
        binding = self.base_model.flame_binding
        can_center, can_orien, can_scaling = binding.canonical_face_frames()
        return self.get_gaussians_for_mesh(flame_verts, degenerate_fallback=(can_center, can_orien, can_scaling))

    def _get_deformed_gaussians_hybrid(self, flame_verts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Hybrid model: _xyz is a WORLD-space offset around the barycentric surface
        anchor; deformation moves the anchor along the deformed surface and applies
        only the RELATIVE per-triangle rotation (canonical->deformed) to the offset,
        with a clamped relative stretch on scales. More tolerant of per-frame
        tracking noise than the strict local mapping."""
        base = self.base_model
        binding = base.flame_binding
        faces = binding.faces
        if flame_verts.dim() == 3:
            flame_verts = flame_verts.squeeze(0)
        canonical_verts = binding.get_canonical_mesh_vertices()
        _, orien_can, scaling_can = binding.canonical_face_frames()
        center_def, orien_def, scaling_def = compute_face_properties(flame_verts, faces)
        self.face_center = center_def
        self.face_orien_mat = orien_def
        self.face_scaling = scaling_def
        rel_rot = torch.bmm(orien_def, orien_can.transpose(1, 2))
        bad = degenerate_faces(canonical_verts, faces) | degenerate_faces(flame_verts, faces)
        eye = torch.eye(3, device=rel_rot.device, dtype=rel_rot.dtype).expand(rel_rot.shape[0], 3, 3)
        rel_rot = torch.where(bad.view(-1, 1, 1), eye, rel_rot)
        fi = binding.face_indices
        rel_rot_pts = rel_rot[fi]
        bary_deformed = binding.get_points_on_surface(flame_verts)
        rotated_offset = torch.bmm(rel_rot_pts, base._xyz.unsqueeze(-1)).squeeze(-1)
        final_xyz = bary_deformed + rotated_offset
        rel_rot_quat = base.rotation_activation(matrix_to_quaternion(rel_rot_pts))
        local_rot = base.rotation_activation(base._rotation)
        final_rotation = base.rotation_activation(quaternion_multiply(rel_rot_quat, local_rot))
        stretch = (scaling_def / (scaling_can + 1e-8))[fi].clamp(0.5, 2.0)
        final_scale = base.scaling_activation(base._scaling) * stretch
        raw_scale = base.scaling_inverse_activation(final_scale)
        return {'xyz': final_xyz, 'rotation': final_rotation, 'scale': final_scale, 'raw_scale': raw_scale}

    def get_deformed_xyz(self, flame_verts: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        return self.get_deformed_gaussians(flame_verts)['xyz']

    def get_deformed_rotation(self, flame_verts: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        return self.get_deformed_gaussians(flame_verts)['rotation']

    def get_deformed_scale(self, flame_verts: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        return self.get_deformed_gaussians(flame_verts)['scale']
