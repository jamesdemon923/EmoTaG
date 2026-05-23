"""Internal helper."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import pdb

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Internal helper."""
    return torch.sum(x * y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float=1e-20) -> torch.Tensor:
    """Internal helper."""
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))

def safe_normalize(x: torch.Tensor, eps: float=1e-20) -> torch.Tensor:
    """Internal helper."""
    return x / length(x, eps)

class FLAMEGaussianModel:
    """Internal helper."""

    def __init__(self, base_gaussian_model):
        """Internal helper."""
        self.base_model = base_gaussian_model
        self.face_center = None
        self.face_orien_mat = None
        self.face_scaling = None

    def get_deformed_gaussians(self, flame_verts: torch.Tensor, flame_pose: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """Internal helper."""
        faces = self.base_model.flame_binding.faces
        canonical_verts = self.base_model.flame_binding.get_canonical_mesh_vertices()
        face_center_canonical, face_orien_mat_canonical, face_scaling_canonical = self._compute_face_properties(canonical_verts, faces)
        face_center_deformed, face_orien_mat_deformed, face_scaling_deformed = self._compute_face_properties(flame_verts, faces)
        self.face_center = face_center_deformed
        self.face_orien_mat = face_orien_mat_deformed
        self.face_scaling = face_scaling_deformed
        point_face_indices = self.base_model.flame_binding.face_indices
        local_xyz = self.base_model._xyz
        rotated_xyz = torch.bmm(self.face_orien_mat[point_face_indices], local_xyz.unsqueeze(-1)).squeeze(-1)
        scaled_xyz = rotated_xyz * self.face_scaling[point_face_indices]
        final_xyz = scaled_xyz + self.face_center[point_face_indices]
        face_orien_quat = self._matrix_to_quaternion(self.face_orien_mat[point_face_indices])
        local_rotation_normalized = self.base_model.rotation_activation(self.base_model._rotation)
        final_rotation = self._quaternion_multiply(face_orien_quat, local_rotation_normalized)
        face_scale = self.face_scaling[point_face_indices]
        local_scale_activated = self.base_model.scaling_activation(self.base_model._scaling)
        final_scale = local_scale_activated * face_scale
        raw_scale = self.base_model.scaling_inverse_activation(final_scale)
        return {'xyz': final_xyz, 'rotation': final_rotation, 'scale': final_scale, 'raw_scale': raw_scale}

    def _compute_face_properties(self, verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal helper."""
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

    def _matrix_to_quaternion(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        trace = rotation_matrices[:, 0, 0] + rotation_matrices[:, 1, 1] + rotation_matrices[:, 2, 2]
        w = torch.sqrt(torch.clamp(1.0 + trace, min=1e-06)) / 2.0
        x = (rotation_matrices[:, 2, 1] - rotation_matrices[:, 1, 2]) / (4.0 * w + 1e-06)
        y = (rotation_matrices[:, 0, 2] - rotation_matrices[:, 2, 0]) / (4.0 * w + 1e-06)
        z = (rotation_matrices[:, 1, 0] - rotation_matrices[:, 0, 1]) / (4.0 * w + 1e-06)
        quaternions = torch.stack([w, x, y, z], dim=-1)
        return self.base_model.rotation_activation(quaternions)

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        w1, x1, y1, z1 = (q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3])
        w2, x2, y2, z2 = (q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3])
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        result = torch.stack([w, x, y, z], dim=-1)
        return self.base_model.rotation_activation(result)

    def get_deformed_xyz(self, flame_verts: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        return self.get_deformed_gaussians(flame_verts)['xyz']

    def get_deformed_rotation(self, flame_verts: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        return self.get_deformed_gaussians(flame_verts)['rotation']

    def get_deformed_scale(self, flame_verts: torch.Tensor) -> torch.Tensor:
        """Internal helper."""
        return self.get_deformed_gaussians(flame_verts)['scale']
