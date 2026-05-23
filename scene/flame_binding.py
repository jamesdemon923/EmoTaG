import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
import trimesh
import pdb

def safe_normalize(x: torch.Tensor, eps: float=1e-20) -> torch.Tensor:
    """Internal helper."""
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def length(x: torch.Tensor, eps: float=1e-20) -> torch.Tensor:
    """Internal helper."""
    return torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Internal helper."""
    return torch.sum(x * y, -1, keepdim=True)

class FLAMEBinding(nn.Module):
    """
    A new, correct implementation of FLAMEBinding based on barycentric coordinates.
    This version accepts pre-computed binding info to avoid alignment issues.
    """

    def __init__(self, flame_model, face_indices, bary_coords, canonical_vertices, device='cuda'):
        """
        Initializes the binding with pre-computed face indices and barycentric coordinates.
        This bypasses the need for ICP alignment.
        
        Args:
            flame_model: The FLAME wrapper instance.
            face_indices (torch.Tensor): Tensor of shape [N,] with the face index for each Gaussian.
            bary_coords (torch.Tensor): Tensor of shape [N, 3] with the barycentric coordinates for each Gaussian.
            canonical_vertices (torch.Tensor): The original mesh vertices used for sampling (not necessarily centered).
            device (str): The target device for tensors.
        """
        super().__init__()
        self.device = device
        self.flame_model = flame_model
        self.faces = flame_model.faces_tensor.to(device)
        self.num_faces = len(self.faces)
        self.register_buffer('face_indices', face_indices.long().to(device))
        self.register_buffer('bary_coords', bary_coords.float().to(device))
        self.register_buffer('canonical_mesh_vertices', canonical_vertices.float().to(device))
        self.register_buffer('canonical_points', self.get_points_from_bary_coords(self.canonical_mesh_vertices))

    def get_points_from_bary_coords(self, verts):
        """Helper function to compute point positions from vertices using barycentric coordinates."""
        face_verts = verts[self.faces[self.face_indices]]
        points = (face_verts * self.bary_coords.unsqueeze(-1)).sum(dim=1)
        return points

    def get_canonical_points(self):
        """Returns the canonical 3D location for each Gaussian based on its face binding."""
        return self.canonical_points

    def get_canonical_mesh_vertices(self):
        """Returns the canonical mesh vertices that were used for sampling and binding."""
        return self.canonical_mesh_vertices

    def transform_gaussians(self, current_verts):
        """Internal helper."""
        if current_verts.dim() == 3:
            current_verts = current_verts.squeeze(0)
        transformed_xyz = self.get_points_from_bary_coords(current_verts)
        return transformed_xyz

    def update_binding(self, operation_type, mask, N=None):
        """
        Updates the binding information based on densification or pruning operations.
        """
        if operation_type == 'clone':
            cloned_face_indices = self.face_indices[mask]
            cloned_bary_coords = self.bary_coords[mask]
            self.face_indices = torch.cat([self.face_indices, cloned_face_indices])
            self.bary_coords = torch.cat([self.bary_coords, cloned_bary_coords])
        elif operation_type == 'split':
            if N is None:
                raise ValueError("N must be provided for 'split' operation.")
            split_face_indices = self.face_indices[mask].repeat_interleave(N, dim=0)
            split_bary_coords = self.bary_coords[mask].repeat_interleave(N, dim=0)
            self.face_indices = torch.cat([self.face_indices, split_face_indices])
            self.bary_coords = torch.cat([self.bary_coords, split_bary_coords])
        elif operation_type == 'prune':
            valid_mask = ~mask
            self.face_indices = self.face_indices[valid_mask]
            self.bary_coords = self.bary_coords[valid_mask]
        else:
            raise ValueError(f'Unsupported binding operation: {operation_type}')
        self._update_canonical_points()

    def _update_canonical_points(self):
        """
        Recalculates and updates the canonical_points buffer based on the current binding.
        """
        self.canonical_points = self.get_points_from_bary_coords(self.canonical_mesh_vertices)

    def get_points_on_surface(self, flame_verts):
        """
        Calculates the 3D positions of the Gaussian points on the surface of a deformed FLAME mesh.

        Args:
            flame_verts (torch.Tensor): The deformed vertices of the FLAME mesh, shape [V, 3], 
                                      where V is the number of vertices.

        Returns:
            torch.Tensor: The calculated 3D positions of the Gaussian points on the mesh surface,
                          shape [N, 3], where N is the number of Gaussian points.
        """
        triangle_verts = flame_verts[self.faces[self.face_indices]]
        points_on_surface = (triangle_verts * self.bary_coords.unsqueeze(-1)).sum(dim=1)
        return points_on_surface
