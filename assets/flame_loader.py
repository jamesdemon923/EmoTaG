#!/usr/bin/env python3
"""
Script to generate initial point cloud and FLAME data from video processing outputs.
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
import cv2
import scipy.io as sio
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from typing import Dict
EMOTAG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EMOTAG_ROOT not in sys.path:
    sys.path.insert(0, EMOTAG_ROOT)
VHAP_ROOT = os.environ.get('EMOTAG_VHAP_ROOT')
if VHAP_ROOT and VHAP_ROOT not in sys.path:
    sys.path.insert(0, VHAP_ROOT)
try:
    from vhap.model.flame import FlameHead
    from utils.graphics_utils import BasicPointCloud
except ImportError as e:
    print(f'Failed to import VHAP FlameHead. Set EMOTAG_VHAP_ROOT or install VHAP. Error: {e}')
    sys.exit(1)

def storePly(path, xyz, rgb):
    import numpy as np
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

class FlameDataProcessor:
    """
    Handles loading of FLAME parameters and generation of initial mesh data.
    """

    def __init__(self, flame_model_path: str):
        """
        Initializes the processor with the FLAME model path.
        """
        self.flame_model_path = flame_model_path
        self.flame_model = None
        self._create_simple_flame(flame_model_path)
        print(f'Loaded FLAME model from: {flame_model_path}')

    def _create_simple_flame(self, model_path):
        """Creates a simplified FLAME model instance using VHAP's FlameHead."""
        try:
            self.flame_model = FlameHead(shape_params=300, expr_params=100, flame_model_path=model_path, flame_lmk_embedding_path=os.path.join(os.path.dirname(model_path), 'landmark_embedding.npy'))
            self._mouth_vertex_indices = None
            self._mouth_face_indices = None
        except Exception as e:
            raise RuntimeError(f'Failed to create VHAP FlameHead from {model_path}: {e}')

    def load_flame_params(self, data_root: str, frame_id: int) -> Dict[str, torch.Tensor]:
        """
        Loads all available FLAME parameters for a specific frame from a .npz file.
        """
        frame_name = f'{frame_id}.npz'
        npz_path = os.path.join(data_root, frame_name)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f'FLAME parameter file not found: {npz_path}')
        data = np.load(npz_path)
        if 'rotation' not in data:
            raise KeyError(f'Missing rotation in {npz_path}')
        global_rot = data['rotation']
        if 'jaw_pose' not in data:
            raise KeyError(f'Missing jaw_pose in {npz_path}')
        jaw_pose = data['jaw_pose']
        pose = np.concatenate([global_rot.reshape(1, -1), jaw_pose.reshape(1, -1)], axis=1)
        if 'translation' not in data:
            raise KeyError(f'Missing translation in {npz_path}')
        translation = data['translation']
        if 'neck_pose' not in data:
            raise KeyError(f'Missing neck_pose in {npz_path}')
        neck_pose = data['neck_pose']
        if 'eyes_pose' not in data:
            raise KeyError(f'Missing eyes_pose in {npz_path}')
        eyes_pose = data['eyes_pose']
        static_offset = data.get('static_offset', None)
        shape_params = torch.from_numpy(data['shape']).float()
        if shape_params.dim() == 1:
            shape_params = shape_params.unsqueeze(0)
        exp_params = torch.from_numpy(data['expr']).float()
        if exp_params.dim() == 1:
            exp_params = exp_params.unsqueeze(0)

        def to_2d_tensor(arr):
            tensor = torch.from_numpy(arr).float()
            if tensor.dim() == 1:
                return tensor.unsqueeze(0)
            return tensor
        params = {'shape': to_2d_tensor(data['shape']), 'expression': to_2d_tensor(data['expr']), 'pose': to_2d_tensor(pose), 'rotation': to_2d_tensor(global_rot), 'jaw_pose': to_2d_tensor(jaw_pose), 'translation': to_2d_tensor(translation), 'neck_pose': to_2d_tensor(neck_pose), 'eyes_pose': to_2d_tensor(eyes_pose)}
        if static_offset is not None:
            params['static_offset'] = to_2d_tensor(static_offset)
        return params

    def get_flame_mesh_for_frame(self, flame_params: Dict[str, torch.Tensor]):
        """Internal helper."""
        real_global_rot = flame_params['pose'][:, :3]
        zero_jaw_pose = torch.zeros_like(flame_params['pose'][:, 3:])
        canonical_pose = torch.cat([real_global_rot, zero_jaw_pose], dim=1)
        zero_expression = torch.zeros_like(flame_params['expression'])
        batch_size = flame_params['shape'].shape[0]
        rotation = canonical_pose[:, :3]
        jaw = canonical_pose[:, 3:6]
        neck = flame_params['neck_pose'].squeeze(0) if flame_params['neck_pose'].dim() > 1 else flame_params['neck_pose']
        eyes = flame_params['eyes_pose'].squeeze(0) if flame_params['eyes_pose'].dim() > 1 else flame_params['eyes_pose']
        translation = flame_params['translation'].squeeze(0) if flame_params['translation'].dim() > 2 else flame_params['translation']
        shape = flame_params['shape']
        expression = zero_expression
        if rotation.dim() == 1:
            rotation = rotation.unsqueeze(0)
        if jaw.dim() == 1:
            jaw = jaw.unsqueeze(0)
        if neck.dim() == 1:
            neck = neck.unsqueeze(0)
        if eyes.dim() == 1:
            eyes = eyes.unsqueeze(0)
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
        if shape.dim() == 1:
            shape = shape.unsqueeze(0)
        if expression.dim() == 1:
            expression = expression.unsqueeze(0)
        flame_output = self.flame_model(shape=shape, expr=expression, rotation=rotation, neck=neck, jaw=jaw, eyes=eyes, translation=translation, return_landmarks=False)
        vertices = flame_output
        vertices_tensor = vertices.detach().cpu()
        if vertices_tensor.dim() == 4 and vertices_tensor.shape[0] == 1 and (vertices_tensor.shape[1] == 1):
            vertices_numpy = vertices_tensor.squeeze(0).squeeze(0).numpy()
        elif vertices_tensor.dim() == 3 and vertices_tensor.shape[0] == 1:
            vertices_numpy = vertices_tensor.squeeze(0).numpy()
        elif vertices_tensor.dim() == 2:
            vertices_numpy = vertices_tensor.numpy()
        else:
            raise ValueError(f'Unexpected FLAME vertices shape: {vertices_tensor.shape}')
        return (vertices_numpy, self.flame_model.faces.cpu().numpy())

    def get_mouth_region_indices(self, vertices: np.ndarray, faces: np.ndarray):
        """Internal helper."""
        if self._mouth_vertex_indices is not None and self._mouth_face_indices is not None:
            return (self._mouth_vertex_indices, self._mouth_face_indices)
        model_dir = os.path.dirname(self.flame_model_path)
        lme_path = os.path.join(model_dir, 'landmark_embedding.npy')
        if not os.path.exists(lme_path):
            print(f'landmark_embedding.npy not found at {lme_path}; mouth binding will be empty.')
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        lme = np.load(lme_path, allow_pickle=True, encoding='latin1').item()
        full_lmk_faces_idx = lme['full_lmk_faces_idx'].squeeze().astype(np.int64)
        full_lmk_bary_coords = lme['full_lmk_bary_coords'].squeeze().astype(np.float32)
        lmk3d = []
        for i in range(len(full_lmk_faces_idx)):
            fi = full_lmk_faces_idx[i]
            tri = faces[fi]
            bary = full_lmk_bary_coords[i]
            xyz = (vertices[tri] * bary[:, None]).sum(axis=0)
            lmk3d.append(xyz)
        lmk3d = np.stack(lmk3d, axis=0)
        adj = [[] for _ in range(vertices.shape[0])]
        for tri in faces:
            a, b, c = tri
            adj[a] += [b, c]
            adj[b] += [a, c]
            adj[c] += [a, b]
        complete_lip_vertices = self._get_complete_lip_region(lmk3d, vertices, adj)
        print(f'Lip region vertices: {len(complete_lip_vertices)}')
        connection_points, mouth_interior_bbox = self._find_interior_connection_points_by_bbox_with_bbox_return(vertices, complete_lip_vertices, adj)
        interior_vertices = self._two_stage_interior_expansion(connection_points, vertices, adj, complete_lip_vertices, mouth_interior_bbox)
        print(f'Mouth interior vertices: {len(interior_vertices)}')
        final_mouth_vertices = set(complete_lip_vertices) | interior_vertices
        mouth_vertex_indices = np.array(sorted(final_mouth_vertices), dtype=np.int64)
        mouth_vertex_set = set(mouth_vertex_indices.tolist())
        mouth_face_mask = np.array([tri[0] in mouth_vertex_set or tri[1] in mouth_vertex_set or tri[2] in mouth_vertex_set for tri in faces], dtype=bool)
        mouth_face_indices = np.nonzero(mouth_face_mask)[0].astype(np.int64)
        self._mouth_vertex_indices = mouth_vertex_indices
        self._mouth_face_indices = mouth_face_indices
        print(f'Mouth-bound point faces: {len(mouth_face_indices)}')
        return (mouth_vertex_indices, mouth_face_indices)

    def _get_complete_lip_region(self, lmk3d: np.ndarray, vertices: np.ndarray, adj: list) -> list:
        """Internal helper."""
        from sklearn.neighbors import KDTree
        tree = KDTree(vertices)
        mouth_landmark_ids = list(range(48, 68))
        mouth_lmks = lmk3d[mouth_landmark_ids]
        _, mouth_vertex_ids = tree.query(mouth_lmks, k=1)
        initial_mouth_vertices = mouth_vertex_ids.flatten().tolist()

        def expand_lips(seed_vertices, num_rings=1):
            current = set(seed_vertices)
            for _ in range(num_rings):
                next_ring = set(current)
                for v in list(current):
                    next_ring.update(adj[v])
                current = next_ring
            return list(current)
        complete_lip_vertices = expand_lips(initial_mouth_vertices, num_rings=1)
        return complete_lip_vertices

    def _find_interior_connection_points_by_bbox_with_bbox_return(self, vertices: np.ndarray, complete_lip_vertices: list, adj: list) -> tuple:
        """Internal helper."""
        all_lip_coords = vertices[complete_lip_vertices]
        mouth_interior_bbox = self._compute_mouth_interior_bbox(all_lip_coords)
        connection_points = []
        complete_lip_set = set(complete_lip_vertices)
        lip_center = all_lip_coords.mean(axis=0)
        max_connection_distance = np.linalg.norm(all_lip_coords.max(axis=0) - all_lip_coords.min(axis=0)) * 0.5
        for lip_vertex in complete_lip_vertices:
            for neighbor in adj[lip_vertex]:
                if neighbor not in complete_lip_set:
                    if not self._is_in_bbox(vertices[neighbor], mouth_interior_bbox):
                        continue
                    distance_to_lip_center = np.linalg.norm(vertices[neighbor] - lip_center)
                    if distance_to_lip_center > max_connection_distance:
                        continue
                    if vertices[neighbor][2] > lip_center[2]:
                        continue
                    connection_points.append(neighbor)
        connection_points = list(set(connection_points))
        return (connection_points, mouth_interior_bbox)

    def _compute_mouth_interior_bbox(self, all_lip_coords: np.ndarray) -> dict:
        """Internal helper."""
        x_min, x_max = (all_lip_coords[:, 0].min(), all_lip_coords[:, 0].max())
        y_min, y_max = (all_lip_coords[:, 1].min(), all_lip_coords[:, 1].max())
        z_min, z_max = (all_lip_coords[:, 2].min(), all_lip_coords[:, 2].max())
        y_range = y_max - y_min
        y_padding = y_range * 0.2
        y_interior_min = y_min - y_padding
        y_interior_max = y_max + y_padding
        x_range = x_max - x_min
        x_padding = x_range * 0.1
        x_interior_min = x_min - x_padding
        x_interior_max = x_max + x_padding
        lip_span = np.linalg.norm(all_lip_coords.max(axis=0) - all_lip_coords.min(axis=0))
        reasonable_depth = lip_span * 1.0
        z_min_extended = z_min - reasonable_depth
        bbox = {'x_min': x_interior_min, 'x_max': x_interior_max, 'y_min': y_interior_min, 'y_max': y_interior_max, 'z_min': z_min_extended, 'z_max': z_max}
        return bbox

    def _two_stage_interior_expansion(self, connection_points: list, vertices: np.ndarray, adj: list, complete_lip_vertices: list, mouth_interior_bbox: dict) -> set:
        """Internal helper."""
        lip_coords = vertices[complete_lip_vertices]
        lip_center = lip_coords.mean(axis=0)
        inward_axis = np.array([0, 0, -1])
        core_bbox = self._create_safe_core_bbox(lip_coords)
        stage1_vertices = self._safe_core_expansion(connection_points, vertices, adj, complete_lip_vertices, core_bbox)
        stage2_vertices = self._cautious_boundary_expansion(stage1_vertices, vertices, adj, complete_lip_vertices, mouth_interior_bbox, lip_center, inward_axis)
        total_interior = stage1_vertices | stage2_vertices
        return total_interior

    def _create_safe_core_bbox(self, lip_coords: np.ndarray) -> dict:
        """Internal helper."""
        x_min, x_max = (lip_coords[:, 0].min(), lip_coords[:, 0].max())
        y_min, y_max = (lip_coords[:, 1].min(), lip_coords[:, 1].max())
        z_min, z_max = (lip_coords[:, 2].min(), lip_coords[:, 2].max())
        y_center = (y_min + y_max) / 2
        y_range = y_max - y_min
        y_margin = y_range * 0.5
        y_core_min = y_center - y_margin / 2
        y_core_max = y_center + y_margin / 2
        x_center = (x_min + x_max) / 2
        x_range = x_max - x_min
        x_margin = x_range * 0.8
        x_core_min = x_center - x_margin / 2
        x_core_max = x_center + x_margin / 2
        lip_span = np.linalg.norm(lip_coords.max(axis=0) - lip_coords.min(axis=0))
        safe_depth = lip_span * 0.4
        z_core_min = z_min - safe_depth
        core_bbox = {'x_min': x_core_min, 'x_max': x_core_max, 'y_min': y_core_min, 'y_max': y_core_max, 'z_min': z_core_min, 'z_max': z_max}
        return core_bbox

    def _safe_core_expansion(self, connection_points: list, vertices: np.ndarray, adj: list, complete_lip_vertices: list, core_bbox: dict) -> set:
        """Internal helper."""
        interior_vertices = set()
        current_frontier = set(connection_points)
        complete_lip_set = set(complete_lip_vertices)
        for iteration in range(10):
            next_frontier = set()
            for vertex in current_frontier:
                for neighbor in adj[vertex]:
                    if neighbor in complete_lip_set or neighbor in interior_vertices:
                        continue
                    if self._is_in_bbox(vertices[neighbor], core_bbox):
                        next_frontier.add(neighbor)
            if not next_frontier:
                break
            interior_vertices.update(next_frontier)
            current_frontier = next_frontier
        return interior_vertices

    def _cautious_boundary_expansion(self, core_vertices: set, vertices: np.ndarray, adj: list, complete_lip_vertices: list, full_bbox: dict, lip_center: np.ndarray, inward_axis: np.ndarray) -> set:
        """Internal helper."""
        boundary_vertices = set()
        current_frontier = core_vertices.copy()
        complete_lip_set = set(complete_lip_vertices)
        all_processed = complete_lip_set | core_vertices
        lip_span = np.linalg.norm(vertices[list(complete_lip_vertices)].max(axis=0) - vertices[list(complete_lip_vertices)].min(axis=0))
        max_distance = lip_span * 0.8
        direction_rejected = 0
        distance_rejected = 0
        for iteration in range(15):
            next_frontier = set()
            for vertex in current_frontier:
                for neighbor in adj[vertex]:
                    if neighbor in all_processed:
                        continue
                    if not self._is_in_bbox(vertices[neighbor], full_bbox):
                        continue
                    direction_vector = vertices[neighbor] - lip_center
                    direction_norm = np.linalg.norm(direction_vector)
                    if direction_norm == 0:
                        continue
                    cos_angle = np.dot(direction_vector, inward_axis) / direction_norm
                    angle_degrees = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    if angle_degrees > 60:
                        direction_rejected += 1
                        continue
                    if direction_norm > max_distance:
                        distance_rejected += 1
                        continue
                    next_frontier.add(neighbor)
                    all_processed.add(neighbor)
            if not next_frontier:
                break
            boundary_vertices.update(next_frontier)
            current_frontier = next_frontier
        return boundary_vertices

    def _is_in_bbox(self, point: np.ndarray, bbox: dict) -> bool:
        """Internal helper."""
        x, y, z = point
        return bbox['x_min'] <= x <= bbox['x_max'] and bbox['y_min'] <= y <= bbox['y_max'] and (bbox['z_min'] <= z <= bbox['z_max'])

    def sample_points_with_binding(self, vertices: np.ndarray, faces: np.ndarray, num_points: int=50000) -> dict:
        """
        Sample points from mesh surface and record binding information.
        Returns a dictionary with points, face_indices, barycentric_coordinates, colors,
        and mouth-only indices based on current logic.
        """
        import trimesh
        num_face = num_points
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        sampled_points, face_indices = mesh.sample(num_face, return_index=True)
        triangles = mesh.vertices[mesh.faces[face_indices]]
        bary_coords = trimesh.triangles.points_to_barycentric(triangles, sampled_points)
        mouth_vertex_indices, mouth_face_indices = self.get_mouth_region_indices(vertices, faces)
        mouth_point_mask = np.isin(face_indices, mouth_face_indices)
        mouth_point_indices = np.nonzero(mouth_point_mask)[0].astype(np.int64)
        colors = np.full_like(sampled_points, 128.0)
        return {'points': sampled_points, 'colors': colors, 'binding': {'face_indices': face_indices, 'bary_coords': bary_coords, 'vertices': vertices, 'mouth_vertex_indices': mouth_vertex_indices, 'mouth_face_indices': mouth_face_indices, 'mouth_point_indices': mouth_point_indices}}

    def create_initial_mesh_data(self, data_root: str, frame_id: int=0, num_points: int=50000) -> dict:
        """
        Generate mesh data with binding information for the given frame.
        
        Generates a canonical model at real-world scale without any additional scaling.
        Scaling is now handled by camera extrinsics in dataset_readers.py.
        """
        frame_name = f'{frame_id}.npz'
        npz_path = os.path.join(data_root, frame_name)
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f'FLAME parameter file not found: {npz_path}')
        flame_params = self.load_flame_params(data_root, frame_id)
        vertices, faces = self.get_flame_mesh_for_frame(flame_params)
        geometric_center = vertices.mean(axis=0)
        gc_flat = geometric_center.flatten()
        print(f'Canonical model center: [{gc_flat[0]:.6f}, {gc_flat[1]:.6f}, {gc_flat[2]:.6f}]')
        mesh_data = self.sample_points_with_binding(vertices, faces, num_points)
        mesh_data['mesh'] = {'vertices': vertices, 'faces': faces}
        mesh_data['model_center'] = geometric_center
        print(f'Sampled {mesh_data["points"].shape[0]} FLAME-bound Gaussian points.')
        return mesh_data

    def collect_all_flame_params(self, flame_root: str, frames: list):
        """Internal helper."""
        all_params = {}
        for frame_id in frames:
            try:
                flame_params = self.load_flame_params(flame_root, frame_id)
                frame_params = {}
                for key, value in flame_params.items():
                    if isinstance(value, torch.Tensor):
                        frame_params[key] = value.cpu().numpy()
                    else:
                        frame_params[key] = value
                all_params[frame_id] = frame_params
                if frame_id % 50 == 0 or frame_id == frames[-1]:
                    print(f'Collected FLAME parameters for frame {frame_id}.')
            except Exception as e:
                print(f'Warning: failed to process frame {frame_id}: {e}')
                continue
        print(f'Collected FLAME parameters for {len(all_params)} frames.')
        return all_params

def generate_and_save_initial_mesh_data(flame_root: str, output_dir: str, flame_model: str, num_points: int=50000):
    """Internal helper."""
    processor = FlameDataProcessor(flame_model)
    frames = sorted([int(os.path.splitext(f)[0]) for f in os.listdir(flame_root) if f.endswith('.npz')])
    if not frames:
        raise RuntimeError('ERROR No valid frame (.npz) files found.')
    print(f'Found {len(frames)} FLAME parameter frames.')
    mesh_data = processor.create_initial_mesh_data(data_root=flame_root, frame_id=frames[0], num_points=num_points)
    print('Computing average model center from all frames...')
    centers_list = []
    for i, frame_id in enumerate(frames):
        try:
            flame_params = processor.load_flame_params(flame_root, frame_id)
            vertices, _ = processor.get_flame_mesh_for_frame(flame_params)
            frame_center = vertices.mean(axis=0)
            centers_list.append(frame_center)
            if (i + 1) % 50 == 0 or i == len(frames) - 1:
                print(f'Processed center for {i + 1}/{len(frames)} frames.')
        except Exception as e:
            print(f'Warning: failed to compute center for frame {frame_id}: {e}')
            continue
    if centers_list:
        average_center = np.mean(centers_list, axis=0)
        print(f'Average model center: [{average_center[0]:.6f}, {average_center[1]:.6f}, {average_center[2]:.6f}]')
        mesh_data['model_center'] = average_center
    else:
        print('Warning: using the first-frame model center as fallback.')
    print('Packing FLAME parameters...')
    all_flame_params = processor.collect_all_flame_params(flame_root, frames)
    binding_data = mesh_data['binding']
    face_indices_path = os.path.join(output_dir, 'face_indices.npy')
    bary_coords_path = os.path.join(output_dir, 'bary_coords.npy')
    vertices_path = os.path.join(output_dir, 'vertices.npy')
    mouth_point_indices_path = os.path.join(output_dir, 'mouth_point_indices.npy')
    model_center_path = os.path.join(output_dir, 'model_center.npy')
    flame_params_path = os.path.join(output_dir, 'flame_params.npz')
    np.save(face_indices_path, binding_data['face_indices'])
    np.save(bary_coords_path, binding_data['bary_coords'])
    np.save(vertices_path, binding_data['vertices'])
    np.save(mouth_point_indices_path, binding_data['mouth_point_indices'])
    np.save(model_center_path, mesh_data['model_center'])
    string_keyed_params = {str(k): v for k, v in all_flame_params.items()}
    np.savez_compressed(flame_params_path, **string_keyed_params)
    print(f'Saved face indices: {face_indices_path}')
    print(f'Saved barycentric coordinates: {bary_coords_path}')
    print(f'Saved vertices: {vertices_path}')
    print(f'Saved mouth point indices: {mouth_point_indices_path}')
    print(f'Saved model center: {model_center_path}')
    print(f'Saved FLAME parameters: {flame_params_path}')
    colors_int = mesh_data['colors'].astype(np.uint8)
    ply_path = os.path.join(output_dir, 'points3D.ply')
    storePly(ply_path, mesh_data['points'], colors_int)
    print(f'Saved initial point cloud: {ply_path}')

def main():
    parser = argparse.ArgumentParser(description='Generate initial PLY and FLAME-related parameters from processed video data')
    parser.add_argument('--flame_param_root', required=True, help='Root directory of FLAME parameters (.npz files)')
    parser.add_argument('--flame_model_path', required=True, help='Path to generic_model.pkl (FLAME model)')
    parser.add_argument('--output_root', required=True, help='Directory to save points3D.ply, params and transforms.json')
    parser.add_argument('--num_points', type=int, default=50000, help='Number of points for initial point cloud')
    parser.add_argument('--img_w', type=int, default=512, help='Image width for transforms.json')
    parser.add_argument('--img_h', type=int, default=512, help='Image height for transforms.json')
    args = parser.parse_args()
    for p in [args.flame_param_root, args.flame_model_path]:
        if not os.path.exists(p):
            sys.exit(f'Path does not exist: {p}')
    os.makedirs(args.output_root, exist_ok=True)
    print(f'Writing FLAME-Gaussian initialization to: {args.output_root}')
    generate_and_save_initial_mesh_data(flame_root=args.flame_param_root, output_dir=args.output_root, flame_model=args.flame_model_path, num_points=args.num_points)
    print('FLAME-Gaussian initialization complete.')
if __name__ == '__main__':
    main()
