import torch
import torch.nn as nn
import numpy as np
import os
import sys
from utils.drive_sanitize import clamp_gaze, load_drive_params

N_FLAME_VERTS = 5023
N_FLAME_FACES = 9976


class SimpleFlameWrapper(nn.Module):
    """Per-frame FLAME state of one clip plus the FLAME model that turns it into a mesh."""

    def __init__(self, flame_params_file, device='cuda', flame_model_path=None, add_teeth=False):
        super().__init__()
        self.device = device
        self.teeth = bool(add_teeth)
        if flame_params_file is None:
            raise ValueError('flame_params_file is required for loading FLAME parameters')
        self.flame_params_file = flame_params_file
        self.all_flame_params = self._load_all_flame_params()
        self.flame_model = None
        self._load_flame_model(flame_model_path)

    def _load_all_flame_params(self):
        """Per-frame tracked FLAME state of the clip."""
        if not os.path.exists(self.flame_params_file):
            raise FileNotFoundError(f'FLAME parameter file not found: {self.flame_params_file}')
        print(f'Loading FLAME parameters from: {self.flame_params_file}')
        all_params, self.repaired_frames = load_drive_params(self.flame_params_file)
        print(f'Loaded FLAME parameters for {len(all_params)} frames.')
        if self.repaired_frames:
            print(f'[drive] tracking glitch at frame(s) {self.repaired_frames}: FLAME state taken from the nearest valid frame')
        return all_params

    def isolate_frames(self, frame_ids):
        """Re-load the clip with `frame_ids` (its adaptation frames) sanitised as a clip of their own, so that their."""
        self.all_flame_params, self.repaired_frames = load_drive_params(self.flame_params_file, isolate=frame_ids)

    def clamp_gaze(self, lo, hi):
        """Clamp eyes_pose of every frame to the range covered by the adaptation frames."""
        clamp_gaze(self.all_flame_params, lo, hi)

    def _resolve_flame_model_path(self, flame_model_path=None):
        if flame_model_path:
            return flame_model_path
        env_model_path = os.environ.get('EMOTAG_FLAME_MODEL')
        if env_model_path:
            return env_model_path
        vhap_root = os.environ.get('EMOTAG_VHAP_ROOT')
        if vhap_root:
            return os.path.join(vhap_root, 'asset', 'flame', 'generic_model.pkl')
        raise FileNotFoundError('FLAME model path is not configured. Pass flame_model_path, set EMOTAG_FLAME_MODEL, or set EMOTAG_VHAP_ROOT.')

    def _load_flame_model(self, flame_model_path=None):
        """Load the FLAME model used to generate mesh vertices."""
        model_path = os.path.abspath(self._resolve_flame_model_path(flame_model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'FLAME model file not found: {model_path}')
        try:
            print(f'Loading FLAME model from: {model_path}')
            self._create_simple_flame(model_path)
        except Exception as e:
            raise RuntimeError(f'Failed to load FLAME model: {e}')

    def _create_simple_flame(self, model_path):
        """Create the FLAME model through VHAP's FlameHead implementation."""
        vhap_path = os.environ.get('EMOTAG_VHAP_ROOT')
        if vhap_path and vhap_path not in sys.path:
            sys.path.insert(0, vhap_path)
        try:
            import vhap
            from vhap.model.flame import FlameHead
        except ImportError as e:
            raise ImportError('Could not import VHAP FlameHead. Set EMOTAG_VHAP_ROOT or install VHAP.') from e
        vhap_root = vhap_path or os.path.dirname(os.path.dirname(os.path.abspath(vhap.__file__)))
        lmk_path = os.path.join(os.path.dirname(model_path), 'landmark_embedding.npy')
        if not os.path.exists(lmk_path):
            lmk_path = os.path.join(os.path.dirname(model_path), 'landmark_embedding_with_eyes.npy')
        cwd = os.getcwd()
        try:
            if os.path.isdir(os.path.join(vhap_root, 'asset', 'flame')):
                os.chdir(vhap_root)
            self.flame_model = FlameHead(shape_params=300, expr_params=50, flame_model_path=model_path,
                                         flame_lmk_embedding_path=lmk_path, add_teeth=self.teeth).to(self.device)
        finally:
            os.chdir(cwd)
        self.faces_tensor = self.flame_model.faces
        print(f'FLAME faces: {tuple(self.faces_tensor.shape)}')

    @property
    def n_verts(self):
        return int(self.flame_model.v_template.shape[0])

    def canonical_vertices(self):
        """Canonical mesh of this clip, computed exactly like the scene's vertices.npy (pose of the first frame, zero."""
        p = self.load_flame_params_from_npz(min(self.all_flame_params))
        z = lambda n: torch.zeros(1, n, device=self.device)
        r = lambda k, n: p[k].reshape(1, n)
        out = self.flame_model(shape=p['shape'].reshape(1, -1), expr=z(50), rotation=r('rotation', 3), neck=r('neck_pose', 3),
                               jaw=z(3), eyes=r('eyes_pose', 6), translation=r('translation', 3), return_landmarks=False)
        v = out[0] if isinstance(out, (list, tuple)) else out
        return v.reshape(-1, 3).detach()

    def pad_canonical(self, vertices):
        """Canonical vertices stored without the teeth (scene files and checkpoints written before the teeth geometry)."""
        if vertices.shape[0] >= self.n_verts:
            return vertices
        extra = self.canonical_vertices()[vertices.shape[0]:].to(vertices.device, vertices.dtype)
        return torch.cat([vertices, extra], 0)

    def teeth_face_groups(self):
        """Face indices of the teeth geometry by part."""
        if not self.teeth:
            return {}
        faces = self.faces_tensor
        n_teeth = self.n_verts - N_FLAME_VERTS
        row = n_teeth // 8
        f = faces[N_FLAME_FACES:] - N_FLAME_VERTS
        r = torch.div(f, row, rounding_mode='floor')
        rows = {'upper_front': (0, 2), 'lower_front': (1, 3), 'upper_back': (4, 5), 'lower_back': (6, 7),
                'upper_edge': (2, 5), 'lower_edge': (3, 7)}
        groups = {}
        for name, (a, b) in rows.items():
            ok = ((r == a) | (r == b)).all(1) & (r == a).any(1) & (r == b).any(1)
            groups[name] = torch.nonzero(ok, as_tuple=False).flatten() + N_FLAME_FACES
        return groups

    def override_shape(self, shape):
        """Use one identity shape for every frame (pose/expression stay per frame)."""
        shape = np.asarray(shape.detach().cpu() if hasattr(shape, 'detach') else shape, dtype=np.float32)
        for params in self.all_flame_params.values():
            params['shape'] = shape.reshape(np.asarray(params['shape']).shape)

    def load_flame_params_from_npz(self, frame_id):
        """Internal helper."""
        if frame_id not in self.all_flame_params:
            raise KeyError(f'Frame {frame_id} is missing from {self.flame_params_file}')
        frame_data = self.all_flame_params[frame_id]
        params = {}
        for key, value in frame_data.items():
            if isinstance(value, np.ndarray):
                params[key] = torch.from_numpy(value).float().to(self.device)
            else:
                params[key] = value
        return params

    def forward(self, frame_id, motion_expression=None, motion_jaw_pose=None):
        """Internal helper."""
        flame_params = self.load_flame_params_from_npz(frame_id)
        if motion_expression is not None:
            flame_params['expression'] = motion_expression
        if motion_jaw_pose is not None:
            flame_params['jaw_pose'] = motion_jaw_pose
            flame_params['pose'] = torch.cat([flame_params['rotation'], motion_jaw_pose], dim=-1)
        batch_size = 1
        for key, value in flame_params.items():
            if isinstance(value, torch.Tensor) and value.dim() == 1:
                flame_params[key] = value.unsqueeze(0)
        shape_params = flame_params['shape']
        expression_params = flame_params['expression'][:, :50]
        flame_output = self.flame_model(shape=shape_params, expr=expression_params, rotation=flame_params['rotation'], neck=flame_params['neck_pose'], jaw=flame_params['jaw_pose'], eyes=flame_params['eyes_pose'], translation=flame_params['translation'], return_landmarks=True)
        if isinstance(flame_output, dict):
            flame_verts = flame_output['verts']
            flame_landmarks = flame_output.get('lmks')
        elif isinstance(flame_output, list) and len(flame_output) > 0:
            flame_verts = flame_output[0]
            flame_landmarks = None
        elif isinstance(flame_output, torch.Tensor):
            flame_verts = flame_output
            flame_landmarks = None
        else:
            raise TypeError(f'Unexpected FLAME output type: {type(flame_output)}')
        flame_joints = None
        return (flame_verts, flame_joints, flame_landmarks)

    def get_flame_vertices(self, frame_id, motion_expression=None, motion_jaw_pose=None):
        """Internal helper."""
        flame_verts, _, _ = self.forward(frame_id, motion_expression, motion_jaw_pose)
        return flame_verts.squeeze(0)

    def get_flame_landmarks(self, frame_id, motion_expression=None, motion_jaw_pose=None):
        """Internal helper."""
        _, _, flame_landmarks = self.forward(frame_id, motion_expression, motion_jaw_pose)
        return flame_landmarks
