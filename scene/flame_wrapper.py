import torch
import torch.nn as nn
import numpy as np
import os
import sys

class SimpleFlameWrapper(nn.Module):

    def __init__(self, flame_params_file, device='cuda', flame_model_path=None):
        super().__init__()
        self.device = device
        if flame_params_file is None:
            raise ValueError('flame_params_file is required for loading FLAME parameters')
        self.flame_params_file = flame_params_file
        self.all_flame_params = self._load_all_flame_params()
        self.flame_model = None
        self._load_flame_model(flame_model_path)

    def _load_all_flame_params(self):
        """Internal helper."""
        if not os.path.exists(self.flame_params_file):
            raise FileNotFoundError(f'FLAME parameter file not found: {self.flame_params_file}')
        print(f'Loading FLAME parameters from: {self.flame_params_file}')
        data = np.load(self.flame_params_file, allow_pickle=True)
        all_params = {}
        for key in data.files:
            try:
                frame_id = int(key)
                frame_params = data[key].item()
                all_params[frame_id] = frame_params
            except (ValueError, AttributeError) as e:
                print(f'Skipping invalid FLAME parameter entry {key}: {e}')
                continue
        print(f'Loaded FLAME parameters for {len(all_params)} frames.')
        return all_params

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
        model_path = self._resolve_flame_model_path(flame_model_path)
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
            from vhap.model.flame import FlameHead
        except ImportError as e:
            raise ImportError('Could not import VHAP FlameHead. Set EMOTAG_VHAP_ROOT or install VHAP.') from e
        self.flame_model = FlameHead(shape_params=300, expr_params=50, flame_model_path=model_path, flame_lmk_embedding_path=os.path.join(os.path.dirname(model_path), 'landmark_embedding.npy')).to(self.device)
        self.faces_tensor = self.flame_model.faces
        print(f'FLAME faces: {tuple(self.faces_tensor.shape)}')

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
