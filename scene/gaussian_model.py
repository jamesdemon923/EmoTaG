import torch
import numpy as np
from typing import Dict
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.neural_renderer import GridRenderer
from scene.flame_binding import FLAMEBinding
from scene.flame_wrapper import SimpleFlameWrapper
from scene.flame_gaussian_model import FLAMEGaussianModel
class GaussianModel:

    def setup_functions(self):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        self.scaling_activation = torch.nn.functional.softplus
        self.scaling_inverse_activation = lambda x: x + torch.log(-torch.expm1(-x))
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, args):
        self.args = args
        self.active_sh_degree = 0
        self.max_sh_degree = args.sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.neural_renderer = None
        self.neural_motion_grid = None
        self.flame_binding = None
        self.flame_wrapper = None
        self.flame_gaussian_model = None
        self.setup_functions()
        self.mouth_mask = None
        self.mouth_indices = None
        self.freeze_mouth_set = False
        self.mouth_budget = None
        self.intraoral_mode = 'audio'
        self.intraoral_min_opacity = 0.05

    @property
    def intraoral_rigid(self):
        return self.intraoral_mode == 'rigid'

    def _mouth_growth_blocked(self):
        if not self.freeze_mouth_set or self.mouth_mask is None:
            return False
        if self.mouth_budget is None:
            return True
        return int(self.mouth_mask.sum().item()) >= int(self.mouth_budget)

    def set_mouth_mask(self, mask: torch.Tensor, indices: torch.Tensor=None):
        """Set the mouth subset for this GaussianModel."""
        if mask is None:
            self.mouth_mask = None
            self.mouth_indices = None
            return
        if not isinstance(mask, torch.Tensor):
            raise TypeError('mouth mask must be a torch.Tensor')
        if mask.dtype != torch.bool:
            raise ValueError('mouth mask must be a bool tensor')
        if mask.dim() != 1:
            raise ValueError('mouth mask must be 1-D [N]')
        if self.get_xyz.numel() > 0 and mask.shape[0] != self.get_xyz.shape[0]:
            raise ValueError(f'mouth mask length {mask.shape[0]} does not match Gaussian count {self.get_xyz.shape[0]}')
        self.mouth_mask = mask.to(self._xyz.device)
        if indices is not None:
            if indices.dtype != torch.long:
                raise ValueError('mouth indices must be LongTensor')
            self.mouth_indices = indices.to(self._xyz.device)
        else:
            self.mouth_indices = self._recompute_mouth_indices_from_mask()

    def _recompute_mouth_indices_from_mask(self) -> torch.Tensor:
        if self.mouth_mask is None:
            return None
        if self.mouth_mask.dim() != 1:
            raise ValueError('mouth mask must be 1-D to compute indices')
        if self.mouth_mask.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self._xyz.device)
        return torch.nonzero(self.mouth_mask, as_tuple=False).flatten().long().to(self._xyz.device)

    def capture(self):
        binding_state = None
        if self.flame_binding is not None:
            binding_state = (self.flame_binding.face_indices.detach().cpu(),
                             self.flame_binding.bary_coords.detach().cpu())
        mouth_state = self.mouth_mask.detach().cpu() if self.mouth_mask is not None else None
        identity_state = None
        if self.flame_binding is not None:
            identity_state = {'canonical_vertices': self.flame_binding.canonical_mesh_vertices.detach().cpu()}
            shape = getattr(self, 'identity_flame_shape', None)
            if shape is None and getattr(self, 'flame_wrapper', None) is not None and self.flame_wrapper.all_flame_params:
                first = self.flame_wrapper.all_flame_params[min(self.flame_wrapper.all_flame_params)]
                shape = torch.as_tensor(first['shape']).float().reshape(1, -1)
            identity_state['flame_shape'] = shape.detach().cpu() if shape is not None else None
            identity_state['intraoral_mode'] = self.intraoral_mode
        return (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling, self._rotation, self._opacity, self.max_radii2D, self.xyz_gradient_accum, self.denom, self.optimizer.state_dict(), self.spatial_lr_scale, self.neural_renderer.state_dict(), self.neural_motion_grid.state_dict() if self.neural_motion_grid is not None else None, binding_state, mouth_state, identity_state)

    @staticmethod
    def checkpoint_intraoral_mode(model_args):
        """'audio' or 'rigid'."""
        identity_state = model_args[16] if len(model_args) > 16 else None
        return (identity_state or {}).get('intraoral_mode', 'audio')

    def restore(self, model_args, training_args):
        (self.active_sh_degree, self._xyz, self._features_dc, self._features_rest, self._scaling,
         self._rotation, self._opacity, self.max_radii2D, xyz_gradient_accum, denom, opt_dict,
         self.spatial_lr_scale, neural_renderer_state, neural_motion_grid_state, *rest) = model_args
        if neural_renderer_state is not None:
            self.neural_renderer = GridRenderer()
            self.neural_renderer.recover_from_ckpt(neural_renderer_state)
            self.neural_renderer.cuda()
        if training_args is not None:
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        binding_state = rest[0] if len(rest) > 0 else None
        if binding_state is not None and self.flame_binding is not None:
            face_indices, bary_coords = binding_state
            dev = self.flame_binding.canonical_mesh_vertices.device
            self.flame_binding.face_indices = face_indices.long().to(dev)
            self.flame_binding.bary_coords = bary_coords.float().to(dev)
            self.flame_binding._update_canonical_points()
        elif self.flame_binding is not None and self._xyz.shape[0] != self.flame_binding.face_indices.shape[0]:
            print(f"[restore] WARNING: gaussian count ({self._xyz.shape[0]}) != binding count "
                  f"({self.flame_binding.face_indices.shape[0]}); checkpoint predates the binding fix "
                  f"and cannot be rendered. Re-train to produce a compatible checkpoint.")
        mouth_state = rest[1] if len(rest) > 1 else None
        if mouth_state is not None:
            self.mouth_mask = mouth_state.bool().to(self._xyz.device)
            self.mouth_indices = self._recompute_mouth_indices_from_mask()
        identity_state = rest[2] if len(rest) > 2 else None
        self.restored_identity = identity_state
        self.intraoral_mode = self.checkpoint_intraoral_mode(model_args)
        if self.intraoral_rigid and getattr(self, 'flame_wrapper', None) is not None and not self.flame_wrapper.teeth:
            raise RuntimeError('checkpoint with a rigid intra-oral set, but the FLAME mesh was built without teeth: '
                               'build SimpleFlameWrapper with add_teeth=True')

    def apply_identity(self, identity_state=None, with_shape=False):
        """Re-bind to the adaptation identity's canonical mesh."""
        identity_state = identity_state if identity_state is not None else getattr(self, 'restored_identity', None)
        if not identity_state or self.flame_binding is None:
            return False
        b = self.flame_binding
        can = identity_state.get('canonical_vertices')
        if can is not None:
            can = can.float().to(b.canonical_mesh_vertices.device)
            if getattr(self, 'flame_wrapper', None) is not None:
                can = self.flame_wrapper.pad_canonical(can)
            b.canonical_mesh_vertices = can
            b._canonical_frames = None
            b._update_canonical_points()
        shape = identity_state.get('flame_shape')
        if with_shape and shape is not None and getattr(self, 'flame_wrapper', None) is not None:
            self.flame_wrapper.override_shape(shape)
            self.identity_flame_shape = shape
        return True

    def get_deformed_gaussians(self, flame_verts, flame_pose=None):
        """Internal helper."""
        if self.flame_gaussian_model is None:
            raise RuntimeError('FLAME-Gaussian binding has not been initialized.')
        return self.flame_gaussian_model.get_deformed_gaussians(flame_verts, flame_pose)

    @property
    def get_scaling(self):
        from scene.flame_gaussian_model import binding_mode
        if self.flame_binding is None or binding_mode() == 'hybrid':
            return self.scaling_activation(self._scaling)
        _, _, scaling = self._canonical_frame_per_point()
        return self.scaling_activation(self._scaling) * scaling

    def _canonical_frame_per_point(self):
        center, orien, scaling = self.flame_binding.canonical_face_frames()
        fi = self.flame_binding.face_indices
        return center[fi], orien[fi], scaling[fi]

    @property
    def get_rotation(self):
        from scene.flame_gaussian_model import binding_mode, matrix_to_quaternion, quaternion_multiply
        if self.flame_binding is None or binding_mode() == 'hybrid':
            return self.rotation_activation(self._rotation)
        _, orien, _ = self._canonical_frame_per_point()
        face_quat = self.rotation_activation(matrix_to_quaternion(orien))
        return self.rotation_activation(quaternion_multiply(face_quat, self.rotation_activation(self._rotation)))

    @property
    def get_xyz(self):
        from scene.flame_gaussian_model import binding_mode
        if self.flame_binding is None:
            return self._xyz
        if binding_mode() == 'hybrid':
            return self.get_canonical_xyz
        return self.get_canonical_xyz

    @property
    def get_canonical_xyz(self):
        from scene.flame_gaussian_model import binding_mode
        if binding_mode() == 'hybrid':
            return self.flame_binding.get_canonical_points() + self._xyz
        center, orien, scaling = self._canonical_frame_per_point()
        rotated = torch.bmm(orien, self._xyz.unsqueeze(-1)).squeeze(-1)
        return rotated * scaling + center

    def world_to_local_xyz(self, world_xyz, face_idx=None):
        """Convert world positions into the parent triangles' local coordinates."""
        center, orien, scaling = self.flame_binding.canonical_face_frames()
        fi = self.flame_binding.face_indices if face_idx is None else face_idx
        c, R, k = center[fi], orien[fi], scaling[fi]
        return torch.bmm(R.transpose(1, 2), ((world_xyz - c) / (k + 1e-12)).unsqueeze(-1)).squeeze(-1)

    def scale_world_to_local_log(self, world_scale, face_idx=None):
        """Convert world-space scales into local log-scales (_scaling convention)."""
        _, _, scaling = self.flame_binding.canonical_face_frames()
        fi = self.flame_binding.face_indices if face_idx is None else face_idx
        return self.scaling_inverse_activation(world_scale / (scaling[fi] + 1e-12))

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, binding_data: Dict=None):
        """Creates Gaussians from a PLY point cloud and, optionally, binding data."""
        self.spatial_lr_scale = spatial_lr_scale
        if binding_data is not None:
            print('OK Binding data found. Initializing Gaussians with pre-computed binding.')
            self._create_from_pcd_with_binding(pcd, binding_data)
        else:
            print('ERROR Fatal: no binding data found. Please regenerate mesh with binding information.')
            sys.exit(1)

    def _create_from_pcd_with_binding(self, pcd: BasicPointCloud, binding_data: Dict):
        """Initializes Gaussians and their FLAME binding using pre-computed data."""
        face_indices = torch.tensor(binding_data['face_indices']).cuda()
        bary_coords = torch.tensor(binding_data['bary_coords']).float().cuda()
        canonical_vertices = self.flame_wrapper.pad_canonical(torch.tensor(binding_data['vertices']).float().cuda())
        fused_color = RGB2SH(torch.tensor(pcd.colors, dtype=torch.float, device='cuda'))
        features = torch.zeros((len(pcd.points), 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        dist2 = torch.clamp_min(distCUDA2(torch.tensor(pcd.points, dtype=torch.float, device='cuda')), 1e-07)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((len(pcd.points), 4), device='cuda')
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.99 * torch.ones((len(pcd.points), 1), dtype=torch.float, device='cuda'))
        self._xyz = nn.Parameter(torch.zeros_like(torch.tensor(pcd.points), device='cuda').requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(self._xyz.shape[0], device='cuda')
        self.neural_renderer = GridRenderer(bound=(torch.tensor(pcd.points).max(0).values - torch.tensor(pcd.points).min(0).values).max() / 2 * 1.2, coord_center=torch.tensor(pcd.points).mean(0)).cuda()
        self._init_flame_binding(face_indices, bary_coords, canonical_vertices)
        from scene.flame_gaussian_model import binding_mode
        if binding_mode() == 'local':
            with torch.no_grad():
                bary_points = self.flame_binding.get_canonical_points()
                self._xyz.data = self.world_to_local_xyz(bary_points)
                _, _, k_pt = self._canonical_frame_per_point()
                self._scaling.data = self._scaling.data - torch.log(k_pt + 1e-12)

    @torch.no_grad()
    def add_intraoral_gaussians(self, n_teeth=2000, n_cavity=500, budget=4000, drop_mask=None, seed=0):
        """Rigid intra-oral set."""
        from scene.flame_gaussian_model import binding_mode
        fw, b = self.flame_wrapper, self.flame_binding
        groups = fw.teeth_face_groups()
        if not groups:
            raise RuntimeError('the FLAME mesh has no teeth geometry: build SimpleFlameWrapper with add_teeth=True')
        dev = self._xyz.device
        if drop_mask is not None and bool(drop_mask.any()):
            keep = ~drop_mask.to(dev)
            for name in ('_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity'):
                setattr(self, name, nn.Parameter(getattr(self, name).detach()[keep].requires_grad_(True)))
            b.update_binding('prune', ~keep)
        g = torch.Generator(device='cpu').manual_seed(seed)
        verts, faces = b.canonical_mesh_vertices, b.faces.long()

        def sample(fids, n):
            tri = verts[faces[fids]]
            area = 0.5 * torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
            pick = torch.multinomial(area.cpu() / area.sum().cpu(), n, replacement=True, generator=g).to(dev)
            u = torch.rand(n, 2, generator=g).to(dev)
            su = u[:, 0].sqrt()
            bary = torch.stack([1 - su, su * (1 - u[:, 1]), su * u[:, 1]], 1)
            return fids[pick], bary, float(area.sum())

        def frame(prefix):
            """(back, down) unit vectors of one jaw half."""
            c = lambda k: verts[faces[groups[prefix + k]]].reshape(-1, 3).mean(0)
            back = torch.nn.functional.normalize(c('_back') - c('_front'), dim=0)
            pts = verts[faces[groups[prefix + '_front']]].reshape(-1, 3)
            edge_c = verts[faces[groups[prefix + '_edge']]].reshape(-1, 3).mean(0)
            down = torch.nn.functional.normalize(edge_c - pts.mean(0), dim=0)
            lvl = (pts - pts.mean(0)) @ down
            return back, down, float(lvl.max() - lvl.min()) + 1e-6

        parts = []
        n_front = int(n_teeth * 0.4)
        n_edge = (n_teeth - 2 * n_front) // 2
        for prefix in ('upper', 'lower'):
            for part, n in (('_front', n_front), ('_edge', n_edge)):
                fi, bary, area = sample(groups[prefix + part], n)
                sigma = 0.5 * (area / n) ** 0.5
                parts.append((fi, bary, torch.zeros(n, 3, device=dev), sigma, (0.80, 0.78, 0.72), 0.5))
        n_up = int(n_cavity * 0.6)
        for prefix, n in (('upper', n_up), ('lower', n_cavity - n_up)):
            back, down, h = frame(prefix)
            fi, bary, _ = sample(groups[prefix + '_back'], n)
            u = torch.rand(n, 2, generator=g).to(dev)
            anchor = (verts[faces[fi]] * bary[:, :, None]).sum(1)
            lvl = ((anchor - anchor.mean(0)) @ down)
            want = (u[:, 0] * (2.5 if prefix == 'upper' else 1.0) - 0.5) * h
            off = back[None] * (0.004 + 0.012 * u[:, 1:2]) + down[None] * (want - lvl)[:, None]
            parts.append((fi, bary, off, 0.002, (0.25, 0.10, 0.10), 0.8))
        fi = torch.cat([p_[0] for p_ in parts]); bary = torch.cat([p_[1] for p_ in parts])
        off = torch.cat([p_[2] for p_ in parts])
        sig = torch.cat([torch.full((len(p_[0]), 3), float(p_[3]), device=dev) for p_ in parts])
        rgb = torch.cat([torch.tensor([p_[4]], device=dev).repeat(len(p_[0]), 1) for p_ in parts])
        opa = torch.cat([torch.full((len(p_[0]), 1), float(p_[5]), device=dev) for p_ in parts])
        n_old, n = self._xyz.shape[0], len(fi)
        b.append(fi, bary)
        if binding_mode() == 'local':
            anchor = (verts[faces[fi]] * bary[:, :, None]).sum(1)
            xyz = self.world_to_local_xyz(anchor + off, face_idx=fi)
            scl = self.scale_world_to_local_log(sig, face_idx=fi)
        else:
            xyz, scl = off, self.scaling_inverse_activation(sig)
        f_dc = RGB2SH(rgb)[:, None, :]
        f_rest = torch.zeros((n, self._features_rest.shape[1], 3), device=dev)
        rot = torch.zeros((n, 4), device=dev); rot[:, 0] = 1
        cat = lambda a, x: nn.Parameter(torch.cat([a.detach(), x.to(a)], 0).requires_grad_(True))
        self._xyz, self._scaling, self._rotation = cat(self._xyz, xyz), cat(self._scaling, scl), cat(self._rotation, rot)
        self._features_dc, self._features_rest = cat(self._features_dc, f_dc), cat(self._features_rest, f_rest)
        self._opacity = cat(self._opacity, inverse_sigmoid(opa))
        self.max_radii2D = torch.zeros(self._xyz.shape[0], device=dev)
        mask = torch.zeros(n_old + n, dtype=torch.bool, device=dev)
        mask[n_old:] = True
        self.mouth_mask = mask
        self.mouth_indices = self._recompute_mouth_indices_from_mask()
        self.intraoral_mode = 'rigid'
        self.freeze_mouth_set = True
        self.mouth_budget = int(budget)
        print(f'[intraoral] rigid set: {n} gaussians on {int(sum(len(v) for v in groups.values()))} teeth faces '
              f'({2 * n_front} front, {2 * n_edge} edge, {n_cavity} cavity backing); budget {budget}; total {n_old + n}')
        return n

    def _init_flame_binding(self, face_indices, bary_coords, canonical_vertices):
        """Initializes the FLAME binding system with pre-computed binding info."""
        try:
            if self.flame_wrapper is None:
                raise ValueError('FLAME Wrapper has not been assigned to GaussianModel before binding.')
            self.flame_binding = FLAMEBinding(flame_model=self.flame_wrapper, face_indices=face_indices, bary_coords=bary_coords, canonical_vertices=canonical_vertices, device='cuda')
            self.flame_gaussian_model = FLAMEGaussianModel(self)
            print('OK FLAME Gaussian Model initialized successfully!')
        except Exception as e:
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self._xyz.shape[0], 1), device='cuda')
        l = [{'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, 'name': 'xyz'}, {'params': [self._features_dc], 'lr': training_args.feature_lr, 'name': 'f_dc'}, {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, 'name': 'f_rest'}, {'params': [self._opacity], 'lr': training_args.opacity_lr, 'name': 'opacity'}, {'params': [self._scaling], 'lr': training_args.scaling_lr, 'name': 'scaling'}, {'params': [self._rotation], 'lr': training_args.rotation_lr, 'name': 'rotation'}]
        l += self.neural_renderer.get_params(lr=0.005, lr_net=0.0005)
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale, lr_final=training_args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=training_args.position_lr_delay_mult, max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step."""
        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'xyz':
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_simple_ply(self, points, path):
        """Saves a simple point cloud to a PLY file."""
        from plyfile import PlyData, PlyElement
        import numpy as np
        points_np = points.detach().cpu().numpy()
        vertex_data = np.empty(len(points_np), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_data['x'] = points_np[:, 0]
        vertex_data['y'] = points_np[:, 1]
        vertex_data['z'] = points_np[:, 2]
        vert_element = PlyElement.describe(vertex_data, 'vertex')
        PlyData([vert_element]).write(path)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_deformed_ply(self, deformed_xyz, deformed_rotations, deformed_raw_scale, path):
        """Internal helper."""
        mkdir_p(os.path.dirname(path))
        xyz = deformed_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = deformed_raw_scale.detach().cpu().numpy()
        rotation = deformed_rotations.detach().cpu().numpy()
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        if self.intraoral_rigid and self.mouth_mask is not None:
            opacities_new = torch.where(self.mouth_mask[:, None], self._opacity.detach(), opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, 'opacity')
        self._opacity = optimizable_tensors['opacity']

    def load_ply(self, path):
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]['x']), np.asarray(plydata.elements[0]['y']), np.asarray(plydata.elements[0]['z'])), axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device='cuda').requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device='cuda').requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device='cuda').requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device='cuda').requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state['exp_avg'] = torch.zeros_like(tensor)
                stored_state['exp_avg_sq'] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'neural' in group['name']:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = stored_state['exp_avg'][mask]
                stored_state['exp_avg_sq'] = stored_state['exp_avg_sq'][mask]
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(group['params'][0][mask].requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        if self.flame_binding is not None:
            self.flame_binding.update_binding('prune', mask)
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.mouth_mask is not None:
            if self.mouth_mask.shape[0] != valid_points_mask.shape[0]:
                raise ValueError('mouth_mask length mismatch during prune')
            self.mouth_mask = self.mouth_mask[valid_points_mask]
            self.mouth_indices = self._recompute_mouth_indices_from_mask()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'neural' in group['name']:
                continue
            assert len(group['params']) == 1
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state['exp_avg'] = torch.cat((stored_state['exp_avg'], torch.zeros_like(extension_tensor)), dim=0)
                stored_state['exp_avg_sq'] = torch.cat((stored_state['exp_avg_sq'], torch.zeros_like(extension_tensor)), dim=0)
                del self.optimizer.state[group['params'][0]]
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group['name']] = group['params'][0]
            else:
                group['params'][0] = nn.Parameter(torch.cat((group['params'][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group['name']] = group['params'][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {'xyz': new_xyz, 'f_dc': new_features_dc, 'f_rest': new_features_rest, 'opacity': new_opacities, 'scaling': new_scaling, 'rotation': new_rotation}
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors['xyz']
        self._features_dc = optimizable_tensors['f_dc']
        self._features_rest = optimizable_tensors['f_rest']
        self._opacity = optimizable_tensors['opacity']
        self._scaling = optimizable_tensors['scaling']
        self._rotation = optimizable_tensors['rotation']
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self._xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros(self._xyz.shape[0], device='cuda')

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self._xyz.shape[0]
        padded_grad = torch.zeros(n_init_points, device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        if self._mouth_growth_blocked():
            selected_pts_mask = torch.logical_and(selected_pts_mask, ~self.mouth_mask)
        from scene.flame_gaussian_model import binding_mode
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation[selected_pts_mask]).repeat(N, 1, 1)
        sampled_world = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if binding_mode() == 'hybrid':
            parent_anchor = self.flame_binding.get_canonical_points()[selected_pts_mask].repeat(N, 1)
            new_xyz = sampled_world - parent_anchor
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        else:
            parent_fi = self.flame_binding.face_indices[selected_pts_mask].repeat(N)
            new_xyz = self.world_to_local_xyz(sampled_world, face_idx=parent_fi)
            new_scaling = self.scale_world_to_local_log(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N), face_idx=parent_fi)
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        if self.mouth_mask is not None:
            parents_mouth = self.mouth_mask[selected_pts_mask]
            if parents_mouth.numel() > 0:
                extension = parents_mouth.repeat(N)
                self.mouth_mask = torch.cat([self.mouth_mask, extension], dim=0)
        if self.flame_binding is not None:
            self.flame_binding.update_binding('split', selected_pts_mask, N=N)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device='cuda', dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        if self._mouth_growth_blocked():
            selected_pts_mask = torch.logical_and(selected_pts_mask, ~self.mouth_mask)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        if self.mouth_mask is not None:
            parents_mouth = self.mouth_mask[selected_pts_mask]
            if parents_mouth.numel() > 0:
                self.mouth_mask = torch.cat([self.mouth_mask, parents_mouth], dim=0)
        if self.flame_binding is not None:
            self.flame_binding.update_binding('clone', selected_pts_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if self.intraoral_rigid and self.mouth_mask is not None:
            dead = (self.get_opacity < self.intraoral_min_opacity).squeeze()
            if max_screen_size:
                dead = torch.logical_or(dead, big_points_ws)
            prune_mask = torch.where(self.mouth_mask, dead, prune_mask)
        elif self.freeze_mouth_set and self.mouth_mask is not None:
            prune_mask = torch.logical_and(prune_mask, ~self.mouth_mask)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
