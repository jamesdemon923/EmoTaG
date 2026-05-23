import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend
_gridtype_to_id = {'hash': 0, 'tiled': 1}
_interp_to_id = {'linear': 0, 'smoothstep': 1}

class _grid_encode(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets, per_level_scale, base_resolution, calc_grad_inputs=False, gridtype=0, align_corners=False, interpolation=0):
        inputs = inputs.contiguous()
        B, D = inputs.shape
        L = offsets.shape[0] - 1
        C = embeddings.shape[1]
        S = np.log2(per_level_scale)
        H = base_resolution
        if torch.is_autocast_enabled() and C % 2 == 0:
            embeddings = embeddings.to(torch.half)
        outputs = torch.empty(L, B, C, device=inputs.device, dtype=embeddings.dtype)
        if calc_grad_inputs:
            dy_dx = torch.empty(B, L * D * C, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None
        _backend.grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, gridtype, align_corners, interpolation)
        outputs = outputs.permute(1, 0, 2).reshape(B, L * C)
        ctx.save_for_backward(inputs, embeddings, offsets, dy_dx)
        ctx.dims = [B, D, C, L, S, H, gridtype, interpolation]
        ctx.align_corners = align_corners
        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, embeddings, offsets, dy_dx = ctx.saved_tensors
        B, D, C, L, S, H, gridtype, interpolation = ctx.dims
        align_corners = ctx.align_corners
        grad = grad.view(B, L, C).permute(1, 0, 2).contiguous()
        grad_embeddings = torch.zeros_like(embeddings)
        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None
        _backend.grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation)
        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)
        return (grad_inputs, grad_embeddings, None, None, None, None, None, None, None)
grid_encode = _grid_encode.apply

class GridEncoder(nn.Module):

    def __init__(self, input_dim=3, num_levels=16, level_dim=2, per_level_scale=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=None, gridtype='hash', align_corners=False, interpolation='linear'):
        super().__init__()
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.per_level_scale = per_level_scale
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.output_dim = num_levels * level_dim
        self.gridtype = gridtype
        self.gridtype_id = _gridtype_to_id[gridtype]
        self.interpolation = interpolation
        self.interp_id = _interp_to_id[interpolation]
        self.align_corners = align_corners
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(num_levels):
            resolution = int(np.ceil(base_resolution * per_level_scale ** i))
            params_in_level = min(self.max_params, (resolution if align_corners else resolution + 1) ** input_dim)
            params_in_level = int(np.ceil(params_in_level / 8) * 8)
            offsets.append(offset)
            offset += params_in_level
        offsets.append(offset)
        offsets = torch.from_numpy(np.array(offsets, dtype=np.int32))
        self.register_buffer('offsets', offsets)
        self.n_params = offsets[-1] * level_dim
        self.embeddings = nn.Parameter(torch.empty(offset, level_dim))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.0001
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f'GridEncoder(input_dim={self.input_dim}, levels={self.num_levels}, level_dim={self.level_dim}, params={tuple(self.embeddings.shape)})'

    def forward(self, inputs, bound=1):
        inputs = (inputs + bound) / (2 * bound)
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
        outputs = grid_encode(inputs, self.embeddings, self.offsets, self.per_level_scale, self.base_resolution, inputs.requires_grad, self.gridtype_id, self.align_corners, self.interp_id)
        outputs = outputs.view(prefix_shape + [self.output_dim])
        return outputs

    @torch.cuda.amp.autocast(enabled=False)
    def grad_total_variation(self, weight=1e-07, inputs=None, bound=1, B=1000000):
        D = self.input_dim
        C = self.embeddings.shape[1]
        L = self.offsets.shape[0] - 1
        S = np.log2(self.per_level_scale)
        H = self.base_resolution
        if inputs is None:
            inputs = torch.rand(B, self.input_dim, device=self.embeddings.device)
        else:
            inputs = (inputs + bound) / (2 * bound)
            inputs = inputs.view(-1, self.input_dim)
            B = inputs.shape[0]
        if self.embeddings.grad is None:
            raise ValueError('grad is None, should be called after loss.backward() and before optimizer.step()!')
        _backend.grad_total_variation(inputs, self.embeddings, self.embeddings.grad, self.offsets, weight, B, D, C, L, S, H, self.gridtype_id, self.align_corners)
