"""Motion-conditioned per-gaussian deformation + appearance prior (optional,
enabled with EMOTAG_DEFORM=1).

A decoder shared across identities predicts a small per-gaussian position and
colour correction from (canonical position, GRMN motion feature, identity code).
The motion feature comes from audio+AU in both training and inference, so the
correction transfers to novel driving; adaptation fits only the low-dim identity
code. Both heads are zero-initialized: an untrained decoder is an exact no-op.
"""
import torch
import torch.nn as nn


def _positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """Axis-wise sinusoidal encoding: x [N,3] -> [N, 3 + 6*num_freqs]."""
    out = [x]
    for i in range(num_freqs):
        freq = 2.0 ** i
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    return torch.cat(out, dim=-1)


class _FiLMBlock(nn.Module):
    """Linear -> FiLM(scale,shift from code) -> SiLU."""

    def __init__(self, dim: int, id_dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.to_film = nn.Linear(id_dim, 2 * dim)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)
        self.act = nn.SiLU()

    def forward(self, h: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_film(code).chunk(2, dim=-1)
        h = self.lin(h)
        h = h * (1.0 + gamma) + beta
        return self.act(h)


class ExprDeformDecoder(nn.Module):
    """Shared motion-conditioned per-gaussian position + colour corrector.

    forward(canon_xyz [N,3], motion_feat [1,motion_dim], idx/code) ->
        (delta_xyz [N,3], delta_color [N,3] or None)
    """

    def __init__(self, id_dim: int = 32, n_id: int = 10000, motion_dim: int = 256,
                 motion_proj: int = 64, pe_L: int = 6, hidden: int = 128, n_blocks: int = 3,
                 max_delta: float = 0.01, max_color: float = 0.2, predict_color: bool = True):
        super().__init__()
        self.id_dim = id_dim
        self.motion_dim = motion_dim
        self.pe_L = pe_L
        self.max_delta = max_delta
        self.max_color = max_color
        self.predict_color = predict_color
        self.id_codes = nn.Parameter(torch.randn(n_id, id_dim) * 0.01)
        # Compress the 256-d motion feature to a compact conditioning vector.
        self.motion_proj = nn.Sequential(nn.Linear(motion_dim, motion_proj), nn.SiLU())
        in_dim = (3 + 6 * pe_L) + motion_proj
        self.stem = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([_FiLMBlock(hidden, id_dim) for _ in range(n_blocks)])
        self.xyz_head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.xyz_head.weight)
        nn.init.zeros_(self.xyz_head.bias)
        if predict_color:
            self.color_head = nn.Linear(hidden, 3)
            nn.init.zeros_(self.color_head.weight)
            nn.init.zeros_(self.color_head.bias)
        self.register_buffer('xyz_center', torch.zeros(3))
        self.register_buffer('xyz_scale', torch.full((3,), 0.15))

    def code_for(self, idx: int) -> torch.Tensor:
        return self.id_codes[idx:idx + 1]

    def forward(self, canon_xyz: torch.Tensor, motion_feat: torch.Tensor,
                idx=None, code: torch.Tensor = None):
        """canon_xyz [N,3], motion_feat [*, motion_dim] (mean-pooled internally)."""
        if motion_feat.dim() == 1:
            motion_feat = motion_feat.unsqueeze(0)
        if motion_feat.shape[0] > 1:
            motion_feat = motion_feat.mean(dim=0, keepdim=True)   # pool audio window
        m = self.motion_proj(motion_feat)                         # [1, motion_proj]
        xn = (canon_xyz - self.xyz_center) / self.xyz_scale
        pe = _positional_encoding(xn, self.pe_L)                  # [N, 3+6L]
        if code is None:
            code = self.code_for(int(idx))
        N = canon_xyz.shape[0]
        h = self.stem(torch.cat([pe, m.expand(N, -1)], dim=-1))
        for blk in self.blocks:
            h = blk(h, code)
        delta = torch.tanh(self.xyz_head(h)) * self.max_delta
        dcolor = torch.tanh(self.color_head(h)) * self.max_color if self.predict_color else None
        return delta, dcolor

    def get_params(self, lr_net: float, lr_code: float):
        net_params = [p for n, p in self.named_parameters() if n != 'id_codes']
        return [
            {'params': net_params, 'name': 'deform_net', 'lr': lr_net},
            {'params': [self.id_codes], 'name': 'deform_id_codes', 'lr': lr_code},
        ]
