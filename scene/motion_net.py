import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from encoding import get_encoder

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float=0.0, max_len: int=1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))
        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x

class MotionNetwork(nn.Module):

    def __init__(self, audio_dim=256, ind_dim=0, args=None, num_speakers: int=None):
        super(MotionNetwork, self).__init__()
        if args.audio_extractor != 'wav2vec2':
            raise ValueError('EmoTaG expects --audio_extractor wav2vec2.')
        self.audio_in_dim = 768
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1)
        self.audio_dim = audio_dim
        self.window_size = 16
        self.num_layers = 4
        self.hidden_dim = 256
        self.audio_proj = nn.Linear(self.audio_in_dim, self.audio_dim)
        self.pos_encoder = PositionalEncoding(self.audio_dim, dropout=0.1, max_len=1024)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.audio_dim, nhead=8, dim_feedforward=self.audio_dim * 4, batch_first=True)
        self.audio_transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.au_dim = 6
        self.au_hidden_dim = 64
        self.au_encoder = nn.Sequential(nn.Linear(self.au_dim, self.au_hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.au_hidden_dim, self.au_hidden_dim), nn.ReLU(inplace=True))
        self.au_fusion = nn.Linear(self.au_hidden_dim, self.hidden_dim)
        for module in self.au_encoder:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.au_fusion.weight)
        nn.init.zeros_(self.au_fusion.bias)
        # AdaIN modulation of the audio and AU/expression streams, driven by the
        # AdaFace identity descriptor `s`.
        self.identity_dim = 512
        self.audio_adain = nn.Sequential(nn.Linear(self.identity_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, 2 * self.audio_dim))
        self.au_adain = nn.Sequential(nn.Linear(self.identity_dim, self.hidden_dim), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim, 2 * self.au_hidden_dim))
        for adain in (self.audio_adain, self.au_adain):
            last = adain[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.bound = 0.15
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.in_dim_pos = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz
        self.flame_exp_dim = 50
        self.flame_jaw_dim = 3
        self.backbone_net = MLP(self.audio_dim + self.individual_dim, self.hidden_dim, self.hidden_dim, self.num_layers)
        self.flame_exp_head = nn.Linear(self.hidden_dim, self.flame_exp_dim)
        self.flame_jaw_head = nn.Linear(self.hidden_dim, self.flame_jaw_dim)
        nn.init.normal_(self.flame_exp_head.weight, 0, 0.01)
        nn.init.zeros_(self.flame_exp_head.bias)
        nn.init.normal_(self.flame_jaw_head.weight, 0, 0.01)
        nn.init.zeros_(self.flame_jaw_head.bias)
        self.motion_dim = self.flame_exp_dim + self.flame_jaw_dim
        self.residual_net = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)
        self.residual_motion_head = nn.Linear(self.hidden_dim, self.motion_dim)
        nn.init.normal_(self.residual_motion_head.weight, 0, 0.005)
        nn.init.zeros_(self.residual_motion_head.bias)
        self.gate_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(self.hidden_dim // 2, 1))
        nn.init.normal_(self.gate_net[-1].weight, 0, 0.01)
        nn.init.zeros_(self.gate_net[-1].bias)
        self.emotion_head = nn.Linear(self.hidden_dim, 7)
        nn.init.normal_(self.emotion_head.weight, 0, 0.01)
        nn.init.zeros_(self.emotion_head.bias)
        self.cache = None
        mouth_in_dim = self.in_dim_pos + self.audio_dim + self.individual_dim
        self.mouth_hidden_dim = 64
        self.mouth_mlp = MLP(mouth_in_dim, self.mouth_hidden_dim, self.mouth_hidden_dim, 3)
        self.mouth_head = nn.Linear(self.mouth_hidden_dim, 10)
        nn.init.normal_(self.mouth_head.weight, 0, 0.01)
        nn.init.zeros_(self.mouth_head.bias)

    def encode_audio(self, a, return_logits: bool=False):
        if a is None:
            return None
        if a.dim() != 3:
            raise ValueError(f'Audio features must have shape [B, T, C] or [B, C, T], got {tuple(a.shape)}.')
        batch, dim1, dim2 = a.shape
        if dim1 == self.window_size and dim2 == self.audio_in_dim:
            seq = a
        elif dim1 == self.audio_in_dim and dim2 == self.window_size:
            seq = a.permute(0, 2, 1)
        elif dim1 == self.audio_in_dim:
            seq = a.permute(0, 2, 1)
        else:
            seq = a
        proj = self.audio_proj(seq)
        proj = self.pos_encoder(proj)
        encoded_seq = self.audio_transformer(proj)
        audio_tokens = encoded_seq
        pooled = audio_tokens.mean(dim=1)
        if return_logits:
            return (pooled, None)
        return pooled

    @staticmethod
    def _prepare_identity(identity, device=None, dtype=None):
        if identity is None:
            return None
        identity = torch.as_tensor(identity, device=device, dtype=dtype)
        if identity.dim() == 1:
            identity = identity.unsqueeze(0)
        return identity

    def _adain(self, feat: torch.Tensor, identity: torch.Tensor, mlp: nn.Module) -> torch.Tensor:
        """Adaptive Instance Normalization conditioned on the identity descriptor."""
        params = mlp(identity)
        gamma, beta = params.chunk(2, dim=-1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True) + 1e-05
        normed = (feat - mean) / std
        return (1.0 + gamma) * normed + beta

    def encode_au(self, au: torch.Tensor=None, batch_size: int=1, device=None, dtype=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if au is None:
            raise ValueError('AU features are required for EmoTaG motion generation. Expected tensor shape [B, 6] or [6] with columns [AU01, AU04, AU05, AU06, AU07, AU45].')
        au = torch.as_tensor(au, device=device, dtype=dtype)
        if au.dim() == 1:
            au = au.unsqueeze(0)
        elif au.dim() > 2:
            au = au.reshape(au.shape[0], -1)
        if au.shape[-1] != self.au_dim:
            raise ValueError(f'AU features must have {self.au_dim} channels, got {au.shape[-1]}.')
        if au.shape[0] == 1 and batch_size > 1:
            au = au.repeat(batch_size, 1)
        elif au.shape[0] != batch_size:
            au = au[:1].repeat(batch_size, 1)
        return self.au_encoder(au)

    @staticmethod
    @torch.jit.script
    def _split_xyz(x):
        xy, yz, xz = (x[:, :-1], x[:, 1:], torch.cat([x[:, :1], x[:, -1:]], dim=-1))
        return (xy, yz, xz)

    def encode_x(self, xyz: torch.Tensor, bound: float=None) -> torch.Tensor:
        if bound is None:
            bound = self.bound
        xy, yz, xz = self._split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    def decode_motion(self, aud_feat: torch.Tensor, c: torch.Tensor=None, au: torch.Tensor=None, identity: torch.Tensor=None) -> dict:
        identity = self._prepare_identity(identity, device=aud_feat.device, dtype=aud_feat.dtype)
        if identity is not None:
            aud_feat = self._adain(aud_feat, identity, self.audio_adain)
        h = aud_feat
        if self.individual_dim > 0 and c is not None:
            h = torch.cat([h, c], dim=-1)
        shared = self.backbone_net(h)
        au_feat = self.encode_au(au, batch_size=shared.shape[0], device=shared.device, dtype=shared.dtype)
        if identity is not None:
            au_feat = self._adain(au_feat, identity, self.au_adain)
        shared = shared + self.au_fusion(au_feat)
        base_exp = self.flame_exp_head(shared)
        base_jaw = self.flame_jaw_head(shared)
        base_motion = torch.cat([base_exp, base_jaw], dim=-1)
        residual_hidden = self.residual_net(shared)
        residual_motion = self.residual_motion_head(residual_hidden)
        gate = torch.sigmoid(self.gate_net(shared))
        motion = base_motion + gate * residual_motion
        pred_exp = motion[..., :self.flame_exp_dim]
        pred_jaw = motion[..., self.flame_exp_dim:]
        return {'flame_exp': pred_exp, 'flame_jaw': pred_jaw, 'encoded_motion': shared, 'base_flame_exp': base_exp, 'base_flame_jaw': base_jaw, 'residual_flame_exp': residual_motion[..., :self.flame_exp_dim], 'residual_flame_jaw': residual_motion[..., self.flame_exp_dim:], 'gate': gate, 'emotion_logits': self.emotion_head(shared)}

    def predict_jaw_from_audio(self, a: torch.Tensor, c: torch.Tensor=None, au: torch.Tensor=None, identity: torch.Tensor=None) -> torch.Tensor:
        aud_feat = self.encode_audio(a)
        return self.decode_motion(aud_feat, c=c, au=au, identity=identity)['flame_jaw']

    def forward_mouth(self, xyz: torch.Tensor, a: torch.Tensor, mouth_mask: torch.Tensor=None, c: torch.Tensor=None, encoded_audio: torch.Tensor=None):
        if mouth_mask is not None:
            if mouth_mask.dtype == torch.bool:
                xyz_mouth = xyz[mouth_mask]
            else:
                xyz_mouth = xyz[mouth_mask.long()]
        else:
            xyz_mouth = xyz
        if xyz_mouth.numel() == 0:
            return {'d_mu': torch.empty(0, 3, device=xyz.device), 'd_rot': torch.empty(0, 4, device=xyz.device), 'd_scale': torch.empty(0, 3, device=xyz.device)}
        if not hasattr(self, '_coordinate_range_warned'):
            xyz_min, xyz_max = (xyz_mouth.min().item(), xyz_mouth.max().item())
            if abs(xyz_min) > self.bound or abs(xyz_max) > self.bound:
                self._coordinate_range_warned = True
            else:
                self._coordinate_range_warned = True
        enc_x = self.encode_x(xyz_mouth, bound=self.bound)
        if encoded_audio is not None:
            aud_feat = encoded_audio
        else:
            aud_feat = self.encode_audio(a)
        if aud_feat.shape[0] > 1:
            aud_feat = aud_feat.mean(dim=0, keepdim=True)
        Nm = enc_x.shape[0]
        aud_rep = aud_feat.repeat(Nm, 1)
        if self.individual_dim > 0 and c is not None:
            c_rep = c.repeat(Nm, 1)
            h_in = torch.cat([enc_x, aud_rep, c_rep], dim=-1)
        else:
            h_in = torch.cat([enc_x, aud_rep], dim=-1)
        h1 = self.mouth_mlp(h_in)
        pred = self.mouth_head(h1)
        d_mu = pred[..., :3] * 0.01
        d_rot = pred[..., 3:7]
        d_scale = pred[..., 7:10] * 0.001
        return {'d_mu': d_mu, 'd_rot': d_rot, 'd_scale': d_scale}

    def forward(self, a, c=None, au=None, identity=None):
        aud_feat, spk_logits = self.encode_audio(a, return_logits=True)
        if aud_feat is None:
            device = next(self.parameters()).device
            batch = 1
            return {'flame_exp': torch.zeros(batch, self.flame_exp_dim, device=device), 'flame_jaw': torch.zeros(batch, self.flame_jaw_dim, device=device), 'encoded_audio': None, 'gate': torch.zeros(batch, 1, device=device), 'emotion_logits': torch.zeros(batch, 7, device=device)}
        results = self.decode_motion(aud_feat, c=c, au=au, identity=identity)
        results.update({'encoded_audio': aud_feat})
        self.cache = results
        return results

    def load_state_dict(self, state_dict, strict: bool=True):
        if strict:
            current_keys = set(self.state_dict().keys())
            loaded_keys = set(state_dict.keys())
            grmn_prefixes = ('residual_net.', 'residual_motion_head.', 'gate_net.', 'emotion_head.', 'au_encoder.', 'au_fusion.')
            missing_grmn = [key for key in current_keys - loaded_keys if key.startswith(grmn_prefixes)]
            if missing_grmn:
                warnings.warn('Loading a checkpoint without GRMN residual/gate keys; GRMN parameters keep their initialized values.', RuntimeWarning)
                strict = False
        return super().load_state_dict(state_dict, strict=strict)

    def get_params(self, lr, lr_net, wd=0):
        params = [{'params': self.audio_proj.parameters(), 'name': 'neural_audio_proj', 'lr': lr_net, 'weight_decay': wd}, {'params': self.audio_transformer.parameters(), 'name': 'neural_audio_transformer', 'lr': lr_net, 'weight_decay': wd}, {'params': self.pos_encoder.parameters(), 'name': 'neural_pos_encoder', 'lr': lr_net, 'weight_decay': wd}, {'params': self.au_encoder.parameters(), 'name': 'neural_au_encoder', 'lr': lr_net, 'weight_decay': wd}, {'params': self.au_fusion.parameters(), 'name': 'neural_au_fusion', 'lr': lr_net, 'weight_decay': wd}, {'params': self.audio_adain.parameters(), 'name': 'neural_audio_adain', 'lr': lr_net, 'weight_decay': wd}, {'params': self.au_adain.parameters(), 'name': 'neural_au_adain', 'lr': lr_net, 'weight_decay': wd}, {'params': self.encoder_xy.parameters(), 'name': 'neural_encoder_xy', 'lr': lr}, {'params': self.encoder_yz.parameters(), 'name': 'neural_encoder_yz', 'lr': lr}, {'params': self.encoder_xz.parameters(), 'name': 'neural_encoder_xz', 'lr': lr}, {'params': self.mouth_mlp.parameters(), 'name': 'neural_mouth_mlp', 'lr': lr_net, 'weight_decay': wd}, {'params': self.mouth_head.parameters(), 'name': 'neural_mouth_head', 'lr': lr_net, 'weight_decay': wd}, {'params': self.backbone_net.parameters(), 'name': 'neural_backbone_net', 'lr': lr_net, 'weight_decay': wd}, {'params': self.flame_exp_head.parameters(), 'name': 'neural_flame_exp_head', 'lr': lr_net, 'weight_decay': wd}, {'params': self.flame_jaw_head.parameters(), 'name': 'neural_flame_jaw_head', 'lr': lr_net, 'weight_decay': wd}, {'params': self.residual_net.parameters(), 'name': 'neural_grmn_residual_net', 'lr': lr_net, 'weight_decay': wd}, {'params': self.residual_motion_head.parameters(), 'name': 'neural_grmn_residual_head', 'lr': lr_net, 'weight_decay': wd}, {'params': self.gate_net.parameters(), 'name': 'neural_grmn_gate', 'lr': lr_net, 'weight_decay': wd}, {'params': self.emotion_head.parameters(), 'name': 'neural_grmn_emotion_head', 'lr': lr_net, 'weight_decay': wd}]
        if self.individual_dim > 0:
            params.extend([{'params': self.individual_codes, 'name': 'neural_individual_codes', 'lr': lr, 'weight_decay': wd}])
        return params

    def freeze_for_adaptation(self):
        """Freeze the motion prior and tune only the AdaIN modulation parameters."""
        adain_modules = (self.audio_adain, self.au_adain)
        for param in self.parameters():
            param.requires_grad_(False)
        for module in adain_modules:
            for param in module.parameters():
                param.requires_grad_(True)
