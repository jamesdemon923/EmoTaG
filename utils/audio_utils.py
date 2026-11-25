import torch

def get_audio_features(features, att_mode, index):
    if att_mode == 0:
        return features[[index]]
    elif att_mode == 1:
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        if pad_left > 0:
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
        return auds
    elif att_mode == 2:
        n = features.shape[0]
        left, right = index - 4, index + 4
        lo, hi = max(0, left), min(n, right)
        auds = features[lo:hi] if hi > lo else features[:0]
        pad_left = min(right - left, lo - left)
        pad_right = (right - left) - pad_left - auds.shape[0]
        zeros = lambda k: torch.zeros(k, *features.shape[1:], device=features.device, dtype=features.dtype)
        if pad_left > 0:
            auds = torch.cat([zeros(pad_left), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, zeros(pad_right)], dim=0)
        return auds
    else:
        raise NotImplementedError(f'Unsupported attention mode: {att_mode}')
