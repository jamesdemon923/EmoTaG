import numpy as np


def pack_bool_mask(mask):
    """Pack an HxW bool mask into a bit array for CPU preloading.

    np.bool_ stores one byte per pixel, so the per-frame parsing masks dominate
    preload RAM (~1MB/frame at 512x512). Packing to bits is lossless and ~8x
    smaller; unpack_bool_mask restores the exact original array.
    """
    mask = np.asarray(mask, dtype=bool)
    return {'bits': np.packbits(mask.reshape(-1)), 'shape': mask.shape}


def unpack_bool_mask(packed):
    """Inverse of pack_bool_mask. Passes through legacy unpacked arrays."""
    if isinstance(packed, dict) and 'bits' in packed:
        count = int(np.prod(packed['shape']))
        return np.unpackbits(packed['bits'], count=count).reshape(packed['shape']).astype(bool)
    return packed
