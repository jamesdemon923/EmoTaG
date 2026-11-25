from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


AU_BASE_COLUMNS = ("AU01", "AU04", "AU05", "AU06", "AU07", "AU45")


def detect_au_scale(au_features: np.ndarray) -> str:
    """'pyfeat' ([0, 1] activations) or 'openface' ([0, 5] intensities)."""
    forced = os.environ.get("EMOTAG_AU_SCALE", "auto").lower()
    if forced in ("pyfeat", "openface"):
        return forced
    return "openface" if np.nanmax(au_features) > 1.5 else "pyfeat"


def load_openface_au_csv(path: str | Path, required_len: int | None = None) -> np.ndarray:
    """Load the six EmoTaG AU channels from an OpenFace FeatureExtraction CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing AU feature file: {csv_path}")

    frame = pd.read_csv(csv_path)
    stripped_columns = {column.strip(): column for column in frame.columns}
    values = []
    missing = []
    for base_name in AU_BASE_COLUMNS:
        source_column = None
        for candidate in (f"{base_name}_r", base_name):
            if candidate in stripped_columns:
                source_column = stripped_columns[candidate]
                break
        if source_column is None:
            missing.append(f"{base_name}_r")
            continue
        values.append(frame[source_column].to_numpy(dtype=np.float32))

    if missing:
        raise ValueError(f"{csv_path} is missing AU columns: {', '.join(missing)}")

    au_features = np.stack(values, axis=1).astype(np.float32)
    au_features = np.nan_to_num(au_features, nan=0.0)
    scale = detect_au_scale(au_features)
    if scale == "openface":
        warnings.warn(
            f"{csv_path}: AU values look like OpenFace intensities (max {au_features.max():.2f} > 1.5); "
            "rescaling by 1/5 to the py-feat [0, 1] range the checkpoint was trained on. "
            "Prefer tools/extract_pyfeat_au.py.", RuntimeWarning)
        au_features = np.clip(au_features / 5.0, 0.0, 1.0)
    au45_index = AU_BASE_COLUMNS.index("AU45")
    au_features[:, au45_index] = np.clip(au_features[:, au45_index], 0.0, 2.0)
    if required_len is not None and au_features.shape[0] < required_len:
        pad = np.repeat(au_features[-1:], required_len - au_features.shape[0], axis=0)
        au_features = np.concatenate([au_features, pad], axis=0)
    return au_features
