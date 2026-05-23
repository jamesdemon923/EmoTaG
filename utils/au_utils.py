from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


AU_BASE_COLUMNS = ("AU01", "AU04", "AU05", "AU06", "AU07", "AU45")


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
    if required_len is not None and au_features.shape[0] < required_len:
        pad = np.repeat(au_features[-1:], required_len - au_features.shape[0], axis=0)
        au_features = np.concatenate([au_features, pad], axis=0)
    return au_features
