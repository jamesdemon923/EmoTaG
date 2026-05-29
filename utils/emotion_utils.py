from __future__ import annotations

from pathlib import Path

import numpy as np

# DeepFace returns a categorical distribution over seven basic emotions.
# We keep this canonical ordering everywhere so that the teacher distribution
# stored on disk, the KL target, and the `neutral` index used for the emotion
# intensity score all stay aligned.
EMOTION_LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
NEUTRAL_INDEX = EMOTION_LABELS.index("neutral")


def emotion_score_from_distribution(p_emo: np.ndarray) -> np.ndarray:
    """Scalar emotion intensity score e = 1 - p_emo(neutral)."""
    p_emo = np.asarray(p_emo, dtype=np.float32)
    return (1.0 - p_emo[..., NEUTRAL_INDEX]).astype(np.float32)


def load_emotion_features(path: str | Path, required_len: int | None = None) -> np.ndarray:
    """Load the per-frame DeepFace emotion distribution `emotion_features.npy`.

    Returns an array of shape [N, 7] normalized to sum to one per frame.
    """
    emotion_path = Path(path)
    p_emo = np.load(emotion_path).astype(np.float32)
    if p_emo.ndim != 2 or p_emo.shape[1] != len(EMOTION_LABELS):
        raise ValueError(
            f"{emotion_path} must provide shape [N, {len(EMOTION_LABELS)}], got {p_emo.shape}."
        )
    row_sum = p_emo.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    p_emo = p_emo / row_sum
    if required_len is not None and p_emo.shape[0] < required_len:
        pad = np.repeat(p_emo[-1:], required_len - p_emo.shape[0], axis=0)
        p_emo = np.concatenate([p_emo, pad], axis=0)
    return p_emo
