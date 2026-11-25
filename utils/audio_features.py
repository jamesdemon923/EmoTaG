"""Audio feature types."""
import os

AUDIO_FEATURE_FILES = {'wav2vec2': 'aud_w2v.npy', 'whisper': 'aud_whisper.npy'}


def feature_file(extractor: str) -> str:
    if extractor not in AUDIO_FEATURE_FILES:
        raise ValueError(f'--audio_extractor must be one of {sorted(AUDIO_FEATURE_FILES)}, got {extractor!r}.')
    return AUDIO_FEATURE_FILES[extractor]


def feature_path(scene_dir: str, extractor: str) -> str:
    return os.path.join(scene_dir, feature_file(extractor))


EXTRACTOR_KEY = 'audio_extractor'


def check_prior_extractor(state: dict, extractor: str, path: str = '') -> dict:
    """Remove the audio-feature tag (untagged priors are Wav2Vec2 priors) and stop if it does not match --audio_extractor."""
    state = dict(state)
    tag = state.pop(EXTRACTOR_KEY, 'wav2vec2')
    if tag != extractor:
        raise SystemExit(f'[EmoTaG] the motion prior {path} was trained on {tag} features: '
                         f'pass --audio_extractor {tag} (got {extractor}).')
    return state
