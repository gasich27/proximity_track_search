import numpy as np

from extractors.audio import load_segments


SAMPLE_RATE = 48000
_model = None


def _get_model():
    global _model
    if _model is None:
        from laion_clap import CLAP_Module

        _model = CLAP_Module(enable_fusion=False)
        _model.load_ckpt()
    return _model


def extract(audio_path: str) -> np.ndarray:
    segments = load_segments(audio_path, SAMPLE_RATE)
    embeddings = _get_model().get_audio_embedding_from_data(x=segments, use_tensor=False)
    vector = np.asarray(embeddings, dtype=np.float64).mean(axis=0)
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm
