import numpy as np

from extractors.audio import load_segments, mean_vector


SAMPLE_RATE = 48000
_model = None


def _get_model():
    global _model
    if _model is None:
        import openl3

        _model = openl3.models.load_audio_embedding_model(
            input_repr="mel256",
            content_type="music",
            embedding_size=512,
        )
    return _model


def _extract_segment(audio: np.ndarray) -> np.ndarray:
    import openl3

    embeddings, _ = openl3.get_audio_embedding(
        audio,
        SAMPLE_RATE,
        model=_get_model(),
        center=True,
        hop_size=1.0,
        verbose=False,
    )
    if embeddings.size == 0:
        raise ValueError("OpenL3 returned an empty embedding")
    return embeddings.mean(axis=0)


def extract(audio_path: str) -> np.ndarray:
    return mean_vector([_extract_segment(segment) for segment in load_segments(audio_path, SAMPLE_RATE)])
