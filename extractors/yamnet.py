import numpy as np

from extractors.audio import load_segments, mean_vector


SAMPLE_RATE = 16000
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
_model = None


def _get_model():
    global _model
    if _model is None:
        import tensorflow_hub as hub

        _model = hub.load(MODEL_URL)
    return _model


def _extract_segment(audio: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    _, embeddings, _ = _get_model()(tf.convert_to_tensor(audio, dtype=tf.float32))
    if embeddings.shape[0] == 0:
        raise ValueError("YAMNet returned an empty embedding")
    return tf.reduce_mean(embeddings, axis=0).numpy()


def extract(audio_path: str) -> np.ndarray:
    return mean_vector([_extract_segment(segment) for segment in load_segments(audio_path, SAMPLE_RATE)])
