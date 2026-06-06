"""YAMNet embeddings via TensorFlow Hub."""

from __future__ import annotations

import numpy as np

from music_parser import config
from music_parser.extractors.base_extractor import BaseExtractor

YAMNET_EMBEDDING_DIM = 1024


class YAMNetExtractor(BaseExtractor):
    name = "yamnet"

    def __init__(self, hub_url: str = config.YAMNET_HUB_URL) -> None:
        import tensorflow as tf
        import tensorflow_hub as hub

        self._tf = tf
        model = hub.load(hub_url)
        self._infer = tf.function(lambda waveform: model(waveform), autograph=False)

    def sample_rate(self) -> int:
        return config.YAMNET_SR

    def get_embedding_dim(self) -> int:
        return YAMNET_EMBEDDING_DIM

    def _extract_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        if segment.size == 0:
            raise ValueError("Empty audio segment")
        waveform = self._tf.convert_to_tensor(segment, dtype=self._tf.float32)
        _scores, embeddings, _spectrogram = self._infer(waveform)
        if embeddings.shape[0] == 0:
            raise ValueError("YAMNet returned an empty embedding")
        return self._tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float64)
