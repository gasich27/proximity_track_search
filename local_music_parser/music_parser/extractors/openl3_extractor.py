"""OpenL3 music embeddings."""

from __future__ import annotations

import numpy as np

from music_parser import config
from music_parser.extractors.base_extractor import BaseExtractor


class OpenL3Extractor(BaseExtractor):
    name = "openl3"

    def __init__(
        self,
        embedding_size: int = config.OPENL3_EMBEDDING_SIZE,
        input_repr: str = config.OPENL3_INPUT_REPR,
        content_type: str = config.OPENL3_CONTENT_TYPE,
        hop_size: float = 0.1,
    ) -> None:
        import openl3

        self._openl3 = openl3
        self.embedding_size = embedding_size
        self.hop_size = hop_size
        self._model = openl3.models.load_audio_embedding_model(
            input_repr=input_repr,
            content_type=content_type,
            embedding_size=embedding_size,
        )

    def sample_rate(self) -> int:
        return config.OPENL3_SR

    def get_embedding_dim(self) -> int:
        return self.embedding_size

    def _extract_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        embeddings, _ = self._openl3.get_audio_embedding(
            segment,
            sr,
            model=self._model,
            center=True,
            hop_size=self.hop_size,
            verbose=False,
        )
        if embeddings.size == 0:
            raise ValueError("OpenL3 returned an empty embedding")
        return np.asarray(embeddings, dtype=np.float64).mean(axis=0)
