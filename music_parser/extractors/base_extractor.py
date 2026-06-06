"""Base class for all audio feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from music_parser.segmentation import load_track_segments, mean_pool_vectors


class BaseExtractor(ABC):
    name: str = "base"

    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate required by this extractor."""

    @abstractmethod
    def _extract_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        """Extract a feature vector from one audio segment."""

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Final embedding dimension after pooling."""

    def get_feature_names(self) -> list[str]:
        return [f"emb_{i}" for i in range(self.get_embedding_dim())]

    def extract(self, audio_path: str) -> np.ndarray:
        segments = load_track_segments(audio_path, self.sample_rate())
        if not segments:
            raise ValueError(f"No segments extracted from {audio_path}")
        vectors = [self._extract_segment(segment, self.sample_rate()) for segment in segments]
        pooled = mean_pool_vectors(vectors)
        if pooled.size != self.get_embedding_dim():
            raise RuntimeError(
                f"{self.name}: expected dim {self.get_embedding_dim()}, got {pooled.size}"
            )
        return pooled
