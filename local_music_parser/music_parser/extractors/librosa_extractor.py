"""Classical librosa features: MFCC, Chroma, Spectral Contrast, etc."""

from __future__ import annotations

import warnings

import librosa
import numpy as np

from music_parser import config
from music_parser.extractors.base_extractor import BaseExtractor

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*librosa\.beat\.tempo.*",
)

MFCC_BANDS = 13
CHROMA_BANDS = 12
CONTRAST_BANDS = 7
EMBEDDING_DIM = 1 + (MFCC_BANDS * 2) * 3 + (CHROMA_BANDS * 2) + (CONTRAST_BANDS * 2) + 2 + 2 + 2


def _series_stats(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.array([float(arr.mean()), float(arr.std())], dtype=float)


def _matrix_stats(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([arr.mean(axis=1), arr.std(axis=1)]).astype(float)


class LibrosaExtractor(BaseExtractor):
    name = "librosa"

    def __init__(self, hop_length: int = config.HOP_LENGTH) -> None:
        self.hop_length = hop_length

    def sample_rate(self) -> int:
        return config.LIBROSA_SR

    def get_embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def _extract_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        if segment.size < sr:
            raise ValueError("Audio segment too short")

        tempo = float(librosa.beat.tempo(y=segment, sr=sr, hop_length=self.hop_length)[0])
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=MFCC_BANDS, hop_length=self.hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        chroma = librosa.feature.chroma_stft(y=segment, sr=sr, hop_length=self.hop_length)
        contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, hop_length=self.hop_length)
        centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=self.hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, hop_length=self.hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)[0]

        return np.concatenate(
            [
                np.array([tempo], dtype=float),
                _matrix_stats(mfcc),
                _matrix_stats(mfcc_delta),
                _matrix_stats(mfcc_delta2),
                _matrix_stats(chroma),
                _matrix_stats(contrast),
                _series_stats(centroid),
                _series_stats(rolloff),
                _series_stats(zcr),
            ]
        ).astype(float)
