import librosa
import numpy as np
from librosa.feature.rhythm import tempo

from extractors.audio import load_segments, mean_vector


SAMPLE_RATE = 22050
HOP_LENGTH = 2048


def _row_stats(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values)
    return np.concatenate((values.mean(axis=1), values.std(axis=1)))


def _series_stats(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values)
    return np.array([values.mean(), values.std()])


def _extract_segment(audio: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13, hop_length=HOP_LENGTH)

    return np.concatenate(
        [
            np.array([tempo(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)[0]]),
            _row_stats(mfcc),
            _row_stats(librosa.feature.delta(mfcc)),
            _row_stats(librosa.feature.delta(mfcc, order=2)),
            _row_stats(librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)),
            _row_stats(librosa.feature.spectral_contrast(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)),
            _series_stats(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)[0]),
            _series_stats(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)[0]),
            _series_stats(librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]),
        ]
    )


def extract(audio_path: str) -> np.ndarray:
    return mean_vector([_extract_segment(segment) for segment in load_segments(audio_path, SAMPLE_RATE)])
