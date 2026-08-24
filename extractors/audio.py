import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spoti_an2_numba_cache"))

import librosa
import numpy as np


SEGMENT_SECONDS = 10.0


def load_segments(audio_path: str, sample_rate: int) -> list[np.ndarray]:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    duration = len(audio) / sample_rate

    if duration <= 0:
        raise ValueError("Audio is empty")

    if duration < 30:
        return [audio]

    starts = [10.0, duration / 2 - 5.0, max(0.0, duration - 12.0)]
    segment_size = int(SEGMENT_SECONDS * sample_rate)
    segments = []

    for start in starts:
        first_sample = int(max(0.0, start) * sample_rate)
        segment = audio[first_sample : first_sample + segment_size]
        if len(segment) < segment_size:
            segment = np.pad(segment, (0, segment_size - len(segment)))
        segments.append(segment)

    return segments


def mean_vector(vectors: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(vectors), axis=0).astype(np.float64)
