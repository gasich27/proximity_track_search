"""3-segment audio slicing: start (10-20s), middle, end (minus fade-out)."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from music_parser import config

SEGMENT_DURATION_SEC = float(config.SEGMENT_DURATION)
INTRO_START_SEC = 10.0
INTRO_END_SEC = 20.0
MIN_MULTI_SEGMENT_DURATION_SEC = 30.0
FADE_OUT_BUFFER_SEC = 2.0
MIDDLE_HALF_WINDOW_SEC = 5.0


@dataclass(frozen=True)
class SegmentSpec:
    start_sec: float
    end_sec: float


def compute_segment_specs(duration_sec: float) -> list[SegmentSpec]:
    if duration_sec <= 0:
        return [SegmentSpec(0.0, 0.0)]
    if duration_sec < MIN_MULTI_SEGMENT_DURATION_SEC:
        return [SegmentSpec(0.0, duration_sec)]

    start_seg = SegmentSpec(INTRO_START_SEC, min(INTRO_END_SEC, duration_sec))

    center = duration_sec / 2.0
    mid_start = max(0.0, center - MIDDLE_HALF_WINDOW_SEC)
    mid_end = min(duration_sec, center + MIDDLE_HALF_WINDOW_SEC)
    if mid_end - mid_start < SEGMENT_DURATION_SEC:
        mid_start = max(0.0, mid_end - SEGMENT_DURATION_SEC)
    middle_seg = SegmentSpec(mid_start, mid_end)

    end_boundary = max(0.0, duration_sec - FADE_OUT_BUFFER_SEC)
    end_start = max(0.0, end_boundary - SEGMENT_DURATION_SEC)
    end_seg = SegmentSpec(end_start, end_boundary)

    return [start_seg, middle_seg, end_seg]


def slice_audio(y: np.ndarray, sr: int, spec: SegmentSpec) -> np.ndarray:
    start = int(spec.start_sec * sr)
    end = int(spec.end_sec * sr)
    segment = np.asarray(y[start:end], dtype=np.float32)
    if segment.size == 0:
        return np.asarray(y, dtype=np.float32)
    return segment


def load_track_segments(audio_path: str, sr: int) -> list[np.ndarray]:
    y, sr_out = librosa.load(audio_path, sr=sr, mono=True)
    duration_sec = len(y) / sr_out if sr_out else 0.0
    specs = compute_segment_specs(duration_sec)
    return [slice_audio(y, sr_out, spec) for spec in specs]


def prepare_fixed_length_segment(segment: np.ndarray, sr: int, duration_sec: float = SEGMENT_DURATION_SEC) -> np.ndarray:
    target_samples = max(1, int(duration_sec * sr))
    audio = np.asarray(segment, dtype=np.float32).ravel()
    if audio.size == 0:
        return np.zeros(target_samples, dtype=np.float32)
    if audio.size >= target_samples:
        return audio[:target_samples]
    repeats = int(np.ceil(target_samples / audio.size))
    return np.tile(audio, repeats)[:target_samples]


def mean_pool_vectors(vectors: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors], axis=0)
    return stacked.mean(axis=0)


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return arr
    return arr / norm
