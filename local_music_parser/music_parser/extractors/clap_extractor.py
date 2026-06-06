"""CLAP audio embeddings via laion-clap."""

from __future__ import annotations

import numpy as np
import os
import ssl
import httpx

# Отключаем проверку SSL для HuggingFace
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# Патчим httpx для отключения SSL
_original_client_init = httpx.Client.__init__
def _patched_client_init(self, *args, **kwargs):
    kwargs['verify'] = False
    _original_client_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_client_init

# Патчим ssl для urllib
ssl._create_default_https_context = ssl._create_unverified_context

from music_parser import config
from music_parser.extractors.base_extractor import BaseExtractor
from music_parser.segmentation import l2_normalize, prepare_fixed_length_segment

CLAP_EMBEDDING_DIM = 512


class CLAPExtractor(BaseExtractor):
    name = "clap"

    def __init__(self, enable_fusion: bool = config.CLAP_ENABLE_FUSION) -> None:
        from laion_clap import CLAP_Module

        self._model = CLAP_Module(enable_fusion=enable_fusion)
        self._model.load_ckpt()

    def sample_rate(self) -> int:
        return config.CLAP_SR

    def get_embedding_dim(self) -> int:
        return CLAP_EMBEDDING_DIM

    def _extract_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        prepared = prepare_fixed_length_segment(segment, sr, duration_sec=float(config.SEGMENT_DURATION))
        batched_audio = [prepared]
        embedding = self._model.get_audio_embedding_from_data(x=batched_audio, use_tensor=False)
        vec = np.asarray(embedding, dtype=np.float64)
        if vec.ndim == 0:
            raise ValueError("CLAP returned a scalar embedding")
        if vec.ndim == 1:
            return vec.ravel()
        return vec.mean(axis=0).ravel()

    def extract(self, audio_path: str) -> np.ndarray:
        pooled = super().extract(audio_path)
        return l2_normalize(pooled)
