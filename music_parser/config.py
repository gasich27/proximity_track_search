"""Configuration for music_parser."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Telegram
TG_API_ID = int(os.getenv("TG_API_ID", "32704381"))
TG_API_HASH = os.getenv("TG_API_HASH", "f5c529e17bcfbffc43255ab570a4f8b9")
TG_CHANNEL = os.getenv("TG_CHANNEL", "@KapustnikGasich")
TG_SESSION = os.getenv("TG_SESSION", str(ROOT_DIR / "user_session"))

# Paths
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
STATE_FILE = DATA_DIR / "last_msg_id.txt"

# Segmentation
SEGMENT_DURATION = 10
SEGMENTS_PER_TRACK = 3
MIN_TRACK_SECONDS = 60

# Extractors
EXTRACTORS = ["librosa", "openl3", "yamnet", "clap"]

# Librosa
LIBROSA_SR = 22050
HOP_LENGTH = 2048

# OpenL3
OPENL3_SR = 48000
OPENL3_INPUT_REPR = "mel256"
OPENL3_CONTENT_TYPE = "music"
OPENL3_EMBEDDING_SIZE = int(os.getenv("OPENL3_EMBEDDING_SIZE", "512"))

# YAMNet
YAMNET_SR = 16000
YAMNET_HUB_URL = "https://tfhub.dev/google/yamnet/1"

# CLAP
CLAP_SR = 48000
CLAP_ENABLE_FUSION = False

# Concurrency
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "3"))
COLLECTOR_TEST_LIMIT = int(os.getenv("COLLECTOR_TEST_LIMIT", "10"))
