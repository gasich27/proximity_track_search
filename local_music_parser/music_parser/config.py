"""Configuration for the local folder-based music parser."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Input / output
INPUT_DIR = ROOT_DIR / os.getenv("INPUT_DIR", "input_mp3")
DATA_DIR = ROOT_DIR / os.getenv("DATA_DIR", "data")

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

