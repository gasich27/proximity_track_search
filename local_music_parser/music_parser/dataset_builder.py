"""Build and save feature datasets from extracted embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from music_parser.extractors.base_extractor import BaseExtractor

DATASET_FILES = {
    "librosa": "librosa_dataset.csv",
    "openl3": "openl3_dataset.csv",
    "yamnet": "yamnet_dataset.csv",
    "clap": "clap_dataset.csv",
}


class DatasetBuilder:
    def __init__(self, extractors: dict[str, BaseExtractor]) -> None:
        self.extractors = extractors
        self._rows: dict[str, list[dict[str, Any]]] = {name: [] for name in extractors}

    def process_track(
        self,
        audio_path: str,
        msg_id: int,
        filename: str,
        tags: str = " ",
    ) -> None:
        for name, extractor in self.extractors.items():
            embedding = extractor.extract(audio_path)
            row: dict[str, Any] = {
                "msg_id": int(msg_id),
                "filename": filename,
                "tags": tags,
            }
            for feat_name, value in zip(extractor.get_feature_names(), np.asarray(embedding).ravel()):
                row[feat_name] = float(value)
            self._rows[name].append(row)

    def save_datasets(self, output_dir: str | Path) -> dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        for name, rows in self._rows.items():
            if not rows:
                continue
            path = out / DATASET_FILES[name]
            pd.DataFrame(rows).to_csv(path, index=False)
            saved[name] = path

        return saved

    def append_existing(self, output_dir: str | Path) -> None:
        """Load existing CSVs and prepend stored rows for incremental updates."""
        out = Path(output_dir)
        for name in self.extractors:
            path = out / DATASET_FILES[name]
            if not path.exists():
                continue
            existing = pd.DataFrame(pd.read_csv(path)).to_dict(orient="records")
            self._rows[name] = existing + self._rows[name]

    @property
    def counts(self) -> dict[str, int]:
        return {name: len(rows) for name, rows in self._rows.items()}
