"""Local folder-based entry point for audio feature extraction."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python -m music_parser.main` from inside local_music_parser.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm

from music_parser import config
from music_parser.dataset_builder import DatasetBuilder
from music_parser.extractors import build_extractors

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".mp3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local folder-based audio feature parser")
    parser.add_argument(
        "--input-dir",
        default=str(config.INPUT_DIR),
        help="Folder with local audio files to process",
    )
    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=config.EXTRACTORS,
        default=config.EXTRACTORS,
        help="Which extractors to run (default: all)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolders recursively for audio files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only first N files for a quick test (0 = all files)",
    )
    return parser.parse_args()


def collect_audio_files(input_dir: str | Path, recursive: bool = False) -> list[Path]:
    base = Path(input_dir)
    if not base.exists():
        return []

    pattern = "**/*.mp3" if recursive else "*.mp3"
    files = [path for path in base.glob(pattern) if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    return sorted(files)


def track_id_from_index(index: int) -> int:
    return index + 1


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()
    audio_files = collect_audio_files(args.input_dir, recursive=args.recursive)

    if args.limit > 0:
        audio_files = audio_files[: args.limit]

    if not audio_files:
        logger.info("Не найдено mp3-файлов в %s", Path(args.input_dir).resolve())
        return

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Загружаю экстракторы: %s", ", ".join(args.extractors))

    extractors = build_extractors(args.extractors)
    builder = DatasetBuilder(extractors)
    builder.append_existing(config.DATA_DIR)

    logger.info("Найдено файлов: %s", len(audio_files))

    for index, audio_path in enumerate(tqdm(audio_files, desc="Обработка файлов"), start=0):
        track_id = track_id_from_index(index)
        try:
            builder.process_track(str(audio_path), track_id, audio_path.name, tags=" ")
        except Exception as exc:  # noqa: BLE001
            logger.error("Ошибка при обработке %s: %s", audio_path.name, exc)
            continue

    saved = builder.save_datasets(config.DATA_DIR)
    for name, path in saved.items():
        logger.info("%s: %s rows -> %s", name, builder.counts[name], path)

    if not saved:
        logger.warning("Нечего сохранять")
    else:
        logger.info("Датасеты сохранены в %s", config.DATA_DIR.resolve())


if __name__ == "__main__":
    main()
