"""
Точка входа: Telegram-канал → 4 датасета (librosa, openl3, yamnet, clap).

Запуск:
  python main.py                  (из папки music_parser)
  python -m music_parser.main     (из корня проекта)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Позволяет запускать как `python main.py` из папки music_parser
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tqdm import tqdm

from music_parser import config
from music_parser.dataset_builder import DATASET_FILES, DatasetBuilder
from music_parser.extractors import build_extractors
from music_parser.telegram_client import TelegramAudioDownloader

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular Telegram audio feature parser")
    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=config.EXTRACTORS,
        default=config.EXTRACTORS,
        help="Which extractors to run (default: all 4)",
    )
    parser.add_argument("--test-mode", action="store_true", help="Process only first N tracks")
    parser.add_argument("--reset-state", action="store_true", help="Start from msg_id=0")
    parser.add_argument(
        "--qr-login",
        action="store_true",
        help="Show a QR code in the terminal to authorize Telegram if the session is not authorized",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    if config.TG_API_ID == 0 or not config.TG_API_HASH or not config.TG_CHANNEL:
        raise SystemExit("Set TG_API_ID, TG_API_HASH, TG_CHANNEL environment variables.")

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading extractors: %s", ", ".join(args.extractors))
    extractors = build_extractors(args.extractors)

    builder = DatasetBuilder(extractors)
    builder.append_existing(config.DATA_DIR)

    async with TelegramAudioDownloader(
        api_id=config.TG_API_ID,
        api_hash=config.TG_API_HASH,
        channel=config.TG_CHANNEL,
        allow_qr_login=args.qr_login,
    ) as downloader:
        last_msg_id = 0 if args.reset_state else downloader.get_last_msg_id()
        new_messages = await downloader.get_new_messages(last_msg_id)

        if args.test_mode:
            new_messages = new_messages[: config.COLLECTOR_TEST_LIMIT]
            logger.info("Test mode: %s messages", len(new_messages))

        if not new_messages:
            logger.info("Нет новых треков")
            return

        logger.info("Найдено %s новых треков", len(new_messages))

        for msg in tqdm(new_messages, desc="Обработка треков"):
            msg_id = int(msg.id)
            filename = downloader.message_filename(msg)
            audio_path = config.AUDIO_CACHE_DIR / f"{msg_id}.mp3"

            try:
                await downloader.download_audio(msg, str(audio_path))
                builder.process_track(str(audio_path), msg_id, filename, tags=" ")
                downloader.save_last_msg_id(msg_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("Ошибка при обработке трека %s: %s", msg_id, exc)
                if audio_path.exists():
                    audio_path.unlink(missing_ok=True)
                continue
            finally:
                if audio_path.exists():
                    audio_path.unlink(missing_ok=True)

    saved = builder.save_datasets(config.DATA_DIR)
    for name, path in saved.items():
        logger.info("%s: %s rows → %s", name, builder.counts[name], path)

    if not saved:
        logger.warning("Нечего сохранять")
    else:
        logger.info("Датасеты сохранены в %s", config.DATA_DIR.resolve())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
