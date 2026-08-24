from pathlib import Path

from telethon import TelegramClient


MIN_TRACK_SECONDS = 60


def create_client(api_id: int, api_hash: str) -> TelegramClient:
    return TelegramClient("user_session", api_id, api_hash)


def is_audio_track(message) -> bool:
    file = message.file
    if file is None:
        return False

    mime_type = (file.mime_type or "").lower()
    filename = (file.name or "").lower()
    duration = file.duration or 0

    is_audio = mime_type.startswith("audio/") or filename.endswith(".mp3")
    return is_audio and duration >= MIN_TRACK_SECONDS


def get_filename(message) -> str:
    return message.file.name or f"{message.id}.mp3"


async def iter_tracks(client: TelegramClient, channel: str, processed_message_ids: set[int]):
    async for message in client.iter_messages(channel, reverse=True):
        if message.id not in processed_message_ids and is_audio_track(message):
            yield message


async def download_track(client: TelegramClient, message, temp_dir: Path) -> Path:
    filename = get_filename(message)
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus"}:
        suffix = ".mp3"
    path = temp_dir / f"current_track{suffix}"
    downloaded_path = await client.download_media(message, file=str(path))

    if not downloaded_path or not path.exists():
        raise RuntimeError("Telegram did not return an audio file")

    return path
