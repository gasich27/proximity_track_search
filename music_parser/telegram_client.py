"""Telegram channel audio downloader (Telethon)."""

from __future__ import annotations

import asyncio
import getpass
import logging
from pathlib import Path
from typing import Optional

from telethon import TelegramClient
from telethon import errors
from telethon.tl.types import Message

from music_parser import config

try:
    import qrcode
except ImportError:  # pragma: no cover - optional runtime dependency
    qrcode = None

logger = logging.getLogger(__name__)


def _is_mp3_message(msg: Message) -> bool:
    mime = ""
    filename = ""
    if getattr(msg, "audio", None):
        mime = getattr(msg.audio, "mime_type", "") or ""
        filename = getattr(msg.audio, "file_name", "") or ""
    elif getattr(msg, "document", None):
        mime = getattr(msg.document, "mime_type", "") or ""
        for attr in msg.document.attributes or []:
            if attr.__class__.__name__ == "DocumentAttributeFilename":
                filename = attr.file_name or ""
    mime = mime.lower()
    filename = filename.lower()
    return mime in {"audio/mpeg", "audio/mp3", "audio/x-mpeg"} or filename.endswith(".mp3")


def _message_duration_ms(msg: Message) -> Optional[int]:
    if getattr(msg, "audio", None) and getattr(msg.audio, "duration", None):
        return int(msg.audio.duration * 1000)
    if getattr(msg, "document", None):
        for attr in msg.document.attributes or []:
            if attr.__class__.__name__ == "DocumentAttributeAudio" and getattr(attr, "duration", None):
                return int(attr.duration * 1000)
    return None


def _message_filename(msg: Message) -> str:
    if getattr(msg, "audio", None) and getattr(msg.audio, "file_name", None):
        return msg.audio.file_name
    if getattr(msg, "document", None):
        for attr in msg.document.attributes or []:
            if attr.__class__.__name__ == "DocumentAttributeFilename":
                return attr.file_name or f"{msg.id}.mp3"
    return f"{msg.id}.mp3"


def is_indexable_message(msg: Message, min_seconds: int = config.MIN_TRACK_SECONDS) -> bool:
    duration_ms = _message_duration_ms(msg)
    if duration_ms is None or duration_ms < min_seconds * 1000:
        return False
    return _is_mp3_message(msg)


class TelegramAudioDownloader:
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        channel: str,
        session: str = config.TG_SESSION,
        allow_qr_login: bool = False,
    ) -> None:
        if not api_id or not api_hash or not channel:
            raise ValueError("TG_API_ID, TG_API_HASH and TG_CHANNEL are required")
        self.channel = channel
        self.client = TelegramClient(session, api_id, api_hash)
        self.allow_qr_login = allow_qr_login
        self._connected = False

    @staticmethod
    def _print_qr(url: str) -> None:
        if qrcode is None:
            raise RuntimeError(
                "Для QR-входа нужен пакет `qrcode`. "
                "Установите зависимости из requirements.txt и попробуйте снова."
            )

        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        matrix = qr.get_matrix()

        black = "██"
        white = "  "
        quiet = 2
        border = white * (len(matrix[0]) + quiet * 2)
        lines = [border for _ in range(quiet)]

        for row in matrix:
            rendered_row = [white] * quiet
            rendered_row.extend(black if cell else white for cell in row)
            rendered_row.extend([white] * quiet)
            lines.append("".join(rendered_row))

        lines.extend(border for _ in range(quiet))
        print("\n".join(lines))

    async def _login_via_qr(self) -> None:
        while True:
            qr_login = await self.client.qr_login()
            print(
                "\nОтсканируйте QR-код в Telegram на телефоне.\n"
                "Откройте Telegram -> Настройки -> Устройства -> Подключить устройство.\n"
            )
            self._print_qr(qr_login.url)

            try:
                await qr_login.wait()
                return
            except asyncio.TimeoutError:
                logger.info("QR-код истёк, генерирую новый.")
                continue
            except errors.SessionPasswordNeededError:
                password = getpass.getpass("Введите пароль 2FA: ")
                await self.client.sign_in(password=password)
                return

    async def connect(self, allow_qr_login: Optional[bool] = None) -> None:
        await self.client.connect()
        if not await self.client.is_user_authorized():
            if allow_qr_login is None:
                allow_qr_login = self.allow_qr_login
            if not allow_qr_login:
                raise RuntimeError(
                    "Telegram session is not authorized. "
                    f"Run `python -m music_parser.main --qr-login` to authorize the session "
                    f"stored at {self.client.session.filename!r}."
                )
            try:
                await self._login_via_qr()
            except (EOFError, OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "QR Telegram login failed. "
                    "Run the app in a real terminal and try `python -m music_parser.main --qr-login` again."
                ) from exc
            if not await self.client.is_user_authorized():
                raise RuntimeError(
                    "Telegram login completed, but the session is still not authorized."
                )
        self._connected = True

    async def disconnect(self) -> None:
        if self._connected:
            await self.client.disconnect()
            self._connected = False

    def get_last_msg_id(self) -> int:
        path = Path(config.STATE_FILE)
        if not path.exists():
            return 0
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            return 0

    def save_last_msg_id(self, msg_id: int) -> None:
        path = Path(config.STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(int(msg_id)), encoding="utf-8")

    async def get_new_messages(self, last_msg_id: int) -> list[Message]:
        entity = await self.client.get_entity(self.channel)
        messages: list[Message] = []
        async for msg in self.client.iter_messages(entity, min_id=last_msg_id):
            if is_indexable_message(msg):
                messages.append(msg)
        messages.sort(key=lambda m: int(m.id))
        return messages

    async def download_audio(self, message: Message, save_path: str) -> str:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        await self.client.download_media(message, file=str(path))
        return str(path)

    @staticmethod
    def message_filename(message: Message) -> str:
        return _message_filename(message)

    async def __aenter__(self) -> TelegramAudioDownloader:
        await self.connect()
        return self

    async def __aexit__(self, *args) -> None:
        await self.disconnect()


def run_async(coro):
    return asyncio.run(coro)
