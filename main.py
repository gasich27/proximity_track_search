import asyncio
import csv
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from extractors import clap, librosa, openl3, yamnet
from telegram_parser import create_client, download_track, get_filename, iter_tracks


DATA_DIR = Path("data")
TEMP_DIR = Path("temp")
LABELS_FILE = DATA_DIR / "labels.csv"
EMBEDDING_FILES = {
    "librosa": (DATA_DIR / "librosa_embeddings.csv", 123),
    "openl3": (DATA_DIR / "openl3_embeddings.csv", 512),
    "yamnet": (DATA_DIR / "yamnet_embeddings.csv", 1024),
    "clap": (DATA_DIR / "clap_embeddings.csv", 512),
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def prepare_files() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    files = [(LABELS_FILE, ["track_id", "message_id", "filename", "style", "energy", "darkness"])]
    for path, size in EMBEDDING_FILES.values():
        files.append((path, ["track_id"] + [f"feature_{index}" for index in range(size)]))

    for path, fields in files:
        if not path.exists():
            write_rows(path, fields, [])
        with path.open(encoding="utf-8", newline="") as file:
            if next(csv.reader(file), []) != fields:
                raise RuntimeError(f"Unexpected columns in {path}")

    label_track_ids = {int(row["track_id"]) for row in read_rows(LABELS_FILE)}
    for path, _ in EMBEDDING_FILES.values():
        embedding_track_ids = {int(row["track_id"]) for row in read_rows(path)}
        if not label_track_ids.issubset(embedding_track_ids):
            raise RuntimeError(f"Some labeled tracks are missing from {path}")


def clean_temp() -> None:
    TEMP_DIR.mkdir(exist_ok=True)
    for path in TEMP_DIR.iterdir():
        if path.is_file():
            path.unlink()


def validate_embedding(name: str, vector: np.ndarray) -> np.ndarray:
    expected_size = EMBEDDING_FILES[name][1]
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size != expected_size:
        raise ValueError(f"{name}: expected {expected_size} features, got {vector.size}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name}: embedding contains NaN or infinity")
    return vector


def extract_all(audio_path: Path) -> dict[str, np.ndarray]:
    functions = {
        "librosa": librosa.extract,
        "openl3": openl3.extract,
        "yamnet": yamnet.extract,
        "clap": clap.extract,
    }
    embeddings = {}
    for name, function in functions.items():
        print(f"  Extracting {name}...")
        embeddings[name] = validate_embedding(name, function(str(audio_path)))
    return embeddings


def save_embedding(name: str, track_id: int, vector: np.ndarray) -> None:
    path, size = EMBEDDING_FILES[name]
    fields = ["track_id"] + [f"feature_{index}" for index in range(size)]
    rows = [row for row in read_rows(path) if int(row["track_id"]) != track_id]
    row = {"track_id": track_id}
    row.update({f"feature_{index}": value for index, value in enumerate(vector)})
    rows.append(row)
    rows.sort(key=lambda item: int(item["track_id"]))
    write_rows(path, fields, rows)


def save_track(track_id: int, message_id: int, filename: str, embeddings: dict[str, np.ndarray]) -> None:
    for name, vector in embeddings.items():
        save_embedding(name, track_id, vector)

    fields = ["track_id", "message_id", "filename", "style", "energy", "darkness"]
    rows = read_rows(LABELS_FILE)
    rows.append(
        {
            "track_id": track_id,
            "message_id": message_id,
            "filename": filename,
            "style": "",
            "energy": "",
            "darkness": "",
        }
    )
    rows.sort(key=lambda item: int(item["track_id"]))
    write_rows(LABELS_FILE, fields, rows)


async def run() -> None:
    load_dotenv()
    api_id = int(os.getenv("TG_API_ID", "0"))
    api_hash = os.getenv("TG_API_HASH", "")
    channel = os.getenv("TG_CHANNEL", "")
    track_limit = int(os.getenv("TRACK_LIMIT", "300"))

    if not api_id or not api_hash or not channel:
        raise RuntimeError("Fill TG_API_ID, TG_API_HASH and TG_CHANNEL in .env")
    if track_limit < 1:
        raise RuntimeError("TRACK_LIMIT must be greater than zero")

    prepare_files()
    clean_temp()

    labels = read_rows(LABELS_FILE)
    processed_message_ids = {int(row["message_id"]) for row in labels}
    next_track_id = max((int(row["track_id"]) for row in labels), default=0) + 1
    successful_tracks = len(labels)

    if successful_tracks >= track_limit:
        print(f"Already collected {successful_tracks} tracks.")
        return

    client = create_client(api_id, api_hash)
    await client.start()

    try:
        async for message in iter_tracks(client, channel, processed_message_ids):
            if successful_tracks >= track_limit:
                break

            audio_path = None
            filename = get_filename(message)
            print(f"[{successful_tracks + 1}/{track_limit}] Downloading: {filename}")

            try:
                audio_path = await download_track(client, message, TEMP_DIR)
                embeddings = extract_all(audio_path)
                save_track(next_track_id, message.id, filename, embeddings)
                processed_message_ids.add(message.id)
                successful_tracks += 1
                next_track_id += 1
                print(f"Saved track_id={next_track_id - 1}")
            except Exception as error:
                print(f"Skipped message_id={message.id}: {type(error).__name__}: {error}")
            finally:
                if audio_path and audio_path.exists():
                    audio_path.unlink()

    finally:
        await client.disconnect()
        clean_temp()

    print(f"Done. Collected {successful_tracks}/{track_limit} tracks.")


if __name__ == "__main__":
    asyncio.run(run())
