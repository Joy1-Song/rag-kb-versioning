import hashlib
import json
from datetime import datetime

from src.config import FILE_REGISTRY_PATH, ensure_project_dirs


def calculate_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_registry() -> dict:
    ensure_project_dirs()
    if not FILE_REGISTRY_PATH.exists():
        return {}
    with FILE_REGISTRY_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_registered(file_hash: str) -> bool:
    return file_hash in load_registry()


def register_file(file_hash: str, file_name: str, chunk_count: int) -> None:
    registry = load_registry()
    registry[file_hash] = {
        "file_name": file_name,
        "chunk_count": chunk_count,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with FILE_REGISTRY_PATH.open("w", encoding="utf-8") as file:
        json.dump(registry, file, ensure_ascii=False, indent=2)
