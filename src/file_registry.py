import hashlib
import json
import os
from datetime import datetime
from tempfile import NamedTemporaryFile

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
    registry = load_registry()
    if registry.get("schema_version") == 2:
        return any(item["file_hash"] == file_hash for item in registry["documents"].values())
    return file_hash in registry


def save_registry(registry: dict) -> None:
    FILE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=FILE_REGISTRY_PATH.parent,
                                suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            json.dump(registry, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, FILE_REGISTRY_PATH)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def register_file(file_hash: str, file_name: str, chunk_count: int) -> None:
    registry = load_registry()
    if registry.get("schema_version") == 2:
        raise ValueError("Use IngestionService to update the versioned index")
    registry[file_hash] = {
        "file_name": file_name,
        "chunk_count": chunk_count,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_registry(registry)
