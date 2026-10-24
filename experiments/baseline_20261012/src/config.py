import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UPLOAD_DIR = DATA_DIR / "uploads"
EVAL_DATA_DIR = DATA_DIR / "eval"
OUTPUTS_DIR = BASE_DIR / "outputs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "chroma"

FILE_REGISTRY_PATH = PROCESSED_DATA_DIR / "file_registry.json"
QUERY_LOG_PATH = OUTPUTS_DIR / "logs.csv"
REQUEST_LOG_PATH = OUTPUTS_DIR / "request_logs.csv"
EVAL_REPORT_PATH = OUTPUTS_DIR / "eval_report.csv"

load_dotenv(BASE_DIR / ".env")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_rag_qa")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen-plus")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

SEPARATORS = ["\n\n", "\n", "。", "；", "，", ".", ";", ",", " ", ""]


def ensure_project_dirs() -> None:
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        UPLOAD_DIR,
        EVAL_DATA_DIR,
        OUTPUTS_DIR,
        VECTOR_STORE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
