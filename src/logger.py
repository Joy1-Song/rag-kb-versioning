import csv
import json
from datetime import datetime
from pathlib import Path

from src.config import EVAL_REPORT_PATH, QUERY_LOG_PATH, REQUEST_LOG_PATH, ensure_project_dirs


QUERY_LOG_FIELDS = [
    "created_at",
    "question",
    "retrieved_chunk_ids",
    "retrieved_sources",
    "model_answer",
    "latency_ms",
    "top_k",
]

REQUEST_LOG_FIELDS = [
    "created_at",
    "request_id",
    "question",
    "status",
    "error_type",
    "duration_ms",
    "top_k",
]

EVAL_REPORT_FIELDS = [
    "created_at",
    "question_id",
    "question",
    "gold_answer",
    "retrieved_chunk_ids",
    "is_gold_chunk_retrieved",
    "model_answer",
    "answer_correct",
    "citation_valid",
    "hallucination",
    "completeness_score",
    "failure_type",
    "note",
]


def log_request(
    request_id: str,
    question: str,
    status: str,
    error_type: str,
    duration_ms: int,
    top_k: int,
) -> None:
    row = {
        "created_at": _now(),
        "request_id": request_id,
        "question": question,
        "status": status,
        "error_type": error_type,
        "duration_ms": duration_ms,
        "top_k": top_k,
    }
    _append_csv(REQUEST_LOG_PATH, REQUEST_LOG_FIELDS, row)


def log_query(
    question: str,
    answer: str,
    retrieved_docs: list[dict],
    latency_ms: int,
    top_k: int,
) -> None:
    row = {
        "created_at": _now(),
        "question": question,
        "retrieved_chunk_ids": _json([doc.get("chunk_id", "") for doc in retrieved_docs]),
        "retrieved_sources": _json([doc.get("source", "") for doc in retrieved_docs]),
        "model_answer": answer,
        "latency_ms": latency_ms,
        "top_k": top_k,
    }
    _append_csv(QUERY_LOG_PATH, QUERY_LOG_FIELDS, row)


def log_eval(
    question_id: str,
    question: str,
    gold_answer: str,
    retrieved_docs: list[dict],
    model_answer: str,
    keyword_report: dict,
    citation_valid: str,
    failure_type: str,
    note: str,
) -> None:
    hit_rate = keyword_report.get("hit_rate", 0)
    answer_correct = "是" if hit_rate >= 0.8 else "否"
    hallucination = "是" if failure_type == "资料无答案但模型编造" else "否"
    row = {
        "created_at": _now(),
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "retrieved_chunk_ids": _json([doc.get("chunk_id", "") for doc in retrieved_docs]),
        "is_gold_chunk_retrieved": "人工待判断",
        "model_answer": model_answer,
        "answer_correct": answer_correct,
        "citation_valid": citation_valid,
        "hallucination": hallucination,
        "completeness_score": f"{hit_rate:.2f}",
        "failure_type": failure_type,
        "note": note,
    }
    _append_csv(EVAL_REPORT_PATH, EVAL_REPORT_FIELDS, row)


def _append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    ensure_project_dirs()
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
