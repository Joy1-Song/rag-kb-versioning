"""Serialize local index updates and recover staged writes before reads."""

from contextlib import contextmanager
import hashlib
import json

from filelock import FileLock

from src import file_registry
from src.config import COLLECTION_NAME, EMBEDDING_MODEL_NAME, SEPARATORS
from src.document_loader import load_uploaded_file
from src.splitter import split_documents


class IngestionService:
    def __init__(self, vectors, embedding_model: str = EMBEDDING_MODEL_NAME, collection: str = COLLECTION_NAME):
        self.vectors = vectors
        self.embedding_model = embedding_model
        self.collection = collection

    def _load(self):
        state = file_registry.load_registry()
        if not state and self.vectors.document_ids(None):
            raise ValueError("向量库已有数据但缺少登记表，请恢复配套登记表或在独立目录重建。")
        if state.get("schema_version") == 2:
            if state["embedding_model"] != self.embedding_model or state["collection"] != self.collection:
                raise ValueError("索引模型或集合已改变，请使用独立的向量库和登记表重新入库，不能混用旧索引。")
            return state
        # Legacy manifests cannot prove embedding identity. Keep this limitation explicit.
        documents = {}
        for digest, entry in state.items():
            documents[entry["file_name"]] = {"file_hash": digest, "signature": None,
                                               "legacy": True, "chunk_count": entry["chunk_count"]}
        return {"schema_version": 2, "embedding_model": self.embedding_model, "collection": self.collection,
                "legacy_model_unverified": bool(documents), "documents": documents, "pending": None}

    def _recover(self, state):
        pending = state.get("pending")
        if not pending:
            return
        if pending["phase"] == "writing":
            # The new version was never committed. Old IDs have not been touched.
            self.vectors.delete_documents(pending["new_ids"])
        elif pending["phase"] == "ready":
            # All new chunks were acknowledged before any old chunks were removed.
            self.vectors.delete_documents(pending["old_ids"])
            state["documents"][pending["source"]] = pending["record"]
            state["legacy_model_unverified"] = any(record.get("legacy") for record in state["documents"].values())
        else:
            raise ValueError("Unknown ingestion journal phase")
        state["pending"] = None
        file_registry.save_registry(state)

    @contextmanager
    def guard(self, *, rebuilding=False):
        path = file_registry.FILE_REGISTRY_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # OS-backed file lock coordinates this application's threads and local processes.
        with FileLock(str(path) + ".lock", timeout=10):
            state = self._load()
            self._recover(state)
            if state.get("legacy_model_unverified") and not rebuilding:
                raise ValueError("旧索引未记录 Embedding 模型，请在模型未变的前提下重新上传登记过的全部文件；模型已变则需独立目录重建。")
            yield state

    def add(self, source: str, content: bytes, chunk_size: int, chunk_overlap: int) -> dict:
        if (type(chunk_size) is not int or type(chunk_overlap) is not int
                or chunk_size < 1 or not 0 <= chunk_overlap < chunk_size):
            raise ValueError("chunk_size 必须为正整数，chunk_overlap 必须非负且小于 chunk_size。")
        if not isinstance(source, str) or not source.strip() or not isinstance(content, bytes):
            raise ValueError("需要有效的文件名和文件字节内容。")
        digest = file_registry.calculate_file_hash(content)
        identity = {"pipeline": 2, "source": source, "file_hash": digest, "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap, "separators": SEPARATORS,
                    "embedding_model": self.embedding_model, "collection": self.collection}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.guard(rebuilding=True) as state:
            previous = state["documents"].get(source, {})
            if previous.get("signature") == signature:
                # Detect missing or out-of-band extra vectors, rather than trusting a stale manifest.
                if set(self.vectors.document_ids(source)) != set(previous["chunk_ids"]):
                    raise ValueError("向量记录与登记表不一致，请检查索引完整性。")
                return {"skipped": True, "chunk_count": 0}
            documents = load_uploaded_file(source, content)
            for document in documents:
                document.metadata["file_hash"] = digest
            chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            if not chunks:
                raise ValueError("文件没有可入库文本，原文档保持不变。")
            for index, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = hashlib.sha256(f"{signature}:{index}".encode()).hexdigest()
                chunk.metadata["index_version"] = signature
            new_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
            old_ids = self.vectors.document_ids(source)
            if set(new_ids) & set(old_ids):
                raise ValueError("发现未登记的相同版本片段，请检查索引完整性。")
            record = {**identity, "signature": signature, "chunk_ids": new_ids, "chunk_count": len(chunks)}
            state["pending"] = {"source": source, "phase": "writing", "new_ids": new_ids,
                                "old_ids": old_ids, "record": record}
            file_registry.save_registry(state)
            self.vectors.add_documents(chunks)
            state["pending"]["phase"] = "ready"
            file_registry.save_registry(state)
            self._recover(state)
            return {"skipped": False, "chunk_count": len(chunks)}
