import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src import document_loader, file_registry
from src.ingestion import IngestionService
from src.rag_service import RagService
from src.vector_store import VectorStoreService
from test_ingestion_reliability import MemoryVectors, upload


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(file_registry, "FILE_REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(file_registry, "ensure_project_dirs", lambda: None)
    rag = RagService.__new__(RagService)
    rag.vector_store = MemoryVectors()
    return rag


def test_read_guard_rolls_back_partial_new_version(service):
    upload(service, "Old complete version")
    before = copy.deepcopy(service.vector_store.documents)
    service.vector_store.partial_failure = True
    with pytest.raises(TimeoutError):
        upload(service, "New incomplete revision " * 10)
    with IngestionService(service.vector_store).guard():
        assert service.vector_store.documents == before
    assert file_registry.load_registry()["pending"] is None


def test_final_manifest_failure_recovers_committed_new_revision(service, monkeypatch):
    upload(service, "Old version")
    original = file_registry.save_registry
    count = 0
    def fail_final(state):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("Injected final save failure")
        original(state)
    monkeypatch.setattr(file_registry, "save_registry", fail_final)
    with pytest.raises(OSError):
        upload(service, "New version")
    assert file_registry.load_registry()["pending"]["phase"] == "ready"
    result = upload(service, "New version")
    assert result["skipped_files"] == 1
    assert [d.page_content for d in service.vector_store.documents.values()] == ["New version"]
    assert service.vector_store.add_calls == 2


def test_old_chunk_delete_failure_blocks_read_until_recovered(service, monkeypatch):
    upload(service, "Old version")
    original = service.vector_store.delete_documents
    def fail_delete(ids):
        raise OSError("Injected vector delete failure")
    monkeypatch.setattr(service.vector_store, "delete_documents", fail_delete)
    with pytest.raises(OSError):
        upload(service, "New version")
    with pytest.raises(OSError):
        with IngestionService(service.vector_store).guard():
            pytest.fail("Read must not see mixed revisions")
    monkeypatch.setattr(service.vector_store, "delete_documents", original)
    with IngestionService(service.vector_store).guard():
        assert [d.page_content for d in service.vector_store.documents.values()] == ["New version"]


def test_different_embedding_model_is_rejected_before_write(service):
    upload(service, "Policy")
    with pytest.raises(ValueError, match="索引模型"):
        IngestionService(service.vector_store, embedding_model="different-embedding").add("policy.txt", b"Policy", 40, 5)
    assert service.vector_store.add_calls == 1


def test_corrupted_manifest_fails_closed(service):
    upload(service, "Policy")
    file_registry.FILE_REGISTRY_PATH.write_text("{incomplete", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        upload(service, "New policy")
    assert service.vector_store.add_calls == 1


def test_missing_registered_vectors_do_not_silently_skip(service):
    upload(service, "Policy")
    service.vector_store.documents.clear()
    with pytest.raises(ValueError, match="不一致"):
        upload(service, "Policy")


def test_missing_manifest_cannot_expose_untracked_vectors(service):
    upload(service, "Policy")
    file_registry.FILE_REGISTRY_PATH.unlink()
    with pytest.raises(ValueError, match="缺少登记表"):
        with IngestionService(service.vector_store).guard():
            pytest.fail("Untracked vectors must not be read")


def test_legacy_index_requires_complete_reindex_before_reads(service):
    file_registry.save_registry({"old-a": {"file_name": "a.txt", "chunk_count": 1},
                                 "old-b": {"file_name": "b.txt", "chunk_count": 1}})
    with pytest.raises(ValueError, match="旧索引"):
        with IngestionService(service.vector_store).guard():
            pass
    upload(service, "New a", name="a.txt")
    with pytest.raises(ValueError, match="旧索引"):
        with IngestionService(service.vector_store).guard():
            pass
    upload(service, "New b", name="b.txt")
    with IngestionService(service.vector_store).guard():
        assert not file_registry.load_registry()["legacy_model_unverified"]


def test_pdf_parse_failure_still_deletes_temporary_file(monkeypatch):
    paths = []
    def broken_loader(path):
        paths.append(Path(path))
        return SimpleNamespace(load=lambda: (_ for _ in ()).throw(ValueError("Malformed PDF")))
    monkeypatch.setattr(document_loader, "PyPDFLoader", broken_loader)
    with pytest.raises(ValueError):
        document_loader.load_uploaded_file("bad.pdf", b"not pdf")
    assert paths and not paths[0].exists()


class LocalHashEmbeddings(Embeddings):
    """Deterministic API substitute; deliberately not a semantic quality benchmark."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [float(value) / 255 for value in hashlib.sha256(text.encode()).digest()[:8]]


def test_real_local_chroma_update_and_retry(service, tmp_path):
    vectors = VectorStoreService.__new__(VectorStoreService)
    vectors.vector_store = Chroma(collection_name="test_" + uuid4().hex,
                                 embedding_function=LocalHashEmbeddings(), persist_directory=str(tmp_path / "chroma"))
    service.vector_store = vectors
    try:
        upload(service, "Old policy")
        upload(service, "New policy")
        result = upload(service, "New policy")
        assert result["skipped_files"] == 1
        assert len(vectors.document_ids("policy.txt")) == 1
        assert [doc.page_content for doc in vectors.retrieve("New policy", 4)] == ["New policy"]
    finally:
        vectors.vector_store.delete_collection()
