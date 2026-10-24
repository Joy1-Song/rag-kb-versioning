"""Synthetic local ingestion cases; no hosted model or user database is used."""

from concurrent.futures import ThreadPoolExecutor
import copy
from types import SimpleNamespace
from threading import Event
from unittest.mock import Mock

import pytest

from src import file_registry
from src.rag_service import RagService


class MemoryVectors:
    def __init__(self):
        self.documents = {}
        self.add_calls = 0
        self.partial_failure = False

    def add_documents(self, documents):
        self.add_calls += 1
        for document in documents:
            self.documents[document.metadata["chunk_id"]] = copy.deepcopy(document)
            if self.partial_failure:
                self.partial_failure = False
                raise TimeoutError("Injected partial write")
        return len(documents)

    def document_ids(self, source):
        return [key for key, doc in self.documents.items() if source is None or doc.metadata["source"] == source]

    def delete_documents(self, ids):
        for key in ids:
            self.documents.pop(key, None)


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(file_registry, "FILE_REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(file_registry, "ensure_project_dirs", lambda: None)
    rag = RagService.__new__(RagService)
    rag.vector_store = MemoryVectors()
    rag.chat_model = Mock()
    return rag


def upload(service, text, name="policy.txt", size=40, overlap=5):
    return service.add_uploaded_files([SimpleNamespace(name=name, getvalue=lambda: text.encode())], size, overlap)


def test_exact_retry_does_not_reembed(service):
    upload(service, "Refund policy text.")
    result = upload(service, "Refund policy text.")
    assert result["skipped_files"] == 1
    assert service.vector_store.add_calls == 1


def test_same_filename_update_removes_old_text(service):
    upload(service, "OLD refund policy.")
    upload(service, "NEW refund policy.")
    assert len(service.vector_store.documents) == 1
    assert [d.page_content for d in service.vector_store.documents.values()] == ["NEW refund policy."]


def test_changed_chunk_settings_reindex_identical_file(service):
    text = "first policy sentence. " * 15
    upload(service, text, size=40)
    result = upload(service, text, size=80)
    assert result["skipped_files"] == 0
    assert result["added_chunks"] == len(service.vector_store.documents)
    assert service.vector_store.add_calls == 2


def test_two_named_documents_preserve_both_sources(service):
    upload(service, "Shared text", name="a.txt")
    upload(service, "Shared text", name="b.txt")
    assert {d.metadata["source"] for d in service.vector_store.documents.values()} == {"a.txt", "b.txt"}


def test_partial_write_then_retry_converges_to_one_revision(service):
    upload(service, "OLD terms")
    service.vector_store.partial_failure = True
    with pytest.raises(TimeoutError):
        upload(service, "New policy " * 15)
    result = upload(service, "New policy " * 15)
    assert len(service.vector_store.documents) == result["added_chunks"]
    assert all("OLD" not in d.page_content for d in service.vector_store.documents.values())


def test_blank_upload_cannot_replace_valid_document(service):
    upload(service, "Keep this evidence")
    before = copy.deepcopy(service.vector_store.documents)
    with pytest.raises(ValueError):
        upload(service, " \n\t")
    assert service.vector_store.documents == before


def test_invalid_overlap_rejected_even_if_file_previously_registered(service):
    upload(service, "Existing policy")
    with pytest.raises(ValueError):
        upload(service, "Existing policy", size=40, overlap=40)


def test_duplicate_uploads_are_serialized(service):
    entered, release = Event(), Event()
    original = service.vector_store.add_documents
    def blocked(documents):
        entered.set()
        assert release.wait(10)
        return original(documents)
    service.vector_store.add_documents = blocked
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(upload, service, "Same policy")
        assert entered.wait(10)
        second = pool.submit(upload, service, "Same policy")
        release.set()
        results = [first.result(timeout=15), second.result(timeout=15)]
    assert sum(r["skipped_files"] for r in results) == 1
    assert service.vector_store.add_calls == 1
