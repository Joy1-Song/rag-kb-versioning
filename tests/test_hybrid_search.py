import pytest
from langchain_core.documents import Document
from langchain_chroma import Chroma
from uuid import uuid4

from src import file_registry
from src.hybrid_search import LexicalIndex, reciprocal_rank_fusion
from src.ingestion import IngestionService
from src.retrieval_metrics import ranking_metrics, summarize_rankings
from src.vector_store import VectorStoreService
from test_ingestion_recovery import LocalHashEmbeddings


def doc(key, text):
    return Document(page_content=text, metadata={"chunk_id": key, "source": key + ".md"})


def test_lexical_ranking_distinguishes_similar_identifiers():
    index = LexicalIndex([doc("a", "PAY-E104 means order expired"), doc("b", "PAY-E140 means pending confirmation")])
    assert index.search("PAY-E140", 2)[0].metadata["chunk_id"] == "b"


def test_lexical_index_supports_chinese_without_space_segmentation():
    index = LexicalIndex([doc("a", "课程支持下载后离线观看"), doc("b", "出差报销提供交通发票")])
    assert index.search("离线观看", 1)[0].metadata["chunk_id"] == "a"


def test_empty_index_and_zero_overlap_return_no_lexical_evidence():
    assert LexicalIndex([]).search("query", 2) == []
    assert LexicalIndex([doc("a", "中文课程")]).search("ZZZZZZ", 2) == []


def test_fusion_keeps_unique_ids_and_preserves_sources():
    a, b, c = doc("a", "alpha"), doc("b", "beta"), doc("c", "gamma")
    result = reciprocal_rank_fusion([[a, b, c], [b, b, c]], 2)
    assert [d.metadata["chunk_id"] for d in result] == ["b", "c"]
    assert result[0].metadata["source"] == "b.md"


def test_no_lexical_hits_preserve_dense_order():
    documents = [doc("b", "beta"), doc("a", "alpha")]
    assert reciprocal_rank_fusion([documents, []], 2) == documents


def test_multiple_gold_documents_need_full_recall():
    score = ranking_metrics(["a", "x", "b"], ["a", "b"], 2)
    assert score["recall"] == 0.5 and score["hit"] == 1


def test_duplicate_retrieval_ids_cannot_inflate_metric():
    with pytest.raises(ValueError):
        ranking_metrics(["a", "a"], ["a"], 2)


def test_unanswerable_cases_are_not_counted_as_perfect_recall():
    result = summarize_rankings([{"dense": ["a"], "relevant_ids": []}], "dense", 3)
    assert result["mean_recall"] is None and result["unanswerable_cases"] == 1


def test_invalid_mode_fails_before_query():
    service = VectorStoreService.__new__(VectorStoreService)
    with pytest.raises(ValueError):
        service.retrieve("query", 3, mode="unknown")


@pytest.fixture
def local_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(file_registry, "FILE_REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(file_registry, "ensure_project_dirs", lambda: None)
    service = VectorStoreService.__new__(VectorStoreService)
    service.vector_store = Chroma(collection_name="hybrid_" + uuid4().hex,
                                 embedding_function=LocalHashEmbeddings(),
                                 persist_directory=str(tmp_path / "chroma"))
    try:
        yield service
    finally:
        service.vector_store.delete_collection()


def test_hybrid_empty_database_returns_no_evidence(local_vectors):
    assert local_vectors.retrieve("policy", 3, "hybrid") == []


def test_hybrid_cache_refreshes_after_same_file_update(local_vectors):
    ingest = IngestionService(local_vectors)
    ingest.add("policy.txt", b"OLD policy", 600, 100)
    assert local_vectors.retrieve("policy", 3, "hybrid")[0].page_content == "OLD policy"
    first_cache = local_vectors._lexical_cache
    ingest.add("policy.txt", b"NEW policy", 600, 100)
    result = local_vectors.retrieve("policy", 3, "hybrid")
    assert [document.page_content for document in result] == ["NEW policy"]
    assert local_vectors._lexical_cache is not first_cache
    current_cache = local_vectors._lexical_cache
    local_vectors.retrieve("policy", 3, "hybrid")
    assert local_vectors._lexical_cache is current_cache


def test_hybrid_cache_refreshes_after_another_file_is_added(local_vectors):
    ingest = IngestionService(local_vectors)
    ingest.add("a.txt", b"First policy", 600, 100)
    local_vectors.retrieve("policy", 3, "hybrid")
    ingest.add("b.txt", b"Second policy", 600, 100)
    result = local_vectors.retrieve("policy", 3, "hybrid")
    assert {document.metadata["source"] for document in result} == {"a.txt", "b.txt"}
    assert len({document.metadata["chunk_id"] for document in result}) == 2
