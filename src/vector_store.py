from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import COLLECTION_NAME, VECTOR_STORE_DIR, ensure_project_dirs
from src.llm_client import build_embeddings
from src.ingestion import IngestionService
from src.hybrid_search import LexicalIndex, reciprocal_rank_fusion


class VectorStoreService:
    def __init__(self) -> None:
        ensure_project_dirs()
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=build_embeddings(),
            persist_directory=str(VECTOR_STORE_DIR),
        )

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0
        ids = [document.metadata["chunk_id"] for document in documents]
        self.vector_store.add_documents(documents, ids=ids)
        return len(documents)

    def retrieve(self, query: str, top_k: int, mode: str = "vector") -> list[Document]:
        if mode not in {"vector", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        with IngestionService(self).guard() as state:
            if mode == "vector":
                return self.vector_store.similarity_search(query, k=top_k)
            revision = tuple(sorted((name, record["signature"]) for name, record in state["documents"].items()))
            cache = getattr(self, "_lexical_cache", None)
            if cache is None or cache[0] != revision:
                stored = self.vector_store.get(include=["documents", "metadatas"])
                documents = [Document(page_content=text, metadata={**metadata, "chunk_id": key})
                             for key, text, metadata in zip(stored["ids"], stored["documents"], stored["metadatas"])
                             if text and text.strip()]
                self._lexical_cache = revision, LexicalIndex(documents)
            lexical = self._lexical_cache[1]
            if not lexical.documents:
                return []
            candidates = min(len(lexical.documents), max(top_k * 4, 12))
            dense = self.vector_store.similarity_search(query, k=candidates)
            return reciprocal_rank_fusion([dense, lexical.search(query, candidates)], top_k)

    def document_ids(self, source: str | None) -> list[str]:
        return self.vector_store.get(where={"source": source} if source is not None else None, include=[])["ids"]

    def delete_documents(self, ids: list[str]) -> None:
        if ids:
            self.vector_store.delete(ids=ids)
