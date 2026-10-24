from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import COLLECTION_NAME, VECTOR_STORE_DIR, ensure_project_dirs
from src.llm_client import build_embeddings


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

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        return self.vector_store.similarity_search(query, k=top_k)
