import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, SEPARATORS


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk_hash = hashlib.sha1(
            f"{source}-{index}-{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        chunk.metadata["chunk_id"] = chunk_hash
        chunk.metadata["chunk_index"] = index
    return chunks
