from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


TEXT_SUFFIXES = {".txt", ".md"}


def decode_text(content: bytes) -> str:
    for encoding in ["utf-8", "utf-8-sig", "gb18030"]:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="ignore")


def load_uploaded_file(file_name: str, content: bytes) -> list[Document]:
    suffix = Path(file_name).suffix.lower()

    if suffix in TEXT_SUFFIXES:
        text = decode_text(content)
        return [
            Document(
                page_content=text,
                metadata={"source": file_name, "file_type": suffix.lstrip(".")},
            )
        ]

    if suffix == ".pdf":
        return _load_pdf_bytes(file_name, content)

    raise ValueError(f"暂不支持的文件类型：{suffix}")


def load_many_uploaded_files(files: Iterable) -> list[tuple[str, bytes, list[Document]]]:
    loaded = []
    for uploaded_file in files:
        content = uploaded_file.getvalue()
        documents = load_uploaded_file(uploaded_file.name, content)
        loaded.append((uploaded_file.name, content, documents))
    return loaded


def _load_pdf_bytes(file_name: str, content: bytes) -> list[Document]:
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    for document in documents:
        page = document.metadata.get("page")
        document.metadata.update(
            {
                "source": file_name,
                "file_type": "pdf",
                "page": page + 1 if isinstance(page, int) else page,
            }
        )
    return documents
