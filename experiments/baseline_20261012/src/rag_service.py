import time

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.document_loader import load_many_uploaded_files
from src.file_registry import calculate_file_hash, is_registered, register_file
from src.llm_client import build_chat_model
from src.splitter import split_documents
from src.vector_store import VectorStoreService


SYSTEM_PROMPT = """你是企业知识库问答助手。
请仅基于【资料片段】回答用户问题。
如果资料中没有答案，请回答“资料中没有明确说明”，不要编造。

回答要求：
1. 只基于资料回答。
2. 不确定时明确说明资料不足。
3. 在最后列出引用来源。
4. 不要输出资料之外的推测。
5. 给出可信度：高 / 中 / 低。
"""


class RagService:
    def __init__(self) -> None:
        self.vector_store = VectorStoreService()
        self.chat_model = build_chat_model()

    def add_uploaded_files(
        self,
        uploaded_files,
        chunk_size: int,
        chunk_overlap: int,
    ) -> dict:
        loaded_files = load_many_uploaded_files(uploaded_files)
        added_files = 0
        skipped_files = 0
        added_chunks = 0

        for file_name, content, documents in loaded_files:
            file_hash = calculate_file_hash(content)
            if is_registered(file_hash):
                skipped_files += 1
                continue

            for document in documents:
                document.metadata["file_hash"] = file_hash

            chunks = split_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            self.vector_store.add_documents(chunks)
            register_file(file_hash, file_name, len(chunks))
            added_files += 1
            added_chunks += len(chunks)

        return {
            "added_files": added_files,
            "skipped_files": skipped_files,
            "added_chunks": added_chunks,
        }

    def ask(self, question: str, top_k: int) -> dict:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        question = question.strip()
        if not question:
            raise ValueError("question must not be blank")
        if type(top_k) is not int:
            raise TypeError("top_k must be an integer, not a boolean")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        start_time = time.perf_counter()
        retrieved_docs = self.vector_store.retrieve(question, top_k=top_k)
        retrieved_docs = [document for document in retrieved_docs if document.page_content.strip()]
        # Empty evidence is a program decision, not a question for the model.
        if not retrieved_docs:
            return {
                "answer": "资料中没有明确说明。当前未检索到可用资料，请补充相关文档后重试。",
                "sources": [],
                "retrieved_docs": [],
                "latency_ms": int((time.perf_counter() - start_time) * 1000),
            }
        context = self._format_context(retrieved_docs)
        user_prompt = f"""【资料片段】
{context}

【用户问题】
{question}
"""
        response = self.chat_model.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        answer = response.content if hasattr(response, "content") else str(response)
        return {
            "answer": answer,
            "sources": self._format_sources(retrieved_docs),
            "retrieved_docs": self._serialize_docs(retrieved_docs),
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        if not documents:
            return "无相关资料。"

        blocks = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            chunk_id = document.metadata.get("chunk_id", "")
            page_text = f"，页码：{page}" if page else ""
            blocks.append(
                f"[{index}] 来源：{source}{page_text}，chunk_id：{chunk_id}\n"
                f"内容：{document.page_content}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _format_sources(documents: list[Document]) -> list[str]:
        sources = []
        for document in documents:
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            label = f"{source} 第 {page} 页" if page else source
            if label not in sources:
                sources.append(label)
        return sources

    @staticmethod
    def _serialize_docs(documents: list[Document]) -> list[dict]:
        serialized = []
        for document in documents:
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            source_label = f"{source} 第 {page} 页" if page else source
            serialized.append(
                {
                    "chunk_id": document.metadata.get("chunk_id", ""),
                    "source": source_label,
                    "content": document.page_content,
                    "metadata": document.metadata,
                }
            )
        return serialized
