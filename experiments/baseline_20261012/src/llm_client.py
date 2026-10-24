import os

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from src.config import CHAT_MODEL_NAME, EMBEDDING_MODEL_NAME


def require_dashscope_api_key() -> None:
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请先在 .env 文件中配置。")


def build_embeddings() -> DashScopeEmbeddings:
    require_dashscope_api_key()
    return DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME)


def build_chat_model() -> ChatTongyi:
    require_dashscope_api_key()
    return ChatTongyi(model=CHAT_MODEL_NAME)
