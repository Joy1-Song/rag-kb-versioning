import os
import time
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from src.config import BASE_DIR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K
from src.evaluator import evaluate_keywords
from src.logger import log_eval, log_query, log_request
from src.rag_service import RagService


load_dotenv(BASE_DIR / ".env")
api_ready = bool(os.getenv("DASHSCOPE_API_KEY"))

st.set_page_config(
    page_title="Enterprise RAG QA",
    layout="wide",
)

st.title("企业知识库 RAG 问答与评测系统")
st.caption("支持文档上传、语义检索、基于资料回答、引用来源展示和回答质量评测。")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "last_question" not in st.session_state:
    st.session_state["last_question"] = ""

if "last_answer_payload" not in st.session_state:
    st.session_state["last_answer_payload"] = None

with st.sidebar:
    st.header("知识库构建")
    uploaded_files = st.file_uploader(
        "上传企业文档",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("chunk_size", 300, 1500, DEFAULT_CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("chunk_overlap", 0, 300, DEFAULT_CHUNK_OVERLAP, step=20)
    top_k = st.slider("Top-k 检索片段数", 1, 8, DEFAULT_TOP_K)

    build_clicked = st.button("构建 / 更新知识库", type="primary")

    st.divider()
    st.caption("API Key 从 `.env` 读取，不会显示在页面中。")

if not api_ready:
    st.warning("请先复制 `.env.example` 为 `.env`，并填写 DASHSCOPE_API_KEY。")

if build_clicked:
    if not api_ready:
        st.error("请先配置 DASHSCOPE_API_KEY，再构建知识库。")
    elif not uploaded_files:
        st.error("请先上传至少一个 txt、md 或 pdf 文件。")
    else:
        try:
            if "rag_service" not in st.session_state:
                st.session_state["rag_service"] = RagService()
            with st.spinner("正在解析文件、切分文本并写入向量库..."):
                result = st.session_state["rag_service"].add_uploaded_files(
                    uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            st.success(
                f"处理完成：新增或更新 {result['added_files']} 个文件，"
                f"跳过 {result['skipped_files']} 个重复文件，"
                f"写入 {result['added_chunks']} 个文本片段。"
            )
        except Exception as exc:
            st.error(f"入库未完成：{exc}。已完成的文件保留；恢复连接后可重试原文件。")

left, right = st.columns([0.62, 0.38])

with left:
    st.subheader("问答界面")
    for message in st.session_state["messages"]:
        st.chat_message(message["role"]).markdown(message["content"])

    question = st.chat_input("请输入和企业资料相关的问题")

    if question:
        st.session_state["last_question"] = question
        st.session_state["last_answer_payload"] = None
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").markdown(question)

        request_id = uuid4().hex
        started_at = time.perf_counter()
        answer_payload = None
        status = "failed"
        error_type = ""
        try:
            if not api_ready:
                error_type = "MissingAPIKey"
                st.error("请先配置 DASHSCOPE_API_KEY，再进行问答。")
            else:
                with st.spinner("正在检索资料并生成回答..."):
                    if "rag_service" not in st.session_state:
                        st.session_state["rag_service"] = RagService()
                    answer_payload = st.session_state["rag_service"].ask(
                        question,
                        top_k=top_k,
                    )
                status = "success"
        except Exception as exc:
            error_type = type(exc).__name__
            st.error(f"调用失败：{exc}")
        finally:
            # Count failed requests as well; stop timing before writing the log.
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            try:
                log_request(
                    request_id=request_id,
                    question=question,
                    status=status,
                    error_type=error_type,
                    duration_ms=duration_ms,
                    top_k=top_k,
                )
            except OSError:
                st.warning("本次请求日志写入失败，请检查输出目录的空间和权限。")

        if answer_payload:
            st.session_state["last_answer_payload"] = answer_payload
            st.chat_message("assistant").markdown(answer_payload["answer"])
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer_payload["answer"]}
            )

            try:
                log_query(
                    question=question,
                    answer=answer_payload["answer"],
                    retrieved_docs=answer_payload["retrieved_docs"],
                    latency_ms=answer_payload["latency_ms"],
                    top_k=top_k,
                )
            except OSError:
                st.warning("问答明细日志写入失败，本次回答仍可查看和评测。")

            with st.expander("引用来源"):
                for source in answer_payload["sources"]:
                    st.write(f"- {source}")

            with st.expander("检索到的资料片段"):
                for index, item in enumerate(answer_payload["retrieved_docs"], start=1):
                    st.markdown(f"**片段 {index} | 来源：{item['source']}**")
                    st.write(item["content"])

with right:
    st.subheader("评测面板")

    question_id = st.text_input("question_id", value="q001")
    gold_answer = st.text_area(
        "标准答案 / 期望答案",
        placeholder="例如：员工出差报销需要提交审批单、发票和行程单。",
    )
    expected = st.text_area(
        "期望关键词，用逗号分隔",
        placeholder="例如：审批单, 发票, 行程单",
    )
    citation_valid = st.selectbox("引用是否有效", ["未判断", "是", "否"])
    failure_type = st.selectbox(
        "失败类型",
        [
            "无",
            "检索失败",
            "检索命中但回答错误",
            "引用无效",
            "资料无答案但模型编造",
            "回答不完整",
            "格式不符合要求",
        ],
    )
    note = st.text_area("备注", placeholder="记录失败原因或优化想法。")

    if st.button("评测最近一次回答", key="evaluate_answer"):
        answer_payload = st.session_state["last_answer_payload"]
        if not answer_payload:
            st.error("还没有可评测的回答。")
        else:
            keywords = [item.strip() for item in expected.split(",") if item.strip()]
            report = evaluate_keywords(answer_payload["answer"], keywords)
            st.metric("关键词命中率", f"{report['hit_rate']:.0%}")
            st.write("命中：", ", ".join(report["hit_keywords"]) or "无")
            st.write("未命中：", ", ".join(report["missed_keywords"]) or "无")

            log_eval(
                question_id=question_id,
                question=st.session_state["last_question"],
                gold_answer=gold_answer,
                retrieved_docs=answer_payload["retrieved_docs"],
                model_answer=answer_payload["answer"],
                keyword_report=report,
                citation_valid=citation_valid,
                failure_type=failure_type,
                note=note,
            )
            st.success("已写入 outputs/eval_report.csv。")
