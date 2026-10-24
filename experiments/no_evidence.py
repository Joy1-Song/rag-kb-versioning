"""No evidence, no model call: 10 fixed-retrieval cases (6 with empty or blank evidence, 4 with evidence).

Retrieval is fixed by data/eval/no_evidence_cases.json; only answer generation uses a real model.
Saved runs (2026-10-09, qwen-plus, temperature 0): results/no_evidence/live_before.json (before the
short circuit) and live_after.json (this repository's src/).
python experiments/no_evidence.py --report          -> results/no_evidence/report.md, no request
python experiments/no_evidence.py --approved-live   -> a new run of src/ (at most 10 requests; reads
                                                       DASHSCOPE_API_KEY from the environment)
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data/eval/no_evidence_cases.json"
RESULTS = ROOT / "results/no_evidence"
LIMIT = 10


def calls_and_tokens(run, kind=None):
    cases = [c for c in run["cases"] if kind is None or c["kind"] == kind]
    calls = sum(len(c["calls"]) for c in cases)
    tokens = sum((call.get("usage") or {}).get("total_tokens", 0) for c in cases for call in c["calls"])
    return calls, tokens


def report():
    runs = {name: json.loads((RESULTS / f"live_{name}.json").read_text(encoding="utf-8")) for name in ("before", "after")}
    lines = ["# 无证据短路：模型调用对比", "",
             "10 例自编输入，检索结果固定（6 例为空或只有空白片段，4 例有有效片段），只有回答生成调用真实模型。",
             "这组输入专门构造了无证据的情况，调用减少的比例只描述这 10 例，不代表线上流量、成本或回答准确率。", "",
             "| 运行 | 模型调用（无证据 6 例） | 模型调用（有证据 4 例） | 合计调用 | 接口返回 token |",
             "|---|---:|---:|---:|---:|"]
    for name, run in runs.items():
        empty, _ = calls_and_tokens(run, "no_evidence")
        grounded, _ = calls_and_tokens(run, "with_evidence")
        total, tokens = calls_and_tokens(run)
        lines.append(f"| {name}（{run['recorded_at'][:10]}） | {empty} | {grounded} | {total} | {tokens} |")
    lines += ["", "## 逐例回答", "", "| 例子 | 类型 | before 调用 | before 回答 | after 调用 | after 回答 |", "|---|---|---:|---|---:|---|"]
    after = {c["id"]: c for c in runs["after"]["cases"]}
    for case in runs["before"]["cases"]:
        other = after[case["id"]]
        short = lambda c: (c.get("result") or {}).get("answer", c.get("error_type", "")).replace("\n", " ")[:60]
        lines.append(f"| {case['id']} | {case['kind']} | {len(case['calls'])} | {short(case)} | {len(other['calls'])} | {short(other)} |")
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))


class FixedRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k):
        return self.documents[:top_k]


class LiveChat:
    def __init__(self, client, model, budget):
        self.client, self.model, self.budget = client, model, budget
        self.calls = []

    def invoke(self, messages):
        if self.budget["used"] >= LIMIT:
            raise RuntimeError("Request limit reached; no request sent.")
        self.budget["used"] += 1
        payload = [{"role": "system" if message.type == "system" else "user", "content": message.content}
                   for message in messages]
        started = time.perf_counter()
        entry = {"model": self.model, "max_tokens": 512,
                 "prompt_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()}
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=payload, temperature=0, max_tokens=512,
                extra_body={"enable_thinking": False}, timeout=20)
            entry.update(status="succeeded", finish_reason=response.choices[0].finish_reason,
                         usage=None if response.usage is None else response.usage.model_dump())
            return SimpleNamespace(content=response.choices[0].message.content or "")
        except Exception as error:
            entry.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
            self.calls.append(entry)


def live():
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("No DASHSCOPE_API_KEY in the environment; no request sent.")
    sys.path.insert(0, str(ROOT))
    from langchain_core.documents import Document
    from openai import OpenAI
    from src.rag_service import RagService
    model = os.getenv("CHAT_MODEL_NAME", "qwen-plus")
    stamp = datetime.now(timezone.utc)
    output = RESULTS / f"live_rerun_{stamp:%Y%m%dT%H%M%SZ}.json"
    result = {"phase": "rerun", "recorded_at": stamp.isoformat(), "model": model, "temperature": 0,
              "max_output_tokens": 512, "sdk_retries": 0,
              "fixture_sha256": hashlib.sha256(FIXTURES.read_bytes()).hexdigest(), "cases": []}
    budget = {"used": 0}
    with OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", max_retries=0, timeout=20) as client:
        for case in json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]:
            documents = [Document(page_content=text, metadata={"source": "self_authored_fixture.md", "chunk_id": f"{case['id']}-{i}"})
                         for i, text in enumerate(case["documents"])]
            service = RagService.__new__(RagService)
            service.vector_store = FixedRetriever(documents)
            service.chat_model = LiveChat(client, model, budget)
            row = {"id": case["id"], "kind": case["kind"], "question": case["question"], "documents": case["documents"]}
            try:
                row.update(status="succeeded", result=service.ask(case["question"], 4))
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__)
            row["calls"] = service.chat_model.calls
            result["cases"].append(row)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(case["id"], row["status"], "calls=", len(row["calls"]), flush=True)
    print("->", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--report", action="store_true")
    group.add_argument("--approved-live", action="store_true")
    args = parser.parse_args()
    report() if args.report else live()
