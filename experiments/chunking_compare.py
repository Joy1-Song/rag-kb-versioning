"""Chunk size / overlap / top-k on longer policy documents, scored by whether the needed sentences are retrieved.

Grid and selection rule are fixed in results/chunking/frozen.json, written before any run:
pick on the development set only, then score the frozen test set once for the chosen configuration and for
the current default (600 / 100 / top 4). Chunks come from the app's own splitter (src/splitter.py).

A case is a full hit when every evidence sentence lies inside the union of the retrieved chunks' character
ranges, so two adjacent retrieved chunks may together cover a sentence that a boundary cut in half.
Context length = characters handed to the model (sum of retrieved chunks), a proxy for reading cost and noise.
Retrieval is cosine similarity over text-embedding-v4 vectors, the same ranking as the app's vector mode.

python experiments/chunking_compare.py                  -> results/chunking/report.{json,md}, cached vectors only
python experiments/chunking_compare.py --approved-live  embeds missing texts (at most 500); reads DASHSCOPE_API_KEY
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/eval/chunking"
OUT = ROOT / "results/chunking"
CACHE = OUT / "embedding_cache.json"
FROZEN = OUT / "frozen.json"
EMBEDDING = "text-embedding-v4"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEXT_LIMIT = 500


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    frozen = read_json(FROZEN)
    if sha(DATA / "test.json") != frozen["test_sha256"]:
        raise SystemExit("test.json changed after it was frozen; write a new test set instead.")
    docs = {}
    for path in sorted((DATA / "corpus").glob("*.md")):
        if frozen["corpus_sha256"].get(path.name) != sha(path):
            raise SystemExit(f"{path.name} changed after the freeze; no run.")
        docs[path.name] = path.read_text(encoding="utf-8")
    splits = {name: read_json(DATA / f"{name}.json")["cases"] for name in ("dev", "test")}
    for cases in splits.values():
        for case in cases:
            text = docs[case["doc"]]
            case["spans"] = []
            for sentence in case["evidence"]:
                if text.count(sentence) != 1:
                    raise ValueError(f"{case['id']}: evidence must occur exactly once in {case['doc']}")
                start = text.index(sentence)
                case["spans"].append((start, start + len(sentence)))
    return frozen, docs, splits


def chunk_corpus(docs, size, overlap):
    from langchain_core.documents import Document
    from src.splitter import split_documents
    chunks = []
    for name, text in docs.items():
        cursor = 0
        for piece in split_documents([Document(page_content=text, metadata={"source": name})], size, overlap):
            body = piece.page_content
            start = text.find(body, cursor)
            if start < 0:
                start = text.find(body)
            if start < 0:
                raise ValueError(f"chunk not found in {name}")
            chunks.append({"doc": name, "start": start, "end": start + len(body), "text": body})
            cursor = start + 1
    return chunks


def embed(texts, approved):
    cache = read_json(CACHE) if CACHE.exists() else {"model": EMBEDDING, "dimensions": 1024, "vectors": {}}
    if cache["model"] != EMBEDDING:
        raise ValueError("embedding cache was made with a different model")
    missing = [t for t in dict.fromkeys(texts) if t not in cache["vectors"]]
    if missing:
        if not approved:
            raise SystemExit(f"{len(missing)} texts are not cached; rerun with --approved-live. No request sent.")
        if len(missing) > TEXT_LIMIT:
            raise SystemExit(f"{len(missing)} texts exceed the {TEXT_LIMIT}-text limit; no request sent.")
        key = os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise SystemExit("No DASHSCOPE_API_KEY in the environment; no request sent.")
        from openai import OpenAI
        with OpenAI(api_key=key, base_url=ENDPOINT, max_retries=0, timeout=30) as client:
            for offset in range(0, len(missing), 10):
                batch = missing[offset:offset + 10]
                response = client.embeddings.create(model=EMBEDDING, input=batch, dimensions=1024,
                                                    encoding_format="float")
                for text, row in zip(batch, sorted(response.data, key=lambda r: r.index)):
                    cache["vectors"][text] = row.embedding
                CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"embedded {len(missing)} new texts", flush=True)
    return cache["vectors"]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0))


def covered(span, ranges):
    """True when [start, end) lies inside the union of the given ranges."""
    start, end = span
    point = start
    for lo, hi in sorted(ranges):
        if lo <= point < hi:
            point = hi
        if point >= end:
            return True
    return point >= end


def score(cases, chunks, vectors, k):
    rows = []
    chunk_vectors = [vectors[c["text"]] for c in chunks]
    for case in cases:
        q = vectors[case["question"]]
        order = sorted(range(len(chunks)), key=lambda i: -cosine(q, chunk_vectors[i]))[:k]
        picked = [chunks[i] for i in order]
        ranges = [(c["start"], c["end"]) for c in picked if c["doc"] == case["doc"]]
        hits = [covered(span, ranges) for span in case["spans"]]
        rows.append({"id": case["id"], "full_hit": all(hits), "evidence_hit": sum(hits), "evidence_total": len(hits),
                     "context_chars": sum(len(c["text"]) for c in picked),
                     "same_doc_chunks": len(ranges)})
    n = len(rows)
    return {"cases": n, "full_hit": sum(r["full_hit"] for r in rows),
            "evidence_recall": round(sum(r["evidence_hit"] for r in rows) / sum(r["evidence_total"] for r in rows), 4),
            "mean_context_chars": round(sum(r["context_chars"] for r in rows) / n, 1),
            "misses": [r["id"] for r in rows if not r["full_hit"]], "rows": rows}


def name(cfg):
    return f"{cfg[0]}/{cfg[1]}/top{cfg[2]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-live", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    frozen, docs, splits = load_inputs()
    grid = frozen["grid"]
    layouts = {(s, o): chunk_corpus(docs, s, o) for s in grid["chunk_size"] for o in grid["chunk_overlap"]}
    texts = [c["text"] for chunks in layouts.values() for c in chunks]
    texts += [case["question"] for cases in splits.values() for case in cases]
    vectors = embed(texts, args.approved_live)

    dev = {}
    for (size, overlap), chunks in layouts.items():
        for k in grid["top_k"]:
            dev[(size, overlap, k)] = score(splits["dev"], chunks, vectors, k)
    chosen = min(dev, key=lambda c: (-dev[c]["full_hit"], dev[c]["mean_context_chars"], c[2], c[0]))
    default = tuple(frozen["current_default"][x] for x in ("chunk_size", "chunk_overlap", "top_k"))
    test = {cfg: score(splits["test"], layouts[cfg[:2]], vectors, cfg[2]) for cfg in dict.fromkeys([chosen, default])}

    report = {
        "embedding_model": EMBEDDING, "frozen": frozen,
        "corpus_chars": {n: len(t) for n, t in docs.items()},
        "chunks_per_layout": {f"{s}/{o}": len(c) for (s, o), c in layouts.items()},
        "dev": {name(c): {k: v for k, v in r.items() if k != "rows"} for c, r in dev.items()},
        "chosen": name(chosen), "default": name(default),
        "test": {name(c): r for c, r in test.items()},
        "scope": "Self-written corpus (4 policies, ~3.8k characters) and questions; retrieval only, no answer generation.",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 切分长度 / 重叠 / 返回条数对比", "",
             f"语料：4 篇自编制度文档，共 {sum(report['corpus_chars'].values())} 字；向量模型 {EMBEDDING}；只评检索，不生成回答。",
             "完整命中 = 问题需要的原句全部落在返回片段里；上下文长度 = 交给模型的字符数。", "",
             f"## 开发集（{len(splits['dev'])} 题，用来选配置）", "",
             "| 切分长度 / 重叠 / 返回条数 | 完整命中 | 原句召回 | 平均上下文（字） |", "|---|---:|---:|---:|"]
    for cfg in sorted(dev, key=lambda c: (-dev[c]["full_hit"], dev[c]["mean_context_chars"])):
        r = dev[cfg]
        mark = "（选中）" if cfg == chosen else "（现默认）" if cfg == default else ""
        lines.append(f"| {name(cfg)}{mark} | {r['full_hit']}/{r['cases']} | {r['evidence_recall']:.0%} | {r['mean_context_chars']:.0f} |")
    lines += ["", f"## 冻结测试集（{len(splits['test'])} 题，只评一次）", "",
              "| 配置 | 完整命中 | 原句召回 | 平均上下文（字） | 未完整命中 |", "|---|---:|---:|---:|---|"]
    for cfg, r in test.items():
        label = "选中" if cfg == chosen else "现默认"
        lines.append(f"| {label} {name(cfg)} | {r['full_hit']}/{r['cases']} | {r['evidence_recall']:.0%} | "
                     f"{r['mean_context_chars']:.0f} | {', '.join(r['misses']) or '无'} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
