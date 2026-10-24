"""Pure vector vs character TF-IDF + RRF on the frozen 20-question retrieval set.

Uses the cached text-embedding-v4 vectors in results/retrieval/embedding_cache.json, so both modes see
identical embeddings and the comparison reruns without any model request.
python experiments/retrieval_compare.py              -> results/retrieval/rerun.json
python experiments/retrieval_compare.py --approved-live
    only when the benchmark changed and texts are missing from the cache; reads DASHSCOPE_API_KEY from
    the environment and embeds at most 60 texts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data/eval/retrieval_benchmark.json"
CACHE = ROOT / "results/retrieval/embedding_cache.json"
EMBEDDING = "text-embedding-v4"
ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
TEXT_LIMIT = 60


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def vectors_for(fixture, approved):
    texts = [doc["text"] for doc in fixture["documents"]] + [case["question"] for case in fixture["cases"]]
    digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    cached = read_json(CACHE)
    if cached["model"] != EMBEDDING:
        raise ValueError("Embedding cache was made with a different model")
    missing = [text for text in texts if text not in cached["vectors"]]
    if not missing:
        if cached["fixture_sha256"] != digest:
            print("note: benchmark file changed, but every text is in the cache", flush=True)
        return cached["vectors"]
    if not approved:
        raise SystemExit(f"{len(missing)} texts are not in the cache; rerun with --approved-live. No request sent.")
    if len(missing) > TEXT_LIMIT:
        raise SystemExit(f"{len(missing)} texts exceed the {TEXT_LIMIT}-text limit; no request sent.")
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("No DASHSCOPE_API_KEY in the environment; no request sent.")
    from openai import OpenAI
    with OpenAI(api_key=key, base_url=ENDPOINT, max_retries=0, timeout=20) as client:
        for offset in range(0, len(missing), 10):
            batch = missing[offset:offset + 10]
            response = client.embeddings.create(model=EMBEDDING, input=batch, dimensions=1024, encoding_format="float")
            values = sorted(response.data, key=lambda row: row.index)
            cached["vectors"].update({text: value.embedding for text, value in zip(batch, values)})
    cached["fixture_sha256"] = digest
    CACHE.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
    return cached["vectors"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-live", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "results/retrieval/rerun.json")
    args = parser.parse_args()
    fixture = read_json(FIXTURE)
    vectors = vectors_for(fixture, args.approved_live)
    os.environ["EMBEDDING_MODEL_NAME"] = EMBEDDING
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    sys.path.insert(0, str(ROOT))
    from langchain_core.embeddings import Embeddings
    from langchain_chroma import Chroma
    from src import file_registry
    from src.ingestion import IngestionService
    from src.retrieval_metrics import summarize_rankings
    from src.vector_store import VectorStoreService

    class CachedEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [vectors[text] for text in texts]

        def embed_query(self, text):
            return vectors[text]

    # Chroma keeps its files open on Windows; leftover temp files are harmless.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        scratch = Path(tmp)
        file_registry.FILE_REGISTRY_PATH = scratch / "registry.json"
        file_registry.ensure_project_dirs = lambda: None
        retriever = VectorStoreService.__new__(VectorStoreService)
        retriever.vector_store = Chroma(collection_name="quality_eval", embedding_function=CachedEmbeddings(),
                                       persist_directory=str(scratch / "chroma"))
        source_ids = {doc["source"]: doc["id"] for doc in fixture["documents"]}
        for document in fixture["documents"]:
            IngestionService(retriever, embedding_model=EMBEDDING).add(document["source"], document["text"].encode(), 600, 100)
        report = {"embedding_model": EMBEDDING, "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                  "documents": len(fixture["documents"]), "records": [],
                  "scope": "Real embeddings, cached identically for both modes; no answer generation."}
        for case in fixture["cases"]:
            row = dict(case)
            for mode in ("vector", "hybrid"):
                row[mode] = [source_ids[doc.metadata["source"]] for doc in retriever.retrieve(case["question"], 3, mode)]
            report["records"].append(row)
    report["metrics"] = {mode: {str(k): summarize_rankings(report["records"], mode, k) for k in (1, 3)}
                         for mode in ("vector", "hybrid")}
    report["splits"] = {split: {mode: summarize_rankings([r for r in report["records"] if r["split"] == split], mode, 3)
                               for mode in ("vector", "hybrid")} for split in ("development", "validation")}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode in ("vector", "hybrid"):
        m1, m3 = report["metrics"][mode]["1"], report["metrics"][mode]["3"]
        print(f"{mode:6}  Hit@1 {m1['hit_count']}/{m1['answerable_cases']}  Recall@3 {m3['mean_recall']:.0%}  MRR@3 {m3['mrr']:.5f}")
    print("->", args.out)


if __name__ == "__main__":
    main()
