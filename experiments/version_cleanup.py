"""Upload 10 successive versions of one file, then retry the latest 10 times; count what stays in Chroma.

Runs offline: real local Chroma, hash embeddings instead of a hosted model, network blocked.
  before = experiments/baseline_20261012 (ingestion before versioning)
  after  = this repository's src/
python experiments/version_cleanup.py            -> results/version_cleanup/rerun.json
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CODE = {"before": ROOT / "experiments/baseline_20261012", "after": ROOT}


def block_network():
    import socket
    def blocked(*args, **kwargs):
        raise RuntimeError("Network disabled: local measurement")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = blocked


def measure(code, workdir, output):
    sys.path.insert(0, str(code))
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings
    from src import file_registry
    from src.rag_service import RagService
    from src.vector_store import VectorStoreService

    class CountingEmbeddings(Embeddings):
        def __init__(self):
            self.calls = self.texts = 0

        def embed_documents(self, texts):
            self.calls += 1
            self.texts += len(texts)
            return [self.embed_query(text) for text in texts]

        def embed_query(self, text):
            return [float(value) / 255 for value in hashlib.sha256(text.encode()).digest()[:8]]

    file_registry.FILE_REGISTRY_PATH = workdir / "registry.json"
    file_registry.ensure_project_dirs = lambda: None
    embeddings = CountingEmbeddings()
    service = RagService.__new__(RagService)
    vectors = VectorStoreService.__new__(VectorStoreService)
    vectors.vector_store = Chroma(collection_name="version_cleanup", embedding_function=embeddings,
                                 persist_directory=str(workdir / "chroma"))
    service.vector_store = vectors
    records = []
    for version in range(1, 11):
        text = f"Policy revision {version}: current revision is {version}."
        uploaded = SimpleNamespace(name="policy.txt", getvalue=lambda: text.encode())
        result = service.add_uploaded_files([uploaded], 600, 100)
        stored = vectors.vector_store.get(include=["documents"])
        records.append({"version": version, "text": text, "result": result,
                        "stored_texts": stored["documents"], "stored_chunks": len(stored["ids"])})
    calls_before_retry = embeddings.calls
    retries = [service.add_uploaded_files([uploaded], 600, 100) for _ in range(10)]
    current = vectors.vector_store.get(include=["documents"])
    report = {"scenario": "10 successive versions under one filename, then 10 retries of the latest version",
              "mode": "real local Chroma with hash embeddings; no semantic-quality or billing measurement",
              "records": records, "retries": retries, "versions_uploaded": 10,
              "retained_chunks": len(current["ids"]),
              "stale_chunks": sum(value != text for value in current["documents"]),
              "latest_version_present": text in current["documents"],
              "embedding_batch_calls": embeddings.calls, "embedding_texts": embeddings.texts,
              "extra_embedding_calls_on_10_retries": embeddings.calls - calls_before_retry}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=list(CODE))
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results/version_cleanup/rerun.json")
    args = parser.parse_args()
    if args.worker:
        block_network()
        measure(CODE[args.worker], args.workdir, args.workdir / "measurement.json")
        return
    env = {key: value for key, value in os.environ.items()
           if not any(word in key.upper() for word in ("API_KEY", "SECRET", "TOKEN"))}
    env.update(ANONYMIZED_TELEMETRY="False", PYTHONIOENCODING="utf-8")
    results = {}
    for phase in CODE:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            workdir = Path(tmp)
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--workdir", str(workdir)],
                           env=env, cwd=workdir, check=True, timeout=180)
            results[phase] = json.loads((workdir / "measurement.json").read_text(encoding="utf-8"))
        print(phase, "retained:", results[phase]["retained_chunks"], "stale:", results[phase]["stale_chunks"])
    args.out.write_text(json.dumps({"network_blocked": True, "results": results}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print("->", args.out)


if __name__ == "__main__":
    main()
