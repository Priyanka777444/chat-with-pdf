"""
eval.py - Retrieval accuracy evaluator for the Chat-with-PDF RAG pipeline,
with MLflow experiment tracking.

Usage:
    python eval.py                              # uses the first available index, top_k=1
    python eval.py --pdf my-document-slug --top-k 3

    Edit QA_PAIRS below to match REAL content from the PDF you're testing
    against -- placeholders will all fail and give meaningless numbers.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import mlflow
from sentence_transformers import SentenceTransformer


STORE_ROOT = Path("faiss_store")
EMBED_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# TODO: Replace with REAL Q&A pairs matching the PDF you're testing against.
# Example, for the Employee Management System SRS doc:
#   {"question": "What is this document about?", "expected": "employee management"}
# ---------------------------------------------------------------------------
QA_PAIRS = [
    {"question": "What does AI stand for?", "expected": "artificial intelligence"},
    {"question": "What is the goal of AI?", "expected": "simulate human intelligence"},
    {"question": "What is machine learning?", "expected": "learn from data"},
    {"question": "What are neural networks?", "expected": "inspired by the human brain"},
    {"question": "What is deep learning?", "expected": "multiple layers"},
    {"question": "What is natural language processing?", "expected": "understand human language"},
    {"question": "What is computer vision?", "expected": "interpret visual information"},
    {"question": "What is reinforcement learning?", "expected": "reward"},
    {"question": "What is supervised learning?", "expected": "labeled data"},
    {"question": "What is unsupervised learning?", "expected": "unlabeled data"},
]


def load_index(folder: Path):
    idx_path = folder / "index.faiss"
    docs_path = folder / "docs.pkl"
    if not idx_path.exists() or not docs_path.exists():
        return None, None
    index = faiss.read_index(str(idx_path))
    with open(docs_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def list_indexes():
    folders = []
    for d in STORE_ROOT.iterdir():
        if d.is_dir() and (d / "index.faiss").exists():
            meta = {}
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf8"))
                except Exception:
                    meta = {}
            folders.append((d.name, meta))
    return sorted(folders, key=lambda x: x[0])


def evaluate(index, chunks, embedder, top_k):
    results = []
    passed = 0
    latencies = []

    print(f"{'#':>3}  {'Pass?':<6}  {'Question':<60}  {'Expected in top chunk':<60}")
    print("-" * 140)

    for i, pair in enumerate(QA_PAIRS, start=1):
        q = pair["question"]
        expected = pair["expected"].lower()

        start = time.perf_counter()
        q_emb = embedder.encode([q]).astype("float32")
        k = min(top_k, len(chunks))
        _, indices = index.search(q_emb, k)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

        retrieved_text = " ".join(
            chunks[idx].lower() for idx in indices[0] if idx < len(chunks)
        )
        found = expected in retrieved_text
        if found:
            passed += 1

        label = "✅ PASS" if found else "❌ FAIL"
        print(f"{i:>3}  {label:<6}  {q:<60}  {expected[:60]:<60}")
        results.append({"question": q, "expected": expected, "found": found, "latency_ms": elapsed_ms})

    total = len(results)
    accuracy = passed / total * 100
    avg_latency = sum(latencies) / len(latencies)
    print("-" * 140)
    print(f"\nResults: {passed}/{total}  ({accuracy:.1f}%)  |  Avg retrieval latency: {avg_latency:.1f}ms")
    return accuracy, avg_latency, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy of Chat-with-PDF RAG pipeline.")
    parser.add_argument("--pdf", type=str, default=None, help="PDF slug under faiss_store/.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of chunks to retrieve per query.")
    args = parser.parse_args()

    if args.pdf:
        folder = STORE_ROOT / args.pdf
        if not folder.exists():
            print(f"Error: folder 'faiss_store/{args.pdf}' does not exist.")
            sys.exit(1)
        folder_name = args.pdf
    else:
        indexes = list_indexes()
        if not indexes:
            print("Error: no FAISS indexes found. Upload a PDF through the app first.")
            sys.exit(1)
        folder_name = indexes[0][0]
        folder = STORE_ROOT / folder_name
        print(f"Using first available index: {folder_name}")

    index, chunks = load_index(folder)
    if index is None or chunks is None:
        print(f"Error: could not load index from '{folder}'.")
        sys.exit(1)

    print(f"Loaded index with {len(chunks)} chunks.\n")

    embedder = SentenceTransformer(EMBED_MODEL)

    mlflow.set_experiment("chat-with-pdf-retrieval-eval")
    with mlflow.start_run(run_name=f"{folder_name}_top{args.top_k}"):
        mlflow.log_param("pdf_folder", folder_name)
        mlflow.log_param("embed_model", EMBED_MODEL)
        mlflow.log_param("top_k", args.top_k)
        mlflow.log_param("num_chunks", len(chunks))
        mlflow.log_param("num_questions", len(QA_PAIRS))

        accuracy, avg_latency, results = evaluate(index, chunks, embedder, args.top_k)

        mlflow.log_metric("retrieval_accuracy_pct", accuracy)
        mlflow.log_metric("avg_query_latency_ms", avg_latency)

        results_path = Path("eval_results.json")
        results_path.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(results_path))


if __name__ == "__main__":
    main()