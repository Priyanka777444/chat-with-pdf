"""
eval.py - Retrieval accuracy evaluator for the Chat-with-PDF RAG pipeline.

Usage:
    python eval.py                              # uses the first available index
    python eval.py --pdf my-document-slug       # uses a specific PDF index

This script:
    1. Loads the FAISS index + chunks for the chosen PDF.
    2. Runs a set of sample questions through the retrieval pipeline (without
       calling the Groq LLM -- only the embedding + FAISS search).
    3. For each question, checks whether the expected answer text appears in
       the *top-1 retrieved chunk*.
    4. Prints a per-query pass/fail and an overall accuracy score.

    Edit the QA_PAIRS list below to match the actual content of your PDF(s).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


STORE_ROOT = Path("faiss_store")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 1  # top-1 retrieval accuracy


# ---------------------------------------------------------------------------
# TODO: Replace these with real Q&A pairs from your PDF(s).
# ---------------------------------------------------------------------------
QA_PAIRS = [
    {
        "question": "What is the main topic of this document?",
        "expected": "the main subject or title of the PDF",
    },
    {
        "question": "Who is the author of this document?",
        "expected": "the author's name",
    },
    {
        "question": "What year was this document published?",
        "expected": "the publication year",
    },
    {
        "question": "What is the key conclusion of the document?",
        "expected": "the main conclusion or finding",
    },
    {
        "question": "How many sections does the document have?",
        "expected": "the number of sections or chapters",
    },
    {
        "question": "What methodology is used in this document?",
        "expected": "the research methodology described",
    },
    {
        "question": "What is the sample size discussed?",
        "expected": "the sample size number",
    },
    {
        "question": "What datasets were used in the analysis?",
        "expected": "the dataset name or source",
    },
    {
        "question": "What is a key limitation mentioned?",
        "expected": "a limitation or caveat described",
    },
    {
        "question": "What future work is suggested?",
        "expected": "the proposed future research direction",
    },
]


# ---------------------------------------------------------------------------
# Core retrieval logic (mirrors app.py's pipeline)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(index, chunks, embedder):
    results = []
    passed = 0

    print(f"{'#':>3}  {'Pass?':<6}  {'Question':<60}  {'Expected in top chunk':<60}")
    print("-" * 140)

    for i, pair in enumerate(QA_PAIRS, start=1):
        q = pair["question"]
        expected = pair["expected"].lower()

        # embed query
        q_emb = embedder.encode([q]).astype("float32")
        k = min(TOP_K, len(chunks))
        _, indices = index.search(q_emb, k)
        top_chunk = chunks[indices[0][0]].lower() if indices[0][0] < len(chunks) else ""

        found = expected in top_chunk
        if found:
            passed += 1

        label = "✅ PASS" if found else "❌ FAIL"
        print(f"{i:>3}  {label:<6}  {q:<60}  {expected[:60]:<60}")
        results.append({"question": q, "expected": expected, "found": found})

    total = len(results)
    accuracy = passed / total * 100
    print("-" * 140)
    print(f"\nResults: {passed}/{total}  ({accuracy:.1f}%)")
    return accuracy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval accuracy of Chat-with-PDF RAG pipeline."
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="PDF slug (folder name under faiss_store/). Defaults to first available index.",
    )
    args = parser.parse_args()

    # locate the index
    if args.pdf:
        folder = STORE_ROOT / args.pdf
        if not folder.exists():
            print(f"Error: folder 'faiss_store/{args.pdf}' does not exist.")
            sys.exit(1)
    else:
        indexes = list_indexes()
        if not indexes:
            print("Error: no FAISS indexes found in faiss_store/.")
            print("Upload at least one PDF through the app first, then re-run eval.")
            sys.exit(1)
        folder_name = indexes[0][0]
        folder = STORE_ROOT / folder_name
        print(f"Using first available index: {folder_name}")

    index, chunks = load_index(folder)
    if index is None or chunks is None:
        print(f"Error: could not load index from '{folder}'.")
        sys.exit(1)

    print(f"Loaded index with {len(chunks)} chunks.\n")
    print("=" * 140)
    print("  Retrieval accuracy evaluation (top-1, no LLM call)")
    print("=" * 140)

    embedder = SentenceTransformer(EMBED_MODEL)
    evaluate(index, chunks, embedder)


if __name__ == "__main__":
    main()
