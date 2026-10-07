"""Shared RAG primitives: chunking, PDF text extraction, FAISS index build/load, retrieval.

Every entry point in this repository (app.py, build_index.py, eval.py, eval_v2.py) imports
from here so that a single copy of the pipeline logic is used everywhere.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))
K_RETRIEVE = int(os.environ.get("RAG_K_RETRIEVE", "4"))
STORE_ROOT = Path(os.environ.get("RAG_STORE_ROOT", "faiss_store"))

INDEX_FILE = "index.faiss"
DOCS_FILE = "docs.pkl"
META_FILE = "meta.json"
TEXT_FILE = "source.txt"
STORE_VERSION = 2

__all__ = [
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "DOCS_FILE",
    "EMBED_MODEL",
    "INDEX_FILE",
    "K_RETRIEVE",
    "META_FILE",
    "STORE_ROOT",
    "STORE_VERSION",
    "TEXT_FILE",
    "build_index",
    "chunk_text",
    "encode_texts",
    "ensure_store",
    "extract_pdf_text",
    "index_folder",
    "ingest_document",
    "list_indexes",
    "load_index",
    "read_meta",
    "retrieve",
    "slugify",
]


def ensure_store(root: Path = STORE_ROOT) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def slugify(name: str) -> str:
    """Filesystem-safe, stable identifier for an uploaded document."""
    stem = Path(str(name or "")).stem
    slug = re.sub(r"[^\w\s-]", " ", stem, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")[:80].strip("-")
    if not slug:
        slug = "doc-" + uuid.uuid4().hex[:10]
    return slug


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping fixed-size windows.

    Raises ValueError when the parameters cannot make forward progress, which would
    otherwise turn into an infinite loop.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}) "
            "otherwise the sliding window never advances"
        )

    cleaned = (text or "").replace("\x00", " ")
    step = chunk_size - overlap
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        chunks.append(cleaned[start:start + chunk_size])
        start += step
    return [c.strip() for c in chunks if c.strip()]


def extract_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader  # noqa: F401  (preferred, maintained fork)
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore[no-redef]

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages).strip()


def encode_texts(embedder: Any, texts: Sequence[str], batch_size: int = 64):
    """Encode text into a contiguous float32 matrix using numpy (not torch)."""
    import numpy as np

    items = list(texts)
    if not items:
        raise ValueError("nothing to encode: text list is empty")
    vectors = embedder.encode(
        items,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype="float32")


def read_meta(folder: Path) -> Dict[str, Any]:
    meta_path = Path(folder) / META_FILE
    if not meta_path.exists():
        return {}
    try:
        loaded = json.loads(meta_path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def write_meta(folder: Path, meta: Dict[str, Any]) -> None:
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / META_FILE).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8"
    )


def build_index(
    folder: Path,
    chunks: Sequence[str],
    embedder: Any,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    source_name: Optional[str] = None,
    source_text: Optional[str] = None,
    sha256: Optional[str] = None,
    root: Path = STORE_ROOT,
) -> Dict[str, Any]:
    """Embed chunks and persist an index atomically.

    The index is staged in a temporary sibling directory and swapped into place, so an
    interrupted build can never leave a half-written index that later reads as valid.
    """
    import faiss

    items = [c for c in (chunks or []) if c and c.strip()]
    if not items:
        raise ValueError("refusing to build an index from empty text")

    embeddings = encode_texts(embedder, items)
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    target = Path(folder)
    if not target.is_absolute():
        target = Path(root) / target
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent)))

    meta: Dict[str, Any] = {
        "chunks": len(items),
        "dim": dim,
        "embed_model": EMBED_MODEL,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(overlap),
        "distance": "l2",
        "store_version": STORE_VERSION,
        "source_name": source_name,
        "sha256": sha256,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        faiss.write_index(index, str(staging / INDEX_FILE))
        with open(staging / DOCS_FILE, "wb") as handle:
            pickle.dump(items, handle)
        if source_text is not None:
            (staging / TEXT_FILE).write_text(source_text, encoding="utf8")
        (staging / META_FILE).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8"
        )
        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return meta


def ingest_document(
    folder: Path,
    text: str,
    embedder: Any,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    source_name: Optional[str] = None,
    sha256: Optional[str] = None,
    root: Path = STORE_ROOT,
) -> Tuple[Dict[str, Any], List[str]]:
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    meta = build_index(
        folder,
        chunks,
        embedder,
        chunk_size=chunk_size,
        overlap=overlap,
        source_name=source_name,
        source_text=text,
        sha256=sha256,
        root=root,
    )
    return meta, chunks


def load_index(folder: Path) -> Tuple[Any, Optional[List[str]]]:
    """Return (index, chunks). (None, None) when the store is missing."""
    import faiss

    folder = Path(folder)
    idx_path = folder / INDEX_FILE
    docs_path = folder / DOCS_FILE
    if not idx_path.exists() or not docs_path.exists():
        return None, None
    index = faiss.read_index(str(idx_path))
    with open(docs_path, "rb") as handle:
        chunks = pickle.load(handle)
    return index, list(chunks)


def index_folder(root: Path = STORE_ROOT) -> Path:
    return Path(root)


def list_indexes(root: Path = STORE_ROOT) -> List[Tuple[str, Dict[str, Any]]]:
    """List document stores that actually contain a loadable index."""
    root = Path(root)
    if not root.exists():
        return []
    items: List[Tuple[str, Dict[str, Any]]] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not (entry / INDEX_FILE).exists() or not (entry / DOCS_FILE).exists():
            continue
        items.append((entry.name, read_meta(entry)))
    return items


def retrieve(
    index: Any,
    chunks: Sequence[str],
    embedder: Any,
    query: str,
    k: int = K_RETRIEVE,
) -> List[Dict[str, Any]]:
    """Return the k nearest chunks.

    FAISS pads short result rows with -1; those are filtered out here so a padded row can
    never be turned into a valid-looking lookup by Python's negative indexing.
    """
    if index is None or not chunks:
        return []
    total = min(int(k), len(chunks), int(getattr(index, "ntotal", len(chunks))))
    if total <= 0:
        return []

    query_vec = encode_texts(embedder, [query])
    distances, indices = index.search(query_vec, total)

    hits: List[Dict[str, Any]] = []
    seen = set()
    for distance, position in zip(distances[0], indices[0]):
        position = int(position)
        if position < 0 or position >= len(chunks) or position in seen:
            continue
        seen.add(position)
        hits.append({"index": position, "distance": float(distance), "text": chunks[position]})
    return hits