"""Unit tests for rag_core.py — chunking and retrieval logic.

These tests avoid requiring faiss/sentence-transformers at collection time
where possible, and use a small fake embedder so build_index/retrieve can be
tested without downloading a real model in CI.
"""

import numpy as np
import pytest

import rag_core as core
from rag_core import chunk_text, retrieve


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_basic_split():
    text = "a" * 2500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) > 1
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_text_overlap_present():
    text = "0123456789" * 300  # 3000 chars
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    # step = 800, so chunk[1] should start where chunk[0]'s tail overlaps
    assert chunks[0][-200:] == text[800:1000]


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("", chunk_size=1000, overlap=200) == []


def test_chunk_text_strips_null_bytes():
    text = "hello\x00world" * 50
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert all("\x00" not in c for c in chunks)


def test_chunk_text_raises_on_zero_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)


def test_chunk_text_raises_on_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=-5)


def test_chunk_text_raises_when_overlap_ge_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

class _FakeIndex:
    """Minimal stand-in for a faiss index: returns fixed nearest neighbours."""

    def __init__(self, n_items):
        self.ntotal = n_items

    def search(self, query_vec, k):
        # Pretend chunk order is also distance order (closest first).
        n = min(k, self.ntotal)
        indices = np.arange(n).reshape(1, -1)
        distances = np.arange(n, dtype="float32").reshape(1, -1)
        return distances, indices


class _FakeEmbedder:
    def encode(self, texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False):
        # Deterministic fixed-size fake embedding, shape doesn't need to be realistic.
        return np.ones((len(texts), 8), dtype="float32")


def test_retrieve_returns_k_hits():
    chunks = ["chunk one", "chunk two", "chunk three", "chunk four"]
    index = _FakeIndex(n_items=len(chunks))
    embedder = _FakeEmbedder()

    hits = retrieve(index, chunks, embedder, query="anything", k=2)

    assert len(hits) == 2
    assert hits[0]["text"] == "chunk one"
    assert hits[0]["index"] == 0


def test_retrieve_returns_empty_when_index_is_none():
    assert retrieve(None, ["a", "b"], _FakeEmbedder(), query="x", k=2) == []


def test_retrieve_returns_empty_when_no_chunks():
    index = _FakeIndex(n_items=0)
    assert retrieve(index, [], _FakeEmbedder(), query="x", k=2) == []


# ---------------------------------------------------------------------------
# build_index / ingest_document (store-root prefixing regression)
# ---------------------------------------------------------------------------

class _RandomEmbedder:
    """Fake embedder producing random 8-dim vectors, as specified for path tests."""

    def encode(self, texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False):
        return np.random.rand(len(texts), 8).astype("float32")


def test_ingest_document_relative_folder_not_double_prefixed(monkeypatch, tmp_path):
    """Regression: STORE_ROOT / 'doc' must not become faiss_store/faiss_store/doc."""
    monkeypatch.chdir(tmp_path)
    meta, chunks = core.ingest_document(
        core.STORE_ROOT / "doc",
        "some text to chunk and embed " * 10,
        _RandomEmbedder(),
    )
    assert meta["chunks"] == len(chunks)
    assert [name for name, _ in core.list_indexes()] == ["doc"]


def test_list_sample_indexes_returns_empty_when_sample_store_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert core.list_sample_indexes() == []


def test_list_sample_indexes_lists_valid_sample(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    core.ingest_document(
        core.SAMPLE_ROOT / "sample-doc",
        "sample document text " * 10,
        _RandomEmbedder(),
        root=core.SAMPLE_ROOT,
    )
    assert [name for name, _ in core.list_sample_indexes()] == ["sample-doc"]


def test_list_sample_indexes_ignores_incomplete_folders(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / core.SAMPLE_ROOT / "half-built").mkdir(parents=True)
    assert core.list_sample_indexes() == []