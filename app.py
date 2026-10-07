"""Streamlit UI for the Chat-with-PDF RAG app.

Run with: streamlit run app.py
"""

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

st.set_page_config(page_title="Chat with PDF", page_icon="📘", layout="wide")

import rag_core as core  # noqa: E402  (after set_page_config, which must be the first command)

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
NO_CONTEXT_ANSWER = "No context available to answer this question."
SESSION_TTL_SECONDS = 24 * 60 * 60
SESSION_DIR_RE = re.compile(r"^[0-9a-f]{32}$")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(core.EMBED_MODEL)


@st.cache_resource
def load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def build_prompt(question: str, context: str, history: List[Dict[str, str]]) -> str:
    recent = [
        turn
        for turn in history[-6:]
        if str(turn.get("assistant", "")).strip() and turn.get("assistant") != "(thinking…)"
    ]
    history_text = "\n".join(
        f"User: {turn.get('user', '')}\nAssistant: {turn.get('assistant', '')}" for turn in recent
    )
    return f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the context does not contain the answer, reply exactly: "{NO_CONTEXT_ANSWER}"
Quote short phrases from the context instead of inventing details.

Conversation History:
{history_text or "(none)"}

Context:
{context or "(no context retrieved)"}

Question:
{question}

Answer:"""


def answer_question(question: str, context: str, history: List[Dict[str, str]]) -> Tuple[str, str]:
    """Return (answer, error); exactly one of the two is non-empty."""
    client = load_groq_client()
    if client is None:
        return "", "GROQ_API_KEY is not configured. Set it in .env or the Space secrets."
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": build_prompt(question, context, history)}],
            max_tokens=1024,
        )
        content = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        return "", f"Groq request failed: {exc}"
    if not content:
        return "", "Groq returned an empty answer."
    return content, ""


def extract_pdf_bytes(data: bytes) -> str:
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        handle.write(data)
    finally:
        handle.close()
    return handle.name


def session_root() -> Path:
    """Per-browser-session store: faiss_store/<session_id>/."""
    return core.STORE_ROOT / st.session_state.session_id


def doc_folder(kind: str, slug: str) -> Path:
    """Resolve a document key to its folder (sample vs this session's uploads)."""
    if kind == "sample":
        return core.SAMPLE_ROOT / slug
    return session_root() / slug


def purge_stale_sessions() -> None:
    """Delete session directories untouched for over 24 hours."""
    try:
        cutoff = time.time() - SESSION_TTL_SECONDS
        for entry in core.STORE_ROOT.iterdir():
            if not entry.is_dir() or not SESSION_DIR_RE.match(entry.name):
                continue
            if entry.name == st.session_state.session_id:
                continue
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
    except Exception:
        pass


def ingest_bytes(name: str, data: bytes, digest: str) -> Tuple[str, str, Optional[str]]:
    """Index one PDF into this session's store. Returns (level, message, slug)."""
    if not data:
        return "error", f"{name}: file is empty.", None
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return "error", f"{name}: exceeds the {MAX_UPLOAD_MB}MB limit.", None

    slug = core.slugify(name)
    folder = session_root() / slug

    tmp_path = ""
    try:
        tmp_path = extract_pdf_bytes(data)
        text = core.extract_pdf_text(tmp_path)
    except Exception:
        return "error", f"{name}: could not read this PDF.", None
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not text.strip():
        return "error", f"{name}: no selectable text (scanned PDF?).", None

    try:
        meta, _ = core.ingest_document(
            folder,
            text,
            embedder,
            source_name=name,
            sha256=digest,
            root=session_root(),
        )
    except Exception:
        return "error", f"{name}: indexing failed.", None
    return "success", f"{name}: indexed {meta['chunks']} chunks.", slug


def show_sources(sources: List[str]) -> None:
    with st.expander("Sources"):
        if not sources:
            st.caption("No chunks were retrieved.")
        for position, text in enumerate(sources, start=1):
            st.markdown(f"**[{position}]** {text}")


embedder = load_embedder()
core.ensure_store()

st.session_state.setdefault("session_id", uuid.uuid4().hex)
st.session_state.setdefault("history", [])
st.session_state.setdefault("processed", {})
st.session_state.setdefault("hashes", set())
st.session_state.setdefault("active", None)
purge_stale_sessions()

with st.sidebar:
    st.title("Chat with PDF")

    uploaded = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=["pdf"])
    if uploaded:
        pending = [
            item for item in uploaded
            if f"{item.name}:{item.size}" not in st.session_state.processed
        ]
        notices: List[Tuple[str, str]] = []
        newest: Optional[Tuple[str, str]] = None
        if pending:
            with st.spinner("Extracting text and embedding chunks…"):
                for item in pending:
                    st.session_state.processed[f"{item.name}:{item.size}"] = True
                    data = item.getvalue()
                    digest = core.content_hash(data)
                    if digest in st.session_state.hashes:
                        notices.append(("info", f"{item.name}: same content already uploaded."))
                        continue
                    level, message, slug = ingest_bytes(item.name, data, digest)
                    notices.append((level, message))
                    if slug:
                        st.session_state.hashes.add(digest)
                        newest = ("user", slug)
        if newest:
            st.session_state.active = newest
        for level, message in notices:
            getattr(st, level)(message)

    samples = core.list_sample_indexes()
    uploads = core.list_indexes(session_root())
    docs = [("sample", name, meta) for name, meta in samples]
    docs += [("user", name, meta) for name, meta in uploads]
    keys = [(kind, name) for kind, name, _ in docs]
    labels = {
        (kind, name): (
            f"{name} (sample) · {meta.get('chunks', '?')} chunks"
            if kind == "sample"
            else f"{name} · {meta.get('chunks', '?')} chunks"
        )
        for kind, name, meta in docs
    }

    if keys:
        if st.session_state.active not in keys:
            user_keys = [key for key in keys if key[0] == "user"]
            st.session_state.active = user_keys[-1] if user_keys else keys[0]
        if len(keys) == 1:
            st.write(labels[keys[0]])
        else:
            selected = st.selectbox(
                "Document",
                options=keys,
                index=keys.index(st.session_state.active),
                format_func=lambda key: labels[key],
            )
            st.session_state.active = selected
    else:
        st.session_state.active = None

    with st.expander("Export"):
        if st.session_state.history:
            st.download_button(
                "⬇ history.txt",
                data="\n".join(
                    f"User: {t['user']}\nAssistant: {t['assistant']}\n"
                    for t in st.session_state.history
                ),
                file_name="chat_history.txt",
                mime="text/plain",
            )
            st.download_button(
                "⬇ history.json",
                data=json.dumps(st.session_state.history, indent=2, ensure_ascii=False),
                file_name="chat_history.json",
                mime="application/json",
            )
        if st.button("Clear conversation"):
            st.session_state.history = []

st.title("📘 Chat with PDF")
st.caption(f"FAISS + SentenceTransformers ({core.EMBED_MODEL}) + Groq ({GROQ_MODEL})")

if not keys:
    st.info("Upload a PDF in the sidebar to get started.")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])
        show_sources(turn.get("sources", []))

question = st.chat_input("Ask a question about the document…")
if question:
    active = st.session_state.active
    if not active:
        st.warning("Upload a PDF in the sidebar first.")
    else:
        kind, slug = active
        index, chunks = core.load_index(doc_folder(kind, slug))
        if index is None:
            st.error("Index missing for this document. Upload the PDF again.")
        else:
            hits = core.retrieve(index, chunks, embedder, question)
            context = "\n\n".join(hit["text"] for hit in hits)
            reply, failure = answer_question(question, context, st.session_state.history)
            if failure:
                st.error(failure)
            else:
                sources = [" ".join(hit["text"].split())[:300] for hit in hits]
                st.session_state.history.append(
                    {"user": question, "assistant": reply, "sources": sources}
                )
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    show_sources(sources)
