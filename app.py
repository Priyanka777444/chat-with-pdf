"""Streamlit UI for the Chat-with-PDF RAG app.

Run with: streamlit run app.py
"""

import json
import os
import tempfile
from typing import Dict, List, Tuple

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
        return "", "GROQ_API_KEY is not configured. Add it to .env and restart the app."
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


def ingest_upload(uploaded_file, force: bool) -> Tuple[str, str]:
    """Return (level, message) where level is 'success', 'info' or 'error'."""
    data = uploaded_file.getvalue()
    if not data:
        return "error", f"{uploaded_file.name}: file is empty."
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return "error", f"{uploaded_file.name}: exceeds the {MAX_UPLOAD_MB}MB limit."

    slug = core.slugify(uploaded_file.name)
    folder = core.STORE_ROOT / slug
    if (folder / core.INDEX_FILE).exists() and not force:
        return "info", f"{uploaded_file.name}: already indexed as '{slug}' (tick re-ingest to rebuild)."

    tmp_path = ""
    try:
        tmp_path = extract_pdf_bytes(data)
        text = core.extract_pdf_text(tmp_path)
    except Exception as exc:
        return "error", f"{uploaded_file.name}: could not read PDF ({exc})."
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not text.strip():
        return "error", f"{uploaded_file.name}: no selectable text (scanned PDF?)."

    try:
        meta, _ = core.ingest_document(
            folder,
            text,
            embedder,
            source_name=uploaded_file.name,
            sha256=core.content_hash(data),
        )
    except Exception as exc:
        return "error", f"{uploaded_file.name}: indexing failed ({exc})."
    return "success", f"{uploaded_file.name}: indexed {meta['chunks']} chunks as '{slug}'."


def rebuild_active(slug: str) -> Tuple[str, str]:
    folder = core.STORE_ROOT / slug
    source_file = folder / core.TEXT_FILE
    if not source_file.exists():
        return "error", "Stored text is missing for this document; re-upload the PDF instead."
    previous = core.read_meta(folder)
    try:
        text = source_file.read_text(encoding="utf8")
        meta, _ = core.ingest_document(
            folder,
            text,
            embedder,
            chunk_size=int(previous.get("chunk_size", core.CHUNK_SIZE)),
            overlap=int(previous.get("chunk_overlap", core.CHUNK_OVERLAP)),
            source_name=previous.get("source_name"),
            sha256=previous.get("sha256"),
        )
    except Exception as exc:
        return "error", f"Rebuild failed: {exc}"
    return "success", f"Rebuilt '{slug}' with {meta['chunks']} chunks."


embedder = load_embedder()
core.ensure_store()

st.session_state.setdefault("history", [])
st.session_state.setdefault("active_pdf", None)

with st.sidebar:
    st.header("Documents")
    if load_groq_client() is None:
        st.warning("GROQ_API_KEY missing — set it in .env to enable answers.")

    uploaded = st.file_uploader("Upload PDF(s)", accept_multiple_files=True, type=["pdf"])
    force = st.checkbox("Re-ingest even if already indexed", value=False)

    if uploaded and st.button("Ingest", type="primary"):
        with st.spinner("Extracting text and embedding chunks…"):
            notices = [ingest_upload(item, force) for item in uploaded]
        for level, message in notices:
            getattr(st, level)(message)

    st.markdown("---")
    st.subheader("Active document")
    indexes = core.list_indexes()
    slugs = [name for name, _ in indexes]
    if slugs:
        if st.session_state.active_pdf not in slugs:
            st.session_state.active_pdf = slugs[0]
        labels = {name: f"{name} ({meta.get('chunks', '?')} chunks)" for name, meta in indexes}
        selected = st.selectbox(
            "Document", options=slugs, format_func=lambda slug: labels.get(slug, slug)
        )
        st.session_state.active_pdf = selected
        meta = core.read_meta(core.STORE_ROOT / selected)
        st.markdown(f"**Folder:** `{selected}`")
        st.markdown(f"**Chunks:** {meta.get('chunks', '?')}")
        st.markdown(f"**Embedding model:** {meta.get('embed_model', core.EMBED_MODEL)}")
        st.markdown(
            f"**Chunking:** {meta.get('chunk_size', '?')} chars, "
            f"{meta.get('chunk_overlap', '?')} overlap"
        )
        st.markdown(f"**Indexed at:** {meta.get('created_at', '?')}")
        if st.button("Rebuild index"):
            with st.spinner("Rebuilding index…"):
                level, message = rebuild_active(selected)
            getattr(st, level)(message)
    else:
        st.info("No documents indexed yet.")

    st.markdown("---")
    st.subheader("Data")
    if st.checkbox("Prepare chunk export") and st.session_state.active_pdf:
        _, chunks = core.load_index(core.STORE_ROOT / st.session_state.active_pdf)
        if chunks:
            st.download_button(
                "⬇ chunks.json",
                data=json.dumps({"chunks": chunks}, indent=2, ensure_ascii=False),
                file_name=f"{st.session_state.active_pdf}_chunks.json",
                mime="application/json",
            )
        else:
            st.error("No chunks found for the active document.")

    if st.button("Clear conversation"):
        st.session_state.history = []

    if st.session_state.history:
        st.download_button(
            "⬇ history.txt",
            data="\n".join(
                f"User: {t['user']}\nAssistant: {t['assistant']}\n" for t in st.session_state.history
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

st.title("📘 Chat with PDF")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["user"])
    with st.chat_message("assistant"):
        st.markdown(turn["assistant"])
        st.caption(f"source: {turn.get('pdf', '-')}")

if not st.session_state.history:
    st.info("No messages yet — ask a question about the active document.")

question = st.chat_input("Ask a question about the active document…")
if question:
    active = st.session_state.active_pdf
    if not active:
        st.warning("Select or ingest a document first.")
    else:
        index, chunks = core.load_index(core.STORE_ROOT / active)
        if index is None:
            st.error("Index missing for this document. Re-upload the PDF.")
        else:
            hits = core.retrieve(index, chunks, embedder, question)
            context = "\n\n".join(hit["text"] for hit in hits)
            reply, failure = answer_question(question, context, st.session_state.history)
            if failure:
                st.error(failure)
            else:
                st.session_state.history.append(
                    {"user": question, "assistant": reply, "pdf": active}
                )
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    st.markdown(reply)
                    st.caption(f"source: {active} · {len(hits)} chunks used")

st.markdown("---")
st.caption(f"FAISS + SentenceTransformers ({core.EMBED_MODEL}) + Groq ({GROQ_MODEL})")