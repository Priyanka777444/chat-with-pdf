# app.py
import os
import re
import time
import json
import tempfile
import faiss
import pickle
import hashlib
from datetime import datetime
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from groq import Groq

# ----------- Config ------------
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

STORE_ROOT = Path("faiss_store")
STORE_ROOT.mkdir(exist_ok=True)
EMBED_MODEL = "all-MiniLM-L6-v2"
K_RETRIEVE = 4

# ----------- Session State ------------
if "history" not in st.session_state:
    st.session_state.history = []

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None

# ----------- Utilities ------------
def slugify(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    name = re.sub(r"[-\s]+", "-", name)
    if not name:
        name = hashlib.sha1(os.urandom(16)).hexdigest()[:8]
    return name[:100]

def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
    return "\n\n".join(pages_text)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()

def build_index_for_pdf(folder: Path, chunks: list):
    embeddings = embedder.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    folder.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(folder / "index.faiss"))
    with open(folder / "docs.pkl", "wb") as f:
        pickle.dump(chunks, f)
    meta = {"chunks": len(chunks), "created_at": datetime.utcnow().isoformat()}
    with open(folder / "meta.json", "w", encoding="utf8") as f:
        json.dump(meta, f)

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
    items = []
    for d in STORE_ROOT.iterdir():
        if d.is_dir():
            meta = {}
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf8"))
                except Exception:
                    meta = {}
            items.append((d.name, meta))
    return sorted(items, key=lambda x: x[0])

def groq_request_answer(question: str, context: str, history: list):
    history_text = "\n".join(f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history[-10:])
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer the user's question. 
If the context doesn't contain the answer, reply: "No context available to answer this question."

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Answer:"""

    if client is None:
        return "Groq client not configured."

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Groq request error: {str(e)}"

# ----------- Streamlit UI ------------
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("📘 Chat with PDF")

# CSS for chat bubbles
bubble_css = """
<style>
body { background-color: #f6f8fa; }
.chat-container { max-width:900px; margin: 0 auto; }
.user-bubble { background: #0b81ff; color: #fff; padding:12px; border-radius:16px; float:right; margin:8px; max-width:80%; }
.assistant-bubble { background: #e6e6e6; color: #000; padding:12px; border-radius:16px; float:left; margin:8px; max-width:80%; }
.clear { clear: both; }
</style>
"""
st.markdown(bubble_css, unsafe_allow_html=True)

# -------- Sidebar ------------
with st.sidebar:
    st.header("Controls")
    st.subheader("Upload PDF(s)")
    uploaded_files = st.file_uploader("Choose one or more PDFs", accept_multiple_files=True, type=["pdf"])
    if uploaded_files:
        for up in uploaded_files:
            display_name = up.name
            slug = slugify(display_name)
            folder = STORE_ROOT / slug
            if (folder / "index.faiss").exists():
                st.info(f"Already ingested: {display_name}")
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(up.read())
                tmp_path = tmp.name
            st.info(f"Ingesting {display_name}...")
            text = extract_pdf_text(tmp_path)
            chunks = chunk_text(text, chunk_size=1000, overlap=200)
            build_index_for_pdf(folder, chunks)
            st.success(f"Ingested {display_name} ({len(chunks)} chunks)")

    st.markdown("---")
    st.subheader("Select active PDF")
    indexes = list_indexes()
    if indexes:
        slugs = [t[0] for t in indexes]
        options = [f"{name} ({meta.get('chunks','?')} chunks)" for (name, meta) in indexes]

        selected_slug = st.selectbox(
            "Active PDF",
            options=slugs,
            format_func=lambda x: options[slugs.index(x)]
        )
        st.session_state.active_pdf = selected_slug

    st.markdown("---")
    st.subheader("Chat History")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.success("Chat history cleared.")
    if st.session_state.history:
        txt_lines = [f"User: {i['user']}\nAssistant: {i['assistant']}\n" for i in st.session_state.history]
        st.download_button("⬇ Download history (TXT)", data="\n".join(txt_lines), file_name="chat_history.txt", mime="text/plain")
        st.download_button("⬇ Download history (JSON)", data=json.dumps(st.session_state.history, indent=2, ensure_ascii=False),
                           file_name="chat_history.json", mime="application/json")

# -------- Main Chat ----------
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")
    chat_placeholder = st.empty()

    def render_history():
        html = "<div class='chat-container'>"
        for item in st.session_state.history:
            html += f"<div class='assistant-bubble'>{item['assistant']}</div><div class='clear'></div>"
            html += f"<div class='user-bubble'>{item['user']}</div><div class='clear'></div>"
        html += "</div>"
        chat_placeholder.markdown(html, unsafe_allow_html=True)

    render_history()

    # Use form to avoid session_state issues
    with st.form(key="ask_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about the active PDF:")
        submit_btn = st.form_submit_button("Ask")
        if submit_btn:
            if not st.session_state.active_pdf:
                st.error("Please select or ingest a PDF first.")
            elif not user_input.strip():
                st.warning("Write a question first.")
            else:
                folder = STORE_ROOT / st.session_state.active_pdf
                index, chunks = load_index(folder)
                if index is None:
                    st.error("Index missing. Re-upload PDF.")
                else:
                    q_emb = embedder.encode([user_input]).astype("float32")
                    k = min(K_RETRIEVE, len(chunks))
                    distances, indices = index.search(q_emb, k)
                    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
                    context = "\n\n".join(retrieved)

                    # Add placeholder in history
                    st.session_state.history.append({"user": user_input, "assistant": "(thinking...)", "pdf": st.session_state.active_pdf})
                    render_history()

                    # Get answer
                    full_answer = groq_request_answer(user_input, context, st.session_state.history[:-1])

                    # Simulated streaming
                    displayed = ""
                    chunk_chars = 16
                    delay = 0.02
                    for i in range(0, len(full_answer), chunk_chars):
                        displayed += full_answer[i:i+chunk_chars]
                        st.session_state.history[-1]["assistant"] = displayed
                        render_history()
                        time.sleep(delay)
                    st.session_state.history[-1]["assistant"] = full_answer
                    render_history()

with col2:
    st.subheader("Active PDF Info")
    if not st.session_state.active_pdf:
        st.info("No active PDF selected.")
    else:
        folder = STORE_ROOT / st.session_state.active_pdf
        meta = {}
        meta_path = folder / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf8"))
            except Exception:
                meta = {}
        st.markdown(f"**PDF folder:** `{folder.name}`")
        st.markdown(f"**Chunks:** {meta.get('chunks', '?')}")
        st.markdown(f"**Indexed at:** {meta.get('created_at', '?')}")
        st.markdown("---")
        if st.button("Rebuild index for active PDF"):
            st.info("Rebuild requires re-uploading the PDF from sidebar.")
        if st.button("Download indexed chunks (JSON)"):
            idx, ch = load_index(folder)
            if ch:
                data_json = json.dumps({"chunks": ch}, indent=2, ensure_ascii=False)
                st.download_button("⬇ Download chunks JSON", data=data_json, file_name=f"{folder.name}_chunks.json", mime="application/json")
            else:
                st.error("Index not found.")

st.markdown("---")
st.caption("Built with FAISS + SentenceTransformers + Groq")
