"""
gui.py — Streamlit GUI for Advanced RAG for Research Papers.

Provides a dark-themed, single-page chat interface to the existing RAG backend
in src/. Supports uploading papers, building/rebuilding the FAISS index, asking
natural-language questions, and viewing grounded answers with source transparency.

Launch: streamlit run gui.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# Ensure project root is on sys.path for src imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(override=True)

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from src.voice import (
    transcribe_audio,
    is_voice_transcription_available,
    AVAILABLE_WHISPER_MODELS,
    DEFAULT_WHISPER_MODEL,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"}
PERSIST_DIR = "faiss_store_medquad"
DATA_DIR = "data"

AVAILABLE_LLM_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "groq/compound",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Color tokens from PRD Section 7.2
COLORS = {
    "accent": "#7C5CFF",
    "success": "#3DD68C",
    "warning": "#F5A623",
    "error": "#FF5C5C",
    "muted": "#9AA0AC",
    "user_bubble": "#1F2430",
    "assistant_bubble": "#161A23",
}

# ---------------------------------------------------------------------------
# Page Configuration (F-16)
# ---------------------------------------------------------------------------

def configure_page() -> None:
    """Set Streamlit page config — must be the first Streamlit call."""
    st.set_page_config(
        page_title="MedAssist — Consumer Health Advisor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# Custom CSS (Section 7.4 — minimal, only what config.toml can't reach)
# ---------------------------------------------------------------------------

def inject_custom_css() -> None:
    """Inject minimal custom CSS for chat bubble styling and source panels."""
    st.markdown(
        f"""
        <style>
        /* Assistant message left border accent */
        .stChatMessage [data-testid="stChatMessageAvatarAssistant"] {{
            /* Handled via border on the container */
        }}

        /* Style assistant chat bubbles with left accent border */
        .assistant-bubble {{
            border-left: 3px solid {COLORS["accent"]};
            padding-left: 0.75rem;
            margin-bottom: 0.25rem;
        }}

        /* Error message in chat */
        .error-bubble {{
            border-left: 3px solid {COLORS["error"]};
            padding-left: 0.75rem;
            color: {COLORS["error"]};
        }}

        /* Source chunk text — monospace/code-like */
        .source-chunk {{
            font-family: 'SFMono-Regular', 'Consolas', 'Liberation Mono', 'Menlo', monospace;
            font-size: 0.82rem;
            line-height: 1.5;
            color: {COLORS["muted"]};
            background-color: #1a1f2b;
            padding: 0.6rem 0.8rem;
            border-radius: 6px;
            margin-top: 0.3rem;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        /* Status badges */
        .status-badge {{
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .badge-success {{
            background-color: rgba(61, 214, 140, 0.15);
            color: {COLORS["success"]};
        }}
        .badge-warning {{
            background-color: rgba(245, 166, 35, 0.15);
            color: {COLORS["warning"]};
        }}
        .badge-error {{
            background-color: rgba(255, 92, 92, 0.15);
            color: {COLORS["error"]};
        }}

        /* Sidebar section headers */
        .sidebar-header {{
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["muted"]};
            margin-bottom: 0.5rem;
        }}

        /* Spoken query badge pill */
        .spoken-pill {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(124, 92, 255, 0.15);
            color: #A78BFA;
            border: 1px solid rgba(124, 92, 255, 0.3);
            padding: 2px 10px;
            border-radius: 12px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-bottom: 6px;
        }}

        /* Empty state centered */
        .empty-state {{
            text-align: center;
            padding: 4rem 2rem;
            color: {COLORS["muted"]};
        }}
        .empty-state h2 {{
            color: #E6E6E6;
            margin-bottom: 0.5rem;
        }}
        .empty-state p {{
            font-size: 1.05rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session State Initialization (Section 11)
# ---------------------------------------------------------------------------

def init_session_state() -> None:
    """Initialize all session_state keys with defaults."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = _check_index_exists()
    if "last_indexed_at" not in st.session_state:
        st.session_state.last_indexed_at = _get_index_timestamp()
    if "rebuild_in_progress" not in st.session_state:
        st.session_state.rebuild_in_progress = False
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = _get_chunk_count()
    if "last_processed_audio_id" not in st.session_state:
        st.session_state.last_processed_audio_id = None
    if "pending_voice_query" not in st.session_state:
        st.session_state.pending_voice_query = None
    if "whisper_model" not in st.session_state:
        st.session_state.whisper_model = DEFAULT_WHISPER_MODEL


def _check_index_exists() -> bool:
    """Check whether a FAISS index is present on disk."""
    faiss_path = os.path.join(PERSIST_DIR, "faiss.index")
    meta_path = os.path.join(PERSIST_DIR, "metadata.pkl")
    return os.path.exists(faiss_path) and os.path.exists(meta_path)


def _get_index_timestamp() -> Optional[datetime]:
    """Return the modification time of the FAISS index, if it exists."""
    faiss_path = os.path.join(PERSIST_DIR, "faiss.index")
    if os.path.exists(faiss_path):
        return datetime.fromtimestamp(os.path.getmtime(faiss_path))
    return None


def _get_chunk_count() -> int:
    """Return the number of stored chunks by loading metadata, or 0."""
    import pickle
    meta_path = os.path.join(PERSIST_DIR, "metadata.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)
            return len(metadata)
        except Exception:
            return 0
    return 0


# ---------------------------------------------------------------------------
# Cached Backend Resources
# ---------------------------------------------------------------------------

@st.cache_resource
def load_vectorstore(persist_dir: str, embedding_model: str) -> FaissVectorStore:
    """Load or create a FaissVectorStore instance (cached per process)."""
    store = FaissVectorStore(persist_dir, embedding_model)
    faiss_path = os.path.join(persist_dir, "faiss.index")
    meta_path = os.path.join(persist_dir, "metadata.pkl")
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        store.load()
    return store


@st.cache_resource
def _cached_rag_search(
    persist_dir: str, embedding_model: str, llm_model: str, api_key: str
) -> Optional[RAGSearch]:
    """Internal cached helper keyed by configuration and api_key."""
    try:
        rag = RAGSearch(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            llm_model=llm_model,
        )
        return rag
    except Exception as e:
        st.error(f"Failed to initialize RAGSearch: {e}")
        return None


def load_rag_search(
    persist_dir: str, embedding_model: str, llm_model: str
) -> Optional[RAGSearch]:
    """Load a RAGSearch instance. Avoids caching None when API key is missing."""
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return _cached_rag_search(persist_dir, embedding_model, llm_model, api_key)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_data_files() -> list[dict]:
    """Recursively scan the data directory and return file info dicts."""
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return []

    files = []
    for file_path in sorted(data_path.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            size_bytes = file_path.stat().st_size
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

            files.append({
                "Filename": file_path.name,
                "Type": file_path.suffix.upper().lstrip("."),
                "Size": size_str,
            })
    return files


def has_api_key() -> bool:
    """Check if GROQ_API_KEY is set in the environment or .env file."""
    load_dotenv(override=True)
    key = os.getenv("GROQ_API_KEY")
    return bool(key and key.strip())


def save_uploaded_file(uploaded_file) -> bool:
    """
    Save an uploaded file to the appropriate data/ subdirectory.
    PDFs → data/pdf/, everything else → data/text_files/.
    Returns True on success, False if unsupported type.
    """
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False

    if ext == ".pdf":
        target_dir = Path(DATA_DIR) / "pdf"
    else:
        target_dir = Path(DATA_DIR) / "text_files"

    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / uploaded_file.name

    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return True


# ---------------------------------------------------------------------------
# Sidebar Rendering (F-1 through F-8)
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    """Render the full sidebar with all sections separated by dividers."""
    with st.sidebar:
        st.markdown("## MedAssist")
        st.caption("Ask questions across your research paper library.")

        st.divider()

        # --- Documents section ---
        _render_document_section()

        st.divider()

        # --- Index section ---
        _render_index_section()

        st.divider()

        # --- Settings section ---
        _render_settings_section()

        st.divider()

        # --- Session section ---
        _render_session_section()


def _render_document_section() -> None:
    """Sidebar: file uploader (F-1) and indexed files table (F-2)."""
    st.markdown(
        '<p class="sidebar-header">📁 Documents</p>', unsafe_allow_html=True
    )

    # F-1: File uploader
    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=list(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        key="file_uploader",
        disabled=st.session_state.rebuild_in_progress,
    )

    if uploaded_files:
        success_count = 0
        for uf in uploaded_files:
            ext = Path(uf.name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                st.error(
                    f"❌ Unsupported file type: `{ext}`. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            elif save_uploaded_file(uf):
                success_count += 1

        if success_count > 0:
            st.success(f"✅ {success_count} file(s) uploaded successfully.")

    # F-2: Indexed files table
    files = get_data_files()
    if files:
        st.dataframe(
            files,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No files in the data directory yet.")


def _render_index_section() -> None:
    """Sidebar: rebuild index button (F-3) and index status badge (F-4)."""
    st.markdown(
        '<p class="sidebar-header">🗂️ Index</p>', unsafe_allow_html=True
    )

    # F-4: Index status badge
    if st.session_state.rebuild_in_progress:
        st.markdown(
            '<span class="status-badge badge-warning">⏳ Rebuilding…</span>',
            unsafe_allow_html=True,
        )
    elif st.session_state.index_ready:
        chunk_count = st.session_state.chunk_count
        ts = st.session_state.last_indexed_at
        ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "unknown"
        st.markdown(
            f'<span class="status-badge badge-success">✅ Index loaded — {chunk_count} chunks</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Last built: {ts_str}")
    else:
        st.markdown(
            '<span class="status-badge badge-warning">⚠️ Index not built</span>',
            unsafe_allow_html=True,
        )

    # F-3: Rebuild Index button
    if st.button(
        "🔄 Rebuild Index",
        use_container_width=True,
        disabled=st.session_state.rebuild_in_progress,
    ):
        _rebuild_index()


def _rebuild_index() -> None:
    """Execute the full index rebuild pipeline with staged progress (F-3)."""
    files = get_data_files()
    if not files:
        st.warning("⚠️ No documents found — upload files first.")
        return

    st.session_state.rebuild_in_progress = True
    embedding_model = st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

    with st.status("Building index…", expanded=True) as status:
        # Stage 1: Loading documents
        status.update(label="Loading MedQuAD health QA records…")
        try:
            from src.medquad_loader import load_medquad_documents
            docs = load_medquad_documents(limit=3000)
            if Path(DATA_DIR).exists():
                try:
                    custom_docs = load_all_documents(DATA_DIR)
                    docs.extend(custom_docs)
                except Exception:
                    pass
        except Exception as e:
            status.update(label="❌ Failed to load documents", state="error")
            st.error(f"Error loading documents: {e}")
            st.session_state.rebuild_in_progress = False
            return

        if not docs:
            status.update(label="⚠️ No documents could be loaded", state="error")
            st.warning("No loadable documents found.")
            st.session_state.rebuild_in_progress = False
            return

        st.write(f"📄 Loaded {len(docs)} document(s)")

        # Stage 2: Chunking
        status.update(label="Chunking…")
        store = FaissVectorStore(PERSIST_DIR, embedding_model)
        from src.embedding import EmbeddingPipeline
        emb_pipe = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=store.chunk_size,
            chunk_overlap=store.chunk_overlap,
        )
        chunks = emb_pipe.chunk_documents(docs)
        st.write(f"✂️ Created {len(chunks)} chunks")

        # Stage 3: Generating embeddings
        status.update(label="Generating embeddings…")
        import numpy as np
        embeddings = emb_pipe.embed_chunks(chunks)
        st.write(f"🧬 Generated {len(embeddings)} embeddings")

        # Stage 4: Building index
        status.update(label="Building index…")
        metadatas = []
        for chunk in chunks:
            meta = dict(chunk.metadata) if hasattr(chunk, "metadata") and chunk.metadata else {}
            meta["text"] = chunk.page_content
            meta.setdefault("source", "MedQuAD / NIH")
            metadatas.append(meta)
        store.add_embeddings(np.array(embeddings).astype("float32"), metadatas)
        store.save()

        # Update session state
        st.session_state.index_ready = True
        st.session_state.last_indexed_at = datetime.now()
        st.session_state.chunk_count = len(chunks)
        st.session_state.rebuild_in_progress = False

        # Clear cached resources so they reload with the new index
        load_vectorstore.clear()
        _cached_rag_search.clear()

        status.update(label=f"Done ✅ — {len(chunks)} chunks indexed", state="complete")


def _render_settings_section() -> None:
    """Sidebar: top_k slider (F-5) and model settings (F-6)."""
    st.markdown(
        '<p class="sidebar-header">⚙️ Settings</p>', unsafe_allow_html=True
    )

    # F-5: Retrieval top_k slider
    st.slider(
        "Top-K results",
        min_value=1,
        max_value=10,
        value=3,
        key="top_k",
        help="Number of document chunks to retrieve per query.",
    )

    # F-6: LLM model selector
    st.selectbox(
        "LLM Model",
        options=AVAILABLE_LLM_MODELS,
        index=0,
        key="llm_model",
        help="Groq-hosted LLM for generating answers.",
    )

    # Response Language & Explaining Tone selector
    st.selectbox(
        "Response Style & Language",
        options=[
            "Auto-detect (Safe Advisory / Hinglish & English)",
            "Hinglish (Safe Advisory / आसान सलाह)",
            "English (Safe Advisory / Patient Guidance)",
        ],
        index=0,
        key="response_language_mode",
        help="Compassionate, non-diagnostic patient education from MedQuAD.",
    )

    # Voice Input Whisper Model
    st.selectbox(
        "🎙️ Speech / Whisper Model",
        options=AVAILABLE_WHISPER_MODELS,
        index=0,
        key="whisper_model",
        help="Groq Whisper model used for fast multilingual voice input transcription.",
    )

    # F-6: Embedding model
    st.text_input(
        "Embedding Model",
        value=DEFAULT_EMBEDDING_MODEL,
        key="embedding_model",
        help="Changing this requires a full re-index.",
    )
    if st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL) != DEFAULT_EMBEDDING_MODEL:
        st.warning("⚠️ Custom embedding model — you must rebuild the index for it to take effect.")


def _render_session_section() -> None:
    """Sidebar: clear chat (F-7) and API key status (F-8)."""
    st.markdown(
        '<p class="sidebar-header">💬 Session</p>', unsafe_allow_html=True
    )

    # F-7: Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_processed_audio_id = None
        st.session_state.pending_voice_query = None
        st.rerun()

    # F-8: API key status indicator
    if has_api_key():
        st.markdown(
            '<span class="status-badge badge-success">🔑 API key detected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge badge-error">🔑 API key not found</span>',
            unsafe_allow_html=True,
        )
        st.caption("Set `GROQ_API_KEY` in your `.env` file.")


# ---------------------------------------------------------------------------
# Main Panel — Chat Interface (F-9 through F-14)
# ---------------------------------------------------------------------------

def render_empty_state() -> None:
    """Show a friendly empty state when no index is built (F-13)."""
    st.markdown(
        """
        <div class="empty-state">
            <h2>🩺 No medical guidance indexed yet</h2>
            <p>Click <strong>🔄 Rebuild Index</strong> in the sidebar to index trusted health records from MedQuAD (NIH / MedlinePlus).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Render the full conversation from session_state (F-9, F-12)."""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources")
        is_error = msg.get("is_error", False)

        with st.chat_message(role):
            if role == "assistant":
                if is_error:
                    st.markdown(
                        f'<div class="error-bubble">{content}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="assistant-bubble">{content}</div>',
                        unsafe_allow_html=True,
                    )

                # F-12: Source transparency panel
                if sources:
                    _render_sources_expander(sources)
            else:
                st.markdown(content, unsafe_allow_html=True)


def _render_sources_expander(sources: list[dict]) -> None:
    """Render the sources expander below an assistant message (F-12)."""
    with st.expander(f"📎 Verified Sources ({len(sources)})"):
        for i, src in enumerate(sources):
            source_file = src.get("source", "NIH / MedlinePlus")
            source_url = src.get("document_url", "")
            distance = src.get("distance", 0.0)
            text = src.get("text", "")
            preview = text[:350] + ("…" if len(text) > 350 else "")

            if source_url:
                st.markdown(
                    f"**{i + 1}. [{source_file}]({source_url})** 🔗 · "
                    f'<span style="color: {COLORS["muted"]}">distance: {distance:.4f}</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"**{i + 1}. {source_file}** · "
                    f'<span style="color: {COLORS["muted"]}">distance: {distance:.4f}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="source-chunk">{preview}</div>',
                unsafe_allow_html=True,
            )
            if i < len(sources) - 1:
                st.divider()


def handle_new_query(query: str, is_voice: bool = False) -> None:
    """Process a new user query: retrieve sources, generate answer (F-10, F-11, F-12, F-14)."""
    display_content = f'<div class="spoken-pill">🎙️ Spoken Query</div>\n\n{query}' if is_voice else query

    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": display_content,
        "raw_query": query,
        "is_voice": is_voice,
        "sources": None,
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(display_content, unsafe_allow_html=True)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                top_k = st.session_state.get("top_k", 3)
                llm_model = st.session_state.get("llm_model", AVAILABLE_LLM_MODELS[0])
                embedding_model = st.session_state.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

                rag = load_rag_search(PERSIST_DIR, embedding_model, llm_model)
                if rag is None:
                    raise RuntimeError(
                        "Cannot generate answers — `GROQ_API_KEY` is not set. "
                        "Add it to your `.env` file and restart the app."
                    )

                # Parse response language mode
                selected_mode = st.session_state.get(
                    "response_language_mode", "Auto-detect (Hinglish / English)"
                )
                if "Hinglish" in selected_mode and "Auto" not in selected_mode:
                    lang_mode = "hinglish"
                elif "English" in selected_mode and "Auto" not in selected_mode:
                    lang_mode = "english"
                else:
                    lang_mode = "auto"

                # Get sources using English-reformulated query if Hinglish
                source_results, search_query_used = rag.retrieve(query, top_k=top_k)
                sources = []
                for r in source_results:
                    meta = r.get("metadata", {}) or {}
                    sources.append({
                        "source": meta.get("source", meta.get("document_source", "MedQuAD/NIH")),
                        "document_url": meta.get("document_url", ""),
                        "distance": r.get("distance", 0.0),
                        "text": meta.get("text", ""),
                    })

                # Get LLM answer with explaining tone in Hinglish/English
                answer = rag.search_and_summarize(
                    query, top_k=top_k, language_mode=lang_mode
                )

                # Render answer
                st.markdown(
                    f'<div class="assistant-bubble">{answer}</div>',
                    unsafe_allow_html=True,
                )

                # Render sources
                if sources:
                    _render_sources_expander(sources)

                # Store in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = f"⚠️ An error occurred: {e}"
                st.markdown(
                    f'<div class="error-bubble">{error_msg}</div>',
                    unsafe_allow_html=True,
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": None,
                    "is_error": True,
                })


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def _handle_voice_submission(audio_file, text_input: str = "") -> None:
    """Transcribe audio from chat input and submit as a spoken query."""
    if not has_api_key():
        st.error("🔑 `GROQ_API_KEY` is required for voice transcription. Add it to `.env`.")
        return

    with st.spinner("🎧 Transcribing your voice with Groq Whisper..."):
        try:
            audio_bytes = audio_file.getvalue()
            whisper_model = st.session_state.get("whisper_model", DEFAULT_WHISPER_MODEL)
            transcribed_text = transcribe_audio(
                audio_bytes,
                filename=getattr(audio_file, "name", "recording.wav") or "recording.wav",
                model=whisper_model,
            )
            if transcribed_text and transcribed_text.strip():
                final_query = (
                    f"{text_input} {transcribed_text.strip()}".strip()
                    if text_input
                    else transcribed_text.strip()
                )
                handle_new_query(final_query, is_voice=True)
            else:
                st.warning("⚠️ No speech could be detected in the recording. Please speak clearly and try again.")
        except Exception as e:
            st.error(f"⚠️ Voice transcription error: {e}")


def main() -> None:
    """Application entry point — orchestrates all UI components."""
    configure_page()
    inject_custom_css()
    init_session_state()

    # F-15: App title & tagline
    st.title("🩺 MedAssist — Consumer Health Advisor")
    st.caption("Evidence-based consumer health guidance powered by MedQuAD (NIH, CDC, MedlinePlus).")

    # Medical Disclaimer Banner
    st.info(
        "ℹ️ **Medical Disclaimer:** This assistant provides general health education from public medical sources (MedQuAD/NIH). "
        "It is **not** a substitute for professional medical diagnosis, advice, or treatment. "
        "If you are having a medical emergency, call your local emergency services or consult a physician immediately."
    )

    # Sidebar
    render_sidebar()

    # Main panel: empty state vs chat
    if not st.session_state.index_ready:
        render_empty_state()

        # Also block chat if API key is missing
        if not has_api_key():
            st.error(
                "🔑 **API key not found.** Set `GROQ_API_KEY` in your `.env` file to enable chat."
            )
    else:
        # Show API key warning inline if missing (chat will still show but queries will fail gracefully)
        if not has_api_key():
            st.error(
                "🔑 **API key not found.** Set `GROQ_API_KEY` in your `.env` file. "
                "You can browse the index, but questions will fail."
            )

        # Render existing chat history
        render_chat_history()

        # Process any newly transcribed voice query (from external or legacy triggers)
        if pending_voice := st.session_state.get("pending_voice_query"):
            st.session_state.pending_voice_query = None
            handle_new_query(pending_voice, is_voice=True)

        # F-10: Unified chat & voice input in the exact same typing space
        if st.session_state.rebuild_in_progress:
            st.info("⏳ Index rebuild in progress — please wait…")
        else:
            chat_val = st.chat_input(
                "Ask a health question in English or Hinglish (type or tap 🎙️ to speak)…",
                accept_audio=True,
            )

            if chat_val:
                audio_file = getattr(chat_val, "audio", None) or (
                    chat_val.get("audio") if isinstance(chat_val, dict) else None
                )
                text_input = getattr(chat_val, "text", None) or (
                    chat_val.get("text", "")
                    if isinstance(chat_val, dict)
                    else (chat_val if isinstance(chat_val, str) else "")
                )
                text_input = (text_input or "").strip()

                if audio_file is not None:
                    _handle_voice_submission(audio_file, text_input=text_input)
                elif text_input:
                    handle_new_query(text_input, is_voice=False)


if __name__ == "__main__":
    main()
