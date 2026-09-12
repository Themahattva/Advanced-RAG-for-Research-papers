# Product Requirements Document (PRD)
## Streamlit GUI for Advanced RAG for Research Papers

| Field | Value |
|---|---|
| Project | Advanced RAG for Research Papers — Streamlit GUI |
| Repo | `Themahattva/Advanced-RAG-for-Research-papers` |
| Author | Mahattva |
| Version | 1.0 |
| Status | Ready for implementation |
| Target Build Environment | Claude (Antigravity) |
| Target Entry File | `gui.py` (repo root) |

---

## 1. Purpose

Build a **dark-themed, single-page Streamlit web application** that provides a conversational (chat-style) interface to the existing RAG backend in `src/`. The GUI must let a user upload research papers, trigger indexing, ask natural-language questions, and see grounded answers along with the exact source chunks used to generate them — without touching any code or notebooks.

This PRD is a complete, self-contained implementation brief. The build agent should implement **exactly** what is specified here, reusing the existing backend modules (`src/data_loader.py`, `src/embedding.py`, `src/vectorstore.py`, `src/search.py`) without modifying their public interfaces unless explicitly instructed in [Section 9 — Required Backend Fixes](#9-required-backend-fixes-before-gui-integration).

---

## 2. Background / Existing System

The repo already implements a working RAG pipeline:

```mermaid
flowchart LR
    A[data/ folder] -->|load_all_documents| B[Document Loaders]
    B --> C[Chunking - EmbeddingPipeline]
    C --> D[Embeddings - all-MiniLM-L6-v2]
    D --> E[(FAISS Index)]
    E --> F[RAGSearch.search_and_summarize]
    F --> G[ChatGroq gemma2-9b-it]
    G --> H[Grounded Answer]
```

Key existing classes the GUI must call into:

| Class / Function | File | Method(s) the GUI uses |
|---|---|---|
| `load_all_documents(data_dir)` | `src/data_loader.py` | Reload all documents after new uploads |
| `FaissVectorStore` | `src/vectorstore.py` | `build_from_documents()`, `load()`, `query()` |
| `EmbeddingPipeline` | `src/embedding.py` | Used internally by `FaissVectorStore` |
| `RAGSearch` | `src/search.py` | `search_and_summarize(query, top_k)` |

The GUI is a **presentation layer only** — it must not reimplement chunking, embedding, indexing, or generation logic. All of that stays in `src/`.

---

## 3. Goals

1. Ship a working, installable Streamlit app (`streamlit run gui.py`) that talks to the existing RAG backend.
2. Dark theme, applied consistently and by default (not user-togglable in v1).
3. Support the full lifecycle: **upload papers → build/rebuild index → ask questions → view sourced answers → manage chat history**.
4. Make retrieval transparent: every answer must show which chunks (with filename + similarity score) were used.
5. Be resilient to common failure states (missing API key, empty index, unsupported files, empty query).

## 4. Non-Goals (v1)

- No user authentication / multi-user support.
- No light-mode toggle (dark only, per requirement).
- No cloud vector DB (FAISS local index only).
- No streaming token-by-token LLM output (v1 shows the full answer once ready — may be added in v2 via `st.write_stream`).
- No mobile-specific responsive design (Streamlit's default responsiveness is acceptable).

---

## 5. Target User & Use Case

**Primary user:** A student/researcher (the repo owner) who has a folder of research PDFs and wants to semantically query them ("What does this paper say about X?", "Compare methodology across these papers") and get a grounded, LLM-generated answer with visible sources — instead of manually searching PDFs.

---

## 6. Feature List (Functional Requirements)

### 6.1 Sidebar — Document Management Panel

| ID | Feature | Requirement |
|---|---|---|
| F-1 | File uploader | `st.file_uploader` accepting multiple files: `.pdf`, `.txt`, `.csv`, `.xlsx`, `.docx`, `.json`. Uploaded files are saved into `data/` (route by extension into subfolders matching existing structure, e.g. PDFs → `data/pdf/`, everything else → `data/text_files/` or a new appropriately named subfolder). |
| F-2 | Indexed files table | `st.dataframe` listing all files currently present in `data/` (recursive scan), showing filename, type, and file size. Refreshes after upload/rebuild. |
| F-3 | Rebuild Index button | A prominent button (`st.sidebar.button("🔄 Rebuild Index")`) that calls `load_all_documents("data")` → `FaissVectorStore.build_from_documents(docs)`. Must show a progress indicator (`st.status` or `st.spinner`) with stage labels: "Loading documents…", "Chunking…", "Generating embeddings…", "Building index…", "Done ✅". |
| F-4 | Index status indicator | A small status badge showing: index not built / index loaded (with chunk count + last-built timestamp) / rebuilding in progress. |
| F-5 | Retrieval settings | `st.sidebar.slider` for `top_k` (range 1–10, default 3). |
| F-6 | Model settings | `st.sidebar.selectbox` for LLM model (values: `gemma2-9b-it`, plus any other Groq chat models the user wants to add later — implement as a simple editable list/constant). `st.sidebar.text_input` (or dropdown) for embedding model, defaulting to `all-MiniLM-L6-v2`, editable but with a warning that changing it requires a full re-index. |
| F-7 | Clear chat button | Clears `st.session_state.messages` and resets the chat panel. |
| F-8 | API key status | A read-only indicator showing whether `GROQ_API_KEY` is detected in the environment (green check / red warning). Never display the key value itself. |

### 6.2 Main Panel — Chat Interface

| ID | Feature | Requirement |
|---|---|---|
| F-9 | Chat history display | Use `st.chat_message("user")` / `st.chat_message("assistant")` to render the full conversation stored in `st.session_state.messages`. |
| F-10 | Chat input | `st.chat_input("Ask a question about your papers…")` pinned at the bottom. |
| F-11 | Answer generation | On submit: append user message → show spinner ("Thinking…") → call `RAGSearch.search_and_summarize(query, top_k)` → append and render assistant message. |
| F-12 | Source transparency panel | Directly below each assistant answer, an `st.expander("📎 Sources (n)")` listing each retrieved chunk: source filename, similarity/distance score, and a truncated text preview (~300 chars) of the chunk. Requires the backend metadata fix described in Section 9. |
| F-13 | Empty-state screen | When no index exists yet, the main panel shows a friendly empty state ("No documents indexed yet — upload papers and click Rebuild Index in the sidebar") instead of the chat input. |
| F-14 | Guard rails | Disable/ignore chat input while an index rebuild is in progress. Show inline error (not a crash) if `search_and_summarize` raises an exception (e.g., missing API key, empty context). |

### 6.3 Header / Branding

| ID | Feature | Requirement |
|---|---|---|
| F-15 | App title & tagline | Page title: **"MedAssist"**, subtitle: "Ask questions across your research paper library." |
| F-16 | Page config | `st.set_page_config(page_title="MedAssist", page_icon="📚", layout="wide", initial_sidebar_state="expanded")` |

---

## 7. Design Requirements — Dark Theme

### 7.1 Theme configuration

Create `.streamlit/config.toml` in the repo root (Streamlit's native theming mechanism — do **not** rely only on custom CSS overrides):

```toml
[theme]
base = "dark"
primaryColor = "#7C5CFF"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#161A23"
textColor = "#E6E6E6"
font = "sans serif"
```

### 7.2 Color palette

| Token | Hex | Usage |
|---|---|---|
| Background (primary) | `#0E1117` | Main app background |
| Background (secondary) | `#161A23` | Sidebar, cards, expanders |
| Accent / primary | `#7C5CFF` | Buttons, active states, links, badges |
| Text (primary) | `#E6E6E6` | Body text |
| Text (muted) | `#9AA0AC` | Captions, metadata, source scores |
| Success | `#3DD68C` | Index built / API key found |
| Warning | `#F5A623` | Rebuild needed / stale index |
| Error | `#FF5C5C` | Missing API key / failed query |
| Chat bubble — user | `#1F2430` | User message background |
| Chat bubble — assistant | `#161A23` with `#7C5CFF` left border accent | Assistant message background |

### 7.3 Layout

- `layout="wide"`, sidebar expanded by default.
- Sidebar sections separated with `st.divider()`: **Documents** → **Index** → **Settings** → **Session**.
- Consistent iconography using emoji (no external icon library needed): 📁 uploads, 🔄 rebuild, ⚙️ settings, 📎 sources, 🗑️ clear chat, 🔑 API key status.
- Source expander content styled as a monospace/code-like block for chunk previews to visually distinguish retrieved text from generated answer text.

### 7.4 Optional custom CSS (inject via `st.markdown(..., unsafe_allow_html=True)`)

Use sparingly, only to refine what `config.toml` theming can't reach (e.g., rounding chat bubble corners, expander borders). Must not break Streamlit's native component behavior.

---

## 8. Tech Stack

| Layer | Choice |
|---|---|
| App framework | `streamlit` (latest stable) |
| Backend (unchanged) | `langchain`, `langchain-community`, `sentence-transformers`, `faiss-cpu`, `langchain-groq`, `python-dotenv` |
| State management | `st.session_state` |
| Caching | `st.cache_resource` for `RAGSearch` / `FaissVectorStore` instances (load once per session/process) |
| Config | `.streamlit/config.toml` for theme; `.env` for `GROQ_API_KEY` (loaded via existing `python-dotenv` usage in `src/search.py`) |
| Data display | `st.dataframe`, `st.expander`, `st.status`/`st.spinner` |

### New dependency to add to `requirment.txt`

```text
streamlit
```

---

## 9. Required Backend Fixes (before GUI integration)

The build agent must make these small backend corrections first, since the GUI depends on correct behavior:

1. **`src/search.py`** — `groq_api_key = ""` is hardcoded and empty. Change to read from environment: `groq_api_key = os.getenv("GROQ_API_KEY")`. If missing, the GUI must surface a clear error state (per F-8, F-14) rather than let the app crash.
2. **`src/vectorstore.py`** — the `__main__` block imports `from data_loader import load_all_documents` (relative, breaks when run as `src.vectorstore`). Fix to `from src.data_loader import load_all_documents`. This does not affect the GUI directly but should be fixed for consistency while in the file.
3. **Metadata enrichment for source transparency (F-12)** — Currently `FaissVectorStore.build_from_documents` stores only `{"text": chunk.page_content}` as metadata. Extend this to also include the source filename, e.g.:
   ```python
   metadatas = [
       {
           "text": chunk.page_content,
           "source": chunk.metadata.get("source", "unknown"),
       }
       for chunk in chunks
   ]
   ```
   This requires `chunk.metadata["source"]` to already be populated by the LangChain loaders (it is, by default, for `PyPDFLoader`, `TextLoader`, etc. — verify and pass through). Update `FaissVectorStore.search()`/`query()` return handling only if needed to pass this field through (it already returns full `metadata` dicts, so no change needed there beyond storing the richer metadata at build time).
4. **`RAGSearch.search_and_summarize`** — should optionally return (or the GUI should independently fetch) both the retrieved chunk metadata *and* the LLM summary, so the GUI can render sources alongside the answer. Recommended approach: **do not change the method's return type** (to avoid breaking `app.py`); instead, in `gui.py`, call `rag.vectorstore.query(query, top_k=top_k)` directly to get chunk metadata for the sources panel, in addition to calling `search_and_summarize()` for the answer text.

---

## 10. Application Structure

```text
Advanced-RAG-for-Research-papers/
├── .streamlit/
│   └── config.toml          # NEW — dark theme config
├── gui.py                    # NEW — Streamlit entry point
├── app.py                    # unchanged (CLI example)
├── requirment.txt            # UPDATED — add `streamlit`
├── data/
│   ├── pdf/
│   └── text_files/
└── src/
    ├── data_loader.py
    ├── embedding.py
    ├── vectorstore.py        # metadata fix (Section 9.3)
    └── search.py              # env var fix (Section 9.1)
```

`gui.py` should be organized into clear functions/sections:

```python
# gui.py — structure outline (implement fully per Section 6 & 7)

import streamlit as st
# ... imports from src ...

def configure_page(): ...          # F-16
def render_sidebar(): ...          # F-1..F-8
def render_empty_state(): ...      # F-13
def render_chat_history(): ...     # F-9, F-12
def handle_new_query(query): ...   # F-10, F-11, F-12, F-14
def main():
    configure_page()
    render_sidebar()
    if index_not_ready:
        render_empty_state()
    else:
        render_chat_history()
        if query := st.chat_input(...):
            handle_new_query(query)

if __name__ == "__main__":
    main()
```

---

## 11. State Model

| `st.session_state` key | Type | Purpose |
|---|---|---|
| `messages` | `list[dict]` | `[{"role": "user"|"assistant", "content": str, "sources": list[dict] | None}]` — full chat history |
| `index_ready` | `bool` | Whether a FAISS index currently exists/loaded |
| `last_indexed_at` | `datetime | None` | Timestamp of last successful rebuild, shown in status badge |
| `rebuild_in_progress` | `bool` | Guards concurrent actions during indexing |

---

## 12. Error / Edge Case Handling

| Scenario | Required Behavior |
|---|---|
| `GROQ_API_KEY` missing | Sidebar shows red "API key not found" badge; chat input disabled with inline message pointing to `.env` setup |
| No files in `data/` and user clicks Rebuild Index | Show warning: "No documents found — upload files first." Do not attempt to build an empty index. |
| Query submitted with no index built | Chat input is not shown (empty state per F-13) — this state should be unreachable, but if hit, show inline warning instead of calling the backend |
| Unsupported file type uploaded | Reject at upload time with `st.error` listing supported extensions (from Section 6.1 table) |
| `search_and_summarize` throws an exception | Catch and render as an assistant "error" message bubble (styled with the Error color token) instead of crashing the app |
| Very large PDF causing slow embedding | Progress indicator (F-3) must remain visible/responsive; no hard requirement to background-thread it in v1, but the UI must not appear frozen without feedback |

---

## 13. Acceptance Criteria

The implementation is considered complete when all of the following are true:

- [ ] `streamlit run gui.py` launches without errors, using dark theme from `.streamlit/config.toml`.
- [ ] User can upload at least PDF and TXT files via the sidebar and see them appear in the indexed-files table.
- [ ] Clicking "Rebuild Index" runs the full pipeline and shows staged progress, ending in a success state with a chunk count.
- [ ] Asking a question in the chat input returns a grounded answer generated via `RAGSearch`.
- [ ] Each assistant answer has an expandable "Sources" section showing filename + score + text preview for each retrieved chunk.
- [ ] `top_k` slider visibly changes the number of sources shown/used.
- [ ] Missing `GROQ_API_KEY` produces a clear, non-crashing error state in the UI.
- [ ] "Clear Chat" empties the visible conversation without affecting the FAISS index.
- [ ] All three backend fixes in Section 9 are applied and do not break `app.py`'s existing CLI flow.
- [ ] No secrets (API keys) are ever rendered in the UI or logs.

---

## 14. Implementation Phases (suggested order for the build agent)

1. **Backend fixes** (Section 9) — env var fix, import fix, metadata enrichment.
2. **Theme setup** — `.streamlit/config.toml` + page config.
3. **Sidebar skeleton** — uploader, indexed-files table, settings controls (no wiring yet).
4. **Index build wiring** — connect Rebuild Index button to real pipeline with progress states.
5. **Chat core** — session state, chat history rendering, query handling, answer rendering.
6. **Source transparency panel** — wire `vectorstore.query()` results into the sources expander.
7. **Error/edge-case handling** — Section 12 checklist.
8. **Polish pass** — custom CSS touches, empty states, badges, copy/wording review against Section 6/7.
9. **Update `requirment.txt`** and add a short "Running the GUI" section to `README.md`.

---

## 15. Out of Scope / Future Enhancements (v2 ideas, do not build now)

- Streaming token-by-token answers.
- Multi-conversation / saved sessions.
- Per-document delete/remove from index (currently full rebuild only).
- Support for switching vector backends (ChromaDB/Typesense) from the UI.
- Authentication for multi-user deployments.
- Deployment config for Streamlit Community Cloud / Docker.
