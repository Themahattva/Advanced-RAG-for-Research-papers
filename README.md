# 🩺 MedAssist — Consumer Health & Medical Guidance RAG

> A modular **Retrieval-Augmented Generation (RAG)** pipeline purpose-built for querying consumer health, symptom guidance, and medical information using the **MedQuAD** dataset (NIH, CDC, MedlinePlus) with semantic search over a FAISS vector store and Groq-hosted LLMs — answering in a **safe, non-diagnostic advisory tone** in both **English and Hinglish**.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white">
  <img alt="FAISS" src="https://img.shields.io/badge/VectorDB-FAISS-00A1D6">
  <img alt="Groq" src="https://img.shields.io/badge/LLM-Groq-F55036">
  <img alt="Dataset" src="https://img.shields.io/badge/Dataset-MedQuAD_(NIH)-blue">
  <img alt="Streamlit" src="https://img.shields.io/badge/GUI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Dataset Attribution](#-dataset-attribution-medquad)
- [Architecture](#-architecture)
- [Safe Advisory Framework](#-safe-advisory-framework)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 🔍 Overview

**MedAssist** ingests trusted medical knowledge from the **MedQuAD** dataset (47,400+ QA pairs curated from 12 NIH institutes including MedlinePlus, CDC, NIDDK, and Cancer.gov), indexes the vectors in a **FAISS** similarity store, and answers user queries in a **safe, non-diagnostic, and empathetic advisory tone**.

**Core capabilities:**

| Capability | Description |
|---|---|
| 🩺 Trusted Medical QA Corpus | Grounded in 47,400+ validated QA pairs from 12 NIH institutes (MedlinePlus, CDC, GHR, Cancer.gov) |
| 🛡️ Safe & Non-Diagnostic | Evidence-based patient education without prescribing drug dosages or offering personal clinical diagnoses |
| ⚠️ Red-Flag Warning Triage | Explicitly highlights emergency symptoms that require urgent medical attention |
| 🇮🇳 Bilingual Support (Hinglish & English) | Intelligently understands queries in Hinglish and delivers warm, clear explanations in Roman-script Hindi |
| ⚡ Semantic Vector Search | FAISS flat L2 index over `all-MiniLM-L6-v2` dense embeddings |
| 🎙️ Voice & Microphone Input | Native browser audio recording + fast multilingual transcription via Groq Whisper (`whisper-large-v3-turbo`) |
| 🔗 Source Transparency | Direct citation badges and links to original NIH/MedlinePlus references for every answer |
| 🖥️ Streamlit Web GUI | Dark-themed health consultation chat interface with medical disclaimer banner |

---

## 📚 Dataset Attribution: MedQuAD

This project utilizes the **MedQuAD (Medical Question Answering Dataset)**:
- **Hugging Face**: [`lavita/MedQuAD`](https://huggingface.co/datasets/lavita/MedQuAD)
- **GitHub**: [`abachaa/MedQuAD`](https://github.com/abachaa/MedQuAD)
- **Source Material**: 47,457 medical question-answer pairs compiled from 12 trusted National Institutes of Health (NIH) websites, including **MedlinePlus**, **CDC**, **Cancer.gov**, **NIDDK**, and **GHR**.

---

## 🛡️ Safe Advisory Framework

Healthcare RAG requires responsible AI principles:
1. **Non-Diagnostic**: The system informs rather than diagnoses, stating potential causes according to medical literature without asserting personal clinical certainty.
2. **Non-Prescriptive**: Avoids prescribing specific drug dosages or advising users to alter prescribed regimens.
3. **Emergency Triage**: Explicitly identifies red-flag symptoms requiring emergency medical evaluation.
4. **Universal Medical Disclaimer**: Attached to all responses and prominently displayed across the UI.

---

## 🏗️ Architecture

### End-to-end pipeline

```mermaid
flowchart TD
    A[📦 MedQuAD Dataset<br/>lavita/MedQuAD Parquet · 47.4k QA Pairs] -->|load_medquad_documents| B[Document Processor<br/>NIH · MedlinePlus · CDC Sources]
    B --> C[Standardized Medical Documents]
    C -->|RecursiveCharacterTextSplitter| D[Medical Knowledge Chunks]
    D -->|SentenceTransformer all-MiniLM-L6-v2| E[Dense Vector Embeddings]
    E -->|IndexFlatL2.add| F[(FAISS Vector Index<br/>+ metadata.pkl)]
    F -->|save / load| G[faiss_store_medquad/ on disk]

    H[🔎 Health Query / Hinglish or English] -->|reformulate_query| I[English Medical Keywords]
    I -->|index.search top_k| F
    F --> J[Top-K MedQuAD Chunks + URLs]
    J --> K[Safe Advisory Prompting<br/>Non-Diagnostic + Red Flags + Disclaimer]
    K -->|Groq LLM| L[💬 Safe Consumer Health Guidance]
```

### Component / class relationship

```mermaid
classDiagram
    class data_loader {
        +load_all_documents(data_dir) List~Document~
    }
    class EmbeddingPipeline {
        -model_name: str
        -chunk_size: int
        -chunk_overlap: int
        -model: SentenceTransformer
        +chunk_documents(documents) List~Chunk~
        +embed_chunks(chunks) ndarray
    }
    class FaissVectorStore {
        -persist_dir: str
        -index: faiss.Index
        -metadata: List
        -embedding_model: str
        +build_from_documents(documents)
        +add_embeddings(embeddings, metadatas)
        +save()
        +load()
        +search(query_embedding, top_k) List
        +query(query_text, top_k) List
    }
    class RAGSearch {
        -vectorstore: FaissVectorStore
        -llm: ChatGroq
        +search_and_summarize(query, top_k) str
    }
    class app_py {
        +main()
    }

    data_loader --> EmbeddingPipeline : documents
    EmbeddingPipeline --> FaissVectorStore : embeddings
    FaissVectorStore --> RAGSearch : retrieved chunks
    RAGSearch --> app_py : summary
    data_loader --> FaissVectorStore : (build_from_documents)
```

### Query sequence

```mermaid
sequenceDiagram
    participant U as User
    participant App as app.py / gui.py
    participant RS as RAGSearch
    participant VS as FaissVectorStore
    participant EM as SentenceTransformer
    participant FI as FAISS Index
    participant LLM as ChatGroq

    U->>App: Run query ("What is an attention mechanism?")
    App->>RS: search_and_summarize(query, top_k)
    RS->>VS: query(query_text, top_k)
    VS->>EM: encode(query_text)
    EM-->>VS: query_embedding
    VS->>FI: search(query_embedding, top_k)
    FI-->>VS: top-k (index, distance, metadata)
    VS-->>RS: results
    RS->>RS: assemble context + prompt
    RS->>LLM: invoke(prompt)
    LLM-->>RS: grounded summary
    RS-->>App: summary text
    App-->>U: printed summary
```

### Ingestion flow

```mermaid
flowchart LR
    A[Data dir] -->|load_all_documents| B[Documents]
    B -->|RecursiveCharacterTextSplitter| C[Chunks]
    C -->|all-MiniLM-L6-v2| D[Embeddings]
    D -->|FAISS IndexFlatL2| E[(faiss_store/)]
```

### Voice query flow

```mermaid
flowchart LR
    U[🎙️ Mic] -->|audio_input| T[Groq Whisper]
    T -->|transcript| Q[Query text]
    Q -->|reformulate + search| R[FAISS + LLM]
    R -->|response| U
```

---

## ⚙️ How It Works

1. **Ingestion** — `src/data_loader.py` recursively scans the `data/` directory and loads every supported file type into standardized LangChain `Document` objects, with per-file error handling and debug logging.
2. **Chunking** — `src/embedding.py`'s `EmbeddingPipeline` splits documents using `RecursiveCharacterTextSplitter` (default `chunk_size=1000`, `chunk_overlap=200`) so that context windows stay within embedding/LLM limits while preserving semantic continuity.
3. **Embedding** — Each chunk is encoded into a dense vector using the `all-MiniLM-L6-v2` Sentence-Transformer model (384-dim embeddings).
4. **Indexing** — `src/vectorstore.py`'s `FaissVectorStore` stores vectors in a FAISS `IndexFlatL2` index and keeps parallel chunk metadata (`text`, `source`) in a pickled list, persisting both to `faiss_store/`.
5. **Retrieval** — At query time, the query string is embedded with the same model and compared against the index via L2 distance to fetch the `top_k` most similar chunks.
6. **Generation** — `src/search.py`'s `RAGSearch` concatenates retrieved chunk text into a context block and prompts a Groq-hosted LLM (via `langchain-groq`'s `ChatGroq`) to produce a grounded, query-specific summary.

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | **Python 3.10+** | Core implementation |
| GUI | **Streamlit** | Dark-themed chat web interface |
| Orchestration | **LangChain** (`langchain`, `langchain-core`, `langchain-community`) | Document loaders, text splitting, chaining |
| PDF Parsing | **pypdf**, **pymupdf** | Extracting text from research paper PDFs |
| Embeddings | **sentence-transformers** (`all-MiniLM-L6-v2`) | Dense semantic vector generation |
| Vector Store | **FAISS** (`faiss-cpu`) | Fast approximate/exact nearest-neighbor similarity search |
| Alt. Vector DB | **ChromaDB**, **Typesense** | Available for extension beyond FAISS |
| LLM Provider | **Groq** (`langchain-groq`) | Fast low-latency inference for summarization |
| Alt. LLM Provider | **langchain-openai** | Optional OpenAI-backed generation |
| Agentic Extension | **LangGraph** | Available for building multi-step / agentic RAG flows |
| Config | **python-dotenv** | Environment variable / API key management |
| Notebooks | **Jupyter** (`document.ipynb`, `pdf_loader.ipynb`) | Exploration and prototyping |

### `requirement.txt` (dependency manifest)

```text
langchain
langchain-core
langchain-community
pypdf
pymupdf
sentence-transformers
faiss-cpu
chromadb
langchain-groq
python-dotenv
typesense
langchain_openai
langgraph
streamlit
```
---

## 📁 Project Structure

```text
Advanced-RAG-for-Research-papers/
├── .streamlit/
│   └── config.toml             # Dark theme configuration
├── gui.py                      # Streamlit web interface (chat GUI)
├── app.py                      # CLI entry point
├── main.py                     # (placeholder / reserved entry point)
├── requirement.txt             # Python dependencies
├── data/
│   ├── pdf/                    # Source research paper PDFs
│   └── text_files/             # Plain-text source documents
├── notebook/
│   ├── document.ipynb          # Document loading experiments
│   └── pdf_loader.ipynb        # PDF-loading exploration notebook
└── src/
    ├── __init__.py
    ├── data_loader.py          # Multi-format document ingestion
    ├── embedding.py            # Chunking + embedding pipeline
    ├── vectorstore.py          # FAISS index build/save/load/search
    └── search.py               # Retrieval + Groq LLM summarization (RAGSearch)
```

### Module responsibility matrix

| File | Key Class / Function | Responsibility |
|---|---|---|
| `src/data_loader.py` | `load_all_documents(data_dir)` | Discover and parse PDF/TXT/CSV/XLSX/DOCX/JSON files into `Document` objects |
| `src/embedding.py` | `EmbeddingPipeline` | Chunk documents and generate embeddings |
| `src/vectorstore.py` | `FaissVectorStore` | Build, persist, load, and query the FAISS index |
| `src/search.py` | `RAGSearch` | Orchestrate retrieval + LLM-based summarization |
| `src/voice.py` | `transcribe_audio` | Multilingual speech-to-text service powered by Groq Whisper |
| `app.py` | — | Example driver script tying all components together |
| `gui.py` | — | Streamlit chat GUI with microphone voice input & source transparency |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Themahattva/Advanced-RAG-for-Research-papers.git
cd Advanced-RAG-for-Research-papers

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirement.txt

# 4. Configure environment variables
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

> 🔑 **API Key required**: `src/search.py` uses `ChatGroq`, which needs a valid Groq API key. Set it via a `.env` file (loaded with `python-dotenv`).

---

## ▶️ Usage

### 1. Add your research papers

Place PDFs, text files, spreadsheets, Word docs, or JSON files inside `data/` (subfolders like `data/pdf/` and `data/text_files/` are supported since loading is recursive).

### 2. Run the Streamlit GUI (recommended)

```bash
streamlit run gui.py
```

This launches a dark-themed chat application where you can:

- **Upload** research papers (PDF, TXT, CSV, XLSX, DOCX, JSON) via the sidebar
- **Build/rebuild** the FAISS vector index with a single click
- **Ask questions via text or voice**:
  - Type questions in the bottom chat bar in English or Hinglish.
  - Or tap the **🎙️ microphone button directly in the chat bar** to record your voice using your microphone; Groq Whisper transcribes speech in real time.
- **View sources** — every answer shows the exact document chunks used, with filenames and similarity scores

### 3. Build the vector index (CLI)

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

docs = load_all_documents("data")
store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)   # chunks, embeds, indexes, and saves to disk
```

### 4. Query and summarize (CLI)

```python
from src.search import RAGSearch

rag_search = RAGSearch(persist_dir="faiss_store_medquad")

# 1. Ask in English (safe consumer health guidance)
advice_en = rag_search.search_and_summarize(
    "What are the common symptoms of asthma and how is it managed?",
    top_k=3
)
print(advice_en)

# 2. Ask in Hinglish (automatic language detection & empathetic advisory tone)
advice_hi = rag_search.search_and_summarize(
    "High blood pressure ke kya lakshan hote hain aur ghar par kya savdhaniyan bartein?",
    top_k=3,
    language_mode="auto"  # options: 'auto', 'hinglish', 'english'
)
print(advice_hi)
```

### 5. Run the example driver script

```bash
python app.py
```

Expected flow: loads the persisted MedQuAD FAISS store (`faiss_store_medquad`) → executes a consumer health query with automatic query reformulation for NIH semantic lookup → prints a grounded, safe, non-diagnostic advisory answer in Hinglish or English.

---

## 🧩 Module Reference

### `FaissVectorStore`

| Method | Description |
|---|---|
| `build_from_documents(documents)` | Full pipeline: chunk → embed → index → save |
| `add_embeddings(embeddings, metadatas)` | Add raw vectors + metadata to the index |
| `save()` | Persist `faiss.index` and `metadata.pkl` to `persist_dir` |
| `load()` | Load a previously persisted index + metadata |
| `search(query_embedding, top_k)` | Raw vector search returning index/distance/metadata |
| `query(query_text, top_k)` | Convenience wrapper — embeds text then searches |

### `EmbeddingPipeline`

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `all-MiniLM-L6-v2` | Sentence-Transformer model used for embeddings |
| `chunk_size` | `1000` | Max characters per chunk |
| `chunk_overlap` | `200` | Overlap between consecutive chunks |

### `RAGSearch`

| Parameter | Default | Description |
|---|---|---|
| `persist_dir` | `faiss_store` | Path to the FAISS index directory |
| `embedding_model` | `all-MiniLM-L6-v2` | Embedding model for query encoding |
| `llm_model` | `llama-3.3-70b-versatile` | Groq-hosted model used for summarization |

---

## 📄 Supported File Types

| Format | Loader Used | Extension |
|---|---|---|
| PDF | `PyPDFLoader` | `.pdf` |
| Plain Text | `TextLoader` | `.txt` |
| CSV | `CSVLoader` | `.csv` |
| Excel | `UnstructuredExcelLoader` | `.xlsx` |
| Word | `Docx2txtLoader` | `.docx` |
| JSON | `JSONLoader` | `.json` |

---

## 🔧 Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| `GROQ_API_KEY` | `.env` | — | Required for `ChatGroq` LLM calls |
| Embedding model | `EmbeddingPipeline(model_name=...)` | `all-MiniLM-L6-v2` | Swap for any Sentence-Transformer model |
| Chunk size / overlap | `EmbeddingPipeline(chunk_size, chunk_overlap)` | `1000` / `200` | Tune for longer/shorter context windows |
| Vector store path | `FaissVectorStore(persist_dir=...)` | `faiss_store` | Directory for `faiss.index` + `metadata.pkl` |
| LLM model | `RAGSearch(llm_model=...)` | `llama-3.3-70b-versatile` | Any Groq-supported chat model |
| Top-K results | `search_and_summarize(query, top_k=...)` | `5` | Number of retrieved chunks per query |

---

## 🗺️ Roadmap

- [x] Move hardcoded `groq_api_key = ""` in `src/search.py` to `.env`-driven config
- [x] Fix relative import in `vectorstore.py`'s `__main__` block
- [x] Add a Streamlit front-end with dark theme and source transparency
- [x] Add source-citation metadata (filenames) to generated summaries
- [x] Support Hinglish queries with automatic English retrieval reformulation & explanatory tone
- [ ] Swap `IndexFlatL2` for an approximate index (`IndexIVFFlat` / `HNSW`) for large corpora
- [ ] Add automated tests and CI
- [ ] Streaming token-by-token answers in the GUI

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m "Add my feature"`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📜 License

This project currently has no explicit license file. Consider adding an [MIT License](https://choosealicense.com/licenses/mit/) or similar to clarify usage terms for contributors and downstream users.

---

<p align="center"><i>Built for researchers who'd rather ask their papers questions than re-read them.</i></p>
