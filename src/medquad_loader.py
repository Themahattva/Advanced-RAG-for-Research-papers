import os
import sys
import ssl
import urllib.request
from pathlib import Path
from typing import List, Optional
import pandas as pd
import certifi
from langchain_core.documents import Document

MEDQUAD_PARQUET_URL = (
    "https://huggingface.co/datasets/lavita/MedQuAD/resolve/main/data/train-00000-of-00001-e36383d177026d53.parquet"
)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "medquad"
DEFAULT_PARQUET_FILE = DEFAULT_CACHE_DIR / "medquad.parquet"


def download_medquad(target_path: Optional[Path] = None) -> Path:
    """Download the MedQuAD dataset parquet file from Hugging Face if not already cached."""
    target_path = target_path or DEFAULT_PARQUET_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size > 1000000:
        print(f"[INFO] MedQuAD dataset already cached at: {target_path}")
        return target_path

    print(f"[INFO] Downloading MedQuAD dataset from Hugging Face to {target_path}...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        MEDQUAD_PARQUET_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MedQuAD-Downloader"}
    )
    with urllib.request.urlopen(req, context=ctx) as response, open(target_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 64
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({percent:.1f}%)")
                sys.stdout.flush()

    print("\n[INFO] Download completed successfully!")
    return target_path


def load_medquad_documents(
    parquet_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> List[Document]:
    """
    Load MedQuAD QA pairs as LangChain Document objects.
    
    Args:
        parquet_path: Path to the cached parquet file.
        limit: Max number of QA pairs to load. If None, loads all ~47,441 pairs.
    """
    parquet_file = download_medquad(parquet_path)
    print(f"[INFO] Loading MedQuAD records from {parquet_file} (limit={limit})...")
    df = pd.read_parquet(parquet_file, engine="pyarrow")

    if limit and limit < len(df):
        # Prioritize diverse and common health categories/types
        df = df.head(limit)

    documents: List[Document] = []
    for _, row in df.iterrows():
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        focus = str(row.get("question_focus") or "").strip()
        source = str(row.get("document_source") or "NIH/MedlinePlus").strip()
        url = str(row.get("document_url") or "").strip()
        q_type = str(row.get("question_type") or "General Information").strip()

        if not question or not answer:
            continue

        # Format content for optimal semantic retrieval and LLM context
        page_content = (
            f"Topic / Condition: {focus}\n"
            f"Question: {question}\n"
            f"Category / Type: {q_type}\n"
            f"Source: {source}\n"
            f"Medical Guidance:\n{answer}"
        )

        metadata = {
            "source": f"{source} - {focus}" if focus else source,
            "document_source": source,
            "document_url": url,
            "question": question,
            "question_focus": focus,
            "question_type": q_type,
            "text": answer,
        }

        documents.append(Document(page_content=page_content, metadata=metadata))

    print(f"[INFO] Successfully created {len(documents)} MedQuAD documents.")
    return documents


if __name__ == "__main__":
    docs = load_medquad_documents(limit=5)
    print("Sample document:")
    print("Content:\n", docs[0].page_content)
    print("Metadata:\n", docs[0].metadata)
