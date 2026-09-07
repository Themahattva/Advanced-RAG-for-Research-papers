import os
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

HINGLISH_KEYWORDS = {
    "kya", "kaise", "kyun", "hai", "hain", "batao", "samjhao", "karo", "hoga",
    "yeh", "ye", "woh", "wo", "me", "mein", "aur", "ka", "ki", "ke", "ko",
    "hota", "hoti", "hote", "bhai", "kuch", "iske", "uske", "chahiye", "raha",
    "rahi", "rahe", "karte", "karta", "karti", "wale", "wala", "wali", "iska",
    "unki", "unke", "kab", "kisko", "kahan", "kitna", "kitne", "pe", "par",
    "se", "saath", "liye", "bhi", "tha", "thi", "the", "dard", "bukhar",
    "gale", "pet", "sar", "sirdard", "khansi", "dawa", "dawai", "upchaar",
    "ilaj", "bimari", "lakshan"
}

def is_hinglish_query(query: str) -> bool:
    """Detect if the query contains common Hinglish words."""
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
    matches = tokens.intersection(HINGLISH_KEYWORDS)
    return len(matches) > 0

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store_medquad",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Load or build vectorstore from MedQuAD
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            print(f"[INFO] MedQuAD index not found at '{persist_dir}'. Building from MedQuAD dataset...")
            from src.medquad_loader import load_medquad_documents
            docs = load_medquad_documents(limit=3000)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not llm_model:
            llm_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.llm_model = llm_model
        self.groq_api_key = groq_api_key
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model, max_tokens=750)
        self.reformulate_llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model, max_tokens=50, temperature=0.0)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def reformulate_query(self, query: str) -> str:
        """
        Translates/reformulates Hinglish and conversational health queries into
        concise medical keywords for accurate FAISS vector retrieval against MedQuAD.
        """
        if not is_hinglish_query(query):
            return query

        reformulate_prompt = (
            "Translate and reformulate the following health/medical query into concise English medical keywords "
            "optimized for searching trusted consumer health databases (MedlinePlus, CDC, NIH). "
            "Output ONLY the English search keywords without explanation or punctuation.\n\n"
            f"Query: {query}\n"
            "English Search Keywords:"
        )
        try:
            res = self.reformulate_llm.invoke([reformulate_prompt])
            english_query = res.content.strip().strip('"').strip("'")
            print(f"[INFO] Reformulated health query '{query}' -> '{english_query}' for retrieval")
            return english_query if english_query else query
        except Exception as e:
            print(f"[WARNING] Query reformulation failed: {e}. Falling back to original query.")
            return query

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve relevant health QA chunks from MedQuAD vector store.
        Returns (results, search_query_used).
        """
        search_query = self.reformulate_query(query)
        results = self.vectorstore.query(search_query, top_k=top_k)
        return results, search_query

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5,
        language_mode: str = "auto",
    ) -> str:
        """
        Retrieve MedQuAD guidance and synthesize a safe, non-diagnostic advisory response.
        
        language_mode:
            - 'auto': automatically respond in Hinglish if query is Hinglish, else English.
            - 'hinglish': always respond in warm, advisory Hinglish.
            - 'english': always respond in clear, advisory English.
        """
        results, search_query = self.retrieve(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        context = "\n\n---\n\n".join(texts)

        # Determine target language
        mode_lower = language_mode.lower()
        if mode_lower == "hinglish":
            respond_in_hinglish = True
        elif mode_lower == "english":
            respond_in_hinglish = False
        else:
            respond_in_hinglish = is_hinglish_query(query)

        if not context:
            if respond_in_hinglish:
                return (
                    "Hamare trusted medical sources (MedlinePlus/NIH) me is specific sawal ke baare me "
                    "paryapt jankari nahi mili. Kripya apna sawal thoda rephrase karein ya related medical terms try karein.\n\n"
                    "⚠️ *Yadi aap ya koi anya vyakti gambhir lakshan mehsoos kar rahe hain, toh turant kisi chikitsak ya aapatkaaleen seva se sampark karein.*"
                )
            return (
                "No relevant consumer health documents found for your query in the MedQuAD database. "
                "Please rephrase or try general health terms.\n\n"
                "⚠️ *If you are experiencing severe symptoms or a medical emergency, please contact a doctor or local emergency services immediately.*"
            )

        if respond_in_hinglish:
            prompt = f"""You are a General Health Information Assistant. Give crisp, structured, non-diagnostic health information in natural Hinglish (Roman script Hindi mixed with English medical terms).

Context from Health Sources:
{context}

User's Question:
{query}

Answer in EXACTLY this structure in natural Hinglish, crisp and to-the-point (no fluff, no repetition):

**What it is:** 1-2 line plain definition.

**Common Symptoms:** 3-5 bullet points, most common first.

**Self-Care / Precautions:** 2-4 bullet points — general measures (rest, hydration, hygiene, diet) and OTC medicine *categories* only (e.g. "antihistamines," "antacids," "paracetamol for fever/pain") — never name a specific brand, exact drug, or dosage.

**When to see a doctor:** 1-2 lines. If symptoms are severe, persistent (>2-3 days), or match any red-flag/emergency sign in the context, say clearly: "Consult a [relevant specialist, e.g. General Physician/Cardiologist/Dermatologist] promptly."

Rules:
- Ground everything strictly in the provided context — no outside knowledge, no guessing.
- Never name a specific drug + dosage.
- If context has no relevant info, say so — don't fabricate.
- Keep total response under ~120 words.
"""
        else:
            prompt = f"""You are a General Health Information Assistant. Give crisp, structured, non-diagnostic health information.

Context from Health Sources:
{context}

User's Question:
{query}

Answer in EXACTLY this structure, crisp and to-the-point (no fluff, no repetition):

**What it is:** 1-2 line plain definition.

**Common Symptoms:** 3-5 bullet points, most common first.

**Self-Care / Precautions:** 2-4 bullet points — general measures (rest, hydration, hygiene, diet) and OTC medicine *categories* only (e.g. "antihistamines," "antacids," "paracetamol for fever/pain") — never name a specific brand, exact drug, or dosage.

**When to see a doctor:** 1-2 lines. If symptoms are severe, persistent (>2-3 days), or match any red-flag/emergency sign in the context, say clearly: "Consult a [relevant specialist, e.g. General Physician/Cardiologist/Dermatologist] promptly."

Rules:
- Ground everything strictly in the provided context — no outside knowledge, no guessing.
- Never name a specific drug + dosage.
- If context has no relevant info, say so — don't fabricate.
- Keep total response under ~120 words.
"""

        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    
    # Test English query
    print("\n--- Testing English Health Query ---")
    query_en = "What are the common symptoms of asthma and how is it managed?"
    print("Q:", query_en)
    print("A:\n", rag_search.search_and_summarize(query_en, top_k=3))

    # Test Hinglish query
    print("\n--- Testing Hinglish Health Query ---")
    query_hi = "Blood pressure high hone ke kya lakshan hote hain aur ghar par kya savdhaniyan bartein?"
    print("Q:", query_hi)
    print("A:\n", rag_search.search_and_summarize(query_hi, top_k=3))