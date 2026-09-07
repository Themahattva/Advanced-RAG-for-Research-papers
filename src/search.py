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
    "se", "saath", "liye", "bhi", "tha", "thi", "the"
}

def is_hinglish_query(query: str) -> bool:
    """Detect if the query contains common Hinglish words."""
    tokens = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
    matches = tokens.intersection(HINGLISH_KEYWORDS)
    return len(matches) > 0

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: Optional[str] = None,
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
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
        Translates/reformulates Hinglish queries to English keywords for accurate
        semantic vector retrieval against English academic papers.
        """
        if not is_hinglish_query(query):
            return query

        reformulate_prompt = (
            "Translate and reformulate the following question into concise English search keywords "
            "optimized for retrieving relevant context from academic research papers. "
            "Output ONLY the English search keywords without explanation or punctuation.\n\n"
            f"Question: {query}\n"
            "English Search Query:"
        )
        try:
            res = self.reformulate_llm.invoke([reformulate_prompt])
            english_query = res.content.strip().strip('"').strip("'")
            print(f"[INFO] Reformulated Hinglish query '{query}' -> '{english_query}' for retrieval")
            return english_query if english_query else query
        except Exception as e:
            print(f"[WARNING] Query reformulation failed: {e}. Falling back to original query.")
            return query

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve relevant chunks from the FAISS vector store using an English-optimized query.
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
        Retrieve chunks and generate an answer.
        Supports Hinglish with an explaining tone if query is in Hinglish or language_mode is 'hinglish'.
        
        language_mode:
            - 'auto': automatically respond in Hinglish if query is Hinglish, else English.
            - 'hinglish': always explain in Hinglish (Roman script Hindi + English technical terms).
            - 'english': always explain in English.
        """
        results, search_query = self.retrieve(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r.get("metadata")]
        context = "\n\n".join(texts)

        # Determine target language
        mode_lower = language_mode.lower()
        if mode_lower == "hinglish":
            respond_in_hinglish = True
        elif mode_lower == "english":
            respond_in_hinglish = False
        else:
            # Auto-detect from original query
            respond_in_hinglish = is_hinglish_query(query)

        if not context:
            if respond_in_hinglish:
                return (
                    "Maaf kijiye, provided research papers me aapke sawal se related koi relevant "
                    "information nahi mili. Kripya apna sawal thoda rephrase karein ya related keywords try karein."
                )
            return "No relevant documents found in the corpus for your query. Please rephrase or try related keywords."

        if respond_in_hinglish:
            prompt = f"""You are an AI Clinical Research Specialist. 
Your persona is like an expert DOCTOR or NURSE explaining a diagnosis and writing a crisp, to-the-point medical prescription (Rx).
Your response MUST be TO-THE-POINT, CRISP, and free of fluff or lengthy storytelling.

Context from Research Papers:
{context}

User's Question:
{query}

Guidelines:
1. Tone: Like an experienced doctor or triage nurse giving a clear diagnosis and prescribing the exact solution. Direct, authoritative, professional, and crisp.
2. Language: Natural, punchy Hinglish (Roman Hindi mixed with essential technical English terms).
3. Structure (Follow this prescription format):
   🩺 **Diagnosis / Core Concept:** (1-2 line direct, sharp answer)
   🔬 **Clinical Findings / Mechanism:** (2-3 crisp bullet points on how it works and what the paper found)
   💊 **Doctor's Rx / Prescription:** (1-2 bullet points with key takeaways, advantages, or limitations/caution)
4. Grounding: Strictly based on the provided research context. No unnecessary preamble or repetitive filler.

Ab ek doctor ya nurse ke prescription style me crisp aur to-the-point Hinglish answer do:"""
        else:
            prompt = f"""You are an AI Clinical Research Specialist. 
Your persona is like an expert DOCTOR or NURSE explaining a diagnosis and writing a crisp, to-the-point medical prescription (Rx).
Your response MUST be TO-THE-POINT, CRISP, and structured cleanly without fluff.

Context from Research Papers:
{context}

User's Question:
{query}

Structure:
🩺 **Diagnosis / Core Concept:** (1-2 line direct, sharp answer)
🔬 **Clinical Findings / Mechanism:** (2-3 crisp bullet points on how it works and what the paper found)
💊 **Doctor's Rx / Prescription:** (1-2 bullet points with key takeaways, advantages, or limitations/caution)

Answer strictly based on the context in a crisp clinical prescription style:"""

        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    
    # Test English query
    print("\n--- Testing English Query ---")
    query_en = "What is attention mechanism?"
    print("Q:", query_en)
    print("A:\n", rag_search.search_and_summarize(query_en, top_k=3))

    # Test Hinglish query
    print("\n--- Testing Hinglish Query ---")
    query_hi = "Attention mechanism kya hota hai aur transformers me iska kya kaam hai, aasan shabdon me samjhao?"
    print("Q:", query_hi)
    print("A:\n", rag_search.search_and_summarize(query_hi, top_k=3))