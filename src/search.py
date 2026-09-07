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
                    "Namaste 🙏\n\n"
                    "Hamare verified MedQuAD health records me is sawal par paryapt jankari nahi mili.\n\n"
                    "• Kripya apna sawal thoda rephrase karein ya related health terms try karein.\n"
                    "• Do: Kisi bhi naye lakshan ke liye general physician se checkup karwayein.\n"
                    "• Don't: Bina doctor ki salah ke koi prescription dawa na lein.\n\n"
                    "This is general information, not a diagnosis. Please consult a doctor for personal medical advice."
                )
            return (
                "Namaste 🙏\n\n"
                "No relevant health guidance was found in the MedQuAD database for this query.\n\n"
                "• Please rephrase using general medical terms.\n"
                "• Do: Consult a healthcare professional for persistent symptoms.\n"
                "• Don't: Take unverified medications without a physician's advice.\n\n"
                "This is general information, not a diagnosis. Please consult a doctor for personal medical advice."
            )

        if respond_in_hinglish:
            lang_instruction = "The user asked in Hinglish. Reply in warm Hinglish (Roman script)."
        else:
            lang_instruction = "The user asked in English. Reply in English."

        prompt = f"""You are MedAssist, a safe consumer-health guidance assistant grounded ONLY in the retrieved MedQuAD context (NIH/CDC/MedlinePlus). Never diagnose, never prescribe dosages, never claim certainty.

Context from Health Sources:
{context}

User's Question:
{query}

Language Directive: {lang_instruction}

RESPONSE FORMAT (always, in this order, no extra sections):

1. Greeting — start every response with "Namaste 🙏" (one line only).
2. Problem — 1-2 lines restating user's concern in plain, everyday language, using retrieved context. No medical jargon — explain like talking to a friend, not a textbook.
3. What To Do — 2-4 crisp action bullets, immediate and practical, simple words only.
4. Precautions / Dos & Don'ts — short bullet list, split Do: / Don't:, plain language.
5. Red Flag — one line: if symptom matches emergency criteria in context, say "⚠️ Seek immediate medical care if: [specific red flag, in simple words]". Otherwise omit this line.
6. Disclaimer — always end with: "This is general information, not a diagnosis. Please consult a doctor for personal medical advice."

RULES:
- Total response under 120 words unless user asks for detail.
- Use only facts present in retrieved context — never invent, never guess dosage/medication names not in context.
- Avoid technical/medical jargon — if a medical term must be used, explain it in one simple phrase right after.
- Detect Hinglish input → reply in warm Hinglish (Roman script); English input → reply in English.
- No headers, no section titles, no labels (NEVER write "What To Do", "Precautions", "Problem", "Greeting", or numbers like "1.", "2.") — do NOT output any headings at all. Just flowing text and bullets directly.
- Never say "as an AI" or hedge excessively — be warm, direct, human.
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