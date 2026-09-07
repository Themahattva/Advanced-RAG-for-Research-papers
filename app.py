from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    rag_search = RAGSearch()

    # Query in Hinglish
    query_hinglish = "Attention mechanism kya hota hai aur research papers ke according yeh kaise kaam karta hai?"
    print("=" * 70)
    print("🔍 Hinglish Query:", query_hinglish)
    print("=" * 70)
    
    summary = rag_search.search_and_summarize(query_hinglish, top_k=3, language_mode="auto")
    print("\n💡 Explaining Answer (Hinglish):\n")
    print(summary)