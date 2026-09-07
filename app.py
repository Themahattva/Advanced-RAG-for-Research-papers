from src.search import RAGSearch

if __name__ == "__main__":
    rag_search = RAGSearch(persist_dir="faiss_store_medquad")

    # Example 1: Consumer health question in Hinglish
    query_hinglish = "Blood pressure high hone par kya lakshan dikhte hain aur gharelu dekhbhal ke liye kya karein?"
    print("=" * 70)
    print("🩺 Consumer Health Query (Hinglish):", query_hinglish)
    print("=" * 70)
    
    advice = rag_search.search_and_summarize(query_hinglish, top_k=3, language_mode="auto")
    print("\n💡 Safe Advisory Guidance (MedQuAD / NIH):\n")
    print(advice)