"""retriever.py — Simple vector-only retriever (used by HybridRetriever internally)."""
from app.retrieval.vector_store import ChromaVectorStore
from typing import List, Dict, Optional

class SimpleRetriever:
    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5,
                 filter_metadata: Optional[Dict] = None) -> List[Dict]:
        return self.vector_store.search(query, k=k,
                                        filter_metadata=filter_metadata)
