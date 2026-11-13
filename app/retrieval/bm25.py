"""
bm25.py
-------
BM25 sparse retriever with:
  - Content quality filter (same filter as vector store)
  - Disk persistence (pickle) so index survives restarts
  - Per-collection isolation
"""

import os
import pickle
import logging
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi

from app.retrieval.vector_store import is_quality_chunk

logger = logging.getLogger("financial_rag.bm25")


def _tokenize(text: str) -> List[str]:
    """
    Simple whitespace + lowercase tokenization.
    Strips punctuation to improve recall on financial terms.
    """
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    return text.split()


class BM25Retriever:
    """
    BM25Okapi retriever with persistence and quality filtering.

    Args:
        collection_name : used as filename for persistence
        persist_dir     : directory to save/load the index
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_dir: str = "chroma_db",
    ):
        self.collection_name = collection_name
        self.persist_dir     = persist_dir
        self.documents:      List[Dict] = []
        self.tokenized_docs: List[List[str]] = []
        self.bm25:           Optional[BM25Okapi] = None

        # Auto-load if saved index exists
        self._load()

    # ── Persistence path ──────────────────────────────────────────────────────

    @property
    def _index_path(self) -> str:
        safe = self.collection_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.persist_dir, f"bm25_{safe}.pkl")

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: List[Dict]) -> int:
        """
        Build BM25 index from chunks. Applies quality filter.
        Returns number of indexed chunks.
        """
        good = [c for c in chunks if is_quality_chunk(c["content"])]
        dropped = len(chunks) - len(good)
        if dropped:
            logger.warning("BM25 quality filter dropped %d/%d chunks", dropped, len(chunks))

        if not good:
            logger.warning("No quality chunks for BM25 index")
            return 0

        self.documents      = good
        self.tokenized_docs = [_tokenize(c["content"]) for c in good]
        self.bm25           = BM25Okapi(self.tokenized_docs)

        self._save()
        logger.info("BM25 index built: %d chunks in '%s'",
                    len(good), self.collection_name)
        return len(good)

    # ── Upsert (add to existing) ──────────────────────────────────────────────

    def upsert(self, chunks: List[Dict]) -> int:
        """Add new chunks to an existing index without full rebuild."""
        good = [c for c in chunks if is_quality_chunk(c["content"])]
        if not good:
            return 0

        self.documents.extend(good)
        self.tokenized_docs.extend([_tokenize(c["content"]) for c in good])
        self.bm25 = BM25Okapi(self.tokenized_docs)

        self._save()
        logger.info("BM25 upserted %d chunks, total=%d", len(good), len(self.documents))
        return len(good)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        BM25 keyword search.

        Returns: list of {content, metadata, score, source}
                 score = BM25 relevance score (higher = more relevant)
        """
        if self.bm25 is None or not self.documents:
            raise ValueError("BM25 index not built. Call build() or upsert() first.")

        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            {
                "content":  self.documents[i]["content"],
                "metadata": self.documents[i]["metadata"],
                "score":    float(scores[i]),
                "source":   "bm25",
            }
            for i in ranked
            if scores[i] > 0   # skip zero-score results
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        try:
            with open(self._index_path, "wb") as f:
                pickle.dump({
                    "documents":      self.documents,
                    "tokenized_docs": self.tokenized_docs,
                }, f)
            logger.debug("BM25 index saved to %s", self._index_path)
        except Exception as e:
            logger.warning("BM25 save failed: %s", e)

    def load(self):
        """Public method to explicitly load index from disk (used by disk fallback)."""
        self._load()

    def _load(self):
        if not os.path.exists(self._index_path):
            return
        try:
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            self.documents      = data["documents"]
            self.tokenized_docs = data["tokenized_docs"]
            self.bm25           = BM25Okapi(self.tokenized_docs)
            logger.info("BM25 index loaded: %d chunks in '%s'",
                        len(self.documents), self.collection_name)
        except Exception as e:
            logger.warning("BM25 load failed: %s", e)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self.documents)

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None and len(self.documents) > 0

    def delete(self):
        self.documents      = []
        self.tokenized_docs = []
        self.bm25           = None
        if os.path.exists(self._index_path):
            os.remove(self._index_path)
        logger.info("BM25 index deleted for '%s'", self.collection_name)