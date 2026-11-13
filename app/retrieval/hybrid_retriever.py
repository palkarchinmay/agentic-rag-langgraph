"""
hybrid_retriever.py — Hybrid retriever using Reciprocal Rank Fusion (RRF).

CHANGE from original: replaced direct ChromaVectorStore instantiation
with get_vector_store() factory. Everything else is identical.

  VECTOR_BACKEND=chroma → uses ChromaVectorStore (local dev)
  VECTOR_BACKEND=qdrant → uses QdrantVectorStore (production)

WHY RRF INSTEAD OF WEIGHTED SCORE
==================================
BM25 scores (0→~20) and vector cosine distances (0→1) are incomparable.
Weighted blending is meaningless across different scales.
RRF uses RANKS not raw scores — immune to scale differences.
RRF formula: score(d) = Σ 1 / (k + rank(d))  where k=60 (empirically best).

RETRIEVAL MODES
===============
  'hybrid' : vector + BM25 merged via RRF (default, best overall)
  'vector' : pure dense semantic search
  'bm25'   : pure sparse keyword search
"""

import logging
import os
from typing import Dict, List, Literal, Optional

from app.retrieval.bm25     import BM25Retriever
from app.retrieval.reranker import Reranker

logger = logging.getLogger("financial_rag.hybrid_retriever")

RetrievalMode = Literal["hybrid", "vector", "bm25"]
_RRF_K = 60


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


def reciprocal_rank_fusion(
    vector_results: List[Dict],
    bm25_results:   List[Dict],
) -> List[Dict]:
    scores: Dict[str, float] = {}
    docs:   Dict[str, Dict]  = {}

    for rank, result in enumerate(vector_results, start=1):
        key = result["content"][:200]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        docs[key]   = result

    for rank, result in enumerate(bm25_results, start=1):
        key = result["content"][:200]
        scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
        if key not in docs:
            docs[key] = result

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**docs[key], "score": score, "source": "hybrid"}
        for key, score in merged
    ]


class HybridRetriever:
    """
    Hybrid retriever: vector + BM25 merged via RRF, with cross-encoder rerank.

    Args:
        collection_name    : per-user/session isolation
        embedding_provider : passed to vector store factory
        embedding_model    : passed to vector store factory
        persist_dir        : used only for BM25 pickle persistence
    """

    def __init__(
        self,
        collection_name: str = "default",
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        from app.retrieval.vector_store_factory import get_vector_store
        from app.config.settings import (
            DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL, CHROMA_PERSIST
        )

        self.collection_name = collection_name
        self._persist_dir    = persist_dir or CHROMA_PERSIST

        # ── Vector store (ChromaDB or Qdrant via factory) ─────────────────────
        self.vector_store = get_vector_store(
            collection_name=collection_name,
            embedding_provider=embedding_provider or DEFAULT_EMBED_PROVIDER,
            embedding_model=embedding_model or DEFAULT_EMBED_MODEL,
        )

        # ── BM25 (always local — sparse index, persisted as pickle) ───────────
        self.bm25 = BM25Retriever(
            collection_name=collection_name,
            persist_dir=self._persist_dir,
        )

        # ── Cross-encoder reranker ─────────────────────────────────────────────
        self.reranker = Reranker()

    # ── Build from scratch ────────────────────────────────────────────────────

    def build_index(self, chunks: List[Dict]) -> int:
        """Index all chunks into vector store + BM25. Returns count indexed."""
        n_vec = self.vector_store.build(chunks)
        n_bm  = self.bm25.build(chunks)
        logger.info(
            "HybridRetriever built: vector=%d bm25=%d collection='%s' backend=%s",
            n_vec, n_bm, self.collection_name,
            os.getenv("VECTOR_BACKEND", "chroma")
        )
        return n_vec

    # Backward-compatible alias — some callers may use the shorter name
    build = build_index

    def upsert(self, chunks: List[Dict]) -> int:
        """Add chunks without wiping. Returns count added."""
        n_vec = self.vector_store.upsert(chunks)
        self.bm25.upsert(chunks)
        return n_vec

    # Backward-compatible alias — index.py router calls add_documents()
    add_documents = upsert

    def load_existing(self) -> None:
        """
        Load from existing backend without rebuilding.
        Called by retriever_service.py fallback loader.
        """
        # Vector store auto-loads in its __init__ if data exists
        # BM25 auto-loads its pickle in its __init__
        logger.info(
            "HybridRetriever loaded existing collection '%s' (backend=%s)",
            self.collection_name, os.getenv("VECTOR_BACKEND", "chroma")
        )

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 5,
        mode: RetrievalMode = "hybrid",
        rerank: bool = True,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks.

        Args:
            query           : natural language query
            k               : number of final results
            mode            : 'hybrid' | 'vector' | 'bm25'
            rerank          : apply cross-encoder reranking
            filter_metadata : optional metadata filter (passed to vector store)

        Returns: list of {content, metadata, score, source}
        """
        fetch_k = k * 3   # over-fetch for RRF / rerank

        if mode == "vector":
            candidates = self.vector_store.search(
                query, k=fetch_k, filter_metadata=filter_metadata
            )

        elif mode == "bm25":
            candidates = self.bm25.retrieve(query, k=fetch_k)

        else:  # hybrid
            vector_results = self.vector_store.search(
                query, k=fetch_k, filter_metadata=filter_metadata
            )
            bm25_results   = self.bm25.retrieve(query, k=fetch_k)
            candidates     = reciprocal_rank_fusion(vector_results, bm25_results)

        if rerank and len(candidates) > 1:
            candidates = self.reranker.rerank(query, candidates, top_k=k)
        else:
            candidates = candidates[:k]

        logger.debug(
            "Retrieved %d docs (mode=%s rerank=%s collection=%s backend=%s)",
            len(candidates), mode, rerank, self.collection_name,
            os.getenv("VECTOR_BACKEND", "chroma")
        )
        return candidates

    # ── Utilities ─────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self.vector_store.is_ready

    def count(self) -> int:
        return self.vector_store.count()