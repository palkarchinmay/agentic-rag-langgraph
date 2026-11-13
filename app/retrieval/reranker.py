"""
reranker.py
-----------
Cross-encoder reranker with lazy model loading.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22MB, runs on CPU
  - Trained on MS MARCO passage ranking
  - Good balance of speed and accuracy for financial QA

The model is loaded ONLY on first rerank() call — not at import time.
This prevents slow startup when the reranker isn't being used.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger("financial_rag.reranker")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model = None   # lazy singleton


def _get_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model '%s'...", RERANKER_MODEL)
            _model = CrossEncoder(RERANKER_MODEL, max_length=512)
            logger.info("Reranker loaded")
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"Reranker load failed: {e}")
    return _model


class Reranker:
    """
    Cross-encoder reranker.

    Usage:
        reranker = Reranker()
        results  = reranker.rerank(query, retrieved_chunks, top_k=5)
    """

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank retrieved chunks using cross-encoder.

        Args:
            query   : user query string
            results : list of {content, metadata, score, ...}
            top_k   : number of results to return after reranking

        Returns:
            Reranked list with added 'rerank_score' field.
            Sorted by rerank_score descending.
        """
        if not results:
            return []

        if len(results) == 1:
            results[0]["rerank_score"] = 1.0
            return results

        model = _get_model()

        pairs = [(query, r["content"][:512]) for r in results]

        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.warning("Reranker predict failed: %s — returning original order", e)
            for r in results:
                r["rerank_score"] = r.get("score", 0.0)
            return results[:top_k]

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        logger.debug("Reranked %d → %d results", len(results), top_k)

        return reranked[:top_k]

    @property
    def is_loaded(self) -> bool:
        return _model is not None
