"""rerank_node.py — Cross-encoder reranking with lazy model loading."""

import logging
from app.graph.state import GraphState
from app.retrieval.reranker import Reranker

logger = logging.getLogger("financial_rag.rerank_node")

# Lazy singleton — loaded on first rerank call, not at import time
_reranker = None

def _get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def rerank_node(state: GraphState) -> GraphState:

    if state.get("blocked"):
        return state

    query  = state.get("query", "")
    docs   = state.get("retrieved_docs", [])
    config = state.get("runtime_config", {})
    k      = config.get("top_k", 5)
    do_rerank = config.get("rerank", True)

    if not docs:
        logger.warning("No docs to rerank")
        return {**state, "reranked_docs": []}

    if not do_rerank or len(docs) <= 1:
        return {**state, "reranked_docs": docs[:k]}

    try:
        import time as _time
        t0       = _time.time()
        reranker  = _get_reranker()
        reranked  = reranker.rerank(query, docs, top_k=k)
        logger.info("Reranked %d → %d docs in %.1fs", len(docs), len(reranked),
                    _time.time() - t0)
        return {**state, "reranked_docs": reranked}
    except Exception as e:
        logger.warning("Reranking failed, using original order: %s", e)
        return {**state, "reranked_docs": docs[:k]}