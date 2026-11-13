"""
retrieval_node.py
-----------------
Retrieves relevant chunks using the registered HybridRetriever.

The retriever is fetched from the module-level registry by collection_name —
NOT stored in GraphState — because GraphState is msgpack-serialized by
LangGraph's MemorySaver and Python objects are not serializable.
"""

import logging
from app.graph.state import GraphState
from app.services.retriever_service import get_retriever

logger = logging.getLogger("financial_rag.retrieval_node")


def retrieval_node(state: GraphState) -> GraphState:

    if state.get("blocked"):
        return state

    query    = state.get("query", "")
    config   = state.get("runtime_config", {})
    metadata = state.get("metadata", {})

    k               = config.get("top_k", 5)
    mode            = config.get("retrieval_mode", "hybrid")
    collection_name = config.get("collection_name", "default")

    # Look up the live retriever object from the registry
    retriever = get_retriever(collection_name)
    if retriever is None:
        logger.error("No retriever for collection '%s'", collection_name)
        return {
            **state,
            "error":         f"Retriever not found for collection '{collection_name}'.",
            "retrieved_docs": [],
        }

    try:
        results = retriever.retrieve(
            query,
            k=k * 3,      # fetch 3x for reranker to trim
            mode=mode,
            rerank=False,  # reranking done in separate node
        )
        logger.info("Retrieved %d docs (collection=%s, mode=%s)",
                    len(results), collection_name, mode)
        return {
            **state,
            "retrieved_docs": results,
            "metadata": {
                **metadata,
                "retrieval": {
                    "collection": collection_name,
                    "mode":       mode,
                    "fetched":    len(results),
                    "k":          k,
                },
            },
        }

    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        return {**state, "error": str(e), "retrieved_docs": []}
