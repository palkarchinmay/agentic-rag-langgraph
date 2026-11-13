"""
cache_node.py
-------------
cache_check_node : look up query in semantic cache before retrieval
cache_store_node : store fresh LLM response in cache after validation

Fixes:
  - cache_check restores validation + citations from cached metadata
    so downstream display shows correct grounding status on hits
  - cache_store saves validation + citations so they survive cache roundtrip
"""

import logging
from app.graph.state import GraphState
from app.cache.semantic_cache import get_cache

logger = logging.getLogger("financial_rag.cache_node")


def _get_embedding_fn(config: dict):
    try:
        from app.providers.embeddings.factory import get_embeddings
        emb = get_embeddings(
            config.get("embedding_provider", "openai"),
            config.get("embedding_model", "text-embedding-3-small"),
        )
        return lambda text: emb.embed_query(text)
    except Exception as e:
        logger.warning("Cache embedding init failed: %s", e)
        return None


def cache_check_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

    config = state.get("runtime_config", {})
    if config.get("skip_cache"):
        logger.info("Cache SKIPPED: skip_cache=True (e.g. golden dataset evaluation)")
        return {**state, "metadata": {**state.get("metadata", {}), "cache_hit": False}}

    query_type = state.get("metadata", {}).get("query_type", "rag")
    if query_type != "rag":
        return {**state, "metadata": {**state.get("metadata", {}), "cache_hit": False}}

    # ── Conversation-awareness guard (Step 4) ─────────────────────────────────
    # The semantic cache has no concept of conversation_id or prior turns —
    # it matches purely on query-text similarity within a collection. A
    # follow-up question ("what was that number?") could wrongly match a
    # DIFFERENT conversation's cached answer to a similarly-worded question.
    # So: skip the cache entirely once a conversation has any history,
    # since only history-dependent answers are at risk of this collision.
    # Fresh, first-turn questions are still cached/served normally.
    chat_history = state.get("chat_history", [])
    if chat_history:
        logger.info(
            "Cache SKIPPED: conversation has %d prior messages — "
            "bypassing cache to avoid cross-conversation answer collision",
            len(chat_history)
        )
        return {
            **state,
            "metadata": {
                **state.get("metadata", {}),
                "cache_hit":          False,
                "conversation_aware": True,
            },
        }

    query      = state.get("query", "")
    config     = state.get("runtime_config", {})
    metadata   = state.get("metadata", {})
    collection = config.get("collection_name", "default")
    model      = config.get("llm_model", "")

    emb_fn = _get_embedding_fn(config)
    cache  = get_cache(collection, embedding_fn=emb_fn)
    cached = cache.get(query, model=model)

    if cached:
        cached_meta = cached.get("metadata", {})
        logger.info("Cache HIT (sim=%.3f): '%s'", cached.get("cache_similarity", 0), query[:60])

        return {
            **state,
            "response":      cached["response"],
            "reranked_docs": cached_meta.get("reranked_docs", []),
            "metadata": {
                **metadata,
                "cache_hit":          True,
                "conversation_aware": False,
                "cache_similarity":   cached.get("cache_similarity", 0),
                # ── Restore full metadata from cached entry ────────────────
                "usage":      cached_meta.get("usage", {}),
                "citations":  cached_meta.get("citations", []),
                "validation": cached_meta.get("validation",
                              {"valid": True, "reason": "served from cache"}),
            },
        }

    return {**state, "metadata": {**metadata, "cache_hit": False, "conversation_aware": False}}


def cache_store_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

    metadata   = state.get("metadata", {})
    query_type = metadata.get("query_type", "rag")
    cache_hit  = metadata.get("cache_hit", False)
    valid      = metadata.get("validation", {}).get("valid", False)

    if query_type != "rag" or cache_hit or not valid:
        return state

    query      = state.get("query", "")
    response   = state.get("response", "")
    config     = state.get("runtime_config", {})

    if not query or not response:
        return state

    collection = config.get("collection_name", "default")
    model      = config.get("llm_model", "")

    emb_fn = _get_embedding_fn(config)
    cache  = get_cache(collection, embedding_fn=emb_fn)

    # Store everything needed to fully reconstruct the response on cache hit
    cache.put(
        query=query,
        response=response,
        model=model,
        metadata={
            "usage":      metadata.get("usage", {}),
            "citations":  metadata.get("citations", []),
            "validation": metadata.get("validation", {}),
            "reranked_docs": [
                {"content": d.get("content","")[:200], "metadata": d.get("metadata",{})}
                for d in state.get("reranked_docs", [])
            ],
        },
    )
    return state