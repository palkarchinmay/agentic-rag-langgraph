"""
workflow.py — Updated LangGraph pipeline with all features.

Full flow:
  security
    ↓ (blocked → END)
  query_router
    ├─ general → direct_llm → ragas → END
    └─ rag:
         cache_check
           ├─ hit  → ragas → END
           └─ miss:
                retrieval → rerank → prompt → llm
                  → validation → citation → cache_store → ragas → END

Features:
  - Domain-agnostic (domain + assistant_name in runtime_config)
  - Semantic caching (skipped on cache miss, stored after fresh response)
  - General task handling (direct_llm, no retrieval)
  - RAGAS evaluation (only when diagnostics=True in runtime_config)
  - IAM-aware (user_role in runtime_config controls general_chat permission)
"""

import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import GraphState
from app.graph.nodes.security_node      import security_node
from app.graph.nodes.query_router_node  import query_router_node
from app.graph.nodes.cache_node         import cache_check_node, cache_store_node
from app.graph.nodes.retrieval_node     import retrieval_node
from app.graph.nodes.rerank_node        import rerank_node
from app.graph.nodes.prompt_node        import prompt_node
from app.graph.nodes.llm_node           import llm_node
from app.graph.nodes.validation_node    import validation_node
from app.graph.nodes.citation_node      import citation_node
from app.graph.nodes.direct_llm_node    import direct_llm_node
from app.graph.nodes.ragas_node         import ragas_node

logger = logging.getLogger("financial_rag.workflow")


# ── Routing functions ─────────────────────────────────────────────────────────

def _route_security(state: GraphState) -> str:
    """Block → END immediately. Safe → router."""
    return "blocked" if state.get("blocked") else "query_router"


def _route_query_type(state: GraphState) -> str:
    """RAG task → cache check. General task → direct LLM."""
    query_type = state.get("metadata", {}).get("query_type", "rag")
    return "direct_llm" if query_type == "general" else "cache_check"


def _route_cache(state: GraphState) -> str:
    """Cache hit → skip to ragas. Cache miss → full retrieval pipeline."""
    return "ragas" if state.get("metadata", {}).get("cache_hit") else "retrieval"


def build_graph():
    builder = StateGraph(GraphState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("security",     security_node)
    builder.add_node("query_router", query_router_node)
    builder.add_node("cache_check",  cache_check_node)
    builder.add_node("retrieval",    retrieval_node)
    builder.add_node("rerank",       rerank_node)
    builder.add_node("prompt",       prompt_node)
    builder.add_node("llm",          llm_node)
    builder.add_node("validation",   validation_node)
    builder.add_node("citation",     citation_node)
    builder.add_node("cache_store",  cache_store_node)
    builder.add_node("direct_llm",   direct_llm_node)
    builder.add_node("ragas",        ragas_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    builder.set_entry_point("security")

    # ── Edges ─────────────────────────────────────────────────────────────────

    # security → blocked:END or query_router
    builder.add_conditional_edges(
        "security", _route_security,
        {"blocked": END, "query_router": "query_router"},
    )

    # query_router → rag:cache_check or general:direct_llm
    builder.add_conditional_edges(
        "query_router", _route_query_type,
        {"cache_check": "cache_check", "direct_llm": "direct_llm"},
    )

    # cache_check → hit:ragas or miss:retrieval
    builder.add_conditional_edges(
        "cache_check", _route_cache,
        {"ragas": "ragas", "retrieval": "retrieval"},
    )

    # RAG pipeline (linear)
    builder.add_edge("retrieval",  "rerank")
    builder.add_edge("rerank",     "prompt")
    builder.add_edge("prompt",     "llm")
    builder.add_edge("llm",        "validation")
    builder.add_edge("validation", "citation")
    builder.add_edge("citation",   "cache_store")
    builder.add_edge("cache_store","ragas")

    # General task
    builder.add_edge("direct_llm", "ragas")

    # End
    builder.add_edge("ragas", END)

    memory = MemorySaver()
    graph  = builder.compile(checkpointer=memory)
    logger.info("LangGraph pipeline compiled (full feature set)")
    return graph


graph = build_graph()
