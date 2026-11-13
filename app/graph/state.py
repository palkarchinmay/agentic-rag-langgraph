"""
state.py — LangGraph GraphState

runtime_config schema (all serializable — no Python objects):
{
    # LLM
    "llm_provider":    "openai" | "azure" | "bedrock" | "vertex" | "google"
    "llm_model":       model identifier string
    "temperature":     float (0.0–1.0)

    # Retrieval
    "collection_name": string — key in retriever_service registry
    "top_k":           int — final results after reranking
    "retrieval_mode":  "hybrid" | "vector" | "bm25"
    "rerank":          bool

    # Domain / Persona (new — makes system domain-agnostic)
    "domain":          e.g. "financial documents", "legal contracts",
                            "research papers", "technical manuals",
                            "technical documentation", "research papers"
    "assistant_name":  e.g. "FinanceBot", "LegalAssist", "Document Assistant"

    # Embedding (used at ingestion time, stored for reference)
    "embedding_provider": string
    "embedding_model":    string
}
"""

from typing import TypedDict, List, Dict, Any


class GraphState(TypedDict, total=False):

    # ── Input ─────────────────────────────────────────────────────────────────
    query:          str
    chat_history:   List[Dict]      # [{role, content}, ...]
    runtime_config: Dict[str, Any]  # all serializable — see schema above

    # ── Security ──────────────────────────────────────────────────────────────
    blocked:        bool
    block_reason:   str

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_docs: List[Dict]
    reranked_docs:  List[Dict]

    # ── Generation ────────────────────────────────────────────────────────────
    final_prompt:   str
    raw_response:   str
    response:       str

    # ── Metadata ──────────────────────────────────────────────────────────────
    metadata:       Dict[str, Any]

    # ── Error ─────────────────────────────────────────────────────────────────
    error:          str