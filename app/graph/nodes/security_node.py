"""security_node.py — Prompt injection + PII check on incoming query."""

import logging
from app.graph.state import GraphState
from app.security.prompt_guard import detect_prompt_injection
from app.security.pii import detect_pii

logger = logging.getLogger("financial_rag.security_node")


def security_node(state: GraphState) -> GraphState:
    query    = state.get("query", "").strip()
    metadata = state.get("metadata", {})

    # ── Injection check ───────────────────────────────────────────────────────
    blocked, reason = detect_prompt_injection(query)
    if blocked:
        logger.warning("Query BLOCKED — injection: %s | query='%s'", reason, query[:80])
        return {
            **state,
            "blocked":    True,
            "block_reason": reason,
            "response":   f"⚠️ Query blocked: {reason}",
            "metadata":   {**metadata, "security": {"blocked": True, "reason": reason}},
        }

    # ── PII check on query itself ─────────────────────────────────────────────
    pii_found = detect_pii(query)
    if pii_found:
        logger.info("PII detected in query: %s", list(pii_found.keys()))
        # Don't block — log and strip PII from metadata, proceed with query
        metadata = {
            **metadata,
            "security": {
                "blocked":    False,
                "pii_in_query": list(pii_found.keys()),
            }
        }
    else:
        metadata = {**metadata, "security": {"blocked": False}}

    logger.debug("Query passed security check: '%s'", query[:80])
    return {
        **state,
        "blocked":  False,
        "metadata": metadata,
    }
