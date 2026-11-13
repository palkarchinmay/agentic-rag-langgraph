"""
direct_llm_node.py
------------------
Handles GENERAL_TASK queries — direct LLM call, no document retrieval.

Explicitly clears retrieved_docs and reranked_docs to prevent
LangGraph MemorySaver from carrying over docs from a previous
checkpoint into this response.
"""

import time
import logging
from app.graph.state import GraphState
from app.providers.llm.factory import get_llm
from app.monitoring.cost_tracker import estimate_cost, format_cost
from app.security.llm_guard import scan_response, safe_fallback_response

logger = logging.getLogger("rag_platform.direct_llm")

_GENERAL_SYSTEM = (
    "You are a helpful, knowledgeable assistant. "
    "Answer the user's question clearly and concisely. "
    "If conversation history is provided, use it to answer questions about "
    "previous exchanges (e.g. 'what did I ask before?', 'summarize our conversation'). "
    "For factual questions not covered by conversation history, use your general knowledge. "
    "Do not make up facts. If unsure, say so."
)


def _format_history(history: list) -> str:
    if not history:
        return ""
    lines = []
    for msg in history:  # use ALL history, not just last 6
        role    = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def direct_llm_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

    config   = state.get("runtime_config", {})
    query    = state.get("query", "")
    history  = state.get("chat_history", [])
    metadata = state.get("metadata", {})

    provider    = config.get("llm_provider", "openai")
    model       = config.get("llm_model",    "gpt-4o-mini")
    temperature = config.get("temperature",  0.3)

    history_text = _format_history(history)

    if history_text:
        history_sec = (
            f"\n\nCONVERSATION HISTORY (use this to answer questions about "
            f"previous exchanges):\n{history_text}\n"
        )
    else:
        history_sec = ""

    prompt = f"{_GENERAL_SYSTEM}{history_sec}\nUser: {query}\nAssistant:"

    try:
        llm      = get_llm(provider=provider, model=model, temperature=temperature)
        t0       = time.time()
        response = llm.invoke(prompt)
        latency  = round(time.time() - t0, 2)

        content = response.content

        is_safe, scan_reason = scan_response(content)
        if not is_safe:
            logger.warning("Direct LLM scan failed: %s", scan_reason)
            content = safe_fallback_response()

        usage_meta        = getattr(response, "response_metadata", {}) or {}
        usage             = usage_meta.get("token_usage") or usage_meta.get("usage") or {}
        prompt_tokens     = usage.get("prompt_tokens",     usage.get("input_tokens",  0))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        total_tokens      = usage.get("total_tokens", prompt_tokens + completion_tokens)
        cost              = estimate_cost(model, prompt_tokens, completion_tokens)

        logger.info("Direct LLM: tokens=%d cost=%s latency=%ss history_msgs=%d",
                    total_tokens, format_cost(cost), latency, len(history))

        return {
            **state,
            "raw_response":  content,
            "response":      content,
            "retrieved_docs": [],
            "reranked_docs":  [],
            "metadata": {
                **metadata,
                "usage": {
                    "provider":           provider,
                    "model":              model,
                    "prompt_tokens":      prompt_tokens,
                    "completion_tokens":  completion_tokens,
                    "total_tokens":       total_tokens,
                    "estimated_cost_usd": cost,
                    "latency_seconds":    latency,
                },
                "conversation_aware": len(history) > 0,  # True when history was used
                "security_scan": {"passed": is_safe,  "reason": scan_reason},
                "validation":    {"valid":  True,
                                  "reason": "general task — no grounding required"},
                "citations":     [],
            },
        }

    except Exception as e:
        logger.exception("Direct LLM failed: %s", e)
        return {
            **state,
            "error":         str(e),
            "response":      "An error occurred. Please try again.",
            "retrieved_docs": [],
            "reranked_docs":  [],
        }