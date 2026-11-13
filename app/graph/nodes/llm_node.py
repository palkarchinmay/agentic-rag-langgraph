"""
llm_node.py — RAG LLM node with graceful fallback to general knowledge.

When docs are uploaded but answer not found in context:
  1. LLM says "not found in document"
  2. Pipeline detects this
  3. Automatically retries via general knowledge
  4. Response clearly labels source: "ℹ️ From general knowledge (not in your document)"
"""

import time
import logging
from app.graph.state import GraphState
from app.providers.llm.factory import get_llm
from app.monitoring.cost_tracker import estimate_cost, format_cost
from app.security.llm_guard import scan_response, safe_fallback_response
from app.services.llm_resilience import (
    guarded_invoke, RateLimitError, CircuitOpenError
)

logger = logging.getLogger("rag_platform.llm_node")

_NO_ANSWER_PHRASES = [
    "i could not find", "not in the documents", "not mentioned",
    "no information", "cannot find", "does not appear",
    "not available in", "not provided in",
    "information is unavailable", "is unavailable", "not available",
    "could not find", "unable to find", "not found in",
    "does not contain", "not present in", "no relevant",
]

_GENERAL_SYSTEM = (
    "You are a helpful, knowledgeable assistant. "
    "Answer the question clearly and concisely using your knowledge. "
    "Do not reference any specific documents. "
    "Do not make up facts. If unsure, say so honestly."
)


def llm_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

    final_prompt = state.get("final_prompt", "")
    config       = state.get("runtime_config", {})
    metadata     = state.get("metadata", {})
    query        = state.get("query", "")

    if not final_prompt:
        return {**state, "error": "No prompt generated",
                "response": "An error occurred — no prompt was built."}

    provider    = config.get("llm_provider",  "openai")
    model       = config.get("llm_model",     "gpt-4o-mini")
    temperature = config.get("temperature",   0.0)
    has_docs    = bool(
        config.get("collection_name") and
        config.get("collection_name") not in ("default", "", None)
    )

    try:
        llm      = get_llm(provider=provider, model=model, temperature=temperature)
        user_id  = config.get("user_id", config.get("user_role", "default"))
        t0       = time.time()

        # Guarded invoke: retries + rate limit + circuit breaker
        response = guarded_invoke(llm, final_prompt, user_id=user_id)
        latency  = round(time.time() - t0, 2)
        logger.info("LLM response in %.1fs", latency)
        content  = response.content

        is_safe, scan_reason = scan_response(content)
        if not is_safe:
            content = safe_fallback_response()

        # ── Graceful fallback when answer not in docs ─────────────────────────
        response_lower = content.lower()
        not_found = any(phrase in response_lower for phrase in _NO_ANSWER_PHRASES)

        # Before falling back, check if retrieved docs have relevant context.
        # If they do, try a softer prompt that answers with related info.
        if not_found and has_docs:
            reranked_docs = state.get("reranked_docs", [])
            if reranked_docs:
                # Re-ask with a softer instruction — answer with related info
                # Use all retrieved chunks (not just top 3) for soft prompt
                # Increase context window to 800 chars per chunk for better coverage
                soft_ctx = "\n\n---\n\n".join(
                    d.get("content", "")[:800] for d in reranked_docs
                )
                soft_prompt = (
                    f"Extract the answer to the question from the context below.\n\n"
                    f"EXTRACTION RULES:\n"
                    f"1. Look for the answer in any form: direct statements, "
                    f"parenthetical values, statistical notation (N=x, n=x, %=x), "
                    f"table entries, or values mentioned in passing.\n"
                    f"2. The document may use different terminology — match by meaning, "
                    f"not exact words (e.g. 'total records' = 'number of entries').\n"
                    f"3. If you find a value or fact that answers the question even "
                    f"indirectly, state it clearly and explain where it came from.\n"
                    f"4. Only say the information is unavailable if the context "
                    f"contains absolutely nothing related to the question.\n\n"
                    f"Context:\n{soft_ctx}\n\n"
                    f"Question: {query}\nAnswer:"
                )
                soft_resp  = guarded_invoke(llm, soft_prompt, user_id=user_id)
                soft_text  = soft_resp.content.strip()
                soft_lower = soft_text.lower()
                still_not_found = any(
                    phrase in soft_lower for phrase in _NO_ANSWER_PHRASES
                )
                if not still_not_found and len(soft_text) > 20:
                    # Soft prompt found related answer — use it
                    content   = soft_text
                    not_found = False
                    metadata  = {**metadata, "answered_from": "rag_related"}
                    logger.info("Soft prompt found related answer from docs")

        if not_found and has_docs:
            logger.info("RAG: answer not in docs, falling back to general (%.1fs so far)", latency)

            # ── Step 1: Try query expansion before giving up on docs ──────────
            # Rephrase the query and retry retrieval once — catches cases where
            # wording differs between query and document (semantic mismatch)
            expanded_content = None
            try:
                expand_prompt = (
                    f"Rephrase this question using 2-3 alternative wordings "
                    f"a document might use to express the same concept. "
                    f"Use simpler, broader, or synonymous terms. "
                    f"Return only the rephrased questions, one per line, no numbering.\n"
                    f"Question: {query}"
                )
                expanded_resp = guarded_invoke(llm, expand_prompt, user_id=user_id)
                alt_queries   = [q.strip() for q in
                                 expanded_resp.content.strip().split("\n")
                                 if q.strip()][:2]   # try top 2 variants

                if alt_queries:
                    from app.retrieval.hybrid_retriever import HybridRetriever
                    from app.services.retriever_service import get_retriever
                    retriever = get_retriever(
                        state.get("runtime_config", {}).get("collection_name", "default")
                    )
                    if retriever:
                        for alt_q in alt_queries:
                            alt_docs = retriever.retrieve(alt_q, top_k=3)
                            if alt_docs:
                                # Build a quick answer from expanded docs
                                alt_ctx = "\n\n".join(
                                    d.get("content", "")[:500] for d in alt_docs[:3]
                                )
                                alt_prompt = (
                                    f"Answer this question using only the context below.\n"
                                    f"If the context does not contain the answer, say NOT_FOUND.\n\n"
                                    f"Context:\n{alt_ctx}\n\n"
                                    f"Question: {query}\nAnswer:"
                                )
                                alt_ans = guarded_invoke(llm, alt_prompt, user_id=user_id)
                                alt_text = alt_ans.content.strip()
                                if "NOT_FOUND" not in alt_text.upper() and len(alt_text) > 30:
                                    expanded_content = alt_text
                                    logger.info(
                                        "Query expansion found answer via '%s'", alt_q
                                    )
                                    break
            except Exception as exp_err:
                logger.debug("Query expansion skipped: %s", exp_err)

            if expanded_content:
                # Expansion succeeded — use doc answer, mark as RAG
                content  = expanded_content
                metadata = {**metadata, "answered_from": "rag_expanded"}
            else:
                # ── Step 2: Full fallback to general knowledge ────────────────
                general_prompt = f"{_GENERAL_SYSTEM}\n\nQuestion: {query}\n\nAnswer:"
                general_resp   = guarded_invoke(llm, general_prompt, user_id=user_id)
                latency        = round(time.time() - t0, 2)
                logger.info("General fallback done in %.1fs total", latency)
                general_content = general_resp.content

                is_safe2, _ = scan_response(general_content)
                if not is_safe2:
                    general_content = safe_fallback_response()

                content = (
                    f"_This information was not found in your uploaded document. "
                    f"The following answer is from general knowledge:_\n\n"
                    f"{general_content}"
                )
                metadata = {**metadata, "answered_from": "general_fallback"}

        usage_meta        = getattr(response, "response_metadata", {}) or {}
        usage             = usage_meta.get("token_usage") or usage_meta.get("usage") or {}
        prompt_tokens     = usage.get("prompt_tokens",     usage.get("input_tokens",  0))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        total_tokens      = usage.get("total_tokens", prompt_tokens + completion_tokens)
        cost              = estimate_cost(model, prompt_tokens, completion_tokens)

        return {
            **state,
            "raw_response": content,
            "response":     content,
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
                "security_scan": {"passed": is_safe, "reason": scan_reason},
            },
        }

    except RateLimitError as e:
        logger.warning("Rate limited: %s", e)
        return {**state, "error": str(e),
                "response": str(e)}

    except CircuitOpenError as e:
        logger.warning("Circuit open: %s", e)
        return {**state, "error": str(e),
                "response": str(e)}

    except Exception as e:
        logger.exception("LLM node failed: %s", e)
        return {**state, "error": str(e),
                "response": "An error occurred. Please try again."}

