"""
validation_node.py — Domain-agnostic groundedness check.

Validates that the LLM response is grounded in retrieved document context.
Works for any domain: financial, medical, legal, technical, etc.

GROUNDEDNESS APPROACH (Step 5 improvement):
============================================
Previous: Word overlap (Jaccard similarity) — lexical match only.
Problem:  "N=57,740" and "total number of claims" share zero words
          despite meaning the same thing → false negatives.

Current: Embedding cosine similarity — semantic match.
Benefit: Catches semantic equivalence across different phrasings.
         "57,740 claims" ≈ "total number of claims" → high similarity.
         No extra dependencies — reuses OpenAI embeddings already in stack.

Threshold: 0.60 (industry standard for RAG groundedness checks).
- LlamaIndex default faithfulness threshold: 0.6
- RAGAS context_precision threshold: 0.6
- Lower than semantic cache (0.85) since we check relevance not exact match.

FALLBACK ORDER when groundedness fails:
1. validation_node detects low similarity → sets general_fallback
2. Response clearly labelled "not in your document, from general knowledge"

Numbers and amounts verified using normalized comparison (strips
currency symbols, commas, spaces) so formatting differences don't
cause false negatives.
"""

import re
import logging
from typing import Tuple, List, Optional
from app.graph.state import GraphState

logger = logging.getLogger("rag_platform.validation_node")

_NO_ANSWER_PHRASES = [
    "i could not find", "not in the documents", "not mentioned",
    "no information", "cannot find", "does not appear",
    "not available in", "not provided in",
    "information is unavailable", "is unavailable", "not available",
    "could not find", "unable to find", "not found in",
    "does not contain", "not present in", "no relevant",
]

_HALLUCINATION_MARKERS = [
    "based on my training", "generally speaking", "from my knowledge",
    "i believe", "as far as i know", "it is likely", "in general,",
    "typically,", "usually,",
    "i can only answer questions about",  # security block — treat as no-answer
]

# Numeric values with currency/formatting — not list markers (7., 8.) or bare integers
_NUMERIC_VALUE_RE = re.compile(
    r"(?:[\$₹£€]\s*[\d,]+(?:\.\d+)?)"           # ₹3,60,000
    r"|(?:\b\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\b)"  # 3,60,000 (comma-formatted)
    r"|(?:\b\d+(?:\.\d+)?%\b)"                   # 15%
)

# Semantic similarity threshold for groundedness
# Industry standard: 0.60 (LlamaIndex, RAGAS default)
_SEMANTIC_SIMILARITY_THRESHOLD = 0.60


def _normalize_num(n: str) -> str:
    """
    Strip currency, spaces, commas for comparison.
    e.g. "₹3,60,000" → "360000" and context "3,60,000" → "360000"
    """
    return re.sub(r"[\$₹£€\s,]", "", n).lower()


def _extract_numeric_values(text: str) -> List[str]:
    return _NUMERIC_VALUE_RE.findall(text)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    import math
    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _get_semantic_similarity(
    response: str,
    context: str,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
) -> Optional[float]:
    """
    Compute semantic similarity between response and combined context
    using whatever embedding model is configured in runtime_config.

    Supports: openai, huggingface, azure, google
    Returns float 0-1, or None if embedding fails (graceful degradation).
    Graceful degradation means: if embedding is unavailable, trust the response.
    """
    try:
        import os

        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            result = client.embeddings.create(
                model=model,
                input=[response[:2000], context[:4000]],
            )
            vec_response = result.data[0].embedding
            vec_context  = result.data[1].embedding
            return _cosine_similarity(vec_response, vec_context)

        elif provider == "azure":
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
            result = client.embeddings.create(
                model=model,
                input=[response[:2000], context[:4000]],
            )
            vec_response = result.data[0].embedding
            vec_context  = result.data[1].embedding
            return _cosine_similarity(vec_response, vec_context)

        elif provider == "huggingface":
            # Use sentence-transformers if available
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                st_model = SentenceTransformer(model)
                vecs = st_model.encode(
                    [response[:2000], context[:4000]],
                    normalize_embeddings=True,
                )
                return float(np.dot(vecs[0], vecs[1]))
            except ImportError:
                # sentence-transformers not installed — use HuggingFace API
                import httpx
                hf_token = os.getenv("HUGGINGFACE_API_KEY", "")
                headers  = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
                api_url  = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
                resp = httpx.post(
                    api_url,
                    json={"inputs": [response[:2000], context[:4000]]},
                    headers=headers,
                    timeout=15,
                )
                if resp.status_code == 200:
                    vecs = resp.json()
                    # Average pooling over token dimension if needed
                    import math
                    def mean_pool(v):
                        if isinstance(v[0], list):
                            n = len(v)
                            dim = len(v[0])
                            return [sum(v[i][j] for i in range(n)) / n for j in range(dim)]
                        return v
                    vec_r = mean_pool(vecs[0])
                    vec_c = mean_pool(vecs[1])
                    return _cosine_similarity(vec_r, vec_c)
                return None

        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
            result = genai.embed_content(
                model=f"models/{model}" if not model.startswith("models/") else model,
                content=[response[:2000], context[:4000]],
                task_type="SEMANTIC_SIMILARITY",
            )
            vecs = result["embedding"]
            return _cosine_similarity(vecs[0], vecs[1])

        else:
            # Unknown provider — graceful degradation
            logger.debug("Unknown embedding provider '%s' — skipping semantic check", provider)
            return None

    except Exception as e:
        logger.debug("Semantic similarity check failed (graceful degradation): %s", e)
        return None


# ── Tier 1 thresholds ─────────────────────────────────────────────────────────
# Reranker score proxy — if top chunk scores above this, likely grounded.
# Cross-encoder scores are log-odds (can be negative), typical range -5 to +5.
# 0.0 means roughly 50/50 relevant — we use a low threshold as a pre-filter.
_RERANKER_SCORE_THRESHOLD = 0.0

# High-confidence reranker bypass — skip Tier 2 entirely when reranker is
# very confident. Score > 2.0 means the cross-encoder is strongly confident
# the chunk is relevant. Tier 2 semantic similarity cannot reliably reject
# short factual answers (e.g. "57,740 claims") that have low cosine similarity
# with long chunks despite being correctly extracted from them.
_RERANKER_HIGH_CONFIDENCE_THRESHOLD = 2.0

# ── Tier 2 threshold ──────────────────────────────────────────────────────────
# Embedding cosine similarity — industry standard 0.60 for RAG groundedness.
# Only reached when Tier 1 passes AND reranker score is not high-confidence.


def _get_top_reranker_score(docs: List[dict]) -> Optional[float]:
    """
    Tier 1: Extract top reranker score from retrieved docs.
    Cross-encoder rerank scores are already computed — zero extra cost.
    Returns the highest score among all docs, or None if unavailable.
    """
    scores = []
    for doc in docs:
        score = doc.get("rerank_score", doc.get("score"))
        if score is not None:
            try:
                scores.append(float(score))
            except (TypeError, ValueError):
                pass
    return max(scores) if scores else None


def _response_is_grounded(
    response: str,
    docs: List[dict],
    config: dict = None,
) -> Tuple[bool, str]:
    """
    Two-tier groundedness check:

    Tier 1 — Reranker score proxy (zero cost, already computed)
    ─────────────────────────────────────────────────────────────
    If the top retrieved chunk scores above threshold, retrieval found
    relevant content. Fast pre-filter — avoids embedding API call when
    retrieval quality is clearly good or clearly bad.

    Tier 2 — Embedding cosine similarity (one API call)
    ─────────────────────────────────────────────────────
    Checks whether the LLM's response is semantically consistent with
    the retrieved context. Catches cases where retrieval was good but
    the LLM diverged or hallucinated in its response.

    Both tiers use graceful degradation — if a check can't run,
    it passes through rather than blocking a valid response.
    """
    if not response or not response.strip():
        return False, "Empty response"

    response_lower = response.lower()

    # ── Not-found phrases → pass through to llm_node soft prompt ─────────────
    # Don't block here — llm_node.py handles soft prompt retry for these cases
    if any(phrase in response_lower for phrase in _NO_ANSWER_PHRASES):
        return True, "Model correctly indicated missing information"

    # ── Hallucination markers → fail immediately ──────────────────────────────
    for marker in _HALLUCINATION_MARKERS:
        if marker in response_lower:
            logger.warning("Hallucination marker: '%s'", marker)
            return False, f"Hallucination marker: '{marker}'"

    if not docs:
        return False, "No context documents for grounding check"

    combined = " ".join(doc.get("content", "") for doc in docs)

    # ── Numeric value grounding (symmetric normalization) ────────────────────────
    numeric_vals = _extract_numeric_values(response)
    if numeric_vals:
        combined_normalized = _normalize_num(combined)
        ungrounded = [
            n for n in numeric_vals
            if _normalize_num(n) not in combined_normalized
        ]
        if len(ungrounded) > 2:
            return False, f"Numeric values not found in context: {ungrounded[:3]}"

    # ══ TIER 1: Reranker score proxy ══════════════════════════════════════════
    # Zero extra cost — rerank scores already computed during retrieval.
    top_score = _get_top_reranker_score(docs)

    if top_score is not None:
        logger.debug("Tier 1 reranker score: %.3f (threshold: %.2f)",
                     top_score, _RERANKER_SCORE_THRESHOLD)
        if top_score < _RERANKER_SCORE_THRESHOLD:
            # Tier 1 failed — retrieved chunks not relevant enough
            # Skip Tier 2 (no point checking semantic similarity of irrelevant chunks)
            logger.info(
                "Tier 1 FAILED: low reranker score (%.3f < %.2f) — grounding suspect",
                top_score, _RERANKER_SCORE_THRESHOLD
            )
            # Still run Tier 2 to confirm — a good response can overcome low rerank
            # (e.g. hybrid retrieval where BM25 found the right chunk but reranker
            # scored it low due to phrasing mismatch)
        else:
            logger.debug("Tier 1 PASSED: reranker score %.3f", top_score)
    else:
        logger.debug("Tier 1 SKIPPED: no reranker scores available")

    # ══ TIER 2: Embedding cosine similarity ═══════════════════════════════════
    # Skip Tier 2 when reranker is high-confidence — cross-encoder score > 2.0
    # means the chunk is strongly relevant. Short factual answers (numbers,
    # dates, names extracted from chunks) naturally have lower cosine similarity
    # than the threshold, so Tier 2 would incorrectly reject correct answers.
    if top_score is not None and top_score >= _RERANKER_HIGH_CONFIDENCE_THRESHOLD:
        logger.info(
            "Tier 2 BYPASSED: high-confidence reranker score (%.3f >= %.1f)",
            top_score, _RERANKER_HIGH_CONFIDENCE_THRESHOLD
        )
        return True, f"Grounded (reranker high-confidence: {top_score:.2f}, Tier 2 bypassed)"

    # One API call using the same embedding model as retrieval.
    # Catches semantic divergence between response and retrieved context.
    cfg            = config or {}
    embed_provider = cfg.get("embedding_provider", "openai")
    embed_model    = cfg.get("embedding_model", "text-embedding-3-small")

    similarity = _get_semantic_similarity(
        response, combined,
        provider=embed_provider,
        model=embed_model,
    )

    if similarity is not None:
        logger.debug("Tier 2 semantic similarity: %.3f (threshold: %.2f)",
                     similarity, _SEMANTIC_SIMILARITY_THRESHOLD)

        # Combined decision: use both signals when both are available
        if top_score is not None and top_score < _RERANKER_SCORE_THRESHOLD:
            # Both tiers weak — fail
            if similarity < _SEMANTIC_SIMILARITY_THRESHOLD:
                return False, (
                    f"Both tiers failed: reranker={top_score:.2f}, "
                    f"semantic similarity={similarity:.0%}"
                )
            else:
                # Tier 2 rescued Tier 1 — response is semantically grounded
                # even though reranker scored low (phrasing mismatch in retrieval)
                logger.info(
                    "Tier 2 rescued Tier 1: semantic similarity=%.0f%% overrides "
                    "low reranker score=%.2f", similarity * 100, top_score
                )
        elif similarity < _SEMANTIC_SIMILARITY_THRESHOLD:
            # Tier 1 passed but Tier 2 failed — response diverged from context
            return False, (
                f"Tier 2 FAILED: low semantic similarity ({similarity:.0%}) — "
                f"response not grounded in retrieved context"
            )
        else:
            logger.debug("Tier 2 PASSED: semantic similarity %.0f%%", similarity * 100)
    else:
        # Tier 2 unavailable — rely on Tier 1 alone
        logger.debug("Tier 2 SKIPPED: embedding unavailable — relying on Tier 1")
        if top_score is not None and top_score < _RERANKER_SCORE_THRESHOLD:
            return False, (
                f"Tier 1 FAILED: low reranker score ({top_score:.2f}) "
                f"and Tier 2 unavailable"
            )

    # Determine pass reason for logging
    tier1_str = f"reranker={top_score:.2f}" if top_score is not None else "reranker=N/A"
    tier2_str = f"similarity={similarity:.0%}" if similarity is not None else "similarity=N/A"
    return True, f"Grounded ({tier1_str}, {tier2_str})"


def validation_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

    raw_response = state.get("raw_response", "")
    docs         = state.get("reranked_docs", [])
    metadata     = state.get("metadata", {})
    config       = state.get("runtime_config", {})

    # Skip validation for general fallback — intentionally uses general knowledge
    if metadata.get("answered_from") == "general_fallback":
        logger.info("Validation skipped: general_fallback")
        return {**state, "metadata": {**metadata,
                "validation": {"valid": True, "reason": "general_fallback"}}}

    if not raw_response:
        return {**state, "metadata": {**metadata,
                "validation": {"valid": False, "reason": "No response generated"}}}

    is_valid, reason = _response_is_grounded(raw_response, docs, config)
    logger.info("Validation: valid=%s reason='%s'", is_valid, reason)

    if not is_valid:
        logger.warning("Groundedness failed: %s — attempting general fallback", reason)

        query          = state.get("query", "")
        general_response = None

        try:
            from app.providers.llm.factory import get_llm
            from app.services.llm_resilience import guarded_invoke

            provider    = config.get("llm_provider", "openai")
            model       = config.get("llm_model", "gpt-4o-mini")
            temperature = config.get("temperature", 0.0)
            user_id     = config.get("user_role", "default")

            llm = get_llm(provider=provider, model=model, temperature=temperature)
            general_prompt = (
                "You are a helpful assistant. Answer the following question "
                "clearly and concisely using your knowledge. "
                "Do not reference any specific documents. "
                "If you are uncertain, say so honestly.\n\n"
                f"Question: {query}\n\nAnswer:"
            )
            resp = guarded_invoke(llm, general_prompt, user_id=user_id)
            general_response = (
                "_This information was not found in your uploaded document. "
                "The following answer is from general knowledge:*\n\n"
                + resp.content.strip()
            )
            logger.info("Validation fallback: general answer generated")
        except Exception as e:
            logger.warning("Validation general fallback failed: %s", e)
            general_response = (
                "_This information was not found in your uploaded document._ "
                "Please try rephrasing your question or check if the relevant "
                "document has been indexed."
            )

        return {
            **state,
            "response": general_response,
            "metadata": {
                **metadata,
                "answered_from": "general_fallback",
                "validation":    {"valid": False, "reason": reason},
                "query_type":    "general",
            },
        }

    return {**state, "metadata": {**metadata,
            "validation": {"valid": True, "reason": reason}}}