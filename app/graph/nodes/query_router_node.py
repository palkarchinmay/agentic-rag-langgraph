"""
query_router_node.py — Domain-agnostic query classifier.

Routes queries to: rag | general | calculation | blocked

Design principles:
- Zero domain-specific terms (no HR, financial, medical, legal hardcoding)
- Routing based on query STRUCTURE and INTENT, not topic keywords
- Calculation detected by math verbs + numbers, not by domain terms
- RAG triggered by document-reference signals, not topic keywords
- Conversational follow-ups detected by structural signals (pronouns +
  meta-communication verbs), not by topic keywords
"""

import re
import logging
from app.graph.state import GraphState

logger = logging.getLogger("rag_platform.query_router")


# ── Calculation detection ──────────────────────────────────────────────────────
_CALC_VERB_RE = re.compile(
    r"\b(calculate|compute|convert|how much is|what is \d[\d,.]*|evaluate|solve)\b",
    re.IGNORECASE,
)
_HAS_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")


def _is_calculation_query(query: str) -> bool:
    return bool(_CALC_VERB_RE.search(query)) and bool(_HAS_NUMBER_RE.search(query))


# ── Conversational meta-query detection — STRUCTURAL, not keyword-based ────────
#
# A conversational meta-query has ALL of these structural properties:
#   1. First/second person pronoun (I, you, we, my, your, our)
#   2. Past-tense or present-perfect meta-communication verb
#      (ask, say, mention, discuss, tell, answer, explain, respond)
#   3. Short query — long queries with nouns/domain terms are likely RAG
#
# This is domain-agnostic because it checks GRAMMATICAL STRUCTURE:
# "what did I ask?" = pronoun + meta-verb + no domain nouns → conversational
# "what is the litigation rate?" = no pronoun + domain noun → RAG
# "what did the document say about rates?" = pronoun but RAG signal → RAG
#
_PRONOUN_RE      = re.compile(r"\b(i|you|we|my|your|our)\b", re.IGNORECASE)
_META_VERB_RE    = re.compile(
    r"\b(ask|asked|say|said|mention|mentioned|discuss|discussed|"
    r"tell|told|answer|answered|explain|explained|respond|responded|"
    r"query|queried|request|requested|previous|prior|last|before|earlier|"
    r"just|already|so far|conversation|history)\b",
    re.IGNORECASE,
)
def _is_conversational_query(query: str) -> bool:
    """
    Detect conversational meta-queries using grammar structure only.

    A conversational query:
    - Has no explicit RAG/document signals (handled by _RAG_PATTERN_RE)
    - Has a first/second person pronoun (I, you, we, my, your, our)
    - Has a meta-communication verb (ask, say, discuss, mention, previous...)

    Domain-agnostic: uses only grammatical structure, no topic keywords.
    No word count limit needed — RAG signals already filter document queries.
    """
    q = query.strip()
    # If it references a document explicitly, it's a RAG query not conversational
    if _RAG_PATTERN_RE.search(q):
        return False
    return bool(_PRONOUN_RE.search(q)) and bool(_META_VERB_RE.search(q))


# ── General-knowledge qualifiers ──────────────────────────────────────────────
_GENERAL_QUALIFIER_RE = re.compile(
    r"\b(in general|typically|usually|normally|industry standard|"
    r"on average|benchmark|market rate|what does .{0,30} mean|"
    r"define|definition of|explain the concept|how does .{0,30} work)\b",
    re.IGNORECASE,
)


# ── RAG signals ───────────────────────────────────────────────────────────────
_RAG_PATTERN_RE = re.compile(
    r"\b(document|file|uploaded|attachment|report|this pdf|this file|"
    r"page|slide|section|paragraph|clause|chapter|appendix|annexure|"
    r"mentioned|stated|written|says|according to|as per|in the document|"
    r"from the document|in this report|summarize|summarise|extract|"
    r"find in|look up|what does it say|what is written)\b",
    re.IGNORECASE,
)


# ── Blocked patterns ──────────────────────────────────────────────────────────
_BLOCKED_RE = re.compile(
    r"\b(ignore (previous|all|above)|disregard|new instructions|"
    r"system prompt|reveal (your|the) (instructions|prompt|rules)|"
    r"act as|you are now|pretend to be|roleplay as|jailbreak|"
    r"DAN mode|developer mode)\b",
    re.IGNORECASE,
)


def _classify_query(query: str, has_docs: bool) -> tuple[str, float, str]:
    """
    Returns (query_type, confidence, method).

    Priority order:
      1. blocked        — security/injection patterns
      2. conversational — structural detection (pronoun + meta-verb, no RAG signals)
      3. calculation    — math verb + number
      4. rag signals    — explicit document-reference words
      5. has_docs       — documents loaded → RAG (overrides general qualifiers)
      6. general        — explicit general-knowledge markers (no docs)
      7. fallback       — general when no docs
    """
    q = query.strip()

    # 1. Blocked
    if _BLOCKED_RE.search(q):
        return "blocked", 0.99, "regex"

    # 2. Conversational meta-query (structural detection)
    # Routes to general → direct_llm uses conversation history from Supabase
    if _is_conversational_query(q):
        return "general", 0.90, "conversational"

    # 3. Calculation
    if _is_calculation_query(q):
        return "calculation", 0.90, "heuristic"

    # 4. Explicit RAG signals
    if _RAG_PATTERN_RE.search(q):
        return "rag", 0.90, "heuristic"

    # 5. Documents loaded → RAG (overrides general qualifiers)
    if has_docs:
        return "rag", 0.60, "heuristic"

    # 6. General knowledge markers (only when no docs)
    if _GENERAL_QUALIFIER_RE.search(q):
        return "general", 0.85, "heuristic"

    # 7. Fallback
    return "general", 0.70, "heuristic"


def query_router_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

    query    = state.get("query", "")
    config   = state.get("runtime_config", {})
    metadata = state.get("metadata", {})

    collection = config.get("collection_name", "")
    has_docs   = bool(collection and collection not in ("default", "", None))

    # TEMPORARY DIAGNOSTIC (2026-08-14) — remove once the has_docs
    # inconsistency across requests in the same session is root-caused.
    # Confirmed via direct testing that _classify_query() itself is
    # correct (routes to "rag" whenever has_docs=True) -- the bug is
    # that has_docs evaluates differently across requests within the
    # SAME chat session despite documents being indexed the whole time.
    logger.warning(
        "ROUTER DIAGNOSTIC: collection=%r has_docs=%s query=%r",
        collection, has_docs, query[:60],
    )

    qtype, conf, method = _classify_query(query, has_docs)

    logger.info(
        "Router: %s (conf=%.2f, method=%s) | '%s'",
        qtype, conf, method, query[:80],
    )

    if qtype == "blocked":
        return {
            **state,
            "blocked":      True,
            "block_reason": "Query contains disallowed patterns.",
            "metadata": {
                **metadata,
                "query_type":    "blocked",
                "router_method": method,
            },
        }

    return {
        **state,
        "metadata": {
            **metadata,
            "query_type":    qtype,
            "router_conf":   conf,
            "router_method": method,
        },
    }