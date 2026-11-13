"""
prompt_guard.py
---------------
Prompt injection detection for user queries.

Covers:
  - Direct instruction override attempts
  - Role-play / persona injection
  - System prompt extraction attempts
  - Jailbreak patterns
  - Indirect injection (instructions embedded in quotes or code blocks)

Returns detailed result so the security_node can log the specific pattern.
"""

import re
import logging
from typing import Tuple, Optional

logger = logging.getLogger("financial_rag.prompt_guard")

# ── Pattern groups ────────────────────────────────────────────────────────────
# Each group has a name and list of patterns (case-insensitive)

_INJECTION_RULES = [

    ("instruction_override", [
        r"ignore\s+(previous|prior|all|above|earlier)\s+instructions?",
        r"disregard\s+(all|previous|your|the)\s+(rules?|instructions?|guidelines?|context)",
        r"forget\s+(everything|all|previous|your)\s*(you|instructions?|context)?",
        r"override\s+(previous|your|all)?\s*(instructions?|rules?|constraints?)",
        r"do\s+not\s+follow\s+(your|the|previous)\s+(rules?|instructions?)",
        r"new\s+(instructions?|rules?|directives?)[\s:]+",
        r"actual\s+instructions?\s+(are|follow|below)",
    ]),

    ("role_persona_injection", [
        r"you\s+are\s+now\s+(a|an|the)\s+\w+",
        r"act\s+as\s+(a|an|the|if)?\s*(unrestricted|unfiltered|evil|hacker|dan|jailbreak)",
        r"pretend\s+(you\s+are|to\s+be)\s+(a|an)?\s*(different|new|unrestricted)",
        r"roleplay\s+as",
        r"switch\s+to\s+(developer|god|admin|root|system)\s+mode",
        r"enter\s+(jailbreak|developer|unrestricted|dan)\s+mode",
        r"you\s+have\s+no\s+(restrictions?|limits?|rules?|guidelines?)",
    ]),

    ("system_prompt_extraction", [
        r"(show|reveal|print|display|repeat|output|tell)(\s+me)?(\s+the)?\s+(your\s+)?(system\s+prompt|instructions?|prompt|guidelines?|rules?)",
        r"what\s+(are\s+your|is\s+your)\s+(system\s+prompt|instructions?|guidelines?|rules?|training)",
        r"(dump|leak|expose)\s+(the\s+)?(system\s+prompt|instructions?|context)",
        r"(ignore|skip)\s+(the\s+)?(system|context|document)\s+(prompt|instructions?)",
    ]),

    ("data_exfiltration", [
        r"(send|email|post|upload|transmit)\s+(all|the|this)\s+(data|documents?|context|information)\s+to",
        r"(extract|copy|export)\s+(all|the)?\s*(confidential|sensitive|private|internal)\s+(data|info|documents?)",
        r"(http|https|ftp|www)\S+",   # URLs in queries are suspicious
    ]),

    ("code_injection", [
        r"```(python|javascript|bash|sh|cmd|powershell|sql)",
        r"<script[\s>]",
        r"(eval|exec|subprocess|os\.system)\s*\(",
        r"__import__\s*\(",
        r"DROP\s+TABLE|SELECT\s+\*\s+FROM|INSERT\s+INTO",  # SQL injection
    ]),

    ("authority_claim", [
        r"(i\s+am|i'm|this\s+is)\s+(the\s+)?(admin|administrator|developer|anthropic|openai|system)",
        r"(as\s+your|i\s+am\s+your)\s+(creator|developer|trainer|owner|master)",
        r"(authorized|permitted|allowed)\s+to\s+(bypass|override|disable|ignore)",
    ]),
]

# Compile all patterns
_COMPILED_RULES = [
    (group_name, [re.compile(p, re.IGNORECASE) for p in patterns])
    for group_name, patterns in _INJECTION_RULES
]

# ── Query quality checks ──────────────────────────────────────────────────────

_MAX_QUERY_LENGTH   = 2000    # chars
_MIN_QUERY_LENGTH   = 2       # chars
_MAX_SPECIAL_RATIO  = 0.40    # if >40% special chars → suspicious


def _check_query_quality(query: str) -> Optional[str]:
    """Returns a rejection reason string, or None if query is fine."""
    if not query or not query.strip():
        return "Empty query"

    stripped = query.strip()

    if len(stripped) < _MIN_QUERY_LENGTH:
        return "Query too short"

    if len(stripped) > _MAX_QUERY_LENGTH:
        return f"Query too long ({len(stripped)} chars, max {_MAX_QUERY_LENGTH})"

    special_count = sum(1 for c in stripped if not c.isalnum() and not c.isspace())
    if len(stripped) > 0 and special_count / len(stripped) > _MAX_SPECIAL_RATIO:
        return "Too many special characters (possible injection)"

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def detect_prompt_injection(query: str) -> Tuple[bool, str]:
    """
    Check query for prompt injection attempts.

    Returns:
        (blocked: bool, reason: str)
        blocked=True  → reject the query, show reason to user
        blocked=False → query is safe to proceed
    """
    # Quality check first
    quality_issue = _check_query_quality(query)
    if quality_issue:
        logger.warning("Query rejected — quality: %s", quality_issue)
        return True, quality_issue

    # Pattern matching
    for group_name, patterns in _COMPILED_RULES:
        for pattern in patterns:
            if pattern.search(query):
                reason = f"Potential prompt injection detected ({group_name})"
                logger.warning("Injection detected — group=%s | query='%s'",
                               group_name, query[:100])
                return True, reason

    return False, ""


def is_safe_query(query: str) -> bool:
    """Convenience wrapper — returns True if query is safe."""
    blocked, _ = detect_prompt_injection(query)
    return not blocked
