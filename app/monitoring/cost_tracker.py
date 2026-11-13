"""
cost_tracker.py — Token counting and cost estimation.

Pricing per 1K tokens (as of mid-2025 — update periodically).
Source: provider pricing pages.
"""

import logging
from typing import Dict

logger = logging.getLogger("financial_rag.cost_tracker")

# ── Pricing table: {model_id: {input: $/token, output: $/token}} ─────────────
MODEL_PRICING: Dict[str, Dict[str, float]] = {

    # ── OpenAI ────────────────────────────────────────────────────────────────
    "gpt-4o":              {"input": 0.0025  / 1000, "output": 0.010   / 1000},
    "gpt-4o-mini":         {"input": 0.00015 / 1000, "output": 0.00060 / 1000},
    "gpt-4-turbo":         {"input": 0.010   / 1000, "output": 0.030   / 1000},
    "gpt-3.5-turbo":       {"input": 0.0005  / 1000, "output": 0.0015  / 1000},

    # ── OpenAI embeddings ─────────────────────────────────────────────────────
    "text-embedding-3-small": {"input": 0.00002 / 1000, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013 / 1000, "output": 0.0},
    "text-embedding-ada-002":  {"input": 0.00010 / 1000, "output": 0.0},

    # ── AWS Bedrock ───────────────────────────────────────────────────────────
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003 / 1000, "output": 0.015 / 1000},
    "anthropic.claude-3-haiku-20240307-v1:0":    {"input": 0.00025/1000, "output": 0.00125/1000},
    "amazon.titan-embed-text-v2:0":              {"input": 0.00002/1000, "output": 0.0},

    # ── GCP Vertex Gemini ─────────────────────────────────────────────────────
    "gemini-2.0-flash-001":   {"input": 0.00015/1000, "output": 0.00060/1000},
    "gemini-1.5-pro":         {"input": 0.00125/1000, "output": 0.00500/1000},
    "gemini-1.5-flash":       {"input": 0.000075/1000,"output": 0.00030/1000},
    "text-embedding-004":     {"input": 0.00001/1000, "output": 0.0},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Estimate cost in USD for a single LLM call.

    Args:
        model            : model identifier string
        prompt_tokens    : input token count
        completion_tokens: output token count

    Returns:
        Estimated cost in USD (float), or 0.0 if model not in pricing table.
    """
    # Try exact match first, then prefix match for Bedrock ARN-style IDs
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        for key in MODEL_PRICING:
            if model.startswith(key) or key.startswith(model.split(":")[0]):
                pricing = MODEL_PRICING[key]
                break

    if not pricing:
        logger.debug("No pricing found for model '%s' — cost=0.0", model)
        return 0.0

    cost = (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])
    return round(cost, 8)


def format_cost(cost_usd: float) -> str:
    """Format cost for display."""
    if cost_usd < 0.001:
        return f"${cost_usd * 100:.4f}¢"
    return f"${cost_usd:.5f}"


def cumulative_cost(usage_records: list) -> Dict[str, float]:
    """
    Aggregate multiple usage records.

    Args:
        usage_records: list of {prompt_tokens, completion_tokens,
                                total_tokens, estimated_cost_usd}
    Returns:
        {total_prompt_tokens, total_completion_tokens,
         total_tokens, total_cost_usd, num_calls}
    """
    result = {
        "total_prompt_tokens":     0,
        "total_completion_tokens": 0,
        "total_tokens":            0,
        "total_cost_usd":          0.0,
        "num_calls":               0,
    }
    for r in usage_records:
        result["total_prompt_tokens"]     += r.get("prompt_tokens", 0)
        result["total_completion_tokens"] += r.get("completion_tokens", 0)
        result["total_tokens"]            += r.get("total_tokens", 0)
        result["total_cost_usd"]          += r.get("estimated_cost_usd", 0.0)
        result["num_calls"]               += 1

    result["total_cost_usd"] = round(result["total_cost_usd"], 8)
    return result
