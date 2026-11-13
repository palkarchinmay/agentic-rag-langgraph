"""
ragas_node.py — RAGAS evaluation support (on-demand model).

ARCHITECTURE CHANGE (Step 5)
==============================
RAGAS no longer runs automatically after every query. Instead:

1. This node generates an eval_key and stores query/answer/docs in Redis.
2. User clicks "Evaluate" in the Diagnostics tab.
3. POST /api/v1/evaluate runs RAGAS synchronously and stores the result
   in Redis under ragas:{user_id}:{eval_key}.
4. Diagnostics panel reads from Redis directly — no polling, no threads.

WHY BACKGROUND THREADS BROKE IN CLOUD RUN
============================================
Cloud Run terminates daemon threads when the HTTP response is sent (~1-2s
after response). RAGAS takes 5-30s — it was always killed before completing.

USER ISOLATION
===============
All Redis keys: ragas:{user_id}:{eval_key} and ragas_data:{user_id}:{eval_key}
user_id from runtime_config["user_id"] (JWT sub claim).
"""

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from app.graph.state import GraphState

logger        = logging.getLogger("financial_rag.ragas_node")
CACHE_DIR     = Path(os.getenv("RAGAS_CACHE_DIR", ".ragas_cache"))
CACHE_DIR.mkdir(exist_ok=True)
RAGAS_TTL_SECONDS = int(os.getenv("RAGAS_RESULT_TTL", str(2 * 60 * 60)))

_redis_client = None
_redis_lock   = threading.Lock()


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    if os.getenv("CACHE_BACKEND", "local").lower() != "redis":
        return None
    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        return None
    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            import redis
            client = redis.from_url(redis_url, decode_responses=True,
                                    socket_connect_timeout=5, socket_timeout=5)
            client.ping()
            _redis_client = client
            logger.info("RAGAS: using Redis result store")
        except Exception as e:
            logger.warning("RAGAS: Redis unavailable (%s), using file cache", e)
            _redis_client = None
    return _redis_client


def _redis_key(user_id: str, eval_key: str) -> str:
    return f"ragas:{(user_id or 'anonymous').replace(':', '_')}:{eval_key}"


def _redis_data_key(user_id: str, eval_key: str) -> str:
    return f"ragas_data:{(user_id or 'anonymous').replace(':', '_')}:{eval_key}"


def _write_result(eval_key: str, data: dict, user_id: str = ""):
    r = _get_redis()
    if r:
        try:
            r.set(_redis_key(user_id, eval_key), json.dumps(data), ex=RAGAS_TTL_SECONDS)
            return
        except Exception as e:
            logger.warning("RAGAS: Redis write failed: %s", e)
    try:
        _cache_path(eval_key).write_text(json.dumps(data))
    except Exception as e:
        logger.warning("RAGAS: file write failed: %s", e)


def _write_query_data(eval_key: str, data: dict, user_id: str = ""):
    """Store query/answer/docs for on-demand evaluation."""
    r = _get_redis()
    if r:
        try:
            r.set(_redis_data_key(user_id, eval_key), json.dumps(data), ex=RAGAS_TTL_SECONDS)
            return
        except Exception as e:
            logger.warning("RAGAS: Redis data write failed: %s", e)
    try:
        (CACHE_DIR / f"{eval_key}_data.json").write_text(json.dumps(data))
    except Exception as e:
        logger.warning("RAGAS: file data write failed: %s", e)


def get_query_data(eval_key: str, user_id: str = "") -> dict:
    """Retrieve stored query/answer/docs for on-demand evaluation."""
    r = _get_redis()
    if r:
        try:
            raw = r.get(_redis_data_key(user_id, eval_key))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("RAGAS: Redis data read failed: %s", e)
    try:
        path = CACHE_DIR / f"{eval_key}_data.json"
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def get_ragas_result(eval_key: str, user_id: str = "") -> dict:
    if not eval_key:
        return {}
    r = _get_redis()
    if r:
        try:
            raw = r.get(_redis_key(user_id, eval_key))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("RAGAS: Redis read failed: %s", e)
    try:
        path = _cache_path(eval_key)
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}


def is_ragas_done(eval_key: str, user_id: str = "") -> bool:
    return get_ragas_result(eval_key, user_id).get("status") in ("done", "error")


def ragas_node(state: GraphState) -> GraphState:
    """
    Generates eval_key and stores query data for on-demand evaluation.
    Does NOT run RAGAS automatically.
    """
    config   = state.get("runtime_config", {})
    metadata = state.get("metadata", {})

    if not config.get("diagnostics", False):
        return {**state, "metadata": {**metadata,
                "ragas": {"skipped": True, "reason": "diagnostics_off"}}}

    if metadata.get("cache_hit"):
        return {**state, "metadata": {**metadata,
                "ragas": {"skipped": True, "reason": "cache_hit"}}}

    if metadata.get("query_type") != "rag":
        return {**state, "metadata": {**metadata,
                "ragas": {"skipped": True, "reason": "not_rag"}}}

    query    = state.get("query", "")
    response = state.get("response", "")
    docs     = state.get("reranked_docs", [])

    if not query or not response or not docs:
        return {**state, "metadata": {**metadata,
                "ragas": {"skipped": True, "reason": "missing_data"}}}

    user_id  = config.get("user_id", "")
    eval_key = f"ragas_{hashlib.md5(f'{query}{time.time()}'.encode()).hexdigest()[:8]}"

    # Store data for on-demand evaluation via POST /api/v1/evaluate
    _write_query_data(eval_key, {
        "question":       query,
        "answer":         response,
        "retrieved_docs": docs,
        "llm_provider":   config.get("llm_provider", "openai"),
        "llm_model":      config.get("llm_model", "gpt-4o-mini"),
        "user_id":        user_id,
    }, user_id)

    # "pending" = ready to evaluate, not yet run
    _write_result(eval_key, {"status": "pending"}, user_id)

    logger.info("RAGAS: ready for on-demand eval [%s] user=%s", eval_key, user_id)

    return {**state, "metadata": {
        **metadata,
        "ragas": {"status": "pending", "eval_key": eval_key, "user_id": user_id}
    }}


def _cache_path(eval_key: str) -> Path:
    return CACHE_DIR / f"{eval_key}.json"


def cleanup_old_results(max_age_hours: int = 24):
    cutoff = time.time() - (max_age_hours * 3600)
    for f in CACHE_DIR.glob("*.json"):
        if f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)