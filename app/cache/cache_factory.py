"""
app/cache/cache_factory.py — Adapter factory for semantic cache backends.

USAGE
=====
    from app.cache.cache_factory import get_cache

    # Returns SemanticCache locally, RedisSemanticCache in production
    cache = get_cache(collection_name="user_alice", embedding_fn=my_embed_fn)
    cache.put(query, response, model="gpt-4o-mini")
    cached = cache.get(query, model="gpt-4o-mini")

SWITCHING BACKENDS
===================
Set CACHE_BACKEND in your environment:

    CACHE_BACKEND=local   → JSON files on disk (local dev, default)
    CACHE_BACKEND=redis   → Redis / Upstash (production)

No other code changes needed anywhere in the app. This mirrors the
exact same pattern used for VECTOR_BACKEND in Step 2
(app/retrieval/vector_store_factory.py).

WHY get_cache() HERE INSTEAD OF semantic_cache.py
====================================================
app/cache/semantic_cache.py already exports a module-level get_cache()
function used by app/graph/nodes/cache_node.py. To avoid touching that
existing import path, semantic_cache.get_cache() itself is updated
(see settings_additions for Step 3) to delegate to this factory when
CACHE_BACKEND=redis, and behave exactly as before when CACHE_BACKEND=local.
This factory is the single source of truth either way.
"""

from __future__ import annotations

import os
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger("financial_rag.cache_factory")

# ── Module-level registry (mirrors semantic_cache.py's own registry) ──────────
# Keeps one cache instance per collection_name, regardless of backend,
# so repeated calls don't reconnect to Redis or reopen files every time.
_cache_registry: Dict[str, object] = {}
_registry_lock = threading.Lock()


def get_cache_backend(collection_name: str, embedding_fn=None):
    """
    Return the correct cache backend based on CACHE_BACKEND env var.
    Cached per collection_name in a module-level registry.

    Args:
        collection_name : isolates cache entries per user/session
        embedding_fn     : callable(text) -> List[float]

    Returns:
        SemanticCache       (CACHE_BACKEND=local or unset)
        RedisSemanticCache  (CACHE_BACKEND=redis)
    """
    backend = os.getenv("CACHE_BACKEND", "local").lower().strip()
    registry_key = f"{backend}:{collection_name}"

    with _registry_lock:
        existing = _cache_registry.get(registry_key)
        if existing is not None:
            return existing

    if backend in ("local", "", "json"):
        logger.debug("Cache backend: local JSON (collection=%s)", collection_name)
        from app.cache.semantic_cache import SemanticCache
        instance = SemanticCache(collection_name, embedding_fn)

    elif backend == "redis":
        logger.debug("Cache backend: Redis (collection=%s)", collection_name)
        from app.cache.redis_semantic_cache import RedisSemanticCache
        instance = RedisSemanticCache(collection_name, embedding_fn)

    else:
        raise ValueError(
            f"Unknown CACHE_BACKEND={backend!r}. "
            "Valid values: 'local' (default) or 'redis'."
        )

    with _registry_lock:
        _cache_registry[registry_key] = instance

    return instance


def invalidate_cache_backend(collection_name: str) -> None:
    """
    Invalidate cache for a collection, regardless of backend.
    Removes the instance from the registry too, so a fresh one is
    created (and reloaded from the backend) on next access.
    """
    backend = os.getenv("CACHE_BACKEND", "local").lower().strip()
    registry_key = f"{backend}:{collection_name}"

    with _registry_lock:
        instance = _cache_registry.pop(registry_key, None)

    if instance is not None:
        instance.invalidate()
        return

    # Not in memory — create a throwaway instance just to invalidate
    # (handles the case where invalidate is called before any get/put)
    try:
        if backend == "redis":
            from app.cache.redis_semantic_cache import RedisSemanticCache
            RedisSemanticCache(collection_name).invalidate()
        else:
            from app.cache.semantic_cache import _invalidate_local_disk_cache
            _invalidate_local_disk_cache(collection_name)
    except Exception as e:
        logger.warning("Could not invalidate cache for '%s': %s", collection_name, e)