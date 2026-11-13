"""
app/cache/redis_semantic_cache.py — Redis-backed semantic query cache.

DROP-IN REPLACEMENT for SemanticCache in production.
Identical public API: get(), put(), invalidate(), stats().

HOW THE ADAPTER WORKS
======================
settings.py reads CACHE_BACKEND from the environment:
  local  → SemanticCache       (local dev, JSON files on disk, unchanged)
  redis  → RedisSemanticCache  (production, Upstash or any Redis-protocol server)

Nothing else in the codebase changes. The factory function
get_cache() in app/cache/cache_factory.py returns whichever
backend is configured, with the exact same method signatures.

WHY THIS DESIGN IS CLOUD-PORTABLE
===================================
This class uses the standard `redis-py` client speaking the plain
Redis wire protocol (RESP) — NOT any Upstash-specific SDK or API.
That means switching providers later is a connection-string change only:

  Upstash:          rediss://default:TOKEN@xxx.upstash.io:6380
  GCP Memorystore:  redis://10.x.x.x:6379          (needs VPC connector)
  AWS ElastiCache:  redis://cluster.xxx.cache.amazonaws.com:6379  (needs VPC)
  Azure Cache:      rediss://name.redis.cache.windows.net:6380

No application code changes for any of the above — only REDIS_URL,
and possibly infra-level VPC networking for the private-IP options.

WHY SEMANTIC (NOT EXACT-MATCH)
================================
"What is the report date?" and "When was the report published?"
should hit the same cache entry. We store query embeddings and
compare via cosine similarity, same logic as the local JSON version.

STORAGE LAYOUT IN REDIS
=========================
Each collection's cache entries are stored as a single Redis key
holding a JSON-encoded list (mirrors the local JSON file structure
exactly, so the data model is identical across both backends):

  key:    semcache:{collection_name}
  value:  JSON list of entries, same shape as the local file

Redis TTL on the key itself is set to CACHE_TTL_SECONDS, so expired
data is automatically reclaimed by Redis — no manual purge loop needed
(unlike the local JSON version, which purges on every load/save).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import redis

logger = logging.getLogger("financial_rag.redis_semantic_cache")

SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))
CACHE_TTL_SECONDS    = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60)))
MAX_CACHE_ENTRIES    = int(os.getenv("MAX_CACHE_ENTRIES", "500"))


def _redis_key_for(collection_name: str) -> str:
    safe = collection_name.replace("/", "_").replace("\\", "_")
    return f"semcache:{safe}"


class RedisSemanticCache:
    """
    Redis-backed semantic cache. Same public API as SemanticCache.

    Args:
        collection_name : isolates cache entries per user/session
        embedding_fn     : callable(text) -> List[float], same as SemanticCache
        redis_url        : overrides REDIS_URL env var if provided
        redis_password   : overrides REDIS_PASSWORD env var if provided
    """

    def __init__(
        self,
        collection_name: str,
        embedding_fn=None,
        redis_url: Optional[str] = None,
        redis_password: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embedding_fn    = embedding_fn

        url      = redis_url      or os.getenv("REDIS_URL", "")
        password = redis_password or os.getenv("REDIS_PASSWORD", "") or None

        if not url:
            raise EnvironmentError(
                "REDIS_URL is not set. Add it to your .env (local) or "
                "GCP Secret Manager as 'rag-redis-url' (production)."
            )

        # decode_responses=True so we get str back, not bytes — simpler JSON handling
        self._client = redis.from_url(
            url,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

        logger.info(
            "RedisSemanticCache ready: collection='%s' ttl=%ds threshold=%.2f",
            collection_name, CACHE_TTL_SECONDS, SIMILARITY_THRESHOLD
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _redis_key(self) -> str:
        return _redis_key_for(self.collection_name)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.embedding_fn is None:
            return None
        try:
            vec  = self.embedding_fn(text)
            arr  = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr)
            return arr / norm if norm > 0 else arr
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return None

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _load_entries(self) -> List[Dict]:
        """Load all entries for this collection from Redis."""
        try:
            raw = self._client.get(self._redis_key)
            if not raw:
                return []
            return json.loads(raw)
        except Exception as e:
            logger.warning("Redis cache load failed: %s", e)
            return []

    def _save_entries(self, entries: List[Dict]) -> None:
        """Save all entries for this collection to Redis with TTL."""
        try:
            self._client.set(
                self._redis_key,
                json.dumps(entries),
                ex=CACHE_TTL_SECONDS,   # Redis auto-expires the whole key
            )
        except Exception as e:
            logger.warning("Redis cache save failed: %s", e)

    # ── Public API (same as SemanticCache) ────────────────────────────────────

    def get(self, query: str, model: str = "") -> Optional[Dict]:
        """
        Look up a semantically similar cached query.
        Returns the cached entry (with cache_similarity added) or None.
        """
        if self.embedding_fn is None:
            return None
        q_emb = self._embed(query)
        if q_emb is None:
            return None

        now      = time.time()
        entries  = self._load_entries()
        best_sim, best_entry, best_idx = 0.0, None, -1

        for idx, entry in enumerate(entries):
            if now - entry["timestamp"] > CACHE_TTL_SECONDS:
                continue
            if entry.get("model", "") != model:
                continue
            sim = self._cosine(q_emb, np.array(entry["query_embedding"], dtype=np.float32))
            if sim > best_sim:
                best_sim, best_entry, best_idx = sim, entry, idx

        if best_entry and best_sim >= SIMILARITY_THRESHOLD:
            # Increment hit counter and persist it back
            entries[best_idx]["hits"] = entries[best_idx].get("hits", 0) + 1
            self._save_entries(entries)
            logger.info("Redis cache HIT sim=%.3f '%s'", best_sim, query[:60])
            return {**best_entry, "cache_similarity": best_sim}

        logger.debug("Redis cache MISS '%s'", query[:60])
        return None

    def put(self, query: str, response: str, model: str = "",
            metadata: Dict = None) -> bool:
        """Store a new query/response pair in the cache."""
        if self.embedding_fn is None:
            return False
        q_emb = self._embed(query)
        if q_emb is None:
            return False

        entry = {
            "query_text":      query,
            "query_embedding": q_emb.tolist(),
            "response":        response,
            "model":           model,
            "metadata":        metadata or {},
            "timestamp":       time.time(),
            "hits":            0,
        }

        now     = time.time()
        entries = self._load_entries()
        entries = [e for e in entries if now - e["timestamp"] <= CACHE_TTL_SECONDS]

        if len(entries) >= MAX_CACHE_ENTRIES:
            entries.sort(key=lambda x: x["timestamp"])
            entries = entries[-(MAX_CACHE_ENTRIES - 1):]

        entries.append(entry)
        self._save_entries(entries)
        logger.info("Redis cache PUT '%s'", query[:60])
        return True

    def invalidate(self) -> None:
        """Clear all cached entries for this collection."""
        try:
            self._client.delete(self._redis_key)
            logger.info("Redis cache invalidated for '%s'", self.collection_name)
        except Exception as e:
            logger.warning("Redis cache invalidate failed: %s", e)

    def stats(self) -> Dict:
        """Return cache statistics, same shape as SemanticCache.stats()."""
        now     = time.time()
        entries = self._load_entries()
        active  = [e for e in entries if now - e["timestamp"] <= CACHE_TTL_SECONDS]
        return {
            "total_entries": len(active),
            "total_hits":    sum(e.get("hits", 0) for e in active),
            "ttl_seconds":   CACHE_TTL_SECONDS,
            "threshold":     SIMILARITY_THRESHOLD,
            "collection":    self.collection_name,
            "backend":       "redis",
        }