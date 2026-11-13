"""
cache/semantic_cache.py
-----------------------
Semantic query cache using embedding cosine similarity.

Why semantic (not exact-match):
  "What is the report date?" and "When was the report published?" → same cache entry.

Cache key  : query embedding (cosine similarity >= threshold)
Storage    : in-memory + JSON disk file (survives process restarts)
TTL        : configurable, default 1 hour
Invalidate : call invalidate_cache() after new document upload
             Works even when collection not loaded in memory (deletes disk file)

STEP 3 CHANGE
=============
get_cache() and invalidate_cache() below now delegate to
app/cache/cache_factory.py, which reads CACHE_BACKEND and returns
either this SemanticCache (local JSON) or RedisSemanticCache
(Upstash/Redis), with the exact same method signatures either way.

The SemanticCache class itself is UNCHANGED — local development
behavior is identical to before Step 3. Only the module-level
get_cache()/invalidate_cache() functions changed, so that
app/graph/nodes/cache_node.py (which imports get_cache from here)
needs zero modifications to support both backends.
"""

import os
import json
import time
import logging
import threading
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path

# Pre-import to avoid threading deadlock when RAGAS background thread
# and main thread both try to import langchain_openai simultaneously
try:
    import langchain_openai  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger("financial_rag.semantic_cache")

SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.85"))  # 0.85 catches paraphrases; 0.92 was too strict
CACHE_TTL_SECONDS    = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60)))
MAX_CACHE_ENTRIES    = int(os.getenv("MAX_CACHE_ENTRIES", "500"))
CACHE_DIR            = os.getenv("CACHE_DIR", ".cache")

os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path_for(collection_name: str) -> str:
    safe = collection_name.replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_DIR, f"cache_{safe}.json")


class SemanticCache:

    def __init__(self, collection_name: str, embedding_fn=None):
        self.collection_name = collection_name
        self.embedding_fn    = embedding_fn
        self._lock           = threading.Lock()
        self._entries: List[Dict] = []
        self._load()

    @property
    def _cache_path(self) -> str:
        return _cache_path_for(self.collection_name)

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

    def get(self, query: str, model: str = "") -> Optional[Dict]:
        if self.embedding_fn is None:
            return None
        q_emb = self._embed(query)
        if q_emb is None:
            return None

        now   = time.time()
        best_sim, best_entry = 0.0, None

        with self._lock:
            for entry in self._entries:
                if now - entry["timestamp"] > CACHE_TTL_SECONDS:
                    continue
                if entry.get("model", "") != model:
                    continue
                sim = self._cosine(q_emb, np.array(entry["query_embedding"], dtype=np.float32))
                if sim > best_sim:
                    best_sim, best_entry = sim, entry

            if best_entry and best_sim >= SIMILARITY_THRESHOLD:
                best_entry["hits"] = best_entry.get("hits", 0) + 1
                logger.info("Cache HIT sim=%.3f '%s'", best_sim, query[:60])
                return {**best_entry, "cache_similarity": best_sim}

        logger.debug("Cache MISS '%s'", query[:60])
        return None

    def put(self, query: str, response: str, model: str = "",
            metadata: Dict = None) -> bool:
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
        with self._lock:
            now            = time.time()
            self._entries  = [e for e in self._entries
                              if now - e["timestamp"] <= CACHE_TTL_SECONDS]
            if len(self._entries) >= MAX_CACHE_ENTRIES:
                self._entries.sort(key=lambda x: x["timestamp"])
                self._entries = self._entries[-(MAX_CACHE_ENTRIES - 1):]
            self._entries.append(entry)
            self._save()
        logger.info("Cache PUT '%s'", query[:60])
        return True

    def invalidate(self):
        with self._lock:
            count          = len(self._entries)
            self._entries  = []
            self._save()
        logger.info("Cache invalidated: %d entries cleared for '%s'",
                    count, self.collection_name)

    def stats(self) -> Dict:
        now    = time.time()
        active = [e for e in self._entries
                  if now - e["timestamp"] <= CACHE_TTL_SECONDS]
        return {
            "total_entries": len(active),
            "total_hits":    sum(e.get("hits", 0) for e in active),
            "ttl_seconds":   CACHE_TTL_SECONDS,
            "threshold":     SIMILARITY_THRESHOLD,
            "collection":    self.collection_name,
            "backend":       "local",
        }

    def _save(self):
        try:
            with open(self._cache_path, "w") as f:
                json.dump(self._entries, f)
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def _load(self):
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            now           = time.time()
            self._entries = [e for e in data
                             if now - e["timestamp"] <= CACHE_TTL_SECONDS]
            purged = len(data) - len(self._entries)
            if purged:
                logger.info("Cache: purged %d expired entries for '%s'",
                            purged, self.collection_name)
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
            self._entries = []


# ── Module-level registry ─────────────────────────────────────────────────────
# STEP 3: get_cache() and invalidate_cache() now delegate to cache_factory.py,
# which reads CACHE_BACKEND and returns SemanticCache (local) or
# RedisSemanticCache (redis) — both with identical method signatures.
# cache_node.py imports get_cache from THIS module and needs no changes.

_cache_registry: Dict[str, SemanticCache] = {}
_registry_lock  = threading.Lock()


def get_cache(collection_name: str, embedding_fn=None):
    """
    Return the configured cache backend for this collection.

    Delegates to app/cache/cache_factory.py, which reads CACHE_BACKEND:
      CACHE_BACKEND=local (default) → SemanticCache (this file, JSON on disk)
      CACHE_BACKEND=redis           → RedisSemanticCache (Upstash/Redis)

    Both backends expose identical get()/put()/invalidate()/stats() methods,
    so callers (cache_node.py) never need to know which one is active.
    """
    from app.cache.cache_factory import get_cache_backend
    return get_cache_backend(collection_name, embedding_fn)


def invalidate_cache(collection_name: str):
    """
    Invalidate cache for a collection, regardless of backend.
    Works whether or not the collection is currently loaded in memory.
    """
    from app.cache.cache_factory import invalidate_cache_backend
    invalidate_cache_backend(collection_name)


def _invalidate_local_disk_cache(collection_name: str):
    """
    Local-disk-specific invalidation, used internally by cache_factory.py
    as a fallback when no SemanticCache instance is currently in memory
    (e.g. invalidate called before any get/put in this process).
    Deletes the JSON file directly so even a fresh process starts clean.
    """
    disk_path = _cache_path_for(collection_name)
    if os.path.exists(disk_path):
        try:
            os.remove(disk_path)
            logger.info("Cache disk file deleted for '%s'", collection_name)
        except Exception as e:
            logger.warning("Could not delete cache file %s: %s", disk_path, e)
    else:
        logger.debug("No cache file to delete for '%s'", collection_name)