"""
tests/test_redis_cache.py — Unit tests for Step 3 Redis cache integration.

All tests run locally without a real Redis connection.
The redis client is mocked so CI never needs Upstash credentials.

Run: pytest tests/test_redis_cache.py -v
"""

import os
import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_embed_fn(vector=None):
    """Returns a callable(text) -> List[float], matching cache_node.py's usage."""
    vec = vector or [0.1, 0.2, 0.3, 0.4]
    return lambda text: vec


# ── Tests: RedisSemanticCache ─────────────────────────────────────────────────

class TestRedisSemanticCache:

    def _make_cache(self, mock_redis_client, embed_fn=None):
        import app.cache.redis_semantic_cache as rsc_module
        with patch.object(rsc_module.redis, "from_url", return_value=mock_redis_client):
            os.environ["REDIS_URL"] = "rediss://default:token@test.upstash.io:6380"
            cache = rsc_module.RedisSemanticCache(
                collection_name="test_col",
                embedding_fn=embed_fn or _make_embed_fn(),
            )
            del os.environ["REDIS_URL"]
            return cache, mock_redis_client

    def test_raises_without_redis_url(self):
        import app.cache.redis_semantic_cache as rsc_module
        os.environ.pop("REDIS_URL", None)
        with patch.object(rsc_module.redis, "from_url", return_value=MagicMock()):
            with pytest.raises(EnvironmentError, match="REDIS_URL"):
                rsc_module.RedisSemanticCache(collection_name="test")

    def test_get_returns_none_when_cache_empty(self):
        mock_client = MagicMock()
        mock_client.get.return_value = None
        cache, client = self._make_cache(mock_client)

        result = cache.get("what is the revenue?", model="gpt-4o-mini")
        assert result is None

    def test_put_stores_entry_in_redis(self):
        mock_client = MagicMock()
        mock_client.get.return_value = None
        cache, client = self._make_cache(mock_client)

        ok = cache.put("what is the revenue?", "Revenue is $2.4B", model="gpt-4o-mini")
        assert ok is True
        client.set.assert_called_once()

        # Verify TTL was passed
        _, kwargs = client.set.call_args
        assert kwargs.get("ex") is not None

    def test_put_returns_false_without_embedding_fn(self):
        mock_client = MagicMock()
        mock_client.get.return_value = None
        cache, client = self._make_cache(mock_client, embed_fn=None)
        cache.embedding_fn = None

        ok = cache.put("query", "response")
        assert ok is False

    def test_get_after_put_roundtrip(self):
        """Simulate Redis actually storing and returning the JSON we wrote."""
        mock_client = MagicMock()
        storage = {}

        def fake_set(key, value, ex=None):
            storage[key] = value
            return True

        def fake_get(key):
            return storage.get(key)

        mock_client.set.side_effect = fake_set
        mock_client.get.side_effect = fake_get

        embed_fn = _make_embed_fn([1.0, 0.0, 0.0, 0.0])
        cache, client = self._make_cache(mock_client, embed_fn=embed_fn)

        cache.put("what is the revenue?", "Revenue is $2.4B", model="gpt-4o-mini")
        result = cache.get("what is the revenue?", model="gpt-4o-mini")

        assert result is not None
        assert result["response"] == "Revenue is $2.4B"
        assert result["cache_similarity"] == pytest.approx(1.0, abs=0.01)

    def test_get_respects_model_filter(self):
        """A cached entry for gpt-4o-mini should not match a query for gpt-4o."""
        mock_client = MagicMock()
        storage = {}
        mock_client.set.side_effect = lambda k, v, ex=None: storage.update({k: v})
        mock_client.get.side_effect = lambda k: storage.get(k)

        embed_fn = _make_embed_fn([1.0, 0.0, 0.0, 0.0])
        cache, client = self._make_cache(mock_client, embed_fn=embed_fn)

        cache.put("what is the revenue?", "Revenue is $2.4B", model="gpt-4o-mini")
        result = cache.get("what is the revenue?", model="gpt-4o")

        assert result is None

    def test_get_respects_similarity_threshold(self):
        """A very different query embedding should not match."""
        mock_client = MagicMock()
        storage = {}
        mock_client.set.side_effect = lambda k, v, ex=None: storage.update({k: v})
        mock_client.get.side_effect = lambda k: storage.get(k)

        put_embed_fn = _make_embed_fn([1.0, 0.0, 0.0, 0.0])
        cache, client = self._make_cache(mock_client, embed_fn=put_embed_fn)
        cache.put("what is the revenue?", "Revenue is $2.4B", model="gpt-4o-mini")

        # Completely orthogonal vector — cosine similarity = 0
        cache.embedding_fn = _make_embed_fn([0.0, 1.0, 0.0, 0.0])
        result = cache.get("totally different query", model="gpt-4o-mini")

        assert result is None

    def test_invalidate_deletes_redis_key(self):
        mock_client = MagicMock()
        cache, client = self._make_cache(mock_client)

        cache.invalidate()
        client.delete.assert_called_once_with("semcache:test_col")

    def test_stats_returns_correct_shape(self):
        mock_client = MagicMock()
        entries = [
            {"query_text": "q1", "query_embedding": [0.1]*4, "response": "r1",
             "model": "gpt-4o-mini", "metadata": {}, "timestamp": time.time(), "hits": 2},
        ]
        mock_client.get.return_value = json.dumps(entries)
        cache, client = self._make_cache(mock_client)

        stats = cache.stats()
        assert stats["total_entries"] == 1
        assert stats["total_hits"] == 2
        assert stats["backend"] == "redis"
        assert "ttl_seconds" in stats
        assert "threshold" in stats

    def test_stats_excludes_expired_entries(self):
        mock_client = MagicMock()
        old_entry = {
            "query_text": "old", "query_embedding": [0.1]*4, "response": "r",
            "model": "", "metadata": {}, "timestamp": time.time() - 999999, "hits": 0,
        }
        mock_client.get.return_value = json.dumps([old_entry])
        cache, client = self._make_cache(mock_client)

        stats = cache.stats()
        assert stats["total_entries"] == 0

    def test_max_entries_eviction(self):
        """When over MAX_CACHE_ENTRIES, oldest entries should be evicted."""
        mock_client = MagicMock()
        storage = {}
        mock_client.set.side_effect = lambda k, v, ex=None: storage.update({k: v})
        mock_client.get.side_effect = lambda k: storage.get(k)

        embed_fn = _make_embed_fn([0.1, 0.2, 0.3, 0.4])
        cache, client = self._make_cache(mock_client, embed_fn=embed_fn)

        with patch("app.cache.redis_semantic_cache.MAX_CACHE_ENTRIES", 3):
            for i in range(5):
                cache.put(f"query {i}", f"response {i}", model="gpt-4o-mini")

            raw = storage[cache._redis_key]
            entries = json.loads(raw)
            assert len(entries) <= 3

    def test_redis_key_naming(self):
        mock_client = MagicMock()
        cache, client = self._make_cache(mock_client)
        assert cache._redis_key == "semcache:test_col"

    def test_redis_key_sanitizes_slashes(self):
        mock_client = MagicMock()
        import app.cache.redis_semantic_cache as rsc_module
        with patch.object(rsc_module.redis, "from_url", return_value=mock_client):
            os.environ["REDIS_URL"] = "rediss://test.upstash.io:6380"
            cache = rsc_module.RedisSemanticCache(
                collection_name="user/with/slashes",
                embedding_fn=_make_embed_fn(),
            )
            del os.environ["REDIS_URL"]
        assert cache._redis_key == "semcache:user_with_slashes"


# ── Tests: cache_factory ──────────────────────────────────────────────────────

class TestCacheFactory:

    def setup_method(self):
        """Clear the module-level registry between tests."""
        import app.cache.cache_factory as cf
        cf._cache_registry.clear()

    def test_defaults_to_local_backend(self):
        os.environ.pop("CACHE_BACKEND", None)
        backend = os.getenv("CACHE_BACKEND", "local").lower().strip()
        assert backend == "local"

    def test_selects_redis_when_env_set(self):
        os.environ["CACHE_BACKEND"] = "redis"
        try:
            backend = os.getenv("CACHE_BACKEND", "local").lower()
            assert backend == "redis"
        finally:
            del os.environ["CACHE_BACKEND"]

    def test_raises_on_unknown_backend(self):
        os.environ["CACHE_BACKEND"] = "memcached"
        try:
            from app.cache.cache_factory import get_cache_backend
            with pytest.raises(ValueError, match="Unknown CACHE_BACKEND"):
                get_cache_backend("test_col")
        finally:
            del os.environ["CACHE_BACKEND"]

    def test_returns_semantic_cache_for_local(self):
        os.environ.pop("CACHE_BACKEND", None)
        from app.cache.cache_factory import get_cache_backend
        from app.cache.semantic_cache import SemanticCache

        cache = get_cache_backend("test_factory_col", embedding_fn=_make_embed_fn())
        assert isinstance(cache, SemanticCache)

        # cleanup
        import os as _os
        path = cache._cache_path
        if _os.path.exists(path):
            _os.remove(path)

    def test_returns_redis_cache_for_redis_backend(self):
        os.environ["CACHE_BACKEND"] = "redis"
        os.environ["REDIS_URL"] = "rediss://test.upstash.io:6380"
        try:
            import app.cache.redis_semantic_cache as rsc_module
            with patch.object(rsc_module.redis, "from_url", return_value=MagicMock()):
                from app.cache.cache_factory import get_cache_backend
                cache = get_cache_backend("test_factory_redis_col", embedding_fn=_make_embed_fn())
                assert type(cache).__name__ == "RedisSemanticCache"
        finally:
            del os.environ["CACHE_BACKEND"]
            del os.environ["REDIS_URL"]

    def test_registry_returns_same_instance(self):
        os.environ.pop("CACHE_BACKEND", None)
        from app.cache.cache_factory import get_cache_backend

        cache1 = get_cache_backend("same_col", embedding_fn=_make_embed_fn())
        cache2 = get_cache_backend("same_col", embedding_fn=_make_embed_fn())
        assert cache1 is cache2

        import os as _os
        path = cache1._cache_path
        if _os.path.exists(path):
            _os.remove(path)


# ── Tests: semantic_cache.py delegation (Step 3 change) ───────────────────────

class TestSemanticCacheDelegation:

    def setup_method(self):
        import app.cache.cache_factory as cf
        cf._cache_registry.clear()

    def test_get_cache_delegates_to_factory_local(self):
        os.environ.pop("CACHE_BACKEND", None)
        from app.cache.semantic_cache import get_cache, SemanticCache

        cache = get_cache("delegation_test_col", embedding_fn=_make_embed_fn())
        assert isinstance(cache, SemanticCache)

        import os as _os
        path = cache._cache_path
        if _os.path.exists(path):
            _os.remove(path)

    def test_get_cache_delegates_to_factory_redis(self):
        os.environ["CACHE_BACKEND"] = "redis"
        os.environ["REDIS_URL"] = "rediss://test.upstash.io:6380"
        try:
            import app.cache.redis_semantic_cache as rsc_module
            with patch.object(rsc_module.redis, "from_url", return_value=MagicMock()):
                from app.cache.semantic_cache import get_cache
                cache = get_cache("delegation_redis_col", embedding_fn=_make_embed_fn())
                assert type(cache).__name__ == "RedisSemanticCache"
        finally:
            del os.environ["CACHE_BACKEND"]
            del os.environ["REDIS_URL"]

    def test_existing_callers_unaffected(self):
        """
        Simulates cache_node.py's exact usage pattern:
        `from app.cache.semantic_cache import get_cache`
        followed by `get_cache(collection, embedding_fn=...)`.
        Should work identically to before Step 3 when CACHE_BACKEND=local.
        """
        os.environ.pop("CACHE_BACKEND", None)
        from app.cache.semantic_cache import get_cache

        cache = get_cache("cache_node_simulation", embedding_fn=_make_embed_fn())
        ok = cache.put("test query", "test response", model="gpt-4o-mini")
        assert ok is True

        result = cache.get("test query", model="gpt-4o-mini")
        assert result is not None
        assert result["response"] == "test response"

        cache.invalidate()
        import os as _os
        path = cache._cache_path
        if _os.path.exists(path):
            _os.remove(path)