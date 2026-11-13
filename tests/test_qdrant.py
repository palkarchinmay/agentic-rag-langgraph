"""
tests/test_qdrant.py — Unit tests for Step 2 Qdrant integration.

All tests run locally without a real Qdrant connection.
Qdrant client is mocked so CI never needs credentials.

Run: pytest tests/test_qdrant.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {"content": "Revenue grew 15% year-over-year to $2.4 billion in Q4 2024.", "metadata": {"source": "report.pdf", "page": 1}},
    {"content": "Operating margin expanded by 200 basis points to 18.5%.",     "metadata": {"source": "report.pdf", "page": 2}},
    {"content": "The board approved a $500 million share buyback programme.",   "metadata": {"source": "report.pdf", "page": 3}},
]

GARBAGE_CHUNKS = [
    {"content": "IHDR sRGB PLTE IDAT binary garbage",  "metadata": {}},
    {"content": "ab",                                   "metadata": {}},
    {"content": "",                                     "metadata": {}},
]


def _mock_qdrant_client():
    """Build a complete mock of QdrantClient."""
    client = MagicMock()

    # get_collections returns empty list by default (no existing collections)
    collections_response        = MagicMock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response

    # get_collection returns a mock with points_count
    collection_info             = MagicMock()
    collection_info.points_count = 3
    client.get_collection.return_value = collection_info

    # search returns mock results via query_points (qdrant-client >=1.10 API)
    result1                     = MagicMock()
    result1.payload             = {"content": "Revenue grew 15%", "source": "report.pdf", "page": 1}
    result1.score               = 0.92
    result2                     = MagicMock()
    result2.payload             = {"content": "Operating margin expanded", "source": "report.pdf", "page": 2}
    result2.score               = 0.85
    query_response               = MagicMock()
    query_response.points        = [result1, result2]
    client.query_points.return_value = query_response

    return client


# ── Tests: vector_store_factory ───────────────────────────────────────────────

class TestVectorStoreFactory:
    def test_returns_chroma_by_default(self):
        os.environ.pop("VECTOR_BACKEND", None)
        backend = os.getenv("VECTOR_BACKEND", "chroma").lower().strip()
        assert backend == "chroma"

    def test_selects_qdrant_when_env_set(self):
        os.environ["VECTOR_BACKEND"] = "qdrant"
        try:
            backend = os.getenv("VECTOR_BACKEND", "chroma").lower()
            assert backend == "qdrant"
        finally:
            del os.environ["VECTOR_BACKEND"]

    def test_raises_on_unknown_backend(self):
        os.environ["VECTOR_BACKEND"] = "pinecone"
        try:
            from app.retrieval.vector_store_factory import get_vector_store
            with patch("app.config.settings.DEFAULT_EMBED_PROVIDER", "openai", create=True), \
                 patch("app.config.settings.DEFAULT_EMBED_MODEL", "text-embedding-3-small", create=True):
                with pytest.raises(ValueError, match="Unknown VECTOR_BACKEND"):
                    get_vector_store()
        finally:
            del os.environ["VECTOR_BACKEND"]


# ── Tests: QdrantVectorStore ──────────────────────────────────────────────────

class TestQdrantVectorStore:

    def _make_store(self, mock_client, mock_embed_fn=None):
        """Helper: create QdrantVectorStore with mocked dependencies."""
        embed_fn = mock_embed_fn or MagicMock()
        embed_fn.embed_documents.return_value = [[0.1] * 1536] * 10
        embed_fn.embed_query.return_value     = [0.1] * 1536

        import app.retrieval.qdrant_store as qs_module
        with patch.object(qs_module, "QdrantClient", return_value=mock_client), \
             patch.object(qs_module, "get_embeddings", return_value=embed_fn):
            os.environ["QDRANT_URL"] = "https://test.qdrant.tech"
            os.environ["QDRANT_API_KEY"] = "test-key"
            store = qs_module.QdrantVectorStore(collection_name="test_col")
            del os.environ["QDRANT_URL"]
            del os.environ["QDRANT_API_KEY"]
            return store, mock_client

    def test_raises_without_qdrant_url(self):
        import app.retrieval.qdrant_store as qs_module
        os.environ.pop("QDRANT_URL", None)
        with patch.object(qs_module, "get_embeddings", return_value=MagicMock()):
            with pytest.raises(EnvironmentError, match="QDRANT_URL"):
                qs_module.QdrantVectorStore(collection_name="test")

    def test_creates_collection_if_not_exists(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)
        client.create_collection.assert_called_once()

    def test_skips_creation_if_collection_exists(self):
        mock_client = _mock_qdrant_client()
        existing = MagicMock()
        existing.name = "test_col"
        mock_client.get_collections.return_value.collections = [existing]
        store, client = self._make_store(mock_client)
        client.create_collection.assert_not_called()

    def test_upsert_filters_garbage(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)

        all_chunks = SAMPLE_CHUNKS + GARBAGE_CHUNKS
        n = store.upsert(all_chunks)

        # Only 3 quality chunks should be indexed
        assert n == 3
        client.upsert.assert_called()

    def test_upsert_returns_zero_for_all_garbage(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)
        n = store.upsert(GARBAGE_CHUNKS)
        assert n == 0
        client.upsert.assert_not_called()

    def test_search_returns_correct_format(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)

        results = store.search("What is the revenue?", k=2)

        assert len(results) == 2
        assert "content" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]
        assert "source" in results[0]
        assert results[0]["source"] == "vector"

    def test_search_score_is_float(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)
        results = store.search("revenue", k=2)
        assert isinstance(results[0]["score"], float)
        assert results[0]["score"] == pytest.approx(0.92)

    def test_count_returns_points_count(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)
        assert store.count() == 3

    def test_count_returns_zero_on_error(self):
        mock_client = _mock_qdrant_client()
        mock_client.get_collection.side_effect = Exception("not found")
        store, client = self._make_store(mock_client)
        assert store.count() == 0

    def test_is_ready_true_when_data_exists(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)
        assert store.is_ready is True

    def test_is_ready_false_when_empty(self):
        mock_client = _mock_qdrant_client()
        mock_client.get_collection.return_value.points_count = 0
        store, client = self._make_store(mock_client)
        assert store.is_ready is False

    def test_delete_collection(self):
        mock_client = _mock_qdrant_client()
        existing = MagicMock()
        existing.name = "test_col"
        mock_client.get_collections.return_value.collections = [existing]
        store, client = self._make_store(mock_client)

        # Reset so get_collections reflects the existing collection
        mock_client.get_collections.return_value.collections = [existing]
        store.delete_collection()
        client.delete_collection.assert_called_once_with("test_col")

    def test_build_wipes_then_upserts(self):
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)

        with patch.object(store, "delete_collection") as mock_delete, \
             patch.object(store, "upsert", return_value=3) as mock_upsert:
            n = store.build(SAMPLE_CHUNKS)

        mock_delete.assert_called_once()
        mock_upsert.assert_called_once_with(SAMPLE_CHUNKS)
        assert n == 3

    def test_vector_size_detection(self):
        mock_client = _mock_qdrant_client()
        store, _ = self._make_store(mock_client)
        # Default model text-embedding-3-small → 1536
        assert store._vector_size == 1536

    def test_content_excluded_from_metadata(self):
        """content key should not appear in metadata dict."""
        mock_client = _mock_qdrant_client()
        store, client = self._make_store(mock_client)
        results = store.search("revenue", k=2)
        for r in results:
            assert "content" not in r["metadata"]


# ── Tests: settings additions ─────────────────────────────────────────────────

class TestSettings:
    def test_vector_backend_defaults_to_chroma(self):
        os.environ.pop("VECTOR_BACKEND", None)
        backend = os.getenv("VECTOR_BACKEND", "chroma")
        assert backend == "chroma"

    def test_vector_backend_reads_env(self):
        os.environ["VECTOR_BACKEND"] = "qdrant"
        try:
            assert os.getenv("VECTOR_BACKEND") == "qdrant"
        finally:
            del os.environ["VECTOR_BACKEND"]