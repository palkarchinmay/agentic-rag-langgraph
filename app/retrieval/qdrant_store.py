"""
app/retrieval/qdrant_store.py — Qdrant Cloud vector store.

DROP-IN REPLACEMENT for ChromaVectorStore in production.
Identical public API: build(), upsert(), search(), count(),
delete_collection(), is_ready property.

HOW THE ADAPTER WORKS
=====================
settings.py reads VECTOR_BACKEND from the environment:
  local / chroma  → ChromaVectorStore  (local dev, unchanged)
  qdrant          → QdrantVectorStore  (production)

Nothing else in the codebase changes. The factory function
get_vector_store() in vector_store_factory.py returns whichever
backend is configured.

QDRANT CONCEPTS
===============
Collection  → equivalent to a ChromaDB collection (per-user/session)
Point       → one vector + payload (metadata + content)
Payload     → arbitrary JSON stored alongside the vector (metadata)

AUTHENTICATION
==============
Local dev:  VECTOR_BACKEND=chroma → Qdrant not called at all
Production: VECTOR_BACKEND=qdrant
            QDRANT_URL    → cluster URL from Qdrant Cloud dashboard
            QDRANT_API_KEY → fetched from GCP Secret Manager at cold-start
                             via app/config/secrets.py (Step 1)
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
)

from app.retrieval.vector_store import is_quality_chunk
from app.providers.embeddings.factory import get_embeddings

logger = logging.getLogger("financial_rag.qdrant_store")


class QdrantVectorStore:
    """
    Qdrant Cloud vector store.

    Args:
        collection_name     : isolates documents per user/session
        embedding_provider  : 'openai' | 'azure' | 'vertex' | 'huggingface'
        embedding_model     : model name for the provider
        qdrant_url          : Qdrant Cloud cluster URL
        qdrant_api_key      : Qdrant Cloud API key
    """

    def __init__(
        self,
        collection_name: str = "default",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embedding_fn    = get_embeddings(embedding_provider, embedding_model)

        # Resolve connection params — env vars are fallback if not passed directly
        url     = qdrant_url     or os.getenv("QDRANT_URL", "")
        api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY", "")

        if not url:
            raise EnvironmentError(
                "QDRANT_URL is not set. Add it to your .env (local) or "
                "GCP Secret Manager as 'rag-qdrant-url' (production)."
            )

        self._client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        self._vector_size = self._detect_vector_size(embedding_provider, embedding_model)

        # Ensure collection exists
        self._ensure_collection()

        logger.info(
            "QdrantVectorStore ready: collection='%s' vector_size=%d url=%s",
            collection_name, self._vector_size, url[:40] + "..."
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_vector_size(self, provider: str, model: str) -> int:
        """Return embedding dimension based on provider/model."""
        size_map = {
            "text-embedding-3-small":    1536,
            "text-embedding-3-large":    3072,
            "text-embedding-ada-002":    1536,
            "text-embedding-gecko@003":  768,
            "all-MiniLM-L6-v2":         384,
            "all-mpnet-base-v2":         768,
        }
        size = size_map.get(model, 1536)
        logger.debug("Vector size for %s/%s: %d", provider, model, size)
        return size

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist yet."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in existing:
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,   # cosine similarity (higher = better)
                ),
            )
            logger.info("Created Qdrant collection: '%s'", self.collection_name)
        else:
            logger.debug("Qdrant collection already exists: '%s'", self.collection_name)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using the configured embedding function."""
        return self.embedding_fn.embed_documents(texts)

    # ── Public API (same as ChromaVectorStore) ────────────────────────────────

    def build(self, chunks: List[Dict]) -> int:
        """
        Index all chunks — wipes existing collection first.
        Use upsert() to add without wiping.
        Returns number of chunks indexed.
        """
        # Wipe first so build() is idempotent
        self.delete_collection()
        self._ensure_collection()
        return self.upsert(chunks)

    def upsert(self, chunks: List[Dict]) -> int:
        """
        Add new chunks without wiping existing data.
        Creates collection if it doesn't exist.
        Returns number of chunks successfully indexed.
        """
        good_chunks = [c for c in chunks if is_quality_chunk(c["content"])]
        dropped     = len(chunks) - len(good_chunks)

        if dropped > 0:
            logger.warning("Quality filter dropped %d/%d chunks", dropped, len(chunks))

        if not good_chunks:
            logger.warning("No quality chunks to index in '%s'", self.collection_name)
            return 0

        texts     = [c["content"]  for c in good_chunks]
        metadatas = [c["metadata"] for c in good_chunks]

        # Embed in batches of 100 to avoid token limits
        batch_size  = 100
        all_vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs  = self._embed_texts(batch)
            all_vectors.extend(vecs)
            logger.debug("Embedded batch %d/%d", i // batch_size + 1,
                         (len(texts) + batch_size - 1) // batch_size)

        # Build Qdrant points
        points = []
        for vector, text, metadata in zip(all_vectors, texts, metadatas):
            payload = {"content": text, **metadata}
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            ))

        # Upsert in batches of 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(
            "Upserted %d chunks into Qdrant collection '%s'",
            len(good_chunks), self.collection_name
        )
        return len(good_chunks)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Semantic similarity search.

        Args:
            query           : natural language query
            k               : number of results
            filter_metadata : optional metadata filter
                              e.g. {"source": "report.pdf"}
                              Maps to Qdrant payload filter.

        Returns: list of {content, metadata, score, source}
                 score = cosine similarity (higher = more similar, range 0-1)

        NOTE: qdrant-client >=1.10 deprecated .search() in favor of
        .query_points(). The note parameter is named differently
        (query= instead of query_vector=) and the response is wrapped
        in a .points attribute instead of being a bare list.
        """
        query_vector = self.embedding_fn.embed_query(query)

        # Build Qdrant filter if metadata filter provided
        qdrant_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=fk, match=MatchValue(value=fv))
                for fk, fv in filter_metadata.items()
            ]
            qdrant_filter = Filter(must=conditions)

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        results = response.points

        return [
            {
                "content":  r.payload.get("content", ""),
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
                "score":    float(r.score),   # cosine similarity (higher = better)
                "source":   "vector",
            }
            for r in results
        ]

    def count(self) -> int:
        """Return number of indexed points in the collection."""
        try:
            info = self._client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def delete_collection(self) -> None:
        """Wipe the collection entirely."""
        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if self.collection_name in existing:
                self._client.delete_collection(self.collection_name)
                logger.info("Deleted Qdrant collection '%s'", self.collection_name)
        except Exception as e:
            logger.warning("Could not delete Qdrant collection '%s': %s",
                           self.collection_name, e)

    @property
    def is_ready(self) -> bool:
        return self.count() > 0