"""
app/retrieval/vector_store_factory.py — Adapter factory for vector stores.

USAGE
=====
    from app.retrieval.vector_store_factory import get_vector_store

    # Returns ChromaVectorStore locally, QdrantVectorStore in production
    store = get_vector_store(collection_name="user_alice")
    store.build(chunks)
    results = store.search("what is the revenue?", k=5)

SWITCHING BACKENDS
==================
Set VECTOR_BACKEND in your environment:

    VECTOR_BACKEND=chroma   → ChromaDB (local dev, default)
    VECTOR_BACKEND=qdrant   → Qdrant Cloud (production)

No other code changes needed anywhere in the app.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Union

logger = logging.getLogger("financial_rag.vector_store_factory")


def get_vector_store(
    collection_name: str = "default",
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
):
    """
    Return the correct vector store based on VECTOR_BACKEND env var.

    Args:
        collection_name    : isolates documents per user/session
        embedding_provider : overrides DEFAULT_EMBED_PROVIDER from settings
        embedding_model    : overrides DEFAULT_EMBED_MODEL from settings

    Returns:
        ChromaVectorStore  (VECTOR_BACKEND=chroma or unset)
        QdrantVectorStore  (VECTOR_BACKEND=qdrant)
    """
    from app.config.settings import DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL

    provider = embedding_provider or DEFAULT_EMBED_PROVIDER
    model    = embedding_model    or DEFAULT_EMBED_MODEL
    backend  = os.getenv("VECTOR_BACKEND", "chroma").lower().strip()

    if backend in ("chroma", "local", ""):
        logger.debug("Vector backend: ChromaDB (collection=%s)", collection_name)
        from app.retrieval.vector_store import ChromaVectorStore
        from app.config.settings import CHROMA_PERSIST
        return ChromaVectorStore(
            collection_name=collection_name,
            embedding_provider=provider,
            embedding_model=model,
            persist_dir=CHROMA_PERSIST,
        )

    if backend == "qdrant":
        logger.debug("Vector backend: Qdrant Cloud (collection=%s)", collection_name)
        from app.retrieval.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(
            collection_name=collection_name,
            embedding_provider=provider,
            embedding_model=model,
        )

    raise ValueError(
        f"Unknown VECTOR_BACKEND={backend!r}. "
        "Valid values: 'chroma' (default) or 'qdrant'."
    )