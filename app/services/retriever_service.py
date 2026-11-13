"""
app/services/retriever_service.py — Thread-safe retriever registry.

CHANGE from original: disk fallback now uses get_vector_store() factory
instead of hardcoding ChromaDB. This means:
  - VECTOR_BACKEND=chroma → loads from ChromaDB (local, unchanged)
  - VECTOR_BACKEND=qdrant → loads from Qdrant Cloud (production)

The registry itself is unchanged — HybridRetriever instances in memory,
keyed by collection_name.
"""

import logging
import os
import threading
from typing import Dict, Optional

logger   = logging.getLogger("financial_rag.retriever_service")
_lock    = threading.Lock()
_registry: Dict[str, object] = {}


def register_retriever(collection_name: str, retriever) -> None:
    with _lock:
        _registry[collection_name] = retriever
    logger.info("Retriever registered: '%s'", collection_name)


def get_retriever(collection_name: str = "default") -> Optional[object]:
    """
    Get retriever from registry. If not found, attempt to load from
    the configured vector backend (ChromaDB or Qdrant).
    """
    with _lock:
        retriever = _registry.get(collection_name)

    if retriever is not None:
        return retriever

    logger.warning(
        "No retriever registered for '%s' — attempting load from %s",
        collection_name,
        os.getenv("VECTOR_BACKEND", "chroma")
    )
    retriever = _load_from_backend(collection_name)
    if retriever:
        with _lock:
            _registry[collection_name] = retriever
        logger.info("Retriever loaded from backend: '%s'", collection_name)
    else:
        logger.warning("No retriever found for collection '%s'", collection_name)

    return retriever


def _load_from_backend(collection_name: str) -> Optional[object]:
    """
    Load a HybridRetriever from the configured vector backend.
    ChromaDB: checks if collection exists on disk first.
    Qdrant:   checks if collection exists in Qdrant Cloud.
    """
    try:
        from app.retrieval.hybrid_retriever import HybridRetriever
        from app.retrieval.vector_store_factory import get_vector_store

        backend = os.getenv("VECTOR_BACKEND", "chroma").lower()

        if backend == "qdrant":
            # Check if collection exists in Qdrant
            from app.retrieval.qdrant_store import QdrantVectorStore
            store = get_vector_store(collection_name=collection_name)
            if store.count() == 0:
                logger.warning(
                    "Qdrant collection '%s' is empty or doesn't exist",
                    collection_name
                )
                return None
        else:
            # ChromaDB: check if collection exists on disk
            import chromadb
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
            client = chromadb.PersistentClient(path=persist_dir)
            existing = [c.name for c in client.list_collections()]
            from app.retrieval.vector_store import chroma_safe_name
            if chroma_safe_name(collection_name) not in existing:
                logger.warning(
                    "ChromaDB collection '%s' not found on disk",
                    collection_name
                )
                return None

        retriever = HybridRetriever(collection_name=collection_name)
        retriever.load_existing()
        return retriever

    except Exception as e:
        logger.error("Failed to load retriever from backend: %s", e)
        return None


def list_collections() -> list:
    with _lock:
        return list(_registry.keys())


def unregister_retriever(collection_name: str) -> None:
    with _lock:
        if collection_name in _registry:
            del _registry[collection_name]
            logger.info("Retriever unregistered: '%s'", collection_name)