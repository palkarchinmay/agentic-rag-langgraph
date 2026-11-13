"""
vector_store.py
---------------
ChromaDB vector store with:
  - Named collections (per-session or per-user isolation)
  - Upsert support (add to existing without full rebuild)
  - Content quality filter (blocks binary garbage before embedding)
  - Configurable embedding provider (OpenAI / Azure / Vertex / local)
"""

import os
import re
import logging
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from app.providers.embeddings.factory import get_embeddings

logger = logging.getLogger("financial_rag.vector_store")


def chroma_safe_name(name: str) -> str:
    """
    Map a logical collection name (e.g. 'user_jane@example.com') to a name
    ChromaDB accepts: 3-512 chars of [a-zA-Z0-9._-], starting and ending
    with an alphanumeric character, and no '..'.
    The logical name stays unchanged everywhere else in the app.
    """
    safe = name.replace("@", "_at_")
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", safe)
    safe = re.sub(r"\.{2,}", ".", safe)
    safe = safe.strip("._-") or "default"
    if len(safe) < 3:
        safe = safe + "_col"
    return safe[:512].rstrip("._-")

# ── Content quality filter ────────────────────────────────────────────────────
# Minimum ratio of printable ASCII chars — binary garbage fails this
_MIN_PRINTABLE_RATIO = 0.80
# Minimum meaningful content length after stripping section tags
_MIN_CONTENT_CHARS   = 30
# Patterns that indicate binary/garbage content
_BINARY_MARKERS      = re.compile(r"IHDR|sRGB|PLTE|IDAT|IEND|cmPP|tRNS")


def is_quality_chunk(content: str) -> bool:
    """
    Returns True if the chunk contains meaningful text worth indexing.
    Rejects: binary garbage, empty chunks, extremely short content.
    """
    if not content or not content.strip():
        return False

    # Strip section tags for length check
    clean = re.sub(r"\[(?:TEXT|TABLES|OCR|CHARTS|NOTES)\]", "", content).strip()
    if len(clean) < _MIN_CONTENT_CHARS:
        return False

    # Binary content check
    if _BINARY_MARKERS.search(content):
        return False

    # Printable ratio check
    printable = sum(1 for c in content if c.isprintable())
    if len(content) > 0 and printable / len(content) < _MIN_PRINTABLE_RATIO:
        return False

    return True


class ChromaVectorStore:
    """
    ChromaDB vector store wrapper.

    Args:
        collection_name     : isolates documents per user/session
        embedding_provider  : 'openai' | 'azure' | 'vertex' | 'huggingface'
        embedding_model     : model name for the provider
        persist_dir         : root directory for ChromaDB storage
    """

    def __init__(
        self,
        collection_name: str = "default",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        persist_dir: str = "chroma_db",
    ):
        self.collection_name    = collection_name
        self.chroma_name        = chroma_safe_name(collection_name)
        self.persist_dir        = persist_dir
        self.embedding_function = get_embeddings(embedding_provider, embedding_model)
        self.db: Optional[Chroma] = None

        # Auto-load if collection already exists
        collection_dir = os.path.join(persist_dir, collection_name)
        if os.path.exists(persist_dir):
            try:
                self._load()
                logger.info("Loaded existing collection '%s'", collection_name)
            except Exception as e:
                logger.debug("No existing collection '%s': %s", collection_name, e)

    # ── Internal load ─────────────────────────────────────────────────────────

    def _load(self):
        self.db = Chroma(
            collection_name=self.chroma_name,
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_function,
        )

    # ── Build from scratch ────────────────────────────────────────────────────

    def build(self, chunks: List[Dict]) -> int:
        """
        Index all chunks into a new (or replaced) collection.
        Returns number of chunks successfully indexed.
        """
        good_chunks = [c for c in chunks if is_quality_chunk(c["content"])]
        dropped     = len(chunks) - len(good_chunks)

        if dropped > 0:
            logger.warning("Quality filter dropped %d/%d chunks", dropped, len(chunks))

        if not good_chunks:
            logger.warning("No quality chunks to index")
            return 0

        texts     = [c["content"]  for c in good_chunks]
        metadatas = [c["metadata"] for c in good_chunks]

        self.db = Chroma.from_texts(
            texts=texts,
            embedding=self.embedding_function,
            metadatas=metadatas,
            collection_name=self.chroma_name,
            persist_directory=self.persist_dir,
        )

        logger.info("Built vector store: %d chunks in '%s'",
                    len(good_chunks), self.collection_name)
        return len(good_chunks)

    # ── Add to existing collection ────────────────────────────────────────────

    def upsert(self, chunks: List[Dict]) -> int:
        """
        Add new chunks to an existing collection without rebuilding.
        Creates collection if it doesn't exist yet.
        """
        good_chunks = [c for c in chunks if is_quality_chunk(c["content"])]
        if not good_chunks:
            return 0

        if self.db is None:
            return self.build(good_chunks)

        texts     = [c["content"]  for c in good_chunks]
        metadatas = [c["metadata"] for c in good_chunks]

        self.db.add_texts(texts=texts, metadatas=metadatas)
        logger.info("Upserted %d chunks into '%s'", len(good_chunks), self.collection_name)
        return len(good_chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5,
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Semantic similarity search.

        Args:
            query           : natural language query
            k               : number of results
            filter_metadata : optional ChromaDB metadata filter
                              e.g. {"source": "report.pdf"}

        Returns: list of {content, metadata, score}
                 score = L2 distance (lower = more similar)
        """
        if self.db is None:
            raise ValueError("Vector store not initialized. Call build() or upsert() first.")

        kwargs = {"query": query, "k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        results_with_scores = self.db.similarity_search_with_score(**kwargs)

        return [
            {
                "content":  doc.page_content,
                "metadata": doc.metadata,
                "score":    float(score),  # L2 distance
                "source":   "vector",
            }
            for doc, score in results_with_scores
        ]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return number of indexed chunks."""
        if self.db is None:
            return 0
        try:
            return self.db._collection.count()
        except Exception:
            return 0

    def delete_collection(self):
        """Wipe the collection (e.g. on session end or re-upload)."""
        if self.db is not None:
            try:
                self.db.delete_collection()
                self.db = None
                logger.info("Deleted collection '%s'", self.collection_name)
            except Exception as e:
                logger.warning("Could not delete collection: %s", e)

    @property
    def is_ready(self) -> bool:
        return self.db is not None and self.count() > 0
