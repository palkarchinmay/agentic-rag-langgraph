"""
api/routers/index.py — POST /api/v1/index

Ingests documents and builds/updates the vector index.
Reuses existing:
  - app/ingestion/loader.py   → load_documents()
  - app/ingestion/chunker.py  → chunk_documents()
  - app/retrieval/hybrid_retriever.py → HybridRetriever
  - app/services/retriever_service.py → register_retriever()
  - app/cache/semantic_cache.py       → invalidate_cache()

Accepts multipart file upload (same files as Streamlit upload widget).
"""

import io
import logging
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from api.middleware.auth import require_auth
from app.config.settings import CHROMA_PERSIST

logger = logging.getLogger("financial_rag.api.index")
router = APIRouter(prefix="/api/v1", tags=["index"])


# ── Response model ────────────────────────────────────────────────────────────

class IndexResponse(BaseModel):
    status:          str           # "indexed" | "error"
    collection_name: str
    files_indexed:   List[str]
    chunks_total:    int
    vectors_stored:  int
    elapsed_seconds: float
    trace_id:        str
    message:         str = ""

    # Aggregate detection signals -- powers the post-upload settings
    # suggestion feature. Detection always runs regardless of settings
    # (see chart_interpreter.py's process_image_region), so these are
    # accurate even when use_vision/caption_images were off.
    total_sections:      int = 0    # pages/slides/sections parsed
    scanned_sections:    int = 0    # of those, how many were scanned
    ocr_method_used:     str = ""
    charts_detected:     int = 0    # sections containing >=1 chart-like image
    charts_interpreted:  int = 0    # of those, how many were actually interpreted
    images_detected:     int = 0    # sections containing >=1 non-chart image
    images_captioned:    int = 0    # of those, how many were actually captioned
    use_vision_used:      bool = False
    caption_images_used:  bool = False


# ── Helpers (copied from upload_panel logic, no Streamlit deps) ───────────────

class NamedBytesIO(io.BytesIO):
    """BytesIO with a .name attribute — matches what load_documents() expects."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Index documents",
    description=(
        "Upload one or more files (PDF, DOCX, PPTX, TXT) and build "
        "a hybrid vector+BM25 index. Call `/api/v1/query` with the returned "
        "`collection_name` to search these documents."
    ),
)
async def index_endpoint(
    files: List[UploadFile] = File(..., description="Files to index (PDF/DOCX/PPTX/TXT)"),
    current_user: dict = Depends(require_auth),

    # Collection — defaults to JWT username so users can't index into others' collections
    username:   str = Form(None, description="User identifier → collection key (defaults to JWT user)"),

    # Ingestion settings
    ocr_method:   str  = Form("easyocr"),
    use_vision:   bool = Form(False),

    # General image captioning -- separate opt-in from use_vision, since
    # this is the path with real per-image API cost (see
    # app/ingestion/image_captioner.py). use_vision gates chart
    # interpretation only (free, local models); this gates VLM captioning.
    caption_images:   bool = Form(False),
    caption_provider: str  = Form("openai"),
    caption_model:    str  = Form("gpt-4o-mini"),

    # Chunking settings
    chunk_size:    int = Form(500),
    chunk_overlap: int = Form(50),
    chunk_method:  str = Form("recursive"),

    # Embedding settings
    embedding_provider: str = Form("openai"),
    embedding_model:    str = Form("text-embedding-3-small"),
) -> IndexResponse:

    trace_id = str(uuid.uuid4())[:8]
    # UI names ('tesseract', 'google_docai') -> internal OCR keys
    from app.ingestion.ocr import normalize_method
    ocr_method = normalize_method(ocr_method)
    # Use JWT username by default — prevents users indexing into others' collections
    effective_username = username or current_user.get("sub", "default")
    collection_name    = f"user_{effective_username}"
    file_names         = [f.filename for f in files]

    logger.info("[%s] index request: files=%s collection=%s",
                trace_id, file_names, collection_name)

    # ── Validate file types ───────────────────────────────────────────────────
    from app.ingestion.loader import supported_extensions
    # supported_extensions() returns [".pdf", ".docx", ...] with leading dots
    allowed_with_dot = supported_extensions()
    allowed_no_dot   = [e.lstrip(".") for e in allowed_with_dot]
    for f in files:
        ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
        if ext not in allowed_no_dot:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Unsupported file type '.{ext}'. Allowed: {sorted(allowed_with_dot)}",
            )

    t0 = time.perf_counter()

    # ── Read files into memory ────────────────────────────────────────────────
    file_objects = []
    for upload in files:
        data = await upload.read()
        file_objects.append(NamedBytesIO(data, upload.filename))
        logger.info("[%s] read %s (%d bytes)", trace_id, upload.filename, len(data))

    # ── Parse documents ───────────────────────────────────────────────────────
    try:
        from app.ingestion.loader import load_documents
        docs = load_documents(
            file_objects,
            ocr_method=ocr_method,
            use_vision=use_vision,
            max_workers=4,
            caption_images=caption_images,
            caption_provider=caption_provider,
            caption_model=caption_model,
        )
    except Exception as e:
        logger.error("[%s] document parsing failed: %s", trace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"Document parsing failed: {e}", "trace_id": trace_id},
        )

    if not docs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "No content extracted. Files may be empty, "
                         "password-protected, or corrupted.",
                "trace_id": trace_id,
            },
        )

    logger.info("[%s] parsed %d sections from %d file(s)",
                trace_id, len(docs), len(files))

    # ── Aggregate detection signals (for the settings-suggestion feature) ─────
    # Computed from metadata that's always populated regardless of
    # use_vision/caption_images settings — detection runs unconditionally
    # in chart_interpreter.py's process_image_region, only the expensive
    # interpretation/captioning calls are gated by the settings.
    total_sections     = len(docs)
    scanned_sections   = sum(1 for d in docs if d.metadata.get("scanned"))
    charts_detected     = sum(1 for d in docs if d.metadata.get("has_charts"))
    charts_interpreted  = sum(1 for d in docs if d.metadata.get("charts_interpreted"))
    images_detected     = sum(1 for d in docs if d.metadata.get("has_images"))
    images_captioned    = sum(1 for d in docs if d.metadata.get("images_captioned"))
    logger.info(
        "[%s] detection: scanned=%d/%d charts=%d(interpreted=%d) images=%d(captioned=%d)",
        trace_id, scanned_sections, total_sections, charts_detected,
        charts_interpreted, images_detected, images_captioned,
    )

    # ── Chunk documents ───────────────────────────────────────────────────────
    try:
        from app.ingestion.chunker import chunk_documents
        chunks = chunk_documents(
            docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            method=chunk_method,
        )
    except Exception as e:
        logger.error("[%s] chunking failed: %s", trace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Chunking failed: {e}", "trace_id": trace_id},
        )

    logger.info("[%s] chunked into %d chunks", trace_id, len(chunks))

    # ── Build / update vector index ───────────────────────────────────────────
    try:
        from app.retrieval.hybrid_retriever import HybridRetriever
        from app.services.retriever_service import (
            get_retriever, register_retriever
        )

        retriever = HybridRetriever(
            collection_name=collection_name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            persist_dir=CHROMA_PERSIST,
        )

        # If collection already exists → add docs, else build fresh
        existing = get_retriever(collection_name)
        if existing:
            try:
                counts = retriever.add_documents(chunks)
                logger.info("[%s] added to existing index: %s", trace_id, counts)
            except Exception as dim_err:
                if "dimension" in str(dim_err).lower():
                    # Embedding dimension mismatch — rebuild with new provider
                    logger.warning(
                        "[%s] Embedding dimension mismatch — rebuilding index "
                        "with new embedding provider '%s'",
                        trace_id, embedding_provider
                    )
                    # Clear old collection and rebuild
                    retriever.vector_store.delete_collection()
                    counts = retriever.build_index(chunks)
                    logger.info("[%s] rebuilt index: %s", trace_id, counts)
                else:
                    raise
        else:
            counts = retriever.build_index(chunks)
            logger.info("[%s] built new index: %s", trace_id, counts)

        register_retriever(collection_name, retriever)

    except Exception as e:
        logger.error("[%s] indexing failed: %s", trace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Indexing failed: {e}", "trace_id": trace_id},
        )

    # ── Invalidate semantic cache (stale answers no longer valid) ─────────────
    try:
        from app.cache.semantic_cache import invalidate_cache
        invalidate_cache(collection_name)
        logger.info("[%s] cache invalidated for '%s'", trace_id, collection_name)
    except Exception as e:
        logger.warning("[%s] cache invalidation failed (non-critical): %s", trace_id, e)

    elapsed = round(time.perf_counter() - t0, 2)
    vectors  = counts.get("vectors", len(chunks)) if isinstance(counts, dict) else len(chunks)

    logger.info("[%s] indexing complete in %.1fs — %d chunks, %d vectors",
                trace_id, elapsed, len(chunks), vectors)

    return IndexResponse(
        status=          "indexed",
        collection_name= collection_name,
        files_indexed=   file_names,
        chunks_total=    len(chunks),
        vectors_stored=  vectors,
        elapsed_seconds= elapsed,
        trace_id=        trace_id,
        message=         f"Successfully indexed {len(files)} file(s) into '{collection_name}'",
        total_sections=      total_sections,
        scanned_sections=    scanned_sections,
        ocr_method_used=     ocr_method,
        charts_detected=     charts_detected,
        charts_interpreted=  charts_interpreted,
        images_detected=     images_detected,
        images_captioned=    images_captioned,
        use_vision_used=     use_vision,
        caption_images_used= caption_images,
    )


@router.delete(
    "/index/{collection_name}",
    summary="Clear index",
    description="Delete a collection and its cache. User must re-index documents after this.",
)
async def delete_index(
    collection_name: str,
    current_user: dict = Depends(require_auth),
) -> dict:
    trace_id = str(uuid.uuid4())[:8]

    # Only allow users to delete their own collection
    # Admins can delete any collection
    # NOTE: collections are named user_{email}, NOT user_{uid} -- this
    # previously compared against current_user.get('sub', '') (the raw
    # Firebase uid), which would never match the actual collection name
    # and would 403 any non-admin user trying to delete their own data.
    expected = f"user_{current_user.get('email', '')}"
    if current_user.get("role") != "admin" and collection_name != expected:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=403,
            detail="You can only delete your own collection"
        )

    logger.info("[%s] delete index: collection=%s user=%s",
                trace_id, collection_name, current_user.get("sub"))

    deleted = False

    # Delete from the ACTUAL configured vector store backend.
    # BUG FIXED (2026-08-14): this used to hardcode a raw
    # chromadb.PersistentClient(...).delete_collection(...) call
    # regardless of VECTOR_BACKEND -- meaning on a Qdrant-backed
    # deployment (confirmed via every real indexing/query log this
    # session), this silently deleted nothing at all: ChromaDB correctly
    # reported "collection does not exist" (true, since Chroma was never
    # used), returned success anyway, and the real Qdrant data was left
    # completely untouched. Every /clear + re-upload cycle was silently
    # ADDING to the same ever-growing collection instead of replacing it.
    try:
        from app.retrieval.vector_store_factory import get_vector_store
        store = get_vector_store(collection_name=collection_name)
        store.delete_collection()
        deleted = True
        logger.info("[%s] deleted vector store collection '%s'", trace_id, collection_name)
    except Exception as e:
        logger.warning("[%s] vector store delete: %s", trace_id, e)

    # Delete the BM25 index too -- a completely separate persisted
    # structure (chroma_db/bm25_{collection}.pkl), not covered by the
    # vector store deletion above at all. Confirmed missing before this
    # fix: BM25's chunk count kept growing across every "cleared" session
    # (303 -> 360 -> ...) even once the vector store side was fixed.
    try:
        from app.retrieval.bm25 import BM25Retriever
        from app.config.settings import CHROMA_PERSIST
        bm25 = BM25Retriever(collection_name=collection_name, persist_dir=CHROMA_PERSIST)
        bm25.delete()
        logger.info("[%s] deleted BM25 index '%s'", trace_id, collection_name)
    except Exception as e:
        logger.warning("[%s] BM25 delete: %s", trace_id, e)

    # Unregister retriever from memory
    try:
        from app.services.retriever_service import unregister_retriever
        unregister_retriever(collection_name)
    except Exception:
        pass

    # Invalidate semantic cache
    try:
        from app.cache.semantic_cache import invalidate_cache
        invalidate_cache(collection_name)
    except Exception as e:
        logger.warning("[%s] cache invalidate: %s", trace_id, e)

    return {
        "status":          "deleted" if deleted else "not_found",
        "collection_name": collection_name,
        "trace_id":        trace_id,
    }