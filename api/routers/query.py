"""
api/routers/query.py — POST /api/v1/query

Calls the existing LangGraph pipeline (graph.invoke) exactly the same
way chat_panel.py does — zero pipeline code is duplicated.

Request:  QueryRequest  (query + config)
Response: QueryResponse (response + metadata)

STEP 4 ADDITIONS:
  - conversation_id (optional) on QueryRequest/QueryResponse — when
    provided, server fetches/builds chat history from Postgres instead
    of requiring the client to resend it. Backward compatible: clients
    that keep sending chat_history directly (e.g. existing Streamlit
    chat_panel.py) are completely unaffected.
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from api.middleware.auth import require_auth

logger = logging.getLogger("financial_rag.api.query")
router = APIRouter(prefix="/api/v1", tags=["query"])


# â”€â”€ Request / Response models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000,
                       description="The question to ask")

    # LLM settings — mirrors sidebar config
    llm_provider:  str   = Field("openai",         description="LLM provider")
    llm_model:     str   = Field("gpt-4o-mini",    description="Model identifier")
    temperature:   float = Field(0.0, ge=0.0, le=1.0)

    # Retrieval settings
    collection_name:  str  = Field("default",  description="ChromaDB collection (user's indexed docs)")
    top_k:            int  = Field(5, ge=1, le=20)
    retrieval_mode:   str  = Field("hybrid",   description="hybrid | vector | bm25")
    rerank:           bool = Field(True)

    # Embedding settings
    embedding_provider: str = Field("openai")
    embedding_model:    str = Field("text-embedding-3-small")

    # Domain / persona
    domain:         str = Field("uploaded documents")
    assistant_name: str = Field("Document Assistant")
    user_role:      str = Field("user", description="user | engineer | admin")

    # Chat history (optional, for multi-turn)
    chat_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous messages [{role, content}, ...]"
    )

    # Conversation persistence (Step 4, optional)
    conversation_id: Optional[str] = Field(
        None,
        description=(
            "If provided, server fetches conversation history from "
            "Postgres automatically (token-aware sliding window) instead "
            "of requiring chat_history to be sent. Omit conversation_id "
            "on the FIRST message and the server returns a new "
            "conversation_id in the response for the client to reuse on "
            "subsequent turns."
        )
    )

    # Cache control
    skip_cache: bool = Field(
        False,
        description=(
            "If true, bypass the semantic cache lookup entirely and force "
            "a fresh retrieval + generation. Used by golden dataset "
            "evaluation so scores reflect the current pipeline, not a "
            "stale cached answer from a prior test run. Normal chat "
            "queries should leave this false to get the cache's cost/"
            "latency benefit."
        )
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the test recall mentioned in the document?",
                "llm_provider": "openai",
                "llm_model": "gpt-4o-mini",
                "collection_name": "user_admin",
                "top_k": 5,
            }
        }


class CitationModel(BaseModel):
    label:    str
    source:   str
    location: str = ""
    type:     str = ""


class ChunkModel(BaseModel):
    label:   str    # matches CitationModel.label for easy lookup
    content: str    # actual retrieved text
    score:   float = 0.0
    rerank_score: Optional[float] = None   # cross-encoder score; None if rerank was off


class UsageModel(BaseModel):
    provider:           str   = ""
    model:              str   = ""
    prompt_tokens:      int   = 0
    completion_tokens:  int   = 0
    total_tokens:       int   = 0
    estimated_cost_usd: float = 0.0
    latency_seconds:    float = 0.0


class QueryResponse(BaseModel):
    # Core answer
    response:   str
    query_type: str = Field(description="rag | general | calculation | blocked")

    # Source attribution
    citations: List[CitationModel] = []

    # Performance
    latency_seconds: float
    cache_hit:       bool = False

    # Step 4 diagnostic: True when cache was bypassed because this turn
    # had conversation history (prevents cross-conversation cache collisions)
    conversation_aware: bool = False

    # Token usage & cost
    usage: UsageModel = Field(default_factory=UsageModel)

    # Quality signals
    faithfulness:     Optional[float] = None
    answer_relevancy: Optional[float] = None

    # Debug (included when user_role is engineer or admin)
    retrieved_chunks_count: int = 0
    router_method:          str = ""
    blocked:                bool = False
    block_reason:           str = ""
    error:                  str = ""

    # Retrieved chunks with content (top 5)
    retrieved_chunks: List[ChunkModel] = []

    # RAGAS evaluation (may be pending — poll using eval_key)
    ragas: dict = {}

    # Whether answer came from documents or general knowledge
    answered_from: str = ""   # "" | "general_fallback"

    # Trace ID for log correlation
    trace_id: str = ""

    # Conversation persistence (Step 4) — client should store this and
    # pass it back on the next turn to continue this conversation
    conversation_id: Optional[str] = None


# â”€â”€ Route handler â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description=(
        "Send a query to the RAG pipeline. "
        "If documents are indexed under `collection_name`, retrieval runs first. "
        "Falls back to general LLM knowledge when answer is not in documents."
    ),
)
async def query_endpoint(
    req: QueryRequest,
    current_user: dict = Depends(require_auth),
) -> QueryResponse:
    trace_id = str(uuid.uuid4())[:8]
    # A non-admin may only search their own collection (user_<email>).
    from api.middleware.auth import assert_collection_access
    assert_collection_access(current_user, req.collection_name)
    # Override user_role from JWT — client can't self-elevate
    req = req.model_copy(update={
        "user_role": current_user.get("role", "user")
    })
    logger.info("[%s] query: %s (provider=%s model=%s collection=%s user=%s role=%s)",
                trace_id, req.query[:80], req.llm_provider,
                req.llm_model, req.collection_name,
                current_user.get("sub"), current_user.get("role"))

    # -- Quota check -- cheapest possible rejection point, before any
    # retrieval/LLM work happens. A user with no quota row set is
    # unlimited (see app/services/usage_service.check_quota). Admins can
    # set quotas on any uid, including other admins -- no built-in role
    # exemption, since that's a deliberate per-user choice made when the
    # quota is set, not something to bake in here.
    from api.middleware.logging import _db_pool as _quota_pool
    if _quota_pool:
        from app.services.usage_service import check_quota
        quota_result = await check_quota(_quota_pool, current_user.get("sub", ""))
        if not quota_result["allowed"]:
            logger.warning("[%s] Quota exceeded for user=%s: %s",
                            trace_id, current_user.get("sub"), quota_result["reason"])
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=quota_result["user_message"],
            )

    # â”€â”€ Build runtime_config — exact same shape as chat_panel.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    runtime_config: Dict[str, Any] = {
        # LLM
        "llm_provider":    req.llm_provider,
        "llm_model":       req.llm_model,
        "temperature":     req.temperature,

        # Retrieval
        "collection_name": req.collection_name,
        "top_k":           req.top_k,
        "retrieval_mode":  req.retrieval_mode,
        "rerank":          req.rerank,

        # Embedding
        "embedding_provider": req.embedding_provider,
        "embedding_model":    req.embedding_model,

        # Domain / persona
        "domain":         req.domain,
        "assistant_name": req.assistant_name,

        # IAM
        "user_role":  req.user_role,
        "user_id":    current_user.get("sub", ""),   # for RAGAS Redis key scoping
        "diagnostics": True,  # Enable for all roles - needed for golden dataset eval

        # Cache control
        "skip_cache": req.skip_cache,

    }

    # â”€â”€ Resolve conversation history (Step 4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # If conversation_id was sent, fetch token-aware history from Postgres.
    # Falls back to client-sent chat_history if no conversation_id, or if
    # the DB isn't configured (graceful degradation — app still works
    # exactly as before Step 4 if AUDIT_DB_URL isn't set).
    resolved_history = req.chat_history
    conversation_id  = req.conversation_id

    from api.middleware.logging import _db_pool
    if _db_pool:
        from app.services import conversation_service as conv_service

        if conversation_id:
            # Existing conversation — verify ownership, fetch context window
            conv = await conv_service.get_conversation(
                _db_pool, conversation_id, current_user.get("sub"),
                is_admin=current_user.get("role") == "admin",
            )
            if conv:
                resolved_history = await conv_service.build_context(
                    _db_pool, conversation_id, model=req.llm_model,
                )
            else:
                # Invalid/inaccessible conversation_id — start fresh rather
                # than failing the whole request
                logger.warning(
                    "[%s] conversation_id=%s not found/inaccessible, starting new",
                    trace_id, conversation_id
                )
                conversation_id = None

        
        if not conversation_id:
            # No conversation_id provided — create one so the client can
            # continue this thread on the next turn
            conversation_id = await conv_service.create_conversation(
                _db_pool, current_user.get("sub"), first_message=req.query,
            )

    state_input = {
        "runtime_config": runtime_config,
        "query":          req.query,
        "chat_history":   resolved_history,
        "metadata":       {},
    }

    # â”€â”€ Call existing LangGraph pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        from app.graph.workflow import graph

        t0     = time.perf_counter()
        result = graph.invoke(
            state_input,
            config={"configurable": {"thread_id": f"api_{trace_id}"}},
        )
        elapsed = round(time.perf_counter() - t0, 2)

    except Exception as e:
        logger.error("[%s] pipeline error: %s", trace_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": str(e), "trace_id": trace_id},
        )

    # â”€â”€ Persist this turn (Step 4) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if _db_pool and conversation_id:
        try:
            await conv_service.add_message(
                _db_pool, conversation_id, "user", req.query, model=req.llm_model,
            )
            await conv_service.add_message(
                _db_pool, conversation_id, "assistant",
                result.get("response", ""), model=req.llm_model,
            )
            # Summarize older turns if the conversation has grown large —
            # keeps the context window from needing to drop history outright
            if await conv_service.should_summarize(_db_pool, conversation_id):
                from app.providers.llm.factory import get_llm

                async def _llm_call(prompt: str) -> str:
                    llm = get_llm(req.llm_provider, req.llm_model, temperature=0.0)
                    resp = await llm.ainvoke(prompt)
                    return resp.content if hasattr(resp, "content") else str(resp)

                await conv_service.summarize_old_messages(
                    _db_pool, conversation_id, _llm_call,
                )
        except Exception as e:
            # Persistence failures should never break the actual answer
            # the user already received — log and move on
            logger.error("[%s] conversation persistence failed: %s", trace_id, e)

    # â”€â”€ Extract results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    response    = result.get("response", "")
    metadata    = result.get("metadata", {})
    blocked     = result.get("blocked", False)
    block_reason = result.get("block_reason", "")
    error       = result.get("error", "")
    reranked    = result.get("reranked_docs", [])

    if blocked:
        response = block_reason or "Query blocked by security policy."

    usage_raw = metadata.get("usage", {})
    usage = UsageModel(
        provider=          usage_raw.get("provider", req.llm_provider),
        model=             usage_raw.get("model", req.llm_model),
        prompt_tokens=     usage_raw.get("prompt_tokens", 0),
        completion_tokens= usage_raw.get("completion_tokens", 0),
        total_tokens=      usage_raw.get("total_tokens", 0),
        estimated_cost_usd=usage_raw.get("estimated_cost_usd", 0.0),
        latency_seconds=   elapsed,
    )

    # Parse citations
    citations = [
        CitationModel(
            label=    c.get("label", ""),
            source=   c.get("source", ""),
            location= c.get("location", ""),
            type=     c.get("type", ""),
        )
        for c in metadata.get("citations", [])
    ]

    # RAGAS scores (available after background eval completes)
    ragas_meta = metadata.get("ragas", {})
    eval_key   = ragas_meta.get("eval_key", "")
    faith = rel = None
    if eval_key:
        try:
            from app.graph.nodes.ragas_node import get_ragas_result, is_ragas_done
            if is_ragas_done(eval_key):
                scores = get_ragas_result(eval_key)
                faith  = scores.get("faithfulness")
                rel    = scores.get("answer_relevancy")
        except Exception:
            pass

    logger.info("[%s] done in %.2fs — type=%s cache=%s tokens=%d cost=$%.5f user=%s",
                trace_id, elapsed,
                metadata.get("query_type", "?"),
                metadata.get("cache_hit", False),
                usage.total_tokens,
                usage.estimated_cost_usd,
                current_user.get("sub"))

    # Write detailed audit entry with query info
    from api.middleware.logging import write_audit_log
    await write_audit_log({
        "trace_id":        trace_id,
        "user_id":         current_user.get("sub"),
        "role":            current_user.get("role"),
        "method":          "POST",
        "endpoint":        "/api/v1/query",
        "query_text":      req.query,
        "query_type":      metadata.get("query_type", "rag"),
        "llm_provider":    req.llm_provider,
        "llm_model":       req.llm_model,
        "status_code":     200,
        "success":         True,
        "latency_ms":      int(elapsed * 1000),
        "tokens_total":    usage.total_tokens,
        "cost_usd":        usage.estimated_cost_usd,
        "chunks_retrieved":len(reranked),
        "cache_hit":       metadata.get("cache_hit", False),
        "faithfulness":    faith,
        "answer_relevancy":rel,
    })

    # Build ragas dict — include eval_key so Streamlit can poll
    ragas_response = {}
    if eval_key:
        if faith is not None:
            ragas_response = {
                "status":           "done",
                "eval_key":         eval_key,
                "faithfulness":     faith,
                "answer_relevancy": rel,
            }
        else:
            ragas_response = {
                "status":   "running",
                "eval_key": eval_key,
            }
    elif ragas_meta.get("skipped"):
        ragas_response = {
            "status":  "skipped",
            "reason":  ragas_meta.get("reason", ""),
        }

    # Determine final query_type
    answered_from = metadata.get("answered_from", "")
    query_type    = metadata.get("query_type", "rag")
    if answered_from in ("rag_expanded", "rag_related"):
        query_type = "rag"
    elif answered_from == "general_fallback":
        query_type = "general"
        citations  = []

    # Strip embedded Sources section from ALL responses.
    # The LLM prompt instructs it to write "Source:" in the text, but
    # the UI also renders citations from metadata — causing duplication.
    # Match: "Source:", "Sources:", "**Sources:**", "Source" alone on a line.
    import re
    _src_re = re.compile(
        r"^\s*\*{0,2}Sources?\s*:?\*{0,2}\s*$",
        re.IGNORECASE,
    )
    _lines = response.split("\n")
    _clean = []
    for _line in _lines:
        if _src_re.match(_line):
            break
        _clean.append(_line)
    response = "\n".join(_clean).rstrip()


    def _build_chunks(docs, cits) -> list:
        """Build chunks with labels that EXACTLY match citation_node label format."""
        from pathlib import Path as _P
        chunks = []
        for doc in docs[:10]:   # fetch more to cover all citations
            dmeta    = doc.get("metadata", {})
            src      = _P(dmeta.get("source", "")).name
            doc_type = dmeta.get("type", "")
            page     = dmeta.get("page")
            slide    = dmeta.get("slide")
            hdg      = dmeta.get("heading", "").strip()

            # Mirror citation_node._build_location() exactly
            if doc_type == "pdf" or isinstance(page, int):
                loc = f"Page {page + 1}" if isinstance(page, int) else ""
            elif doc_type == "pptx" or isinstance(slide, int):
                loc = f"Slide {slide}" + (f" — {hdg}" if hdg else "")                       if isinstance(slide, int) else ""
            elif doc_type == "docx":
                loc = f"§ {hdg}" if hdg else ""
            else:
                loc = f"§ {hdg}" if hdg else ""

            label   = src + (f", {loc}" if loc else "")
            content = doc.get("content", "")
            score   = float(doc.get("score", 0.0))
            rr      = doc.get("rerank_score")
            chunks.append(ChunkModel(label=label, content=content, score=score,
                                     rerank_score=float(rr) if rr is not None else None))
        return chunks

    return QueryResponse(
        response=         response,
        query_type=       query_type,
        citations=        citations,
        latency_seconds=  elapsed,
        cache_hit=          metadata.get("cache_hit", False),
        conversation_aware= metadata.get("conversation_aware", False),
        usage=            usage,
        faithfulness=     faith,
        answer_relevancy= rel,
        retrieved_chunks_count= len(reranked),
        retrieved_chunks=     _build_chunks(reranked, citations),
        router_method=    metadata.get("router_method", ""),
        blocked=          blocked,
        block_reason=     block_reason,
        error=            error,
        ragas=            ragas_response,
        answered_from=    answered_from,
        trace_id=         trace_id,
        conversation_id=  conversation_id,
    )