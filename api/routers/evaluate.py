"""
api/routers/evaluate.py — On-demand RAGAS evaluation endpoint.

WHY ON-DEMAND INSTEAD OF AUTOMATIC
=====================================
RAGAS evaluation takes 5-30 seconds per query (it makes LLM calls to
score faithfulness and relevancy). In Cloud Run's serverless model,
background threads started during a request are killed when the HTTP
response is sent — so automatic background evaluation never completes.

On-demand evaluation solves this cleanly:
  1. Query runs fast (no RAGAS overhead)
  2. User switches to Diagnostics tab when ready
  3. User clicks "Evaluate" — this endpoint runs synchronously
  4. Result stored in Redis under ragas:{user_id}:{eval_key}
  5. Diagnostics panel reads result directly — no polling needed

USER ISOLATION
===============
Results are stored under ragas:{user_id}:{eval_key}, so User A's
evaluations are completely isolated from User B's.
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.middleware.auth import require_auth

logger = logging.getLogger("financial_rag.api.evaluate")
router = APIRouter(prefix="/api/v1", tags=["evaluation"])


# ── Request / Response models ─────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    eval_key:       str   = Field(..., description="Unique key from the original query response")
    question:       str   = Field(..., description="The original question")
    answer:         str   = Field(..., description="The model's answer")
    retrieved_docs: List[Dict] = Field(default_factory=list,
                                       description="Reranked docs from the query")
    reference:      Optional[str] = Field(None,
                                          description="Reference answer for context_recall + correctness")
    llm_provider:   str   = Field("openai")
    llm_model:      str   = Field("gpt-4o-mini")


class EvaluateResponse(BaseModel):
    eval_key:         str
    status:           str   # "done" | "error"
    faithfulness:     Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall:   Optional[float] = None
    answer_correctness: Optional[float] = None
    error:            Optional[str]  = None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    summary="Run RAGAS evaluation on a query-response pair",
    description=(
        "Runs synchronously — blocks until evaluation is complete. "
        "Result is stored in Redis under ragas:{user_id}:{eval_key} "
        "so it persists across Cloud Run instances and Streamlit reruns."
    ),
)
async def evaluate_endpoint(
    req: EvaluateRequest,
    current_user: dict = Depends(require_auth),
):
    user_id = current_user.get("sub", "")

    try:
        from app.evaluation.ragas_evaluator import evaluate_response
        from app.graph.nodes.ragas_node import get_query_data

       	# Fetch stored query data from Redis (ragas_node stored it at query time)
        stored = get_query_data(req.eval_key, user_id)

        # Use Redis data when request fields are empty (e.g. called from Chainlit)
        retrieved_docs = req.retrieved_docs or stored.get("retrieved_docs", [])
        question       = req.question or stored.get("question", "")
        answer         = req.answer   or stored.get("answer", "")

        scores = evaluate_response(
            question=question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            reference=req.reference if req.reference and req.reference.strip() else None,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
        )

        result = {**scores, "status": "done"}

        # Store in Redis with user-scoped key so Streamlit can retrieve it
        from app.graph.nodes.ragas_node import _write_result
        _write_result(req.eval_key, result, user_id)

        logger.info(
            "RAGAS evaluate done: eval_key=%s user=%s faith=%.3f rel=%.3f",
            req.eval_key, user_id,
            scores.get("faithfulness", 0) or 0,
            scores.get("answer_relevancy", 0) or 0,
        )

        return EvaluateResponse(
            eval_key=req.eval_key,
            status="done",
            faithfulness=scores.get("faithfulness"),
            answer_relevancy=scores.get("answer_relevancy"),
            context_recall=scores.get("context_recall"),
            answer_correctness=scores.get("answer_correctness"),
        )

    except Exception as e:
        logger.error("RAGAS evaluate failed: eval_key=%s error=%s", req.eval_key, e)

        # Store error in Redis so UI shows it correctly instead of "running"
        from app.graph.nodes.ragas_node import _write_result
        _write_result(req.eval_key, {"status": "error", "error": str(e)}, user_id)

        return EvaluateResponse(
            eval_key=req.eval_key,
            status="error",
            error=str(e),
        )