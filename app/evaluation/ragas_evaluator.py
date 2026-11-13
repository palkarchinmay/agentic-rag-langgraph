"""
evaluation/ragas_evaluator.py
-----------------------------
RAGAS evaluation — only runs when diagnostics=True.

Metrics without reference (auto-eval after every RAG response):
  faithfulness      : is the answer faithful to retrieved context?
  answer_relevancy  : is the answer relevant to the question?

Metrics requiring reference (full eval with uploaded Q&A dataset):
  context_recall    : did we retrieve all needed context?
  answer_correctness: how correct is the answer vs ground truth?

NOTE: context_precision was removed — it requires 'reference' in
RAGAS >= 0.2. Use full eval mode if you need precision metrics.
"""

import logging
from typing import Dict, List, Optional

# Pre-import at module level — acquires Python module lock on main thread
# before RAGAS background thread tries to import the same modules.
try:
    import langchain_openai  # noqa: F401
except ImportError:
    pass

logger = logging.getLogger("financial_rag.ragas")


def evaluate_response(
    question:       str,
    answer:         str,
    retrieved_docs: List[Dict],
    reference:      Optional[str] = None,
    llm_provider:   str = "openai",
    llm_model:      str = "gpt-4o-mini",
) -> Dict:
    """
    Run RAGAS on a single query-response pair.

    Without reference → faithfulness + answer_relevancy
    With reference    → adds context_recall + answer_correctness
    """
    try:
        # Pre-import to avoid threading module-lock deadlock
        import langchain_openai
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        import os

        contexts = [
            doc.get("content", "")[:1000]
            for doc in retrieved_docs
            if doc.get("content", "").strip()
        ]

        if not contexts:
            return {"error": "No context available for evaluation"}

        # Build dataset — only columns that metrics actually need
        data = {
            "question": [question],
            "answer":   [answer],
            "contexts": [contexts],
        }

        # Only include reference if provided
        # Auto-eval (no reference needed): faithfulness + answer_relevancy only
        # context_precision requires 'reference' in this RAGAS version — skip
        metrics = [faithfulness, answer_relevancy]

        # Reference-based metrics only when reference is provided
        if reference and reference.strip():
            from ragas.metrics import context_recall, answer_correctness
            metrics += [context_recall, answer_correctness]
            data["reference"] = [reference]

        dataset = Dataset.from_dict(data)

        # Wire RAGAS to use same LLM/embeddings as the app
        if llm_provider == "openai":
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            ragas_llm = LangchainLLMWrapper(
                ChatOpenAI(model=llm_model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
            )
            ragas_emb = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
            )
            for m in metrics:
                m.llm        = ragas_llm
                m.embeddings = ragas_emb

        results = ragas_evaluate(dataset=dataset, metrics=metrics, show_progress=False)
        scores  = results.to_pandas().iloc[0].to_dict()

        metric_names = {
            "faithfulness", "answer_relevancy",
            "context_recall", "answer_correctness",
        }
        clean = {
            k: round(float(v), 4) if v == v else None   # NaN → None
            for k, v in scores.items()
            if k in metric_names
        }
        clean["error"] = None
        logger.info("RAGAS complete: %s", clean)
        return clean

    except ImportError as e:
        return {"error": f"RAGAS dependency missing: {e}"}
    except Exception as e:
        logger.exception("RAGAS failed: %s", e)
        return {"error": str(e)}


def evaluate_batch(
    qa_pairs:            List[Dict],
    retrieved_docs_list: List[List[Dict]],
    llm_provider:        str = "openai",
    llm_model:           str = "gpt-4o-mini",
) -> Dict:
    """
    Evaluate a batch of Q&A pairs (for reference dataset upload in diagnostics).
    qa_pairs: list of {question, answer, reference (optional)}
    """
    try:
        import langchain_openai
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        import os

        questions  = [q["question"] for q in qa_pairs]
        answers    = [q["answer"]   for q in qa_pairs]
        references = [q.get("reference", "") for q in qa_pairs]
        contexts   = [
            [d.get("content","")[:800] for d in docs if d.get("content","").strip()]
            for docs in retrieved_docs_list
        ]

        data    = {"question": questions, "answer": answers, "contexts": contexts}
        metrics = [faithfulness, answer_relevancy]

        if any(r.strip() for r in references):
            from ragas.metrics import context_recall, answer_correctness
            data["reference"] = references
            metrics += [context_recall, answer_correctness]

        dataset = Dataset.from_dict(data)
        results = ragas_evaluate(dataset=dataset, metrics=metrics, show_progress=False)
        df      = results.to_pandas()

        metric_cols = [
            c for c in df.columns
            if c in {"faithfulness","answer_relevancy","context_recall","answer_correctness"}
        ]
        return {
            "summary":   {col: round(float(df[col].mean()), 4) for col in metric_cols},
            "per_query": df[metric_cols].to_dict(orient="records"),
            "error":     None,
        }

    except Exception as e:
        logger.exception("Batch RAGAS failed: %s", e)
        return {"error": str(e)}