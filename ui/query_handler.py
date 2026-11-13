"""
ui/query_handler.py — Core RAG query flow + interactive golden dataset
(CSV batch) evaluation.

Called from ui.commands (/golden) and ui.chat_handlers (on_message, for
plain-text queries).
"""

import os
import logging

import chainlit as cl

from app.services.api_client import query as api_query, APIError, API_URL
from ui.helpers import build_progress
from ui.settings_actions import rebuild_settings_widget

logger = logging.getLogger("financial_rag.chainlit")


async def run_golden_evaluation_interactive():
    """Run golden dataset evaluation from an uploaded CSV file."""
    import io
    import pandas as pd
    import httpx

    token = cl.user_session.get("token", "")
    username = cl.user_session.get("username", "default")
    config = cl.user_session.get("config", {})
    api_url = config.get("api_url", API_URL)
    collection = cl.user_session.get("collection_name", f"user_{username}")

    # Check for a pending golden CSV in session
    golden_csv = cl.user_session.get("_golden_csv_data")
    if not golden_csv:
        await cl.Message(content=(
            "🎯 **Golden Dataset Evaluation**\n\n"
            "No CSV uploaded yet. Please:\n"
            "1. Upload a CSV file with columns: `question`, `reference_answer` (optional)\n"
            "2. Type `/golden` to run the evaluation\n\n"
            "**Example CSV:**\n"
            "```\n"
            "question,reference_answer\n"
            "What is the test recall?,0.86\n"
            "What model is used?,Logistic Regression\n"
            "```"
        )).send()
        return

    try:
        # Handle Windows line endings and BOM
        cleaned_csv = golden_csv.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        df = pd.read_csv(io.StringIO(cleaned_csv))
        # Case-insensitive column check
        df.columns = [c.strip().lower() for c in df.columns]
        if "question" not in df.columns:
            cols = list(df.columns)
            await cl.Message(content=f"❌ CSV must have a `question` column. Found: {cols}").send()
            return
        # Rename reference_answer variants
        for alt in ["reference", "answer", "reference_answer", "expected_answer"]:
            if alt in df.columns and "reference_answer" not in df.columns:
                df = df.rename(columns={alt: "reference_answer"})

        df = df.dropna(subset=["question"]).reset_index(drop=True)  # i must be 0..n-1
        total = len(df)
        if total == 0:
            await cl.Message(content="❌ No valid questions found in CSV.").send()
            return

        await cl.Message(content=(
            f"🎯 **Golden Dataset Evaluation** — {total} questions\n\n"
            f"Running queries and RAGAS evaluation..."
        )).send()

        # Live progress tracker — updated after each question
        progress_bar = ["⬜"] * total
        progress_msg = cl.Message(content=build_progress(progress_bar, 0, total, [], "Starting..."))
        await progress_msg.send()

        results = []
        for i, row in df.iterrows():
            question = str(row["question"]).strip()
            reference = str(row.get("reference_answer", "")).strip()

            # Update progress to show current question running
            progress_bar[i] = "🔄"
            await progress_msg.update()
            progress_msg.content = build_progress(
                progress_bar, i, total, results,
                f"Q{i+1}: {question[:50]}..."
            )
            await progress_msg.update()

            try:
                import asyncio
                from functools import partial
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    partial(
                        api_query,
                        query_text=question, token=token,
                        collection_name=collection,
                        llm_provider=config.get("llm_provider", "openai"),
                        llm_model=config.get("llm_model", "gpt-4o-mini"),
                        temperature=config.get("temperature", 0.0),
                        top_k=config.get("top_k", 5),
                        retrieval_mode=config.get("retrieval_mode", "hybrid"),
                        rerank=config.get("rerank", True),
                        embedding_provider=config.get("embedding_provider", "openai"),
                        embedding_model=config.get("embedding_model", "text-embedding-3-small"),
                        domain=config.get("domain", "uploaded documents"),
                        assistant_name=config.get("assistant_name", "Document Assistant"),
                        user_role=cl.user_session.get("role", "user"),
                        chat_history=[],
                        skip_cache=True,  # golden dataset must always reflect the
                                          # current pipeline, never a stale cached
                                          # answer from an earlier test run
                    )
                )
                answer = result.get("response", "")
                cache_hit = result.get("cache_hit", False)
                eval_key = result.get("ragas", {}).get("eval_key", "")

                # Run RAGAS evaluation
                scores = {}
                eval_error = None
                if eval_key:
                    r = await loop.run_in_executor(
                        None,
                        partial(
                            httpx.post,
                            f"{api_url}/api/v1/evaluate",
                            json={"eval_key": eval_key, "question": question,
                                  "answer": answer, "retrieved_docs": [],
                                  "reference": reference},
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=120,
                        )
                    )
                    if r.status_code == 200:
                        scores = r.json()
                        # The endpoint returns HTTP 200 even when evaluation
                        # itself failed internally — check the body's status.
                        if scores.get("status") == "error":
                            eval_error = scores.get("error", "unknown evaluation error")
                            logger.warning(
                                "RAGAS eval failed for Q%d (eval_key=%s): %s",
                                i + 1, eval_key, eval_error,
                            )
                    else:
                        eval_error = f"HTTP {r.status_code}: {r.text[:200]}"
                elif cache_hit:
                    # No eval_key because this answer came from the semantic
                    # cache — RAGAS intentionally skips re-scoring cached
                    # answers (see app/graph/nodes/ragas_node.py). Not an
                    # error, just nothing to show. Note it explicitly so it
                    # doesn't read as a silent failure.
                    eval_error = "⚡ cached answer — RAGAS skipped, clear cache to re-score"

                results.append({
                    "question": question,
                    "answer": answer[:200],
                    "reference": reference,
                    "faithfulness": scores.get("faithfulness"),
                    "answer_relevancy": scores.get("answer_relevancy"),
                    "context_recall": scores.get("context_recall"),
                    "answer_correctness": scores.get("answer_correctness"),
                    "eval_error": eval_error,
                })
                progress_bar[i] = "⚠️" if eval_error else "✅"

            except Exception as e:
                results.append({"question": question, "answer": f"ERROR: {e}",
                                 "reference": reference})
                progress_bar[i] = "❌"

            # Update progress after each question
            done = sum(1 for p in progress_bar if p in ("✅", "❌"))
            pending = sum(1 for p in progress_bar if p == "⬜")
            progress_msg.content = build_progress(
                progress_bar, done, total, results,
                f"Completed {done}/{total} · {pending} remaining"
            )
            await progress_msg.update()

        # Final progress update
        progress_msg.content = build_progress(progress_bar, total, total, results, "All done!")
        await progress_msg.update()

        # Build results table
        lines = [f"✅ **Golden Dataset Complete — {total} questions**\n",
                 "| Q | Question | Faithfulness | Answer Relevancy | Note |",
                 "|---|---|---|---|---|"]
        any_errors = False
        for i, r in enumerate(results, 1):
            faith = f"{round(r.get('faithfulness',0)*100,1)}%" if r.get("faithfulness") is not None else "—"
            rel = f"{round(r.get('answer_relevancy',0)*100,1)}%" if r.get("answer_relevancy") is not None else "—"
            note = ""
            if r.get("eval_error"):
                any_errors = True
                note = f"⚠️ {r['eval_error'][:60]}"
            lines.append(f"| {i} | {r['question'][:40]} | {faith} | {rel} | {note} |")

        if any_errors:
            lines.append(
                "\n⚠️ **Some rows have notes** — see the Note column. "
                "'cached answer' rows are expected (RAGAS intentionally skips "
                "re-scoring cache hits); anything else is a real server-side "
                "evaluation issue — check the API logs for that eval_key."
            )

        await cl.Message(content="\n".join(lines)).send()

        # Export to Excel
        import tempfile, io as _io
        import pandas as _pd
        df_out = _pd.DataFrame(results)
        buf = _io.BytesIO()
        df_out.to_excel(buf, index=False)
        buf.seek(0)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False, prefix="golden_eval_") as tmp:
            tmp.write(buf.getvalue())
            tmp_path = tmp.name
        await cl.Message(
            content=f"📥 Download golden dataset results ({total} questions)",
            elements=[cl.File(name="golden_eval_results.xlsx", path=tmp_path)]
        ).send()
        os.unlink(tmp_path)
        cl.user_session.set("_golden_csv_data", None)

    except Exception as e:
        await cl.Message(content=f"❌ Golden dataset evaluation failed: {e}").send()


async def handle_query(query_text: str):
    token = cl.user_session.get("token", "")
    role = cl.user_session.get("role", "user")
    username = cl.user_session.get("username", "default")
    config = cl.user_session.get("config", {})
    collection_name = cl.user_session.get("collection_name") or f"user_{username}"
    indexed_docs = cl.user_session.get("indexed_docs", [])
    # Force correct collection if docs are indexed but collection somehow cleared
    if indexed_docs and not collection_name:
        collection_name = f"user_{username}"

    # If documents are indexed, ensure collection_name is correct
    # and show a thinking message that hints docs are available
    if indexed_docs:
        thinking = cl.Message(content=f"🤔 *Thinking... (searching {len(indexed_docs)} indexed document(s))*")
    else:
        thinking = cl.Message(content="🤔 *Thinking...*")
    await thinking.send()

    try:
        conversation_id = cl.user_session.get("conversation_id")
        import asyncio
        from functools import partial
        # Run synchronous api_query in thread pool to avoid blocking Chainlit's event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                api_query,
                query_text=query_text,
                token=token,
                collection_name=collection_name,
                llm_provider=config.get("llm_provider", "openai"),
                llm_model=config.get("llm_model", "gpt-4o-mini"),
                temperature=config.get("temperature", 0.0),
                top_k=config.get("top_k", 5),
                retrieval_mode=config.get("retrieval_mode", "hybrid"),
                rerank=config.get("rerank", True),
                embedding_provider=config.get("embedding_provider", "openai"),
                embedding_model=config.get("embedding_model", "text-embedding-3-small"),
                domain=config.get("domain", "uploaded documents"),
                assistant_name=config.get("assistant_name", "Document Assistant"),
                user_role=role,
                chat_history=[],
                conversation_id=conversation_id,
            )
        )

        new_conv_id = result.get("conversation_id")
        if new_conv_id:
            cl.user_session.set("conversation_id", new_conv_id)
            # Persist conversation_id to Redis immediately
            try:
                from app.graph.nodes.ragas_node import _get_redis
                import json as _j2
                _r2 = _get_redis()
                if _r2:
                    _username2 = cl.user_session.get("username", "default")
                    _existing = _r2.get(f"session_state:{_username2}")
                    _state2 = _j2.loads(_existing) if _existing else {}
                    _state2["conversation_id"] = new_conv_id
                    _state2["indexed_docs"] = cl.user_session.get("indexed_docs", [])
                    _state2["collection_name"] = cl.user_session.get("collection_name", f"user_{_username2}")
                    _r2.set(f"session_state:{_username2}", _j2.dumps(_state2), ex=7200)
            except Exception:
                pass
            # Update settings widget with new conversation ID (no sidebar open)
            await rebuild_settings_widget(cl.user_session.get("config", {}))
            # Persist to Redis so conversation survives profile switches
            try:
                from app.graph.nodes.ragas_node import _get_redis
                import json as _json
                r = _get_redis()
                if r:
                    username_key = cl.user_session.get("username", "default")
                    existing = r.get(f"session_state:{username_key}")
                    state = _json.loads(existing) if existing else {}
                    state["conversation_id"] = new_conv_id
                    r.set(f"session_state:{username_key}", _json.dumps(state), ex=7200)
            except Exception:
                pass

        response = result.get("response", "")
        citations = result.get("citations", [])
        cache_hit = result.get("cache_hit", False)
        query_type = result.get("query_type", "rag")
        latency = result.get("latency_seconds", 0)
        usage = result.get("usage", {})
        ragas_meta = result.get("ragas", {})
        eval_key = ragas_meta.get("eval_key", "")
        chunks = result.get("retrieved_chunks", [])

        await thinking.remove()

        # ── Main response (clean — sources only, no diagnostics) ──────────
        citation_text = ""
        if citations and query_type == "rag":
            citation_text = "\n\n---\n**Sources:**\n"
            for i, c in enumerate(citations, 1):
                citation_text += f"{i}. {c.get('label', '')}\n"

        # Response type badge + meta line (plain markdown, no HTML)
        route_badge = {
            "rag": "📄 RAG",
            "general": "💬 General",
            "calculation": "🔢 Calculation",
            "blocked": "🚫 Blocked",
        }.get(query_type, "📄 RAG")

        cache_badge = " | ⚡ cached" if cache_hit else ""
        meta_line = (
            f"\n\n---\n"
            f"{route_badge} | "
            f"⏱ {latency:.1f}s | "
            f"🪙 {usage.get('total_tokens', 0)} tok | "
            f"${usage.get('estimated_cost_usd', 0):.5f}{cache_badge}"
        )

        # RAGAS action (engineer/admin only)
        actions = []
        if eval_key and role in ("admin", "engineer"):
            actions.append(cl.Action(
                name="evaluate_ragas",
                label="📊 Evaluate (RAGAS)",
                payload={"eval_key": eval_key, "token": token,
                         "question": query_text[:300]},
            ))

        await cl.Message(
            content=response + citation_text + meta_line,
            actions=actions if actions else None,
        ).send()

        # ── Diagnostics Step (engineer/admin only — collapsible) ──────────
        # Always show for admin/engineer, even with 0 chunks — a query that
        # fell back to general knowledge with truly empty retrieval is a
        # different failure mode than one with low-scored chunks, and both
        # need to be visible rather than silently hiding the whole panel.
        if role in ("admin", "engineer"):
            async with cl.Step(name="🔍 Diagnostics", type="tool") as diag_step:
                if not chunks:
                    chunk_lines = [
                        f"**Retrieved 0 chunks** (top-k={config.get('top_k', 5)}) "
                        f"— retrieval returned nothing for this query."
                    ]
                else:
                    # Retrieved chunks section
                    chunk_lines = [f"**Retrieved {len(chunks)} chunks (top-k={config.get('top_k', 5)}):**\n"]
                    for i, ch in enumerate(chunks, 1):
                        label = ch.get("label", f"Chunk {i}")
                        content = ch.get("content", "")[:400]
                        # rerank_score is what validation_node.py actually uses to
                        # decide RAG-vs-general routing. Fall back to "score" (the
                        # pre-rerank RRF fusion value) only if rerank wasn't applied.
                        rerank_score = ch.get("rerank_score")
                        fusion_score = ch.get("score", 0)
                        chunk_lines.append(f"**[{i}] {label}**")
                        if rerank_score is not None:
                            chunk_lines.append(
                                f"Rerank score: `{rerank_score:.4f}` "
                                f"(used for grounding decision) · RRF fusion score: `{fusion_score:.4f}`"
                            )
                        else:
                            chunk_lines.append(f"Score: `{fusion_score:.4f}` (rerank not applied)")
                        chunk_lines.append(f"```\n{content}{'...' if len(ch.get('content','')) > 400 else ''}\n```")
                        chunk_lines.append("")

                # Query routing info
                chunk_lines.append(f"**Query type:** `{query_type}`")
                chunk_lines.append(f"**Router method:** `{result.get('router_method', 'heuristic')}`")
                chunk_lines.append(f"**Cache hit:** `{cache_hit}`")
                chunk_lines.append(f"**Conversation ID:** `{result.get('conversation_id', 'N/A')}`")

                if eval_key:
                    chunk_lines.append(f"\n**RAGAS eval key:** `{eval_key}`")
                    chunk_lines.append("_Click 📊 Evaluate above to run RAGAS scoring_")

                diag_step.output = "\n".join(chunk_lines)

    except APIError as e:
        await thinking.remove()
        if e.status_code == 401:
            await cl.Message(content="Session expired. Please refresh and log in again.").send()
        else:
            await cl.Message(content=f"Query failed: {e}").send()
    except Exception as e:
        await thinking.remove()
        logger.exception("Query error: %s", e)
        await cl.Message(content=f"Unexpected error: {e}").send()