"""
ui/evaluation_actions.py — On-demand RAGAS evaluation + evaluation export.

Covers: evaluate_ragas (triggered by the "📊 Evaluate" button on a query
response), run_golden_dataset (prompt), clear_evaluations, export_this,
export_all, export_select action callbacks, and the underlying Excel
export logic.
"""

import logging

import chainlit as cl

from app.services.api_client import API_URL
from ui.helpers import load_evals, save_evals
from ui.settings_actions import rebuild_settings_widget

logger = logging.getLogger("financial_rag.chainlit")


@cl.action_callback("evaluate_ragas")
async def on_evaluate_ragas(action: cl.Action):
    eval_key = action.payload.get("eval_key", "")
    token = action.payload.get("token", "")

    if not eval_key:
        await cl.Message(content="No evaluation key found.").send()
        return

    thinking2 = cl.Message(
        content=(
            "⚡ **Running RAGAS evaluation...**\n\n"
            "`░░░░░░░░░░` Calling LLM to score faithfulness and relevancy...\n"
            "_This takes 15-30 seconds_"
        )
    )
    await thinking2.send()
    try:
        import httpx, time, asyncio
        from functools import partial
        t0 = time.time()
        loop = asyncio.get_event_loop()
        r = await loop.run_in_executor(
            None,
            partial(
                httpx.post,
                f"{API_URL}/api/v1/evaluate",
                json={"eval_key": eval_key, "question": "",
                      "answer": "", "retrieved_docs": []},
                headers={"Authorization": f"Bearer {token}"},
                timeout=120,
            )
        )
        elapsed = round(time.time() - t0, 1)
        await thinking2.remove()

        if r.status_code == 200:
            data = r.json()

            # The endpoint returns HTTP 200 even when evaluation itself
            # failed internally — check the body's status before treating
            # this as success, or real errors get silently hidden as blanks.
            if data.get("status") == "error":
                await cl.Message(content=(
                    f"❌ **RAGAS evaluation failed**\n\n"
                    f"`{data.get('error', 'unknown error')}`\n\n"
                    f"_The query itself succeeded — this is a server-side "
                    f"scoring error. Check the API logs for eval_key `{eval_key}`._"
                )).send()
                return

            faith = data.get("faithfulness")
            rel = data.get("answer_relevancy")
            cr = data.get("context_recall")
            ac = data.get("answer_correctness")

            def _fmt(v):
                if v is None:
                    return "—"
                bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
                grade = "🟢" if v >= 0.8 else "🟡" if v >= 0.6 else "🔴"
                return f"{grade} {v:.3f} `{bar}`"

            # Fetch question from Redis query data (needed for display + storage)
            username = cl.user_session.get("username", "default")
            stored_q = {}
            try:
                from app.graph.nodes.ragas_node import get_query_data
                stored_q = get_query_data(eval_key, username)
            except Exception:
                pass

            stored_question = stored_q.get("question") or action.payload.get("question", "")
            q_header = f"**Q: {stored_question[:80]}**\n\n" if stored_question else ""
            lines = [
                f"📊 **RAGAS Evaluation Results**\n\n{q_header}",
                "| Metric | Score |",
                "|---|---|",
                f"| Faithfulness | {_fmt(faith)} |",
                f"| Answer Relevancy | {_fmt(rel)} |",
            ]
            if cr is not None:
                lines.append(f"| Context Recall | {_fmt(cr)} |")
            if ac is not None:
                lines.append(f"| Answer Correctness | {_fmt(ac)} |")

            # Store evaluation — persist to Redis so Diagnostics panel can read it
            # across profile switches (cl.user_session is profile-scoped)
            # Redis when configured; otherwise keep the list in this session
            # (load_evals returns [] in local mode, which reset the list on
            # every evaluation and hid "Export All").
            evaluations = load_evals(username) or list(cl.user_session.get("evaluations") or [])
            eval_index = len(evaluations)

            evaluations.append({
                "eval_key": eval_key,
                "question": stored_q.get("question", action.payload.get("question", f"Q{eval_index+1}")),
                "faithfulness": faith,
                "answer_relevancy": rel,
                "context_recall": cr,
                "answer_correctness": ac,
                "label": f"Q{eval_index + 1}",
            })
            save_evals(username, evaluations)
            cl.user_session.set("evaluations", evaluations)

            # Build export action buttons
            export_actions = [
                cl.Action(
                    name="export_this",
                    label="📥 Export This",
                    payload={"indices": [eval_index]},
                    tooltip="Download just this evaluation as Excel",
                ),
            ]
            if len(evaluations) > 1:
                export_actions.append(
                    cl.Action(
                        name="export_all",
                        label=f"📊 Export All ({len(evaluations)})",
                        payload={"indices": list(range(len(evaluations)))},
                        tooltip=f"Download all {len(evaluations)} evaluations as Excel",
                    )
                )

            await cl.Message(
                content="\n".join(lines),
                actions=export_actions,
            ).send()
        else:
            await cl.Message(
                content=f"Evaluation failed: HTTP {r.status_code}"
            ).send()

    except Exception as e:
        await thinking2.remove()
        await cl.Message(content=f"Evaluation error: {e}").send()


@cl.action_callback("run_golden_dataset")
async def on_run_golden_dataset(action: cl.Action):
    """Prompt user to upload a golden dataset file."""
    await cl.Message(content=(
        "🎯 **Golden Dataset Evaluation**\n\n"
        "To run a batch evaluation:\n\n"
        "1. Switch to **💬 Chat** tab\n"
        "2. Upload a CSV file with columns: `question`, `reference_answer` (optional)\n"
        "3. Then type `/golden` to trigger batch evaluation\n\n"
        "**CSV format:**\n"
        "```\n"
        "question,reference_answer\n"
        "What is test recall?,0.86 or 86%\n"
        "What model is used?,Logistic Regression\n"
        "```\n\n"
        "_Or upload the CSV and type a question at the same time — "
        "the system will index the CSV as a golden dataset._"
    )).send()


@cl.action_callback("clear_evaluations")
async def on_clear_evaluations(action: cl.Action):
    """Clear all stored evaluations for this user."""
    username = cl.user_session.get("username", "default")
    try:
        from app.graph.nodes.ragas_node import _get_redis
        r = _get_redis()
        if r:
            r.delete(f"evals:{username}")
    except Exception:
        pass
    save_evals(username, [])
    cl.user_session.set("evaluations", [])
    await rebuild_settings_widget(cl.user_session.get("config", {}))
    await cl.Message(content=(
        "🗑️ **Evaluations cleared from storage**\n\n"
        "✅ All evaluation data has been deleted from Redis.\n\n"
        "⚠️ The evaluation messages above are still visible in this chat view "
        "(Chainlit cannot delete sent messages). Their export buttons will no longer "
        "work since the data is deleted.\n\n"
        "**To see a fresh empty Diagnostics view:**\n"
        "Switch to 💬 Chat → then back to 📊 Diagnostics"
    )).send()


@cl.action_callback("export_this")
async def on_export_this(action: cl.Action):
    indices = action.payload.get("indices", [])
    await export_evaluations(indices=indices)


@cl.action_callback("export_all")
async def on_export_all(action: cl.Action):
    indices = action.payload.get("indices", [])
    await export_evaluations(indices=indices)


@cl.action_callback("export_select")
async def on_export_select(action: cl.Action):
    """Show per-evaluation export buttons instead of text input (avoids chat conflict)."""
    all_indices = action.payload.get("all_indices", [])
    labels = action.payload.get("labels", [])
    evaluations = cl.user_session.get("evaluations", [])

    lines = ["**Select evaluations to export — click the ones you want:**\n"]
    for idx, label in zip(all_indices, labels):
        e = evaluations[idx] if idx < len(evaluations) else {}
        faith = f"Faith: {round(e.get('faithfulness',0)*100,0):.0f}%" if e.get("faithfulness") is not None else "—"
        rel = f"Rel: {round(e.get('answer_relevancy',0)*100,0):.0f}%" if e.get("answer_relevancy") is not None else "—"
        lines.append(f"- **Q{idx+1}** {faith} | {rel} — _{label}_")

    await cl.Message(
        content="\n".join(lines),
        actions=[
            cl.Action(
                name="export_this",
                label=f"📥 Export Q{idx+1}",
                payload={"indices": [idx]},
                tooltip=label,
            )
            for idx, label in zip(all_indices, labels)
        ] + [
            cl.Action(
                name="export_all",
                label=f"📊 Export All ({len(all_indices)})",
                payload={"indices": all_indices},
            )
        ]
    ).send()


async def export_evaluations(indices: list = None):
    username = cl.user_session.get("username", "default")
    # Redis when configured; otherwise the list kept in this chat session
    evaluations = load_evals(username) or list(cl.user_session.get("evaluations") or [])
    role = cl.user_session.get("role", "user")

    # Filter to selected indices if provided
    if indices is not None:
        evaluations = [evaluations[i] for i in indices if i < len(evaluations)]

    if role not in ("admin", "engineer"):
        await cl.Message(content="Export is only available to admins and engineers.").send()
        return

    if not evaluations:
        await cl.Message(
            content="No evaluations to export yet. Run some queries with Diagnostics "
                    "enabled and click Evaluate first."
        ).send()
        return

    try:
        import io
        import pandas as pd

        rows = []
        for i, e in enumerate(evaluations, 1):
            rows.append({
                "Q#": i,
                "Question": e.get("question", e.get("label", f"Q{i}")),
                "Eval Key": e.get("eval_key", ""),
                "Faithfulness": round(e["faithfulness"], 4) if e.get("faithfulness") is not None else "",
                "Answer Relevancy": round(e["answer_relevancy"], 4) if e.get("answer_relevancy") is not None else "",
                "Context Recall": round(e["context_recall"], 4) if e.get("context_recall") is not None else "",
                "Answer Correctness": round(e["answer_correctness"], 4) if e.get("answer_correctness") is not None else "",
                "Faith %": f"{round(e['faithfulness']*100,1)}%" if e.get("faithfulness") is not None else "—",
                "Relevancy %": f"{round(e['answer_relevancy']*100,1)}%" if e.get("answer_relevancy") is not None else "—",
            })

        df = pd.DataFrame(rows)
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="RAGAS Evaluations")
            ws = writer.sheets["RAGAS Evaluations"]
            # Auto-width columns
            for col_cells in ws.columns:
                max_len = max(len(str(c.value or "")) for c in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 4, 40)

        output.seek(0)

        # Write to a temp file for Chainlit to serve
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            suffix=".xlsx", delete=False,
            prefix="ragas_evaluations_"
        ) as tmp:
            tmp.write(output.getvalue())
            tmp_path = tmp.name

        elements = [
            cl.File(
                name=f"ragas_evaluations_{len(evaluations)}_queries.xlsx",
                path=tmp_path,
                display="inline",
            )
        ]
        await cl.Message(
            content=f"📥 **RAGAS Evaluations Export** ({len(evaluations)} queries)",
            elements=elements,
        ).send()

        # Cleanup temp file after a delay
        os.unlink(tmp_path)

    except ImportError:
        await cl.Message(
            content="Export requires `openpyxl` and `pandas`. "
                    "Add them to `requirements.chainlit.txt` and restart."
        ).send()
    except Exception as e:
        logger.exception("Export failed: %s", e)
        await cl.Message(content=f"Export failed: {e}").send()