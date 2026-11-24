"""
ui/upload_handler.py — File upload handling: document indexing + golden
dataset CSV intake.

Called from ui.chat_handlers.on_message when a message carries file
elements.
"""

from app.config.vision_models import vision_model_choices, caption_spec, caption_provider_model
import os
import logging

import chainlit as cl

from app.services.api_client import index_files, APIError
from ui.helpers import SUPPORTED_EXTENSIONS, guess_mime
from ui.settings_actions import rebuild_settings_widget

logger = logging.getLogger("financial_rag.chainlit")


async def handle_file_upload(files: list) -> bool:
    token = cl.user_session.get("token", "")
    username = cl.user_session.get("username", "default")
    config = cl.user_session.get("config", {})

    # Separate CSV files (for golden dataset) from indexable documents
    csv_files = [f for f in files if f.name.lower().endswith(".csv")]
    doc_files = [f for f in files if any(f.name.lower().endswith(ext)
                 for ext in SUPPORTED_EXTENSIONS) and not f.name.lower().endswith(".csv")]

    # Handle CSV files for golden dataset
    for f in csv_files:
        try:
            csv_text = None
            # Chainlit stores files at f.path (temp file on disk) — use that first
            if hasattr(f, "path") and f.path and os.path.exists(str(f.path)):
                with open(f.path, "rb") as fh:
                    csv_text = fh.read().decode("utf-8-sig", errors="replace")
            elif hasattr(f, "content") and f.content is not None:
                raw = f.content
                if isinstance(raw, bytes):
                    csv_text = raw.decode("utf-8-sig", errors="replace")
                elif hasattr(raw, "read"):
                    csv_text = raw.read().decode("utf-8-sig", errors="replace")
            if not csv_text:
                await cl.Message(content=f"❌ Could not read `{f.name}` — file appears empty.").send()
                continue
            cl.user_session.set("_golden_csv_data", csv_text)
            await cl.Message(content=(
                f"📊 **CSV file received: `{f.name}`**\n\n"
                f"This looks like a golden dataset file.\n"
                f"Type `/golden` to run batch RAGAS evaluation, or ask questions normally."
            )).send()
            logger.info("Golden CSV stored: %s (%d chars)", f.name, len(csv_text))
        except Exception as e:
            await cl.Message(content=f"⚠️ Could not read CSV `{f.name}`: {e}").send()

    valid = doc_files
    if not valid:
        if csv_files:
            return True  # CSV handled above
        await cl.Message(
            content=f"No supported document files. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        ).send()
        return False

    names = ", ".join(f.name for f in valid)
    indexed_docs = cl.user_session.get("indexed_docs", [])

    # Warn if files already indexed (prevents accidental duplicates)
    already_indexed = [f.name for f in valid if f.name in indexed_docs]
    if already_indexed:
        confirm_msg = cl.Message(
            content=(
                f"⚠️ **Already indexed:** {', '.join(f'`{n}`' for n in already_indexed)}\n\n"
                f"Re-indexing will add duplicate chunks. "
                f"Use **🗑️ Clear My Collection** in Admin Panel first, "
                f"or continue to re-index anyway."
            ),
            actions=[
                cl.Action(name="confirm_reindex", label="✅ Re-index anyway",
                          payload={"files": [f.name for f in valid]}),
                cl.Action(name="cancel_reindex", label="❌ Cancel", payload={}),
            ]
        )
        await confirm_msg.send()
        return False

    vision_on = config.get("vision_mode", False)
    status_msg = cl.Message(
        content=(
            f"📂 **Indexing {names}...**\n"
            f"OCR: `{config.get('ocr_method', 'easyocr')}` · "
            f"Chunk: `{config.get('chunk_size', 500)}` · "
            f"Overlap: `{config.get('chunk_overlap', 50)}` · "
            f"Vision: `{'on' if vision_on else 'off'}`\n"
            f"Sending to API for processing..."
        )
    )
    await status_msg.send()

    try:
        file_tuples = []
        for f in valid:
            if hasattr(f, "path") and f.path and os.path.exists(f.path):
                with open(f.path, "rb") as fh:
                    data = fh.read()
            elif hasattr(f, "content") and f.content:
                data = f.content if isinstance(f.content, bytes) else f.content.read()
            else:
                logger.warning("File %s has no readable content", f.name)
                continue
            file_tuples.append((f.name, data, guess_mime(f.name)))

        if not file_tuples:
            await cl.Message(content="Could not read file content.").send()
            return False

        import asyncio
        from functools import partial
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                index_files,
                files=file_tuples,
                token=token,
                username=username,
                ocr_method=config.get("ocr_method", "easyocr"),
                use_vision=vision_on,
                caption_images=vision_on,
                caption_provider=caption_provider_model(config)[0],
                caption_model=caption_provider_model(config)[1],
                chunk_size=config.get("chunk_size", 500),
                chunk_overlap=config.get("chunk_overlap", 50),
                chunk_method=config.get("chunk_method", "recursive"),
                embedding_provider=config.get("embedding_provider", "openai"),
                embedding_model=config.get("embedding_model", "text-embedding-3-small"),
            )
        )

        chunks = result.get("chunks_total", 0)
        elapsed = result.get("elapsed_seconds", 0)
        collection = result.get("collection_name", "")

        indexed = cl.user_session.get("indexed_docs", [])
        for f in valid:
            if f.name not in indexed:
                indexed.append(f.name)
        cl.user_session.set("indexed_docs", indexed)
        cl.user_session.set("collection_name", collection)

        # Update settings widget to show newly indexed files (no sidebar open)
        config = cl.user_session.get("config", {})
        await rebuild_settings_widget(config)

        # Persist session state to Redis so it survives profile switches
        try:
            from app.graph.nodes.ragas_node import _get_redis
            import json as _json
            r = _get_redis()
            if r:
                r.set(f"session_state:{username}", _json.dumps({
                    "indexed_docs": indexed,
                    "collection_name": collection,
                    "conversation_id": cl.user_session.get("conversation_id"),
                }), ex=7200)  # 2 hour TTL
        except Exception:
            pass

        # CSV files are handled separately before indexing

        # Invalidate semantic cache for this collection so fresh results are returned
        try:
            from app.graph.nodes.ragas_node import _get_redis
            _r = _get_redis()
            if _r:
                cache_key = f"semcache:{collection}"
                _r.delete(cache_key)
                logger.info("Cache invalidated for collection %s after indexing", collection)
        except Exception:
            pass

        await cl.Message(content=(
            f"✅ **Indexed successfully!**\n\n"
            f"Files: `{names}` | Chunks: {chunks} | "
            f"Time: {elapsed:.1f}s | Collection: `{collection}`\n\n"
            f"_You can now ask questions about your documents._"
        )).send()
        return True

    except APIError as e:
        await cl.Message(content=f"Indexing failed: {e}").send()
        return False
    except Exception as e:
        logger.exception("Indexing error: %s", e)
        await cl.Message(content=f"Indexing error: {e}").send()
        return False