"""
app/ingestion/image_captioner.py — VLM-based image/chart interpretation.

UPDATED (2026-08-14): now the PRIMARY interpretation path for BOTH
chart-detected regions and general (non-chart) images, not just general
images. Same mechanism, different prompt (is_chart=True/False) -- one
unified pathway instead of two separate ones (DePlot/Pix2Struct for
charts, this module for everything else).

Why the switch: DePlot+Pix2Struct running locally on CPU measured at
30-90+ seconds per chart even after heavy tuning (see chart_interpreter.py's
perf notes), with Pix2Struct's narrative descriptions consistently
near-empty (1-40 chars across a real 8-chart document) -- a prompt/model
mismatch, since pix2struct-chartqa-base is fine-tuned for short factual
QA, not open-ended description. A hosted multimodal API call (Gemini
Flash by default) measured at 1-3 seconds typically, with real
descriptions, for a cost small enough to be a rounding error against
what 3.5GB of continuously-provisioned Cloud Run memory for the local
models was already costing.

DePlot/Pix2Struct (chart_interpreter.py) remain available as an
explicit zero-API-cost fallback for when no VLM captioning is
configured -- not deleted, just no longer the default path.

COST: every call here is a real LLM API request with real cost --
this is why Vision Mode is opt-in, not on by default.
"""

import base64
import io
import logging

from PIL import Image

logger = logging.getLogger("financial_rag.image_captioner")

_GENERAL_IMAGE_PROMPT = (
    "Describe this image in detail for someone who cannot see it. "
    "If it contains text, transcribe the text. If it's a diagram, "
    "screenshot, or photo, describe what it shows and any relevant "
    "detail a reader might need to answer questions about it. "
    "Be factual and specific — avoid vague descriptions like 'an image "
    "showing some content'. Keep it under 200 words."
)

_CHART_PROMPT = (
    "This image is a chart or graph. Extract its data precisely:\n"
    "1. Chart type (bar, line, pie, scatter, etc.)\n"
    "2. Axis labels and units, if present\n"
    "3. Every data point/value you can read — as a table if there are "
    "multiple series or categories (e.g. 'Category | Value')\n"
    "4. Any title, legend, or source citation shown on the chart\n"
    "5. One sentence on the overall trend or key takeaway\n"
    "Be precise with numbers — do not round or approximate values you "
    "can read directly off the chart. If a value is genuinely illegible, "
    "say so rather than guessing."
)


def _pil_to_data_url(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def caption_image_with_vlm(
    pil_image: Image.Image,
    provider: str = "google",
    model: str = "gemini-2.5-flash",
    context: str = "",
    is_chart: bool = False,
) -> str:
    """Interpret a single image region via a vision-capable LLM call.

    is_chart=True uses a prompt asking for precise structured data
    extraction (axis labels, values, trend) -- matches what DePlot used
    to do, but from a much stronger, current-generation model.
    is_chart=False uses a general "describe this image" prompt.

    Default provider/model: Gemini Flash -- cheapest/fastest option
    with a strong lineage for chart-reading specifically (see the
    session note on why: Google's own research produced DePlot/
    Pix2Struct/MatCha, and Gemini's vision is a much stronger
    descendant of that same work). Verify this model ID against
    Google's current docs before relying on it in production --
    provider model lineups change on their own release cadence.

    Returns "" on any failure (unsupported provider, model without
    vision support, API error) -- callers should treat empty string as
    "captioning unavailable for this image", not raise, since one
    failed image shouldn't abort an entire document's ingestion.
    """
    try:
        from app.providers.llm.factory import get_llm
        from langchain_core.messages import HumanMessage

        llm = get_llm(provider, model, temperature=0.0)
        data_url = _pil_to_data_url(pil_image)

        prompt = _CHART_PROMPT if is_chart else _GENERAL_IMAGE_PROMPT
        if context.strip():
            prompt += f"\n\nSurrounding document context (for reference only): {context[:300]}"

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ])

        response = llm.invoke([message])
        result = (response.content or "").strip()

        if not result:
            logger.warning("VLM captioning returned empty for provider=%s model=%s is_chart=%s",
                            provider, model, is_chart)
        return result

    except Exception as e:
        # Deliberately broad -- covers unsupported providers (e.g. the
        # Bedrock API-key auth path, which can't handle multimodal
        # content at all), models without vision support, and any
        # transient API failure. Ingestion should degrade gracefully,
        # not crash, when a single image's interpretation fails.
        logger.warning("VLM captioning failed (provider=%s model=%s is_chart=%s): %s",
                        provider, model, is_chart, e)
        return ""