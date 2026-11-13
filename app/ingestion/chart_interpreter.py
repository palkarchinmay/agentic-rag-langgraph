"""
chart_interpreter.py
--------------------
Chart/image detection, plus local-model interpretation as a FALLBACK.

UPDATED (2026-08-14): the primary interpretation path is now a hosted
VLM call (see app/ingestion/image_captioner.py), passed in here as
caption_fn. DePlot/Pix2Struct below are used ONLY when caption_fn is
None (e.g. an explicit offline/zero-API-cost mode) -- see
process_image_region's docstring for the full routing logic. Local
models measured at 30-90+ seconds per chart on CPU with the fallback
Pix2Struct step producing near-empty descriptions in practice; a hosted
VLM call measured at 1-3 seconds with real output, for negligible cost
per image.

Local models used when caption_fn is None (all free, local, no API key):
  DePlot      : google/deplot              — chart image → structured data table
  Pix2Struct  : google/pix2struct-chartqa-base — chart → narrative description

Detection (always runs, regardless of settings or which interpretation
path is active):
  compute_chart_score() — OpenCV heuristics score [0,1]
  score >= threshold     → treated as a chart

Models are lazy-loaded (first call only) and cached as singletons.
All local models run on CPU if no GPU is available.

PERFORMANCE NOTE (2026-08-13, part 1): neither .generate() call originally
set num_beams explicitly. Pix2Struct-architecture checkpoints (which
DePlot is built on) often ship a generation_config.json defaulting to
beam search -- confirmed empirically: DePlot calls were taking ~13
minutes EACH on CPU, independent of actual output length (199 vs 608
chars, same ~13 min), the signature of beam-search multiplying
per-token cost rather than output-length driving it. num_beams=1
(greedy) fixed this -- ~6x speedup confirmed on a real chart.

PERFORMANCE NOTE (2026-08-13, part 2): greedy decoding alone isn't
sufficient -- confirmed a real 10+ minute hang with ZERO output on a
conceptual 2x2 matrix diagram (no real tabular data for DePlot to
extract). Almost certainly a degenerate repetition loop: without beam
diversity or sampling, greedy decoding can get stuck repeating the same
token/short phrase, burning the full max_new_tokens budget at real
per-token CPU cost with no natural stopping point. Two fixes: (1)
repetition_penalty=1.3 makes this loop less likely to occur at all, (2)
a wall-clock StoppingCriteria (DEPLOT_TIMEOUT_SECONDS /
PIX2STRUCT_TIMEOUT_SECONDS) guarantees no single image can hang the
pipeline regardless of cause -- the hard safety net, not just a
mitigation.

Works for images extracted from: PDF, DOCX, PPTX, PPT, standalone images.
"""

import os
import io
import time
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, List
from functools import lru_cache

logger = logging.getLogger("financial_rag.chart_interpreter")

# ── Tuning ────────────────────────────────────────────────────────────────────
CHART_SCORE_THRESHOLD     = 0.42
MIN_IMAGE_SIDE_FOR_CHART  = 100    # skip tiny images (bullets, icons)
DEPLOT_MAX_TOKENS         = 256    # was 512 -- charts rarely need this many
                                     # tokens for a data table; lower cap also
                                     # bounds worst-case CPU generation time
PIX2STRUCT_MAX_TOKENS     = 128    # was 256 -- see perf note above
DEPLOT_MODEL_ID           = "google/deplot"
PIX2STRUCT_MODEL_ID       = "google/pix2struct-chartqa-base"

# Hard wall-clock ceiling per generate() call, independent of max_new_tokens.
# CONFIRMED NECESSARY (2026-08-13): a conceptual 2x2 matrix diagram (no real
# tabular data for DePlot to extract) hung for 10+ minutes with zero output
# even after num_beams=1 was set -- almost certainly greedy decoding falling
# into a degenerate repetition loop, a known failure mode when nothing
# (beam diversity, sampling, repetition penalty) pushes generation out of a
# local loop. max_new_tokens alone doesn't protect against this if the loop
# is short (e.g. repeating a single token or short phrase many times still
# burns the full token budget at real CPU per-token cost). This stopping
# criterion checks wall-clock time after every generated token and forces
# a stop once the deadline passes, regardless of cause.
DEPLOT_TIMEOUT_SECONDS     = 90
PIX2STRUCT_TIMEOUT_SECONDS = 45


def _make_timeout_stopping_criteria(timeout_seconds: float):
    """Returns a transformers StoppingCriteriaList containing one criterion
    that stops generation once a wall-clock deadline passes, checked after
    every generated token. Imported lazily, matching this file's existing
    pattern for transformers/torch imports (avoids a hard dependency at
    module-import time)."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    deadline = time.monotonic() + timeout_seconds
    warned = {"done": False}

    class _TimeoutCriteria(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            timed_out = time.monotonic() > deadline
            if timed_out and not warned["done"]:
                logger.warning(
                    "Generation stopped early: exceeded %.0fs timeout "
                    "(likely a degenerate repetition loop or genuinely "
                    "long output -- either way, bounded so it can't hang "
                    "the pipeline)", timeout_seconds
                )
                warned["done"] = True
            return timed_out

    return StoppingCriteriaList([_TimeoutCriteria()])


# ─────────────────────────────────────────────────────────────────────────────
# Lazy model singletons
# ─────────────────────────────────────────────────────────────────────────────

_deplot_model     = None
_deplot_processor = None
_pix2struct_model     = None
_pix2struct_processor = None


def _get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_deplot():
    global _deplot_model, _deplot_processor
    if _deplot_model is None:
        try:
            from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
            logger.info("Loading DePlot model (first load ~2GB download)...")
            _deplot_processor = Pix2StructProcessor.from_pretrained(DEPLOT_MODEL_ID)
            _deplot_model     = Pix2StructForConditionalGeneration.from_pretrained(
                DEPLOT_MODEL_ID
            )
            _deplot_model.eval()
            logger.info("DePlot loaded on %s", _get_device())
        except Exception as e:
            logger.warning("DePlot load failed: %s", e)
            return None, None
    return _deplot_model, _deplot_processor


def _load_pix2struct():
    global _pix2struct_model, _pix2struct_processor
    if _pix2struct_model is None:
        try:
            from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
            logger.info("Loading Pix2Struct model (~1.5GB download)...")
            _pix2struct_processor = Pix2StructProcessor.from_pretrained(PIX2STRUCT_MODEL_ID)
            _pix2struct_model     = Pix2StructForConditionalGeneration.from_pretrained(
                PIX2STRUCT_MODEL_ID
            )
            _pix2struct_model.eval()
            logger.info("Pix2Struct loaded on %s", _get_device())
        except Exception as e:
            logger.warning("Pix2Struct load failed: %s", e)
            return None, None
    return _pix2struct_model, _pix2struct_processor


# ─────────────────────────────────────────────────────────────────────────────
# Chart detection — OpenCV heuristics
# ─────────────────────────────────────────────────────────────────────────────

def compute_chart_score(image_np: np.ndarray) -> Tuple[float, dict]:
    """
    Score [0,1] — how likely is this image a chart/graph/diagram?

    Features:
      edge_density       : charts have many structural edges (axes, bars, lines)
      colour_variance    : charts use distinct colour blocks
      line_ratio         : axis lines are long horizontal/vertical edges
      unique_colour_ratio: distinct colour regions (bars, pie slices)
      text_penalty       : plain-text images penalised (high dark-pixel ratio)
    """
    h, w = image_np.shape[:2]
    if h < MIN_IMAGE_SIDE_FOR_CHART or w < MIN_IMAGE_SIDE_FOR_CHART:
        return 0.0, {"reason": "too_small"}

    grey = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) \
           if len(image_np.shape) == 3 else image_np.copy()

    # ── Edge density ──────────────────────────────────────────────────────
    edges        = cv2.Canny(grey, 50, 150)
    edge_density = float(np.sum(edges > 0)) / (h * w)

    # ── Colour variance ───────────────────────────────────────────────────
    if len(image_np.shape) == 3:
        colour_variance = float(np.std(image_np.reshape(-1, 3), axis=0).mean()) / 128.0
    else:
        colour_variance = 0.0

    # ── Structural lines (axis lines) ─────────────────────────────────────
    kh         = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    kv         = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    horiz      = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kh)
    vert       = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kv)
    line_ratio = float(np.sum(horiz > 0) + np.sum(vert > 0)) / max(float(np.sum(edges > 0)), 1.0)

    # ── Unique colour regions ─────────────────────────────────────────────
    if len(image_np.shape) == 3:
        quantised    = (image_np // 32) * 32
        unique_cols  = len(np.unique(quantised.reshape(-1, 3), axis=0))
        unique_ratio = min(unique_cols / 200.0, 1.0)
    else:
        unique_ratio = 0.0

    # ── Text penalty (dense dark pixels + low colour = plain text) ────────
    _, binary    = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY_INV)
    dark_ratio   = float(np.sum(binary > 0)) / (h * w)
    text_penalty = max(0.0, dark_ratio - 0.3) * (1.0 - colour_variance)

    # Require meaningful structural lines (axis lines) for a chart score to be high.
    # This prevents colourful ID cards/photos from scoring as charts.
    # A photo/ID card has colour but almost no long horizontal/vertical axis lines.
    line_gate = 1.0 if line_ratio > 0.05 else 0.5   # half score if no structural lines

    score = float(np.clip(
        0.30 * min(edge_density * 10, 1.0) * line_gate
      + 0.30 * colour_variance             * line_gate
      + 0.25 * line_ratio
      + 0.15 * unique_ratio
      - 0.40 * text_penalty,
        0.0, 1.0
    ))

    return score, {
        "edge_density":    round(edge_density,    3),
        "colour_variance": round(colour_variance, 3),
        "line_ratio":      round(line_ratio,      3),
        "unique_ratio":    round(unique_ratio,    3),
        "text_penalty":    round(text_penalty,    3),
        "chart_score":     round(score,           3),
    }


def is_chart(image_np: np.ndarray) -> Tuple[bool, float]:
    score, _ = compute_chart_score(image_np)
    return score >= CHART_SCORE_THRESHOLD, score


# ─────────────────────────────────────────────────────────────────────────────
# DePlot — chart → structured data table
# ─────────────────────────────────────────────────────────────────────────────

def extract_chart_data_deplot(pil_image: Image.Image) -> str:
    """
    Convert a chart image to a structured data table using DePlot.

    Example output:
      "Year | Revenue | Profit
       2021 | 120M    | 18M
       2022 | 145M    | 22M
       2023 | 180M    | 31M"

    Returns empty string if model unavailable or extraction fails.
    """
    model, processor = _load_deplot()
    if model is None:
        return ""

    try:
        import torch
        prompt = "Generate underlying data table of the figure below:"
        inputs = processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
            max_patches=512,
        )

        with torch.no_grad():
            predictions = model.generate(
                **inputs,
                max_new_tokens=DEPLOT_MAX_TOKENS,
                num_beams=1,        # greedy decoding -- see module perf note.
                do_sample=False,    # deterministic, and skips sampling overhead
                early_stopping=True,
                repetition_penalty=1.3,   # discourages the degenerate
                                            # repetition-loop failure mode
                                            # greedy decoding is prone to
                                            # without beam diversity/sampling
                stopping_criteria=_make_timeout_stopping_criteria(DEPLOT_TIMEOUT_SECONDS),
            )

        result = processor.decode(predictions[0], skip_special_tokens=True)
        # DePlot uses <0x0A> as row separator — convert to readable newlines
        result = result.replace("<0x0A>", "\n").strip()
        logger.info("DePlot extracted %d chars of table data", len(result))
        return result

    except Exception as e:
        logger.warning("DePlot inference failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Pix2Struct — chart → narrative description
# ─────────────────────────────────────────────────────────────────────────────

def describe_chart_pix2struct(pil_image: Image.Image, question: str = "") -> str:
    """
    Generate a natural-language description of a chart using Pix2Struct.

    question: optional specific question (default → general description)

    Example output:
      "The bar chart shows quarterly revenue for 2023. Q4 had the highest
       revenue at 180M, representing a 50% increase from Q1 (120M)."

    Returns empty string if model unavailable or fails.

    QUALITY NOTE: pix2struct-chartqa-base is fine-tuned for the ChartQA
    benchmark -- short, specific factual answers to specific questions
    ("What is the Q3 value?" -> "180"), NOT open-ended narrative
    description. Confirmed empirically: the generic "describe this
    chart" prompt below produced near-empty output (1-13 chars) on real
    charts. If you need real descriptions, consider either (a) asking a
    specific, ChartQA-style question instead of an open-ended one, or
    (b) dropping this step -- DePlot's structured table above is
    already the higher-value output for RAG purposes (real extractable
    numbers), and this call adds real CPU time for comparatively little
    return as currently prompted.
    """
    model, processor = _load_pix2struct()
    if model is None:
        return ""

    try:
        import torch
        prompt = question if question else "What does this chart show? Describe the key trends and values."

        inputs = processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt",
            max_patches=512,
        )

        with torch.no_grad():
            predictions = model.generate(
                **inputs,
                max_new_tokens=PIX2STRUCT_MAX_TOKENS,
                num_beams=1,        # greedy decoding -- see module perf note.
                do_sample=False,
                early_stopping=True,
                repetition_penalty=1.3,
                stopping_criteria=_make_timeout_stopping_criteria(PIX2STRUCT_TIMEOUT_SECONDS),
            )

        result = processor.decode(predictions[0], skip_special_tokens=True).strip()
        logger.info("Pix2Struct described chart: %d chars", len(result))
        return result

    except Exception as e:
        logger.warning("Pix2Struct inference failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — used by parser.py for every image region
# Works for images from: PDF pages, DOCX, PPTX, PPT (OLE), standalone images
# ─────────────────────────────────────────────────────────────────────────────

def process_image_region(
    image_np: np.ndarray,
    ocr_fn,
    context: str = "",
    use_vision: bool = True,
    caption_fn=None,
    known_chart: bool = False,
) -> Tuple[str, str]:
    """
    Decide whether to run chart interpretation, general image
    interpretation, or OCR on an image region.

    UPDATED (2026-08-14): caption_fn, when provided, is now the PRIMARY
    interpretation path for BOTH chart-detected and non-chart regions --
    not just non-chart images as before. It's expected to be a callable
    of the form fn(pil_image, context, is_chart) -> str (see
    app/ingestion/image_captioner.py's caption_image_with_vlm, typically
    passed in as a closure that fixes provider/model). DePlot+Pix2Struct
    (this file's local models) are now an explicit FALLBACK, used only
    when caption_fn is None -- e.g. an offline/zero-API-cost mode.

    DETECTION ALWAYS RUNS regardless of settings -- the OpenCV chart
    scoring is cheap (no model inference) and its result is reported back
    via the method string even when interpretation is skipped.

    Args:
        image_np    : BGR numpy array from any document type
        ocr_fn      : callable(image_np) -> str  (your OCR function)
        context     : surrounding text (page text, slide title) for context
        use_vision  : gates ALL interpretation (both chart and general
                      image), whichever path (caption_fn or local models)
                      is active. Detection always runs regardless.
        caption_fn  : optional callable(pil_image, context, is_chart) -> str.
                      When provided, this is the primary interpretation
                      path for both chart and non-chart regions. None =
                      fall back to local DePlot+Pix2Struct for charts,
                      and plain OCR for non-chart images (no captioning
                      configured at all).
        known_chart : skip the pixel-based is_chart() scoring and treat
                      this region as a confirmed chart. Use this when the
                      caller already has a stronger signal than pixel
                      heuristics can provide -- e.g. a region found via
                      clustering actual vector-drawn shapes (rects/lines/
                      curves) on a PDF page (see parser.py's
                      _cluster_vector_regions).

    Returns:
        (extracted_text, method)
        method:
          "vlm_chart"     -- chart detected, interpreted via caption_fn
          "vlm_caption"   -- non-chart image, interpreted via caption_fn
          "deplot_chart"  -- chart detected AND interpreted via local
                              DePlot+Pix2Struct (caption_fn was None)
          "chart_skipped" -- chart detected, use_vision was off
          "image_skipped" -- non-chart image, use_vision off or no
                              caption_fn and use_vision off
          "ocr"           -- fallback (interpretation returned nothing)
          "skipped"       -- image too small / invalid to process at all
    """
    if image_np is None or image_np.size == 0:
        return "", "skipped"

    h, w = image_np.shape[:2]
    if h < MIN_IMAGE_SIDE_FOR_CHART or w < MIN_IMAGE_SIDE_FOR_CHART:
        return "", "skipped"

    if known_chart:
        chart_detected, score = True, 1.0
        logger.debug("Chart region pre-confirmed via vector-object clustering (pixel scoring skipped)")
    else:
        chart_detected, score = is_chart(image_np)
        logger.debug("Chart score: %.2f | detected: %s", score, chart_detected)

    if not use_vision:
        # Detection-only mode -- report what was found, OCR as fallback
        # text, but don't spend any interpretation budget (local model
        # time or API cost).
        method = "chart_skipped" if chart_detected else "image_skipped"
        logger.info("%s (score=%.2f) but Vision Mode is off -- OCR fallback",
                     "Chart detected" if chart_detected else "Image detected", score)
        try:
            ocr_text = ocr_fn(image_np)
        except Exception as e:
            logger.warning("OCR fallback failed: %s", e)
            ocr_text = ""
        return ocr_text, method

    pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    if caption_fn is not None:
        # Primary path: unified VLM interpretation, chart-specific prompt
        # when chart_detected, general-description prompt otherwise.
        logger.info("%s -- running VLM interpretation",
                     "Chart detected" if chart_detected else "Image detected")
        result = caption_fn(pil_image, context, chart_detected)
        if result.strip():
            tag = "[Chart Data]" if chart_detected else "[Image Description]"
            method = "vlm_chart" if chart_detected else "vlm_caption"
            return f"{tag}\n{result}", method
        logger.warning("VLM interpretation returned nothing -- falling back to OCR")

    elif chart_detected:
        # No VLM configured -- fall back to local DePlot+Pix2Struct for
        # charts specifically (zero API cost, but see this file's perf
        # notes on local-model speed/quality tradeoffs).
        logger.info("Chart detected (score=%.2f) -- no VLM configured, running local DePlot + Pix2Struct", score)
        parts = []

        table_data = extract_chart_data_deplot(pil_image)
        if table_data.strip():
            parts.append(f"[Chart Data]\n{table_data}")

        question = (
            f"Context: {context[:200]}\n"
            f"What does this chart show? Describe key trends and values."
        ) if context.strip() else ""
        description = describe_chart_pix2struct(pil_image, question)
        if description.strip():
            parts.append(f"[Chart Description]\n{description}")

        if parts:
            return "\n\n".join(parts), "deplot_chart"

        logger.warning("Local chart models returned empty -- falling back to OCR")

    # -- OCR fallback (no caption_fn for non-chart, or interpretation
    # returned nothing above) --------------------------------------------
    try:
        ocr_text = ocr_fn(image_np)
    except Exception as e:
        logger.warning("OCR failed for image region: %s", e)
        ocr_text = ""
    return ocr_text, "ocr"