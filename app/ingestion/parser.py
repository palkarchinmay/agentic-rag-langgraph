"""
parser.py — Document parsers for PDF, DOCX, PPTX, PPT, images, text.

All parse_* functions accept:
    file        : file-like object (NamedBytesIO or open file)
    ocr_method  : 'easyocr' | 'pytesseract' | 'textract' | 'docai' | 'azure_di'
                  | 'vision' (any vision LLM, chosen by VISION_MODEL)
    use_vision  : bool — route chart images through DePlot/Pix2Struct (True)
                         or always use OCR (False, faster, offline)
    caption_images / caption_provider / caption_model
                : send image regions to a vision LLM via image_captioner.
                  provider/model default to VISION_MODEL
                  (provider:model, default anthropic:claude-opus-5-5).
"""

import io
import os
import re
import cv2
import tempfile
import numpy as np
import logging
from PIL import Image

import pdfplumber
from docx import Document as DocxDocument

from app.ingestion.cleaner import clean_ocr_text
from app.ingestion.schema import Document
from app.ingestion.ocr import extract_text, normalize_method
from app.ingestion.chart_interpreter import process_image_region
from app.ingestion.layout import extract_text_reading_order
from app.ingestion import doc_title as _doc_title
from app.security.pii import mask_pii

logger = logging.getLogger("financial_rag.parser")

MIN_TEXT_CHARS_FOR_DIGITAL = 30
MAX_IMAGE_SIDE_PX          = 2000
PDF_RENDER_DPI_LARGE       = 150
PDF_RENDER_DPI_SMALL       = 300


# ── Image helpers ─────────────────────────────────────────────────────────────

def preprocess_for_ocr(image_np: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE_PX:
        scale    = MAX_IMAGE_SIDE_PX / max(h, w)
        image_np = cv2.resize(image_np, (int(w*scale), int(h*scale)),
                              interpolation=cv2.INTER_AREA)
    grey     = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) \
               if len(image_np.shape) == 3 else image_np
    denoised = cv2.fastNlMeansDenoising(grey, h=10, templateWindowSize=7,
                                         searchWindowSize=21)
    binary   = cv2.adaptiveThreshold(denoised, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 31, 10)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _pil_to_np(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


# OCR methods that call a remote API per page (more parallel workers).
_CLOUD_OCR = {"docai", "textract", "azure_di", "vision"}


def _resize_only(image_np: np.ndarray) -> np.ndarray:
    h, w = image_np.shape[:2]
    if max(h, w) > MAX_IMAGE_SIDE_PX:
        scale = MAX_IMAGE_SIDE_PX / max(h, w)
        image_np = cv2.resize(image_np, (int(w*scale), int(h*scale)),
                              interpolation=cv2.INTER_AREA)
    return image_np


def _make_ocr_fn(ocr_method: str):
    method = normalize_method(ocr_method)

    def ocr_fn(image_np):
        if method == "vision":
            # Vision LLMs read colour charts and layout; grayscale +
            # adaptive threshold would destroy exactly that. Resize only.
            # The output is Markdown, so skip the OCR-noise cleaner too.
            return extract_text(_resize_only(image_np), method=method)
        processed = preprocess_for_ocr(image_np)
        raw       = extract_text(processed, method=method)
        return clean_ocr_text(raw)
    return ocr_fn


def _resolve_caption_model(caption_provider=None, caption_model=None):
    """Explicit provider/model win; otherwise VISION_MODEL (provider:model)."""
    if caption_provider and caption_model:
        return caption_provider, caption_model
    from app.providers.llm.factory import parse_model_spec, default_model
    provider, model = parse_model_spec(
        os.getenv("VISION_MODEL", "anthropic:claude-opus-5-5"))
    provider = caption_provider or provider
    return provider, caption_model or model or default_model(provider)


def _make_caption_fn(caption_images: bool, caption_provider=None, caption_model=None):
    """One place that builds the caption callback for every parser."""
    if not caption_images:
        return None
    from app.ingestion.image_captioner import caption_image_with_vlm
    provider, model = _resolve_caption_model(caption_provider, caption_model)

    def caption_fn(pil_img, ctx, is_chart):
        return caption_image_with_vlm(pil_img, provider, model, ctx, is_chart)
    return caption_fn


def _interpret_scanned_page(image_np, ocr_fn, ocr_method, context="",
                            use_vision=True, caption_fn=None) -> list:
    """
    (content, method) pairs for one scanned page.

    - OCR method 'vision': one full-page transcription (text + tables +
      charts as Markdown). Captioning the whole page as well would pay twice.
    - Otherwise as before (process_image_region), with one fix: when the page
      isn't a chart, the caption prompt returns a <200-word description, which
      used to REPLACE the page's OCR text. A scanned text page then lost most
      of its words. Now the OCR text is kept and the description added.
    """
    if normalize_method(ocr_method) == "vision":
        text = ocr_fn(image_np)
        return [(text, "ocr")] if text.strip() else []

    content, method = process_image_region(
        image_np, ocr_fn, context=context,
        use_vision=use_vision, caption_fn=caption_fn,
    )
    outputs = []
    if method in ("vlm_caption", "image_ignored"):
        # image_ignored (model said logo/decorative) on a WHOLE scanned page
        # is a misjudgement worth hedging: keep the page's OCR text.
        ocr_text = ocr_fn(image_np)
        if ocr_text.strip():
            outputs.append((ocr_text, "ocr"))
    if content.strip():
        outputs.append((content, method))
    return outputs


# ── Section assembler ────────────────────────────────────────────────────────

def _assemble(text="", table_text="", image_outputs=None) -> str:
    sections    = []
    ocr_parts   = []
    chart_parts = []

    if text.strip():
        sections.append(f"[TEXT]\n{text.strip()}")
    if table_text.strip():
        sections.append(f"[TABLES]\n{table_text.strip()}")

    for content, method in (image_outputs or []):
        if not content.strip():
            continue
        if method in ("deplot_chart", "vlm_chart"):
            chart_parts.append(content)
        else:
            ocr_parts.append(content)

    if ocr_parts:
        sections.append(f"[OCR]\n" + "\n".join(ocr_parts))
    if chart_parts:
        sections.append(f"[CHARTS]\n" + "\n\n".join(chart_parts))

    return "\n\n".join(sections)


_WORD_RE = None


def _containment(part: str, whole: str) -> float:
    """Share of `part`'s distinct words (3+ letters, or numbers) also in `whole`."""
    global _WORD_RE
    import re
    _WORD_RE = _WORD_RE or re.compile(r"\d[\d.,]*\d%?|\d%?|[a-z][a-z'-]{2,}")
    pw = set(_WORD_RE.findall(part.lower()))
    if not pw:
        return 0.0
    ww = set(_WORD_RE.findall(whole.lower()))
    return len(pw & ww) / len(pw)


def _has_structure(content: str) -> bool:
    """Tables ('a | b' rows) or diagram arrows: the pairing/flow is new
    information even when every word also appears in the page text."""
    table_rows = sum(1 for line in content.splitlines() if line.count("|") >= 2)
    return table_rows >= 2 or "->" in content or "\u2192" in content


def _repeats_page_text(content: str, page_text: str, threshold: float = 0.8) -> bool:
    """True when a region's text/description only repeats the page text:
    >= threshold of its words are on the page and it adds no table/diagram
    structure."""
    if not content.strip() or _has_structure(content):
        return False
    return _containment(content, page_text) >= threshold


def _overlap_ratio(a: str, b: str) -> float:
    wa = set(a.lower().split()); wb = set(b.lower().split())
    if not wa or not wb: return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


# ── PDF chart region detection ──────────────────────────────────────────────
# Charts exported from Excel/PowerPoint/matplotlib to PDF are very often
# drawn with vector graphics (rects for bars, lines for axes/gridlines),
# NOT embedded as a raster image object -- page.images (pdfplumber's
# bitmap-object list) simply can't find them. This clusters nearby vector
# shapes into candidate chart regions, crops the rendered page image to
# just that region, and lets process_image_region() score/interpret the
# isolated crop instead of an entire page (whole-page scoring dilutes the
# chart-specific visual signal below the detection threshold when a chart
# only occupies part of a text-heavy page -- confirmed empirically before
# building this).

_MIN_CLUSTER_OBJECTS  = 6    # minimum vector shapes to call something a chart region
_CLUSTER_MERGE_PAD_PT = 15   # merge shapes within this many PDF points of each other
_REGION_EXPAND_PT     = 20   # expand final region bbox to catch axis labels/titles


def _cluster_vector_regions(page, pad: float = _REGION_EXPAND_PT) -> list:
    """Returns a list of (x0, top, x1, bottom) bounding boxes in PDF point
    coordinates (top-down, matching image conventions) -- one per dense
    cluster of vector-drawn shapes on the page, expanded by pad."""
    objects = []
    for r in page.rects:
        objects.append((r["x0"], r["top"], r["x1"], r["bottom"]))
    for l in page.lines:
        objects.append((l["x0"], l["top"], l["x1"], l["bottom"]))
    for c in page.curves:
        objects.append((c["x0"], c["top"], c["x1"], c["bottom"]))

    n = len(objects)
    if n < _MIN_CLUSTER_OBJECTS:
        return []

    # Union-find over padded-bbox overlap -- standard connected-components
    # clustering, O(n^2) comparisons (fine for the tens-to-low-hundreds of
    # vector objects a typical page has).
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    def overlaps_padded(a, b, pad):
        ax0, ay0, ax1, ay1 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
        bx0, by0, bx1, by1 = b
        return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)

    for i in range(n):
        for j in range(i + 1, n):
            if overlaps_padded(objects[i], objects[j], _CLUSTER_MERGE_PAD_PT):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(objects[i])

    regions = []
    for group in groups.values():
        if len(group) < _MIN_CLUSTER_OBJECTS:
            continue
        x0 = min(o[0] for o in group) - pad
        y0 = min(o[1] for o in group) - pad
        x1 = max(o[2] for o in group) + pad
        y1 = max(o[3] for o in group) + pad
        regions.append((x0, y0, x1, y1))

    return regions


def _crop_region(pil_image: Image.Image, bbox_pt: tuple, dpi: int) -> Image.Image:
    """Crop a rendered page image to a region given in PDF point coordinates,
    converting via the DPI the page was rendered at. Clips to image bounds."""
    scale = dpi / 72.0
    x0, y0, x1, y1 = bbox_pt
    px_box = (
        max(0, int(x0 * scale)),
        max(0, int(y0 * scale)),
        min(pil_image.width,  int(x1 * scale)),
        min(pil_image.height, int(y1 * scale)),
    )
    return pil_image.crop(px_box)


# Region growth: a vector cluster or an embedded image usually stops at the
# plot area, but the chart's title, axis labels, legend and source line are
# text objects printed just outside it. Those words are pulled in when they
# sit within a line or so of the region and overlap it horizontally or
# vertically; repeated until nothing new is close. All thresholds are
# relative to the page's median word height, so they hold at any font size.
_GROW_GAP_LINES   = 2.0   # how far (in word heights) a word may sit from the region
_GROW_MAX_ROUNDS  = 6
_GROW_MAX_WORDS   = 14    # a line above/below longer than this is body text, not a title
_GROW_MAX_ABOVE   = 3     # title lines (figure number, title, subtitle)
_GROW_MAX_BELOW   = 6     # axis labels, legend, source, notes
_GROW_SIDE_WORDS  = 4     # a line left/right longer than this is body text, not a label
_REGION_TEXT_CHARS = 1500


def _median_word_height(words) -> float:
    hs = sorted(w["bottom"] - w["top"] for w in words if w["bottom"] > w["top"])
    return hs[len(hs) // 2] if hs else 10.0


def _word_lines(words, tol):
    """Group words into visual lines: same top within tol, and split where
    the horizontal gap is wider than 1.6*tol (0.8 word heights: wider than
    a word space, narrower than a column gutter), so two columns side by
    side stay two separate lines."""
    rows = []
    for w in sorted(words, key=lambda w: w["top"]):
        if rows and abs(rows[-1][0]["top"] - w["top"]) <= tol:
            rows[-1].append(w)
        else:
            rows.append([w])
    lines = []
    for row in rows:
        cur = None
        for w in sorted(row, key=lambda w: w["x0"]):
            if cur is not None and w["x0"] - cur["x1"] <= 1.6 * tol:
                cur["words"].append(w)
                cur["x1"] = max(cur["x1"], w["x1"])
                cur["top"] = min(cur["top"], w["top"])
                cur["bottom"] = max(cur["bottom"], w["bottom"])
            else:
                cur = {"top": w["top"], "bottom": w["bottom"],
                       "x0": w["x0"], "x1": w["x1"], "words": [w]}
                lines.append(cur)
    return lines


def _inside(w, bbox, slack=0.5):
    cx, cy = (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2
    return bbox[0] - slack <= cx <= bbox[2] + slack and bbox[1] - slack <= cy <= bbox[3] + slack


_CAPTION_LINE_RE = re.compile(
    r"(?i)^\W*(figure|fig\.|chart|table|exhibit|graph|diagram|plate|map)\s*[\dA-Z]+\b")
_FOOTER_LINE_RE = re.compile(r"(?i)^\W*(sources?|notes?|data)\s*[:.]")
_BODY_RUN_LINES = 3      # aligned prose lines in a row = a text column
_PROSE_WORDS    = 4
_PROSE_CHARS    = 20
_BODY_PITCH     = 1.6    # max top-to-top distance of consecutive body lines


def _mark_body_lines(lines, h):
    """Flag lines that belong to running text (a paragraph / column), so a
    figure never grows into them. Running text = at least _BODY_RUN_LINES
    prose-like lines stacked at normal line spacing with the same left
    edge; every line lying inside the column that run occupies (short last
    lines, indented first lines, bullets) belongs to it too. Figure titles (1-2 lines),
    axis labels and legends do not form such runs."""
    def prose(l):
        txt = " ".join(w["text"] for w in l["words"])
        return len(l["words"]) >= _PROSE_WORDS and len(txt) >= _PROSE_CHARS
    for l in lines:
        l["_body"] = False
        l["_incol"] = False
    cand = sorted((l for l in lines if prose(l)), key=lambda l: l["top"])
    runs, used = [], set()
    for i, l in enumerate(cand):
        if i in used:
            continue
        run, last = [l], l
        for j in range(i + 1, len(cand)):
            m = cand[j]
            if m["top"] - last["top"] > 2.2 * h:
                if m["top"] - last["top"] > 6 * h:
                    break
                continue
            if j not in used and abs(m["x0"] - last["x0"]) <= h and m["top"] > last["top"]:
                run.append(m); used.add(j); last = m
        if len(run) >= _BODY_RUN_LINES:
            runs.append(run)
    for run in runs:
        # the column the run occupies; lines fully inside it (indented first
        # lines, bullets, short last lines) are running text too
        xs0 = sorted(l["x0"] for l in run)
        xs1 = sorted(l["x1"] for l in run)
        cx0 = xs0[len(xs0) // 2] - h
        cx1 = xs1[(3 * len(xs1)) // 4] + h
        top, bottom = run[0]["top"] - 1.2 * h, run[-1]["bottom"] + 1.2 * h
        for l in lines:
            if cx0 <= l["x0"] and l["x1"] <= cx1 and top <= l["top"] <= bottom:
                l["_body"] = True
        # carry on through the column while lines follow at line pitch
        # (a bullet list after the paragraph, a short heading inside it)
        col = sorted((l for l in lines if cx0 <= l["x0"] and l["x1"] <= cx1),
                     key=lambda l: l["top"])
        # a heading of this column (left-aligned, inside it) is not a label
        # of a figure next to it; vertical growth may still take it
        for l in col:
            if abs(l["x0"] - (cx0 + h)) <= h:
                l["_incol"] = True
        for seq in (col, col[::-1]):
            prev = None
            for l in seq:
                if l["_body"]:
                    prev = l
                elif prev is not None and abs(l["top"] - prev["top"]) <= _BODY_PITCH * h:
                    l["_body"] = True
                    prev = l
                else:
                    prev = None


def _clip_to_text(bbox, core, lines):
    """Shrink the padding of bbox where it would cut into running text
    lying outside core (the figure itself)."""
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = core
    for l in lines:
        if not l.get("_body"):
            continue
        lx0, ly0, lx1, ly1 = l["x0"], l["top"], l["x1"], l["bottom"]
        if min(x1, lx1) <= max(x0, lx0) or min(y1, ly1) <= max(y0, ly0):
            continue                                     # not in the box
        if min(cx1, lx1) > max(cx0, lx0) and min(cy1, ly1) > max(cy0, ly0):
            continue                                     # overlaps the figure itself
        if lx0 >= cx1 - 1:
            x1 = min(x1, lx0 - 1)
        elif lx1 <= cx0 + 1:
            x0 = max(x0, lx1 + 1)
        elif ly1 <= cy0 + 1:
            y0 = max(y0, ly1 + 1)
        elif ly0 >= cy1 - 1:
            y1 = min(y1, ly0 - 1)
    return (x0, y0, x1, y1)


def _grow_region(bbox, lines, h, page_w=None, page_h=None):
    """Extend bbox over nearby short text lines (titles, axis labels,
    legends, source notes). A line qualifies when it is within
    _GROW_GAP_LINES word heights of the box and lies within its x-span
    (above / below, at most _GROW_MAX_ABOVE / _GROW_MAX_BELOW text rows,
    and not past a caption line such as 'Figure 3 ...') or is
    a short label beside it. Long lines and lines that mostly stick out of
    the box are body text and are left alone, so a chart never swallows the
    paragraph or column next to it."""
    x0, y0, x1, y1 = bbox
    gap = _GROW_GAP_LINES * h
    # a box that already spans (almost) the whole page height has nothing
    # above/below to take, and every line beside it overlaps it: only
    # grow in the direction that still makes sense
    grow_v = not page_h or (y1 - y0) < 0.8 * page_h
    grow_h = not page_w or (x1 - x0) < 0.8 * page_w
    above, below = set(), set()  # text rows added outside the box, per side
    stop_up = stop_down = False  # passed a caption line ("Figure 3 ...")
    row = lambda y: round(y / (0.5 * h))
    for _ in range(_GROW_MAX_ROUNDS):
        grew = False
        for ln in lines:
            if ln.get("_used"):
                continue
            if len(ln["words"]) > _GROW_MAX_WORDS or ln.get("_body"):
                continue
            lx0, ly0, lx1, ly1 = ln["x0"], ln["top"], ln["x1"], ln["bottom"]
            short = len(ln["words"]) <= _GROW_SIDE_WORDS
            x_ov = min(x1, lx1) - max(x0, lx0)
            y_ov = min(y1, ly1) - max(y0, ly0)
            # "belongs to the figure's columns": most of the line lies within
            # the region's x-span, or it starts at the region's left edge
            # (a left-aligned title wider than the plot).
            x_fit = x_ov > 0 and (x_ov >= 0.5 * (lx1 - lx0) or abs(lx0 - x0) <= gap)
            up = x_fit and 0 <= y0 - ly1 <= gap and not stop_up and \
                (len(above) < _GROW_MAX_ABOVE or row(ly0) in above)
            text_ln = " ".join(w["text"] for w in ln["words"])
            # a 'Source: ...' / 'Note: ...' line often sits a little lower
            gap_dn = gap * 1.5 if _FOOTER_LINE_RE.match(text_ln) else gap
            down = x_fit and 0 <= ly0 - y1 <= gap_dn and not stop_down and \
                (len(below) < _GROW_MAX_BELOW or row(ly0) in below)
            near_v = grow_v and (up or down)
            # Side neighbours must be short (axis ticks, legend entries):
            # a body-text column beside the chart is never pulled in.
            near_h = grow_h and grow_v and y_ov > 0 and short and not ln.get("_incol") and \
                (0 <= x0 - lx1 <= gap or 0 <= lx0 - x1 <= gap)
            inside = y_ov > 0 and x_fit
            if near_v or near_h or inside:
                ln["_used"] = True
                if inside and not (near_v or near_h) and \
                        max(x0 - lx0, lx1 - x1, y0 - ly0, ly1 - y1) > gap:
                    # a line that sticks far out of the box (a heading
                    # running across a photo, say): read it, don't widen to it
                    continue
                caption = bool(_CAPTION_LINE_RE.match(text_ln))
                if up and not inside:
                    above.add(row(ly0))
                    stop_up = stop_up or caption      # the figure's own title
                if down and not inside:
                    below.add(row(ly0))
                    stop_down = stop_down or caption  # caption-below style
                nb = (min(x0, lx0), min(y0, ly0), max(x1, lx1), max(y1, ly1))
                if nb != (x0, y0, x1, y1):
                    x0, y0, x1, y1 = nb
                    grew = True
        if not grew:
            break
    return (x0, y0, x1, y1)


def _region_text(words, bbox, h) -> str:
    inside = [w for w in words if _inside(w, bbox)]
    lines = _word_lines(inside, h * 0.5)
    lines.sort(key=lambda l: (l["top"], l["x0"]))
    out = [" ".join(w["text"] for w in sorted(l["words"], key=lambda w: w["x0"])) for l in lines]
    return "\n".join(out)[:_REGION_TEXT_CHARS]


def _region_context(region_text: str, page_text: str) -> str:
    """Context handed to the caption model for one region: the exact words
    printed inside the (grown) region, then the start of the page text."""
    parts = []
    if region_text.strip():
        parts.append("Text printed inside this figure in the PDF (exact, use it "
                     "for the title, labels and numbers):\n" + region_text.strip())
    if page_text.strip():
        parts.append("Start of the page text:\n" + page_text.strip()[:400])
    return "\n\n".join(parts)


def _find_image_regions(page, pil_image: Image.Image, dpi: int) -> list:
    """Returns a list of (cropped_pil_image, is_known_chart, region_text)
    tuples -- one per embedded raster image AND one per detected
    vector-drawn chart cluster on this page.

    Raster images get is_known_chart=False (a raster could be a chart, a
    photo or a logo). Vector clusters get is_known_chart=True.

    Each region is first grown over the nearby short text lines (title,
    axis labels, legend, source) so the crop shows the whole figure, and
    the native PDF words inside the final box are returned as region_text.
    The caption model gets that text as exact context, so titles and labels
    are copied rather than re-read from pixels."""
    try:
        words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
        from app.ingestion.layout import _merge_spaced_words
        words = _merge_spaced_words(
            words, [c for c in page.chars if not c.get("text", "x").strip()])
    except Exception:
        words = []
    h = _median_word_height(words)
    lines = _word_lines(words, h * 0.5) if words else []

    if lines:
        _mark_body_lines(lines, h)

    page_area = float(page.width * page.height) or 1.0
    raw = [((img["x0"], img["top"], img["x1"], img["bottom"]), False) for img in page.images]
    raw += [(bbox, True) for bbox in _cluster_vector_regions(page, pad=0)]

    crops = []
    for core, known in raw:
        area = max(0.0, core[2] - core[0]) * max(0.0, core[3] - core[1])
        if area >= 0.6 * page_area or not lines:
            # Full-page background art: the page text is the context; no
            # growth (it would absorb every line on the page).
            crops.append((_crop_region(pil_image, core, dpi), known, ""))
            continue
        for ln in lines:
            ln.pop("_used", None)
        grown = _grow_region(core, lines, h, float(page.width), float(page.height))
        # Margin for unlabeled tick marks / anti-aliasing, cut back where
        # it would reach into running text beside the figure.
        pad = _REGION_EXPAND_PT if known else 0.4 * h
        bbox = _clip_to_text((grown[0] - pad, grown[1] - pad,
                              grown[2] + pad, grown[3] + pad), grown, lines)
        text = _region_text(words, bbox, h)
        crops.append((_crop_region(pil_image, bbox, dpi), known, text))

    return crops


# ── PDF parser ────────────────────────────────────────────────────────────────

def parse_pdf(file, ocr_method: str = "easyocr", use_vision: bool = True,
               caption_images: bool = False, caption_provider: str = None,
               caption_model: str = None):
    """
    Parse PDF with parallel page processing.
    Cloud OCR backends (docai, textract, azure_di) benefit most from parallelism
    since each page is an independent API call.
    Local OCR (easyocr) uses 2 workers to avoid memory pressure.
    """
    caption_fn = _make_caption_fn(caption_images, caption_provider, caption_model)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    filename = getattr(file, "name", str(file))
    ocr_fn   = _make_ocr_fn(ocr_method)

    # Cloud OCR = more parallel workers (each page = independent API call)
    # Local OCR  = fewer workers (shared GPU/CPU memory)
    cloud_ocr = normalize_method(ocr_method) in _CLOUD_OCR
    max_workers = 4 if cloud_ocr else 2

    def _process_page(args):
        i, page_data = args
        page_w, text, table_text, pil_image, is_scanned, image_regions = page_data

        image_np      = _pil_to_np(pil_image) if pil_image is not None else None
        image_outputs = []

        # ── Digital page: native text + tables + optional chart extraction ────
        if not is_scanned:
            if image_regions:
                # One process_image_region() call per isolated, cropped
                # region (embedded raster image or detected vector-chart
                # cluster) -- NOT one call scoring the whole page. Whole-
                # page scoring dilutes the chart-specific visual signal
                # below the detection threshold once a chart only occupies
                # part of a text-heavy page (confirmed empirically before
                # this fix — a real bar chart on a mostly-text page scored
                # 0.13-0.37, well under the 0.42 detection threshold, even
                # though the same chart cropped in isolation scores high).
                print(f"  📄 Page {i+1}: digital + {len(image_regions)} region(s) to scan ({len(text)} chars)")
                for region, is_known_chart, region_text in image_regions:
                    region_np = _pil_to_np(region)
                    try:
                        content, method = process_image_region(
                            region_np, ocr_fn,
                            context=_region_context(region_text, text),
                            use_vision=use_vision,
                            caption_fn=caption_fn,
                            known_chart=is_known_chart,
                        )
                        if method in ("deplot_chart", "vlm_chart") and content.strip():
                            image_outputs.append((content, method))
                        elif method in ("chart_skipped", "vlm_caption", "image_skipped"):
                            # Always record the detection signal for
                            # has_charts/has_images metadata, even when
                            # fallback OCR content is empty -- common for a
                            # pure bar chart with no readable text in it.
                            # But if the region is a text box / sidebar whose
                            # words are already in the page text, keep only
                            # the signal: a second copy just takes retrieval
                            # slots from other passages. Charts are never
                            # dropped (their label/value pairing is new info
                            # even when the bare numbers are in the text).
                            if _repeats_page_text(content, text):
                                content = ""
                            image_outputs.append((content, method))
                        elif content.strip() and _overlap_ratio(text, content) < 0.4:
                            # Genuinely new content (not duplicate of native text)
                            image_outputs.append((content, method))
                    except Exception as e:
                        logger.debug("Page %d region scan skipped: %s", i+1, e)
            else:
                print(f"  📄 Page {i+1}: digital ({len(text)} chars) — native only")

            assembled = _assemble(text, table_text, image_outputs)
            if not assembled.strip():
                return None
            return (i, Document(
                content=mask_pii(assembled),
                metadata={
                    "source":     filename,
                    "page":       i,
                    "type":       "pdf",
                    "scanned":    False,
                    "ocr_method": "native" + ("+vision" if image_outputs else ""),
                    "has_charts":         any(m in ("deplot_chart", "vlm_chart", "chart_skipped") for _, m in image_outputs),
                    "charts_interpreted": any(m in ("deplot_chart", "vlm_chart") for _, m in image_outputs),
                    "has_images":         any(m in ("vlm_caption", "image_skipped") for _, m in image_outputs),
                    "images_captioned":   any(m == "vlm_caption" for _, m in image_outputs),
                }
            ))

        # ── Scanned page: full OCR required ──────────────────────────────────
        print(f"  📄 Page {i+1}: scanned → {ocr_method}")
        if image_np is None:
            return None
        try:
            image_outputs.extend(_interpret_scanned_page(
                image_np, ocr_fn, ocr_method,
                context=f"Page {i+1} of {filename}",
                use_vision=use_vision, caption_fn=caption_fn,
            ))
        except Exception as e:
            logger.warning("Page %d OCR failed: %s", i+1, e)

        assembled = _assemble("", table_text, image_outputs)
        if not assembled.strip():
            return None
        return (i, Document(
            content=mask_pii(assembled),
            metadata={
                "source":     filename,
                "page":       i,
                "type":       "pdf",
                "scanned":    True,
                "ocr_method": ocr_method,
                "has_charts":         any(m in ("deplot_chart", "vlm_chart", "chart_skipped") for _, m in image_outputs),
                "charts_interpreted": any(m in ("deplot_chart", "vlm_chart") for _, m in image_outputs),
                "has_images":         any(m in ("vlm_caption", "image_skipped") for _, m in image_outputs),
                "images_captioned":   any(m == "vlm_caption" for _, m in image_outputs),
            }
        ))

    # Pre-render pages: scanned pages always, digital pages only if they
    # actually contain embedded images (checked cheaply via page.images,
    # no rendering needed for that check itself). Purely text-only digital
    # pages still skip rendering entirely (fast path preserved).
    page_data = []
    with pdfplumber.open(file) as pdf:
        doc_title = _doc_title.pdf_title(pdf, filename)
        for i, page in enumerate(pdf.pages):
            page_w     = page.width or 595
            dpi        = PDF_RENDER_DPI_LARGE if page_w > 500 else PDF_RENDER_DPI_SMALL
            # Reading order (multi-column safe); see layout.py
            text       = extract_text_reading_order(page)
            is_scanned = len(text.strip()) < MIN_TEXT_CHARS_FOR_DIGITAL

            # Extract tables for digital pages
            table_text = ""
            if not is_scanned:
                for table in (page.extract_tables() or []):
                    for row in table:
                        table_text += " | ".join(str(c) if c else "" for c in row) + "\n"

            # Render image:
            # - Always for scanned pages (need OCR)
            # - For digital pages, only if the page has SOMETHING worth
            #   looking at: embedded raster images (page.images, cheap
            #   metadata check) OR a dense cluster of vector-drawn shapes
            #   (rects/lines/curves) that looks like a chart (charts
            #   exported from Excel/PowerPoint/matplotlib are very often
            #   drawn as vectors, not embedded bitmaps -- page.images
            #   alone can't find those). Both checks are metadata-only,
            #   no rendering needed to decide.
            # This lets detection run independent of use_vision (only the
            # DePlot/Pix2Struct/VLM interpretation itself stays gated,
            # inside process_image_region) while still skipping the real
            # rendering cost for pages that genuinely have neither.
            vector_regions_pt = _cluster_vector_regions(page)
            has_embedded_images = bool(page.images) or bool(vector_regions_pt)
            pil_image = None
            image_regions = []  # list of already-cropped PIL images, filled below
            if is_scanned or use_vision or has_embedded_images:
                try:
                    pil_image = page.to_image(resolution=dpi).original
                    if not is_scanned and pil_image is not None:
                        # Crop NOW, while `page` is still open/valid --
                        # _process_page runs later in a thread pool after
                        # the pdfplumber.open() context has closed, so
                        # page.images/rects/etc. won't be accessible there.
                        image_regions = _find_image_regions(page, pil_image, dpi)
                except Exception as e:
                    logger.warning("Page %d render failed: %s", i+1, e)
                    if is_scanned:
                        continue  # scanned page useless without image
                    # digital page can still use native text

            page_data.append((i, (page_w, text, table_text, pil_image, is_scanned, image_regions)))

    # Process pages in parallel
    # Digital pages complete instantly (no OCR), scanned pages use cloud/local OCR
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_page, pd): pd[0] for pd in page_data}
        for future in as_completed(futures):
            result = future.result()
            if result:
                idx, doc = result
                results[idx] = doc

    # Return in original page order
    return _doc_title.tag([results[i] for i in sorted(results.keys())], doc_title)


# ── DOCX parser ───────────────────────────────────────────────────────────────

def parse_docx(file, ocr_method: str = "easyocr", use_vision: bool = True,
                caption_images: bool = False, caption_provider: str = None,
                caption_model: str = None):
    """
    Parse DOCX with section-aware chunking.
    Splits on Heading 1/2 paragraphs to produce section-level documents
    with heading metadata for citations.
    """
    caption_fn = _make_caption_fn(caption_images, caption_provider, caption_model)

    doc      = DocxDocument(file)
    filename = getattr(file, "name", str(file))
    doc_title = _doc_title.docx_title(doc, filename)
    ocr_fn   = _make_ocr_fn(ocr_method)

    # ── Extract tables ────────────────────────────────────────────────────────
    table_rows = []
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells)
            if row_text.strip():
                table_rows.append(row_text)
    table_text = "\n".join(table_rows)

    # ── Extract images ─────────────────────────────────────────────────────
    # ALWAYS attempt extraction + process_image_region — detection runs
    # unconditionally inside process_image_region (cheap, OpenCV-only);
    # only actual chart interpretation / VLM captioning are gated by
    # use_vision / caption_images respectively. Previously this whole
    # block was skipped when use_vision=False, meaning DOCX images were
    # never even looked at for chart/image detection in that case.
    image_outputs = []
    try:
        all_text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
        for rel in doc.part._rels.values():
            if "image" not in rel.target_ref:
                continue
            try:
                pil_image = Image.open(io.BytesIO(rel.target_part.blob)).convert("RGB")
                image_np  = _pil_to_np(pil_image)
                c, method = process_image_region(
                    image_np, ocr_fn, context=all_text[:300], use_vision=use_vision,
                    caption_fn=caption_fn,
                )
                if c.strip():
                    image_outputs.append((c, method))
            except Exception as e:
                logger.warning("DOCX image OCR failed: %s", e)
    except Exception as e:
        logger.warning("DOCX image extraction failed: %s", e)

    # ── Section-aware splitting ───────────────────────────────────────────────
    # Split on Heading 1 and Heading 2 paragraphs
    sections     = []
    current_head = ""
    current_text = []

    def _is_section_title(para) -> bool:
        """
        Detect section titles even when not styled as Heading.
        Catches: short lines, bold text, ALL CAPS, ends without period.

        EXCLUDES: bold "Label: Value" data lines (e.g. "Mean Claim Cost
        (Litigated): $266,758.15") — these are commonly bolded in docx for
        emphasis but are data points, not section titles. Misclassifying
        them as headings silently drops the actual value from indexed
        content (only the heading text survives, and even that gets
        overwritten if consecutive bold data lines appear back-to-back
        with no body text between them).
        """
        text  = para.text.strip()
        style = para.style.name if para.style else ""

        if not text or len(text) > 80:
            return False
        if style.startswith("Heading"):
            return True

        # A "Label: Value" data line — ends with a number, currency amount,
        # or percentage after a colon. Real section titles don't end this
        # way ("Model Evaluation", "Key Drivers") — data points do
        # ("Mean Claim Cost (Litigated): $266,758.15", "Recall: 0.86").
        import re
        if re.search(r":\s*[\$\-]?[\d,]+\.?\d*%?\s*$", text):
            return False

        # Bold paragraph = likely a title
        if para.runs and all(r.bold for r in para.runs if r.text.strip()):
            return True
        # ALL CAPS short line
        if text.isupper() and len(text) > 3:
            return True
        # Title case short line without period (e.g. "Model Evaluation")
        if (text.istitle() and not text.endswith(".")
                and len(text.split()) <= 6 and len(text) > 5):
            return True
        return False

    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        if _is_section_title(para):
            # Save previous section
            if current_text:
                sections.append((current_head, "\n".join(current_text)))
            current_head = para.text.strip()
            current_text = []
        else:
            current_text.append(para.text)

    # Last section
    if current_text:
        sections.append((current_head, "\n".join(current_text)))

    # If no headings found, treat as single document
    if not sections:
        all_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        sections = [("", all_text)]

    # ── Build Document objects ────────────────────────────────────────────────
    docs = []
    for sec_idx, (heading, text) in enumerate(sections):
        # Append table text to first section
        if sec_idx == 0 and table_text:
            text = text + "\n\n[TABLES]\n" + table_text

        # Append image outputs to first section
        if sec_idx == 0 and image_outputs:
            assembled = _assemble(text, "", image_outputs)
        else:
            assembled = text

        if not assembled.strip():
            continue

        docs.append(Document(
            content=mask_pii(assembled),
            metadata={
                "source":   filename,
                "type":     "docx",
                "section":  sec_idx,
                "heading":  heading,
                "ocr_method": ocr_method,
                # image_outputs is only attached to section 0 (see above) —
                # these flags reflect that, so we don't falsely claim every
                # section "has charts" when only the first one does.
                "has_charts":         sec_idx == 0 and any(m in ("deplot_chart", "vlm_chart", "chart_skipped") for _, m in image_outputs),
                "charts_interpreted": sec_idx == 0 and any(m in ("deplot_chart", "vlm_chart") for _, m in image_outputs),
                "has_images":         sec_idx == 0 and any(m in ("vlm_caption", "image_skipped") for _, m in image_outputs),
                "images_captioned":   sec_idx == 0 and any(m == "vlm_caption" for _, m in image_outputs),
            }
        ))

    return _doc_title.tag(docs if docs else [Document(
        content="",
        metadata={"source": filename, "type": "docx"}
    )], doc_title)


# ── PPTX parser ───────────────────────────────────────────────────────────────

def parse_pptx(file, ocr_method: str = "easyocr", use_vision: bool = True,
                caption_images: bool = False, caption_provider: str = None,
                caption_model: str = None):
    caption_fn = _make_caption_fn(caption_images, caption_provider, caption_model)

    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("pip install python-pptx")

    filename = getattr(file, "name", str(file))
    ext      = os.path.splitext(filename)[-1].lower()

    if ext == ".ppt":
        return _parse_ppt_legacy(file, filename, ocr_method, use_vision)

    prs    = Presentation(file)
    ocr_fn = _make_ocr_fn(ocr_method)
    doc_title = _doc_title.pptx_title(prs, filename)
    docs   = []

    for slide_idx, slide in enumerate(prs.slides):
        text_parts = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                line = " ".join(r.text for r in para.runs if r.text.strip())
                if line.strip():
                    text_parts.append(line)
        slide_text = "\n".join(text_parts)

        notes_text = ""
        try:
            if slide.has_notes_slide:
                notes_text = "\n".join(
                    p.text for p in slide.notes_slide.notes_text_frame.paragraphs
                    if p.text.strip()
                )
        except Exception:
            pass

        image_outputs = []
        for shape in slide.shapes:
            if shape.shape_type == 13:
                try:
                    pil_image = Image.open(io.BytesIO(shape.image.blob)).convert("RGB")
                    image_np  = _pil_to_np(pil_image)
                    content, method = process_image_region(
                        image_np, ocr_fn, context=slide_text[:300],
                        use_vision=use_vision,
                        caption_fn=caption_fn,
                    )
                    if content.strip():
                        image_outputs.append((content, method))
                except Exception as e:
                    logger.warning("Slide %d image failed: %s", slide_idx+1, e)

        assembled = _assemble(slide_text, "", image_outputs)
        if notes_text.strip():
            assembled += f"\n\n[NOTES]\n{notes_text}"

        if not assembled.strip():
            continue

        docs.append(Document(
            content=mask_pii(assembled),
            metadata={
                "source":     filename,
                "slide":      slide_idx + 1,
                "type":       "pptx",
                "ocr_method": ocr_method,
                "has_charts":         any(m in ("deplot_chart", "vlm_chart", "chart_skipped") for _, m in image_outputs),
                "charts_interpreted": any(m in ("deplot_chart", "vlm_chart") for _, m in image_outputs),
                "has_images":         any(m in ("vlm_caption", "image_skipped") for _, m in image_outputs),
                "images_captioned":   any(m == "vlm_caption" for _, m in image_outputs),
            }
        ))
    return _doc_title.tag(docs, doc_title)


# ── PPT legacy (3-tier fallback) ──────────────────────────────────────────────

def _parse_ppt_legacy(file, filename, ocr_method, use_vision):
    docs = _try_libreoffice_convert(file, filename, ocr_method, use_vision)
    if docs is not None:
        return docs
    if hasattr(file, "seek"):
        file.seek(0)
    docs = _try_ole_image_extract(file, filename, ocr_method, use_vision)
    if docs is not None:
        return docs
    if hasattr(file, "seek"):
        file.seek(0)
    docs = _try_olefile_text(file, filename)
    if docs is not None:
        return docs
    print(f"\n  ⚠️  Cannot parse '{filename}' (old .ppt binary).\n"
          f"     Fix: open in PowerPoint → Save As → .pptx → re-upload.\n"
          f"     Or install LibreOffice (soffice on PATH).\n")
    return []


def _try_libreoffice_convert(file, filename, ocr_method, use_vision):
    import shutil
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    try:
        import subprocess
        data = file.read() if hasattr(file, "read") else file
        if hasattr(file, "seek"):
            file.seek(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            ppt_path = os.path.join(tmpdir, filename)
            with open(ppt_path, "wb") as f:
                f.write(data)
            result = subprocess.run(
                [soffice, "--headless", "--convert-to", "pptx",
                 "--outdir", tmpdir, ppt_path],
                capture_output=True, text=True, timeout=60
            )
            pptx_path = ppt_path.replace(".ppt", ".pptx")
            if os.path.exists(pptx_path):
                print(f"  ✅ LibreOffice converted '{filename}' → .pptx")
                with open(pptx_path, "rb") as pf:
                    pf.name = filename.replace(".ppt", ".pptx")
                    return parse_pptx(pf, ocr_method=ocr_method,
                                      use_vision=use_vision)
            logger.warning("LibreOffice failed: %s", result.stderr)
            return None
    except Exception as e:
        logger.warning("LibreOffice error: %s", e)
        return None


def _try_ole_image_extract(file, filename, ocr_method, use_vision):
    try:
        import olefile
        import re as _re
    except ImportError:
        return None

    try:
        data   = file.read() if hasattr(file, "read") else file
        ole    = olefile.OleFileIO(io.BytesIO(data))
        ocr_fn = _make_ocr_fn(ocr_method)
        # Was an undefined name here (NameError -> whole OLE path failed).
        # Legacy .ppt gets no captions: the caption settings aren't passed in.
        caption_fn = None

        image_signatures = {
            b"\xff\xd8\xff": "jpeg",
            b"\x89PNG":      "png",
            b"\x47\x49\x46": "gif",
            b"\x42\x4d":     "bmp",
        }

        extracted_images = []
        for entry in ole.listdir():
            try:
                stream_data = ole.openstream(entry).read()
            except Exception:
                continue
            for sig in image_signatures:
                for match in _re.finditer(_re.escape(sig), stream_data):
                    candidate = stream_data[match.start():match.start()+5_000_000]
                    try:
                        pil_img = Image.open(io.BytesIO(candidate)).convert("RGB")
                        w, h    = pil_img.size
                        if w > 100 and h > 100:
                            extracted_images.append(_pil_to_np(pil_img))
                    except Exception:
                        pass
        ole.close()

        if not extracted_images:
            return None

        print(f"  🖼️  '{filename}': {len(extracted_images)} image(s) from OLE")
        docs = []
        for idx, image_np in enumerate(extracted_images):
            content, method = process_image_region(
                image_np, ocr_fn,
                context=f"Slide {idx+1} of {filename}",
                use_vision=use_vision,
                caption_fn=caption_fn,
            )
            content = clean_ocr_text(content) if method == "ocr" else content
            if not content.strip():
                continue
            docs.append(Document(
                content=mask_pii(f"[OCR]\n{content}" if method == "ocr" else content),
                metadata={"source": filename, "slide": idx+1,
                          "type": "ppt_image", "ocr_method": ocr_method}
            ))
        return docs if docs else None

    except Exception as e:
        logger.warning("OLE image extraction failed: %s", e)
        return None


def _try_olefile_text(file, filename):
    try:
        import olefile
        import re as _re
    except ImportError:
        return None
    try:
        data = file.read() if hasattr(file, "read") else file
        ole  = olefile.OleFileIO(io.BytesIO(data))
        text = ""
        if ole.exists("PowerPoint Document"):
            stream     = ole.openstream("PowerPoint Document").read()
            byte_chunks = _re.findall(rb"[ -~]{4,}", stream)
            text        = " ".join(c.decode("latin1", errors="ignore")
                                   for c in byte_chunks)
        ole.close()
        if not text.strip():
            return None
        print(f"  ⚠️  '{filename}': olefile text fallback — formatting lost")
        return [Document(
            content=mask_pii(f"[TEXT]\n{text.strip()}"),
            metadata={"source": filename, "type": "ppt_legacy", "slide": 1}
        )]
    except Exception as e:
        logger.warning("olefile text failed: %s", e)
        return None


# ── Standalone image ──────────────────────────────────────────────────────────

def parse_image(file, ocr_method: str = "easyocr", use_vision: bool = True,
                 caption_images: bool = False, caption_provider: str = None,
                 caption_model: str = None):
    caption_fn = _make_caption_fn(caption_images, caption_provider, caption_model)

    filename = getattr(file, "name", str(file))
    ocr_fn   = _make_ocr_fn(ocr_method)

    try:
        pil_image = Image.open(file).convert("RGB")
        image_np  = _pil_to_np(pil_image)
        print(f"  🖼️  Processing: {filename}")
        content, method = process_image_region(
            image_np, ocr_fn, context="", use_vision=use_vision,
            caption_fn=caption_fn,
        )
        content = clean_ocr_text(content) if method == "ocr" else content
    except Exception as e:
        logger.warning("Image processing failed %s: %s", filename, e)
        content, method = "", "failed"

    if not content.strip():
        return []

    tag = "[CHARTS]" if method in ("deplot_chart", "vlm_chart") else "[OCR]"
    return [Document(
        content=mask_pii(f"{tag}\n{content}"),
        metadata={
            "source":     filename,
            "page":       0,
            "type":       "image",
            "ocr_method": ocr_method,
            "vision":     method in ("deplot_chart", "vlm_chart"),
            "has_charts":         method in ("deplot_chart", "vlm_chart", "chart_skipped"),
            "charts_interpreted": method in ("deplot_chart", "vlm_chart"),
            "has_images":         method in ("vlm_caption", "image_skipped"),
            "images_captioned":   method == "vlm_caption",
        }
    )]


# ── Plain text ────────────────────────────────────────────────────────────────

def parse_text(file):
    filename = getattr(file, "name", str(file))
    content  = file.read().decode("utf-8", errors="ignore")
    return [Document(
        content=mask_pii(content),
        metadata={"source": filename, "type": "text"}
    )]