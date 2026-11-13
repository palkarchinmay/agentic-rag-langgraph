"""
parser.py — Document parsers for PDF, DOCX, PPTX, PPT, images, text.

All parse_* functions accept:
    file        : file-like object (NamedBytesIO or open file)
    ocr_method  : 'easyocr' | 'pytesseract' | 'textract' | 'docai' | 'azure_di'
    use_vision  : bool — route chart images through DePlot/Pix2Struct (True)
                         or always use OCR (False, faster, offline)
"""

import io
import os
import cv2
import tempfile
import numpy as np
import logging
from PIL import Image

import pdfplumber
from docx import Document as DocxDocument

from app.ingestion.cleaner import clean_ocr_text
from app.ingestion.schema import Document
from app.ingestion.ocr import extract_text
from app.ingestion.chart_interpreter import process_image_region
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


def _make_ocr_fn(ocr_method: str):
    def ocr_fn(image_np):
        processed = preprocess_for_ocr(image_np)
        raw       = extract_text(processed, method=ocr_method)
        return clean_ocr_text(raw)
    return ocr_fn


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


def _cluster_vector_regions(page) -> list:
    """Returns a list of (x0, top, x1, bottom) bounding boxes in PDF point
    coordinates (top-down, matching image conventions) -- one per dense
    cluster of vector-drawn shapes on the page."""
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
        x0 = min(o[0] for o in group) - _REGION_EXPAND_PT
        y0 = min(o[1] for o in group) - _REGION_EXPAND_PT
        x1 = max(o[2] for o in group) + _REGION_EXPAND_PT
        y1 = max(o[3] for o in group) + _REGION_EXPAND_PT
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


def _find_image_regions(page, pil_image: Image.Image, dpi: int) -> list:
    """Returns a list of (cropped_pil_image, is_known_chart) tuples -- one
    per embedded raster image AND one per detected vector-drawn chart
    cluster on this page. Raster images get is_known_chart=False (still
    need pixel-based is_chart() scoring, since a raster image could be a
    chart OR a photo/logo -- no vector-structure confirmation either way).
    Vector-cluster regions get is_known_chart=True (the clustering itself
    is already strong evidence -- see process_image_region's known_chart
    docstring for why pixel re-scoring is skipped for these)."""
    crops = []

    for img in page.images:
        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
        crops.append((_crop_region(pil_image, bbox, dpi), False))

    for bbox in _cluster_vector_regions(page):
        crops.append((_crop_region(pil_image, bbox, dpi), True))

    return crops


# ── PDF parser ────────────────────────────────────────────────────────────────

def parse_pdf(file, ocr_method: str = "easyocr", use_vision: bool = True,
               caption_images: bool = False, caption_provider: str = "openai",
               caption_model: str = "gpt-4o-mini"):
    """
    Parse PDF with parallel page processing.
    Cloud OCR backends (docai, textract, azure_di) benefit most from parallelism
    since each page is an independent API call.
    Local OCR (easyocr) uses 2 workers to avoid memory pressure.
    """
    caption_fn = None
    if caption_images:
        from app.ingestion.image_captioner import caption_image_with_vlm
        def caption_fn(pil_img, ctx, is_chart):
            return caption_image_with_vlm(pil_img, caption_provider, caption_model, ctx, is_chart)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    filename = getattr(file, "name", str(file))
    ocr_fn   = _make_ocr_fn(ocr_method)

    # Cloud OCR = more parallel workers (each page = independent API call)
    # Local OCR  = fewer workers (shared GPU/CPU memory)
    cloud_ocr = ocr_method in {"docai", "textract", "azure_di"}
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
                for region, is_known_chart in image_regions:
                    region_np = _pil_to_np(region)
                    try:
                        content, method = process_image_region(
                            region_np, ocr_fn,
                            context=text[:300],
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
            chart_content, method = process_image_region(
                image_np, ocr_fn,
                context=f"Page {i+1} of {filename}",
                use_vision=use_vision,
                caption_fn=caption_fn,
            )
            if chart_content.strip():
                image_outputs.append((chart_content, method))
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
        for i, page in enumerate(pdf.pages):
            page_w     = page.width or 595
            dpi        = PDF_RENDER_DPI_LARGE if page_w > 500 else PDF_RENDER_DPI_SMALL
            text       = page.extract_text() or ""
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
    return [results[i] for i in sorted(results.keys())]


# ── DOCX parser ───────────────────────────────────────────────────────────────

def parse_docx(file, ocr_method: str = "easyocr", use_vision: bool = True,
                caption_images: bool = False, caption_provider: str = "openai",
                caption_model: str = "gpt-4o-mini"):
    """
    Parse DOCX with section-aware chunking.
    Splits on Heading 1/2 paragraphs to produce section-level documents
    with heading metadata for citations.
    """
    caption_fn = None
    if caption_images:
        from app.ingestion.image_captioner import caption_image_with_vlm
        def caption_fn(pil_img, ctx, is_chart):
            return caption_image_with_vlm(pil_img, caption_provider, caption_model, ctx, is_chart)

    doc      = DocxDocument(file)
    filename = getattr(file, "name", str(file))
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

    return docs if docs else [Document(
        content="",
        metadata={"source": filename, "type": "docx"}
    )]


# ── PPTX parser ───────────────────────────────────────────────────────────────

def parse_pptx(file, ocr_method: str = "easyocr", use_vision: bool = True,
                caption_images: bool = False, caption_provider: str = "openai",
                caption_model: str = "gpt-4o-mini"):
    caption_fn = None
    if caption_images:
        from app.ingestion.image_captioner import caption_image_with_vlm
        def caption_fn(pil_img, ctx, is_chart):
            return caption_image_with_vlm(pil_img, caption_provider, caption_model, ctx, is_chart)

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
    return docs


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
                 caption_images: bool = False, caption_provider: str = "openai",
                 caption_model: str = "gpt-4o-mini"):
    caption_fn = None
    if caption_images:
        from app.ingestion.image_captioner import caption_image_with_vlm
        def caption_fn(pil_img, ctx, is_chart):
            return caption_image_with_vlm(pil_img, caption_provider, caption_model, ctx, is_chart)

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