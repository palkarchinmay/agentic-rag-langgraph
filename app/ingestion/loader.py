"""
loader.py
---------
Routes uploaded files to the correct parser. Supports parallel processing.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.ingestion.parser import (
    parse_pdf, parse_docx, parse_pptx, parse_image, parse_text,
)

logger = logging.getLogger("financial_rag.loader")

EXTENSION_MAP = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".ppt":  "pptx",
    ".txt":  "text",
    ".md":   "text",
    ".csv":  "text",
    ".png":  "image",
    ".jpg":  "image",
    ".jpeg": "image",
    ".tiff": "image",
    ".tif":  "image",
    ".bmp":  "image",
    ".webp": "image",
}


def _get_extension(file) -> str:
    name = getattr(file, "name", str(file))
    return os.path.splitext(name)[-1].lower()


def _parse_single(file, ocr_method: str, use_vision: bool,
                   caption_images: bool = False, caption_provider: str = "openai",
                   caption_model: str = "gpt-4o-mini") -> list:
    ext      = _get_extension(file)
    ftype    = EXTENSION_MAP.get(ext)
    filename = getattr(file, "name", str(file))

    if ftype is None:
        logger.warning("Unsupported type '%s' — skipping %s", ext, filename)
        return []

    logger.info("Parsing %s as %s (ocr=%s, vision=%s, caption_images=%s)",
                filename, ftype, ocr_method, use_vision, caption_images)
    kwargs = dict(ocr_method=ocr_method, use_vision=use_vision,
                  caption_images=caption_images, caption_provider=caption_provider,
                  caption_model=caption_model)
    try:
        if ftype == "pdf":
            return parse_pdf(file, **kwargs)
        elif ftype == "docx":
            return parse_docx(file, **kwargs)
        elif ftype == "pptx":
            return parse_pptx(file, **kwargs)
        elif ftype == "image":
            return parse_image(file, **kwargs)
        elif ftype == "text":
            return parse_text(file)
    except Exception as e:
        logger.exception("Failed to parse %s: %s", filename, e)
    return []


def load_documents(files, ocr_method: str = "easyocr",
                   use_vision: bool = True, max_workers: int = 4,
                   caption_images: bool = False, caption_provider: str = "openai",
                   caption_model: str = "gpt-4o-mini") -> list:
    """
    Load and parse files in parallel.

    Args:
        files            : list of file-like objects
        ocr_method       : 'easyocr' | 'textract' | 'docai'
        use_vision       : interpret charts/graphs via DePlot+Pix2Struct (free, local)
        caption_images   : caption non-chart images via a VLM call (opt-in, real cost)
        caption_provider : LLM provider for captioning, e.g. "openai"
        caption_model    : LLM model for captioning, e.g. "gpt-4o-mini"
        max_workers      : parallel threads (keep ≤ 4 on local machine)

    Returns: flat list of Document objects.
    """
    if not files:
        return []

    caption_kwargs = dict(caption_images=caption_images, caption_provider=caption_provider,
                           caption_model=caption_model)

    if len(files) == 1:
        return _parse_single(files[0], ocr_method, use_vision, **caption_kwargs)

    all_docs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(_parse_single, f, ocr_method, use_vision, **caption_kwargs):
            getattr(f, "name", str(f))
            for f in files
        }
        for future in as_completed(future_to_file):
            fname = future_to_file[future]
            try:
                docs = future.result()
                logger.info("  ✅ %s → %d doc(s)", fname, len(docs))
                all_docs.extend(docs)
            except Exception as e:
                logger.error("  ❌ %s failed: %s", fname, e)

    return all_docs


def supported_extensions() -> list:
    return sorted(EXTENSION_MAP.keys())