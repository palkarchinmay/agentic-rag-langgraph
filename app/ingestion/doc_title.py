"""
doc_title.py -- a short, human title for an uploaded document.

The chunker prepends it to every chunk ("Document: <title>"), so a question
like "what is the title of this report?" or "what does the <report name>
say about X" can match a chunk even when the chunk itself never names the
document. Works for any file; nothing document-specific.

Order of preference:
  1. the file's own metadata title (PDF /Title, DOCX/PPTX core properties),
     unless it is empty or obvious junk ("Untitled", "Microsoft Word - x.doc",
     a file name);
  2. the content: PDF -> the largest text on page 1; DOCX -> the first
     'Title' or 'Heading 1' paragraph; PPTX -> the first slide's title;
  3. the file name, without extension, '_' and '-' turned into spaces.
"""

from __future__ import annotations

import logging
import os
import re
from statistics import median

logger = logging.getLogger("financial_rag.doc_title")

MAX_CHARS = 150
_JUNK = re.compile(
    r"(?i)^(untitled|title|document\d*|presentation\d*|slide \d+|none|unknown)$"
    r"|^microsoft (word|powerpoint|excel)\s*-"
    r"|\.(pdf|docx?|pptx?|xlsx?|txt|rtf|odt|indd|qxd)$")


def _clean(t) -> str:
    t = re.sub(r"\s+", " ", str(t or "")).strip(" \t-|:")
    return t[:MAX_CHARS].rstrip()


def usable(t) -> str:
    """t cleaned, or "" if it is empty or junk."""
    t = _clean(t)
    if len(t) < 3 or _JUNK.search(t) or not re.search(r"[A-Za-z]{2}", t):
        return ""
    return t


def from_filename(filename: str) -> str:
    base = re.split(r"[\\/]", str(filename or ""))[-1]   # Windows or POSIX path
    stem = os.path.splitext(base)[0]
    return _clean(re.sub(r"[_\-]+", " ", stem)) or "Untitled document"


def pdf_title(pdf, filename: str = "") -> str:
    """pdf: an open pdfplumber.PDF."""
    try:
        t = usable((pdf.metadata or {}).get("Title"))
        if t:
            return t
    except Exception:
        pass
    try:
        if pdf.pages:
            t = largest_text(pdf.pages[0])
            if t:
                return t
    except Exception as e:
        logger.debug("PDF title from page 1 failed: %s", e)
    return from_filename(filename)


def largest_text(page, ratio: float = 0.7, min_gain: float = 1.3) -> str:
    """Words on the page set in the largest type (>= ratio x the largest
    size), in reading order -- but only if that type is clearly larger
    (min_gain x) than the page's body text; a page with no display type
    has no visual title."""
    words = page.extract_words(extra_attrs=["size"])
    if not words:
        return ""
    try:
        from app.ingestion.layout import _merge_spaced_words
        spaces = [c for c in page.chars if not c.get("text", "x").strip()]
        words = _merge_spaced_words(words, spaces)
    except Exception:
        pass
    sizes = [w.get("size") or (w["bottom"] - w["top"]) for w in words]
    biggest, body = max(sizes), median(sizes)
    if biggest < min_gain * body:
        return ""
    big = [(w, s) for w, s in zip(words, sizes) if s >= ratio * biggest]
    big.sort(key=lambda ws: (round(ws[0]["top"]), ws[0]["x0"]))
    # split into blocks (big lines within about a line height of each
    # other) and take the one in the largest type that reads as a title; a
    # drop cap or a pull quote further down the page is a separate block
    blocks, cur, last_bottom = [], [], None
    for w, s in big:
        if cur and w["top"] - last_bottom > 0.8 * s:
            blocks.append(cur)
            cur = []
        cur.append((w, s))
        last_bottom = max(last_bottom or 0, w["bottom"]) if cur[:-1] else w["bottom"]
    if cur:
        blocks.append(cur)
    for block in sorted(blocks, key=lambda b: -max(s for _, s in b)):
        t = usable(" ".join(w["text"] for w, _ in block))
        if t:                       # a one-letter drop cap is not usable
            return t
    return ""


def docx_title(doc, filename: str = "") -> str:
    """doc: a python-docx Document."""
    try:
        t = usable(doc.core_properties.title)
        if t:
            return t
    except Exception:
        pass
    try:
        for style in ("Title", "Heading 1"):
            for p in doc.paragraphs:
                if p.style is not None and p.style.name == style and usable(p.text):
                    return usable(p.text)
    except Exception:
        pass
    return from_filename(filename)


def pptx_title(prs, filename: str = "") -> str:
    """prs: a python-pptx Presentation."""
    try:
        t = usable(prs.core_properties.title)
        if t:
            return t
    except Exception:
        pass
    try:
        for slide in list(prs.slides)[:1]:
            shape = slide.shapes.title
            if shape is not None and usable(shape.text):
                return usable(shape.text)
    except Exception:
        pass
    return from_filename(filename)


def tag(docs, title: str) -> list:
    """Set metadata['doc_title'] on every Document (PII-masked)."""
    try:
        from app.security.pii import mask_pii
        title = mask_pii(title)
    except Exception:
        pass
    for d in docs or []:
        try:
            d.metadata["doc_title"] = title
        except Exception:
            pass
    return docs
