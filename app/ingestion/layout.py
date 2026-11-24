"""
layout.py
---------
Reading-order text extraction for digital PDF pages (any layout).

pdfplumber's page.extract_text() reads each line straight across the page.
On multi-column pages that interleaves the columns line by line, so a
sentence that wraps inside one column comes out mixed with the neighbouring
column and can no longer be retrieved or quoted.

This module orders words with a recursive XY-cut, a standard layout-analysis
method:

  1. Split the region at horizontal whitespace bands that run across its
     whole width (paragraph breaks, gaps between a header and the body,
     between a figure and text). Read the bands top to bottom.
  2. Inside each band, split at vertical whitespace gutters that run its
     whole height (column breaks). Read the columns left to right.
  3. Recurse until nothing can be cut.

Nothing is tuned to a particular document. Every threshold scales with the
page's own median word height and character width, so it adapts to font
size, and no page positions are assumed: one, two, three or more columns,
mixed layouts (full-width title over columns, sidebars) and single-column
pages all go through the same rule. A single-column page has no gutter and
comes out in normal line order.

Guards against splitting things that aren't columns:
  - a vertical cut needs at least MIN_LINES lines on each side, and those
    lines must look like prose (average line length >= MIN_LINE_CHARS).
    Tables (short cells) and chart axis labels fail this test, so their rows
    are still read across, as before;
  - pages with fewer than MIN_WORDS words are returned as extract_text().

Scanned pages have no words and never reach this code.
LAYOUT_READING_ORDER=false switches back to plain extract_text().

UPDATED (step 7): letter-spaced display type. pdfplumber joins characters
into a word when they are within x_tolerance = 3 pt, a fixed distance. A
large, tracked-out heading has wider gaps than that between its letters,
so one word comes out as two ("Responsib le"). Neighbouring words on the
same line are now joined when (a) the PDF has NO space character between
them and (b) the gap is below MERGE_GAP_RATIO of their height. A real word
space is either an explicit space glyph or a gap of about a quarter of the
font size, so ordinary words are never joined; at body sizes the threshold
is below 3 pt and changes nothing.
"""

from __future__ import annotations

import logging
import os
from statistics import median
from typing import List, Tuple

logger = logging.getLogger("financial_rag.layout")

MIN_WORDS      = 40     # below this, layout analysis isn't worth it
MIN_LINES      = 3      # lines needed on each side of a column cut
MIN_LINE_CHARS = 15     # avg chars per line for a side to count as prose
GUTTER_CHARS   = 1.5    # column gutter >= this many median character widths
BAND_HEIGHTS   = 1.0    # horizontal cut needs a gap >= this many word heights
MAX_DEPTH      = 12
MERGE_GAP_RATIO = 0.15  # join same-line words whose gap < this x word height

Word = dict  # pdfplumber word: text, x0, x1, top, bottom


def _enabled() -> bool:
    return os.getenv("LAYOUT_READING_ORDER", "true").lower() not in ("0", "false", "no")


# ─────────────────────────────────────────────────────────────────────────────
# Projection gaps
# ─────────────────────────────────────────────────────────────────────────────

def _gaps(intervals: List[Tuple[float, float]], min_gap: float) -> List[Tuple[float, float]]:
    """Empty stretches >= min_gap between merged 1-D intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    gaps, end = [], intervals[0][1]
    for a, b in intervals[1:]:
        if a - end >= min_gap:
            gaps.append((end, a))
        end = max(end, b)
    return gaps


def _lines(words: List[Word], tol: float) -> List[List[Word]]:
    """Group words into lines by vertical position, each sorted left to right."""
    lines: List[List[Word]] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if lines and abs(lines[-1][0]["top"] - w["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(l, key=lambda w: w["x0"]) for l in lines]


def _prose_like(words: List[Word], tol: float) -> bool:
    lines = _lines(words, tol)
    if len(lines) < MIN_LINES:
        return False
    chars = sum(len(w["text"]) + 1 for w in words)
    return chars / len(lines) >= MIN_LINE_CHARS


# ─────────────────────────────────────────────────────────────────────────────
# Recursive XY-cut
# ─────────────────────────────────────────────────────────────────────────────

def _xy_cut(words: List[Word], h: float, cw: float, depth: int = 0) -> List[List[Word]]:
    """Return word blocks in reading order."""
    if depth >= MAX_DEPTH or len(words) < 2:
        return [words]

    # 1) horizontal bands (top -> bottom). The gap is measured against the
    #    text on either side of it, so big headings (whose normal line gap is
    #    larger than a body line) aren't split into one band per line.
    ygaps = [(a, b) for a, b in _gaps([(w["top"], w["bottom"]) for w in words], BAND_HEIGHTS * h)
             if b - a >= BAND_HEIGHTS * _local_height(words, a, b, h)]
    if ygaps:
        cuts = [(a + b) / 2 for a, b in ygaps]
        bands: List[List[Word]] = [[] for _ in range(len(cuts) + 1)]
        for w in words:
            mid = (w["top"] + w["bottom"]) / 2
            bands[sum(1 for c in cuts if mid > c)].append(w)
        out: List[List[Word]] = []
        for band in bands:
            if band:
                out.extend(_xy_cut(band, h, cw, depth + 1))
        return out

    # 2) vertical gutters (left -> right), only between prose-like columns
    xgaps = _gaps([(w["x0"], w["x1"]) for w in words], GUTTER_CHARS * cw)
    if xgaps:
        # try the widest gutter first; accept it only if both sides are prose
        for a, b in sorted(xgaps, key=lambda g: g[1] - g[0], reverse=True):
            cut = (a + b) / 2
            left = [w for w in words if (w["x0"] + w["x1"]) / 2 < cut]
            right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= cut]
            if _prose_like(left, h / 2) and _prose_like(right, h / 2):
                return (_xy_cut(left, h, cw, depth + 1)
                        + _xy_cut(right, h, cw, depth + 1))

    return [words]


PARA_INDENT_CHARS = 1.5   # first-line indent >= this many char widths
PARA_SHORT_LINE   = 0.75  # previous line ends before 75% of the block width


def _local_height(words: List[Word], gap_top: float, gap_bottom: float, h: float) -> float:
    """Height of the tallest word on the line just above / just below a gap."""
    above = [w for w in words if w["bottom"] <= gap_top + 0.5]
    below = [w for w in words if w["top"] >= gap_bottom - 0.5]
    hs = []
    if above:
        last = max(w["bottom"] for w in above)
        hs += [w["bottom"] - w["top"] for w in above if w["bottom"] >= last - 0.5]
    if below:
        first = min(w["top"] for w in below)
        hs += [w["bottom"] - w["top"] for w in below if w["top"] <= first + 0.5]
    return max(hs) if hs else h


def _block_text(block: List[Word], tol: float, cw: float = 5.0) -> str:
    """Lines of one block. A blank line is inserted where typography marks a
    new paragraph, so the chunker can split there instead of mid-paragraph:
      - the line is indented (first-line indent) relative to the block, or
      - the previous line ended a sentence and stopped well short of the
        block's right edge (last line of a paragraph)."""
    lines = _lines(block, tol)
    if len(lines) < 3:
        return "\n".join(" ".join(w["text"] for w in l) for l in lines)
    left = median(l[0]["x0"] for l in lines)
    right = max(l[-1]["x1"] for l in lines)
    width = max(1.0, right - left)
    out = []
    for k, line in enumerate(lines):
        text = " ".join(w["text"] for w in line)
        if k > 0:
            prev = lines[k - 1]
            indented = line[0]["x0"] - left >= PARA_INDENT_CHARS * cw
            prev_short = (prev[-1]["x1"] - left) < PARA_SHORT_LINE * width
            prev_ends = prev[-1]["text"].rstrip("\"'”’)").endswith((".", "!", "?", ":"))
            if indented or (prev_short and prev_ends):
                out.append("")
        out.append(text)
    return "\n".join(out)


def _space_between(spaces, p: Word, w: Word) -> bool:
    for c in spaces:
        # a space glyph that starts in the gap (its advance may overlap the
        # next letter when the font is kerned)
        if p["x1"] - 0.5 <= c["x0"] <= w["x0"] and \
                c["top"] < w["bottom"] and c["bottom"] > w["top"]:
            return True
    return False


def _merge_spaced_words(words: List[Word], spaces=()) -> List[Word]:
    """Join pieces of one word split by letter-spacing (see module notes).
    spaces: the page's space characters (page.chars with text ' ')."""
    if not words:
        return words
    out: List[Word] = []
    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if out:
            p = out[-1]
            hp, hw = p["bottom"] - p["top"], w["bottom"] - w["top"]
            same_line = abs(p["top"] - w["top"]) <= 0.2 * max(hp, hw) and \
                abs(hp - hw) <= 0.2 * max(hp, hw)
            gap = w["x0"] - p["x1"]
            if same_line and -0.05 * hw <= gap < MERGE_GAP_RATIO * min(hp, hw) \
                    and not _space_between(spaces, p, w):
                out[-1] = {**p, "text": p["text"] + w["text"], "x1": w["x1"],
                           "top": min(p["top"], w["top"]),
                           "bottom": max(p["bottom"], w["bottom"])}
                continue
        out.append(dict(w))
    return out


def extract_text_reading_order(page) -> str:
    """
    Text of a pdfplumber page in reading order. Falls back to
    page.extract_text() when disabled, when the page has few words, or on
    any error.
    """
    plain = page.extract_text() or ""
    if not _enabled():
        return plain
    try:
        spaces = [c for c in getattr(page, "chars", []) if not c.get("text", "x").strip()]
        words = _merge_spaced_words(page.extract_words(), spaces)
        if len(words) < MIN_WORDS:
            return plain
        h = median(w["bottom"] - w["top"] for w in words) or 10.0
        cw = median((w["x1"] - w["x0"]) / max(1, len(w["text"])) for w in words) or 5.0
        blocks = _xy_cut(words, h, cw)
        text = "\n\n".join(_block_text(b, h / 2, cw) for b in blocks if b)
        return text or plain
    except Exception as e:  # never lose a page over layout analysis
        logger.debug("Reading-order extraction skipped: %s", e)
        return plain
