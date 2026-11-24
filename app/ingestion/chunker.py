"""
chunker.py — Document chunking with multiple strategies.

Methods:
  recursive : LangChain RecursiveCharacterTextSplitter (default)
  structure : splits on [TEXT]/[TABLES]/[OCR] section boundaries first
  semantic  : embedding-based semantic chunking (requires embedding model)

UPDATED (step 7): figure-aware chunking for recursive and structure.
  Each "[Chart Data]" / "[Image Description]" block the parser writes (one
  per chart, diagram or picture) is chunked on its own, never glued to the
  page text before it, and every chunk of it starts with a header line:
      [Chart Data] <title line>        (first chunk)
      [Chart Data] <title line> (cont.)  (later chunks)
  so a question about a figure's title retrieves the chunk holding its
  numbers. A long Markdown table is split between rows only, and its
  header row is repeated in each later chunk. Nothing here depends on the
  document: it keys only on the tags the parser writes.

UPDATED (step 8): every chunk starts with "Document: <title>" (all methods,
all file types), so questions that name the document or ask for its title
can match chunks that never mention it. The title comes from
metadata['doc_title'] (set by the parser, see doc_title.py), else from the
file name in metadata['source']; a document with neither gets no prefix.
The prefix counts toward chunk_size. Turn off with CHUNK_DOC_TITLE=false
or chunk_documents(..., add_title=False).
"""

import os
import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("financial_rag.chunker")


def chunk_documents(
    docs,
    chunk_size:    int = 500,
    chunk_overlap: int = 50,
    method:        str = "recursive",
    embedding_model=None,
    add_title:     Optional[bool] = None,
) -> List[Dict]:
    """
    Chunk a list of Document objects.

    Args:
        docs           : list of Document objects
        chunk_size     : max chars per chunk
        chunk_overlap  : overlap between chunks
        method         : 'recursive' | 'structure' | 'semantic'
        embedding_model: required for method='semantic'
        add_title      : prepend "Document: <title>" to each chunk;
                         None -> env CHUNK_DOC_TITLE (default true)

    Returns:
        list of {content: str, metadata: dict}
    """
    if not docs:
        return []
    if add_title is None:
        add_title = os.getenv("CHUNK_DOC_TITLE", "true").strip().lower() not in ("0", "false", "no", "off")

    if method == "recursive":
        return _recursive_chunk(docs, chunk_size, chunk_overlap, add_title)
    elif method == "structure":
        return _structure_chunk(docs, chunk_size, chunk_overlap, add_title)
    elif method == "semantic":
        return _semantic_chunk(docs, embedding_model, add_title)
    else:
        logger.warning("Unknown method '%s', defaulting to recursive", method)
        return _recursive_chunk(docs, chunk_size, chunk_overlap, add_title)


# ── Document title prefix ────────────────────────────────────────────────────

_MIN_BODY = 100   # never shrink the body budget below this many chars


def _title_prefix(doc, add_title: bool) -> str:
    if not add_title:
        return ""
    meta = getattr(doc, "metadata", None) or {}
    title = (meta.get("doc_title") or "").strip()
    if not title and meta.get("source"):
        from app.ingestion.doc_title import from_filename
        title = from_filename(meta["source"])
    return f"Document: {title[:150]}\n" if title else ""


def _with_title(chunks: List[Dict], prefix: str) -> List[Dict]:
    if prefix:
        title = prefix[len("Document: "):].strip()
        for c in chunks:
            c["content"] = prefix + c["content"]
            c["metadata"].setdefault("doc_title", title)
    return chunks


# ── Figure blocks ────────────────────────────────────────────────────────────

_FIG_TAGS     = ("[Chart Data]", "[Image Description]")
_SECTION_RE   = re.compile(r"^\[(TEXT|TABLES|OCR|CHARTS|NOTES)\]\s*$")
_FIG_START_RE = re.compile(r"^\[(Chart Data|Image Description)\]")
_BLOCK_END_RE = re.compile(r"^\[(Chart Data|Image Description|Chart Description|"
                           r"TEXT|TABLES|OCR|CHARTS|NOTES)\]")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}")
_TITLE_CHARS  = 160


def _split_figures(content: str):
    """-> (rest_text, [(section_tag, figure_block), ...]).

    rest_text is the content with figure blocks cut out (section tags kept);
    section_tag is the [SECTION] each figure sat in (CHARTS, OCR, ...)."""
    rest, figures = [], []
    section, block = "TEXT", None
    for line in content.splitlines():
        m = _SECTION_RE.match(line.strip())
        if block is not None and _BLOCK_END_RE.match(line.strip()) and \
                not line.strip().startswith("[Chart Description]"):
            figures.append((block[0], "\n".join(block[1]).strip()))
            block = None
        if m:
            section = m.group(1)
            rest.append(line)
            continue
        if _FIG_START_RE.match(line.strip()):
            block = (section, [line])
            continue
        if block is not None:
            block[1].append(line)
        else:
            rest.append(line)
    if block is not None:
        figures.append((block[0], "\n".join(block[1]).strip()))
    return "\n".join(rest), [f for f in figures if f[1]]


_CAPTION_RE = re.compile(r"(?i)^\W*(figure|fig\.|chart|table|exhibit|graph|diagram)\s*[\dA-Z]")


def _figure_title(block: str):
    """-> (title, line_index) or ("", None).

    The title is the '# ...' line the caption prompt asks for, else a line
    that reads like a caption ('Figure 3 ...', 'Table 2 ...'), looked for
    only in the first few lines."""
    lines = block.splitlines()
    seen = 0
    for i, l in enumerate(lines[1:], start=1):
        t = l.strip()
        if not t:
            continue
        seen += 1
        if seen > 3:
            break
        if re.match(r"^#\s", t):
            title = t[1:].strip()
            return ("" if title.lower() in ("(untitled)", "untitled") else title), i
        if _CAPTION_RE.match(t):
            return t.strip("*_ "), None
    return "", None


def _figure_header(block: str) -> str:
    """'[Chart Data] Figure 1: ...' -- the tag plus the figure title, if any."""
    first = block.lstrip().splitlines()[0].strip() if block.strip() else ""
    tag = next((t for t in _FIG_TAGS if first.startswith(t)), "[Chart Data]")
    title, _ = _figure_title(block)
    return (tag + (" " + title[:_TITLE_CHARS] if title else "")).strip()


def _table_header(lines, idx):
    """Header + separator rows of the Markdown table that line idx belongs to."""
    j = idx
    while j > 0 and lines[j - 1].lstrip().startswith("|"):
        j -= 1
    if j + 1 < len(lines) and _TABLE_SEP_RE.match(lines[j + 1]):
        return lines[j] + "\n" + lines[j + 1]
    return ""


def _chunk_figure(block: str, chunk_size: int, chunk_overlap: int):
    """Split one figure block on line boundaries; header on every piece."""
    header = _figure_header(block)
    lines = block.splitlines()
    _, title_idx = _figure_title(block)
    # tag line and '# title' line are in the header already
    body_lines = [l for i, l in enumerate(lines) if i != 0 and i != title_idx]
    tail = lines[0].strip()
    for t in _FIG_TAGS:
        if tail.startswith(t):
            tail = tail[len(t):].strip()
    if tail:                                     # text on the tag line itself
        body_lines.insert(0, tail)
    budget = max(120, chunk_size - len(header) - 12)
    pieces, cur, cur_len = [], [], 0
    for i, line in enumerate(body_lines):
        if not line.strip() and not cur:
            continue
        if cur and cur_len + len(line) + 1 > budget:
            pieces.append(cur)
            cur, cur_len = [], 0
            if line.lstrip().startswith("|"):
                th = _table_header(body_lines, i)
                if th and line.strip() not in th:
                    cur.append(th); cur_len += len(th) + 1
        if len(line) > budget:                  # one huge line: hard wrap
            for k in range(0, len(line), budget):
                pieces.append([line[k:k + budget]])
            continue
        cur.append(line); cur_len += len(line) + 1
    if cur:
        pieces.append(cur)
    out = []
    for n, lines in enumerate(pieces):
        text = "\n".join(lines).strip()
        if not text:
            continue
        h = header if n == 0 else header + " (cont.)"
        out.append(h + "\n" + text)
    if not out and header:
        out.append(header)
    return out


def _is_empty_section(text: str) -> bool:
    return not re.sub(r"\[(TEXT|TABLES|OCR|CHARTS|NOTES)\]", "", text).strip()


def _recursive_chunk(docs, chunk_size, chunk_overlap, add_title=False):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunks = []
    for doc in docs:
        prefix = _title_prefix(doc, add_title)
        size = max(_MIN_BODY, chunk_size - len(prefix))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=min(chunk_overlap, size // 2),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        out = []
        rest, figures = _split_figures(doc.content)
        for split in splitter.split_text(rest):
            if split.strip() and not _is_empty_section(split):
                out.append({"content": split, "metadata": dict(doc.metadata)})
        for _, block in figures:
            for piece in _chunk_figure(block, size, chunk_overlap):
                out.append({"content": piece, "metadata": {**doc.metadata, "is_figure": True}})
        chunks.extend(_with_title(out, prefix))
    return chunks


def _structure_chunk(docs, chunk_size, chunk_overlap, add_title=False):
    """Split on section tags first, then apply recursive within each section."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    tag_pattern = re.compile(r'\[(TEXT|TABLES|OCR|CHARTS|NOTES)\]')
    chunks = []

    for doc in docs:
        prefix = _title_prefix(doc, add_title)
        size = max(_MIN_BODY, chunk_size - len(prefix))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=min(chunk_overlap, size // 2)
        )
        out = []
        rest, figures = _split_figures(doc.content)
        sections = tag_pattern.split(rest)
        current_tag = "TEXT"
        for part in sections:
            if tag_pattern.match(f"[{part}]"):
                current_tag = part
                continue
            if not part.strip():
                continue
            for split in splitter.split_text(part):
                if split.strip():
                    meta = {**doc.metadata, "section_type": current_tag}
                    out.append({"content": split, "metadata": meta})
        for section, block in figures:
            for piece in _chunk_figure(block, size, chunk_overlap):
                meta = {**doc.metadata, "section_type": section, "is_figure": True}
                out.append({"content": piece, "metadata": meta})
        chunks.extend(_with_title(out, prefix))
    return chunks


def _semantic_chunk(docs, embedding_model, add_title=False):
    """Semantic chunking using embedding similarity."""
    if embedding_model is None:
        logger.warning("No embedding model for semantic chunking, falling back to recursive")
        return _recursive_chunk(docs, 500, 50, add_title)

    try:
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(embeddings=embedding_model)
        chunks   = []
        for doc in docs:
            out = []
            for split in splitter.split_text(doc.content):
                if split.strip():
                    out.append({"content": split, "metadata": dict(doc.metadata)})
            chunks.extend(_with_title(out, _title_prefix(doc, add_title)))
        return chunks
    except ImportError:
        logger.warning("langchain-experimental not installed, falling back to recursive")
        return _recursive_chunk(docs, 500, 50, add_title)
