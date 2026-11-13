"""
citation_node.py — Rich citations for all document types.

PDF:   sample.pdf, Page 4
DOCX:  sample.docx, § Model Evaluation
PPTX:  sample.pptx, Slide 3 — Key Findings
TXT:   sample.txt

Also strips any inline "can be found in..." text the LLM adds.
"""

import re
import logging
from pathlib import Path
from app.graph.state import GraphState

logger = logging.getLogger("financial_rag.citation_node")

# Patterns the LLM writes that we want to remove from the answer
_LLM_CITATION_PATTERNS = [
    r"\.\s*The information can be found in [^.]*\.",
    r"\.\s*This information is (found|available|stated|mentioned) in [^.]*\.",
    r"\.\s*\[Source:[^\]]*\]",
    r"\nSource:.*$",
    r"\(Source:[^)]*\)",
    r"\[Source:[^\]]*\]",
]


def _build_location(meta: dict) -> str:
    """Build a human-readable location string from chunk metadata."""
    doc_type = meta.get("type", "")
    page     = meta.get("page")
    slide    = meta.get("slide")
    heading  = meta.get("heading", "").strip()
    section  = meta.get("section")
    para     = meta.get("paragraph")

    # PDF — always has page number
    if doc_type == "pdf" or isinstance(page, int):
        loc = f"Page {page + 1}" if isinstance(page, int) else ""
        return loc

    # PPTX — slide number + optional title
    if doc_type == "pptx" or isinstance(slide, int):
        if isinstance(slide, int):
            loc = f"Slide {slide}"
            if heading:
                loc += f" — {heading}"
            return loc

    # DOCX — section heading (most informative)
    if doc_type == "docx":
        if heading:
            return f"§ {heading}"
        elif isinstance(section, int):
            return f"Section {section + 1}"
        elif isinstance(para, int):
            return f"Para {para + 1}"

    # TXT / fallback
    if heading:
        return f"§ {heading}"

    return ""


def citation_node(state: GraphState) -> GraphState:
    if state.get("blocked"):
        return state

    response   = state.get("response", "")
    validation = state.get("metadata", {}).get("validation", {})
    docs       = state.get("reranked_docs", [])
    metadata   = state.get("metadata", {})

    if not validation.get("valid", False) or not docs:
        return state

    # ── Strip LLM-generated inline citations ──────────────────────────────────
    for pattern in _LLM_CITATION_PATTERNS:
        response = re.sub(pattern, "", response,
                          flags=re.MULTILINE | re.IGNORECASE)
    response = response.strip()

    # ── Build structured citations ────────────────────────────────────────────
    citations = []
    seen      = set()

    for doc in docs:
        meta    = doc.get("metadata", {})
        source  = Path(meta.get("source", "Unknown")).name
        loc     = _build_location(meta)
        label   = f"{source}" + (f", {loc}" if loc else "")

        if label not in seen:
            seen.add(label)
            citations.append({
                "label":   label,
                "source":  meta.get("source", ""),
                "location": loc,
                "type":    meta.get("type", ""),
                "page":    meta.get("page"),
                "slide":   meta.get("slide"),
                "heading": meta.get("heading", ""),
            })

    # ── Append formatted Sources section ─────────────────────────────────────
    if citations:
        lines = ["", "", "**Sources:**"]
        for c in citations:
            icon = {"pdf": "📄", "docx": "📝", "pptx": "📊",
                    "txt": "📃"}.get(c["type"], "📄")
            lines.append(f"- {icon} {c['label']}")
        response = response + "\n".join(lines)

    logger.info("Citations: %d sources", len(citations))

    return {
        **state,
        "response": response,
        "metadata": {**metadata, "citations": citations},
    }