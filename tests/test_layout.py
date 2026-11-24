"""Reading-order extraction on generated PDFs of different layouts (offline).
Needs reportlab only to build the test PDFs: pip install reportlab"""
import re

import pytest

pdfplumber = pytest.importorskip("pdfplumber")
pytest.importorskip("reportlab")
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, PageTemplate, Paragraph, Table

from app.ingestion.layout import extract_text_reading_order

ST = getSampleStyleSheet()


def _sentences(tag, n=4):
    return [f"{tag} sentence {i} continues here with several words so lines wrap across the column width."
            for i in range(n)]


def _pdf(path, ncols, columns, title=None, gap=18, style=None):
    style = style or ST["BodyText"]
    doc = BaseDocTemplate(str(path), pagesize=letter)
    W, H = doc.width, doc.height
    frames, story = [], []
    if title:
        frames.append(Frame(doc.leftMargin, doc.bottomMargin + H - 100, W, 100))
        story += [Paragraph(title, ST["Title"]), FrameBreak()]
        H -= 100
    cw = (W - gap * (ncols - 1)) / ncols
    frames += [Frame(doc.leftMargin + i * (cw + gap), doc.bottomMargin, cw, H) for i in range(ncols)]
    for k, tag in enumerate(columns):
        story.append(Paragraph(" ".join(_sentences(tag)), style))
        if k < len(columns) - 1:
            story.append(FrameBreak())
    doc.addPageTemplates([PageTemplate(frames=frames)])
    doc.build(story)
    with pdfplumber.open(str(path)) as pdf:
        return re.sub(r"\s+", " ", extract_text_reading_order(pdf.pages[0]))


def _check(text, tags):
    for tag in tags:
        for s in _sentences(tag):
            assert s in text, f"broken sentence: {s}"
    starts = [text.index(f"{t} sentence 0") for t in tags]
    assert starts == sorted(starts), "columns out of reading order"


@pytest.mark.parametrize("ncols,tags,kw", [
    (1, ["ONE"], {}),
    (2, ["LEFT", "RIGHT"], {}),
    (3, ["AAA", "BBB", "CCC"], {"gap": 12}),
    (2, ["LEFT", "RIGHT"], {"title": "Annual Report Title Spanning The Full Page Width"}),
    (2, ["SMALLL", "SMALLR"], {"gap": 10, "style": ParagraphStyle("s", fontSize=7, leading=8.5)}),
    (2, ["BIGL", "BIGR"], {"gap": 30, "style": ParagraphStyle("b", fontSize=14, leading=17)}),
])
def test_columns_read_in_order(tmp_path, ncols, tags, kw):
    _check(_pdf(tmp_path / "x.pdf", ncols, tags, **kw), tags)


def test_table_rows_are_not_split(tmp_path):
    path = tmp_path / "t.pdf"
    doc = BaseDocTemplate(str(path), pagesize=letter)
    doc.addPageTemplates([PageTemplate(frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height)])])
    rows = [["Item", "2023", "2024", "Change"]] + [[f"Line item {i}", f"{100+i}", f"{120+i}", f"{i}%"] for i in range(25)]
    doc.build([Table(rows, colWidths=[200, 80, 80, 80])])
    with pdfplumber.open(str(path)) as pdf:
        text = extract_text_reading_order(pdf.pages[0])
    for i in range(25):
        assert f"Line item {i} {100+i} {120+i} {i}%" in text


def test_switch_off(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYOUT_READING_ORDER", "false")
    path = tmp_path / "x.pdf"
    _pdf(path, 2, ["LEFT", "RIGHT"])
    with pdfplumber.open(str(path)) as pdf:
        p = pdf.pages[0]
        assert extract_text_reading_order(p) == (p.extract_text() or "")


@pytest.mark.parametrize("indent,space", [(18, 0), (0, 8)])
def test_paragraph_breaks_are_kept(tmp_path, indent, space):
    """Indented paragraphs (no extra spacing) and spaced paragraphs both get a
    blank line between them, so the chunker can split there."""
    style = ParagraphStyle("p", parent=ST["BodyText"], firstLineIndent=indent, spaceBefore=space, spaceAfter=0)
    path = tmp_path / "p.pdf"
    doc = BaseDocTemplate(str(path), pagesize=letter)
    doc.addPageTemplates([PageTemplate(frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width / 2, doc.height)])])
    paras = [" ".join(_sentences(f"P{k}", 3)) for k in range(4)]
    doc.build([Paragraph(p, style) for p in paras])
    with pdfplumber.open(str(path)) as pdf:
        text = extract_text_reading_order(pdf.pages[0])
    blocks = [re.sub(r"\s+", " ", b).strip() for b in text.split("\n\n") if b.strip()]
    for p in paras:
        assert p in blocks, "paragraph not kept as its own block"
