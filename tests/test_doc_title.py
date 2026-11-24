"""Step 8: document title detection + 'Document: <title>' chunk prefix.
Offline; synthetic files only."""
import io
import pytest

from app.ingestion import doc_title as dt
from app.ingestion import chunker


# ── title cleaning ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,ok", [
    ("Annual Report 2023", "Annual Report 2023"),
    ("  Q3   results\n summary ", "Q3 results summary"),
    ("Untitled", ""), ("Microsoft Word - draft_v2.docx", ""), ("report.pdf", ""),
    ("", ""), (None, ""), ("12345", ""), ("Presentation1", ""),
])
def test_usable(raw, ok):
    assert dt.usable(raw) == ok


def test_from_filename():
    assert dt.from_filename(r"C:\data\Annual_Report-2023.pdf") == "Annual Report 2023"
    assert dt.from_filename("") == "Untitled document"


# ── PDF ──────────────────────────────────────────────────────────────────────

pdfplumber = pytest.importorskip("pdfplumber")


def _pdf(meta_title=None, big=True):
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    if meta_title is not None:
        c.setTitle(meta_title)
    if big:
        c.setFont("Helvetica-Bold", 30)
        c.drawString(72, 760, "Water Risk in")
        c.drawString(72, 726, "Emerging Markets")
    c.setFont("Helvetica", 10)
    for k in range(30):
        c.drawString(72, 680 - k * 12, f"Body line {k} with ordinary words in it.")
    if big:
        c.setFont("Helvetica-Bold", 40)          # drop cap far below the title
        c.drawString(72, 250, "T")
    c.save()
    buf.seek(0)
    return buf


def test_pdf_metadata_title_wins():
    with pdfplumber.open(_pdf("Board Pack March")) as p:
        assert dt.pdf_title(p, "x.pdf") == "Board Pack March"


def test_pdf_junk_metadata_falls_back_to_largest_text():
    with pdfplumber.open(_pdf("Microsoft Word - temp.doc")) as p:
        assert dt.pdf_title(p, "x.pdf") == "Water Risk in Emerging Markets"


def test_pdf_without_display_type_uses_filename():
    with pdfplumber.open(_pdf(None, big=False)) as p:
        assert dt.pdf_title(p, "/tmp/fund_factsheet_2024.pdf") == "fund factsheet 2024"


# ── DOCX / PPTX ──────────────────────────────────────────────────────────────

def test_docx_title_order():
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("intro text")
    d.add_heading("Credit Policy Review", level=1)
    assert dt.docx_title(d, "a.docx") == "Credit Policy Review"
    d.add_paragraph("Group Credit Policy", style="Title")
    assert dt.docx_title(d, "a.docx") == "Group Credit Policy"
    d.core_properties.title = "Credit Policy 2025"
    assert dt.docx_title(d, "a.docx") == "Credit Policy 2025"
    assert dt.docx_title(docx.Document(), "my_notes.docx") == "my notes"


def test_pptx_title_order():
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "Q4 Investor Update"
    assert dt.pptx_title(prs, "deck.pptx") == "Q4 Investor Update"
    prs.core_properties.title = "FY25 Investor Update"
    assert dt.pptx_title(prs, "deck.pptx") == "FY25 Investor Update"
    assert dt.pptx_title(pptx.Presentation(), "board_deck.pptx") == "board deck"


def test_tag_sets_metadata():
    class D:
        def __init__(self): self.metadata = {}
    docs = dt.tag([D(), D()], "Annual Report")
    assert all(d.metadata["doc_title"] == "Annual Report" for d in docs)


# ── chunker prefix ───────────────────────────────────────────────────────────

class Doc:
    def __init__(self, c, m): self.content, self.metadata = c, m


TEXT = "[TEXT]\n" + "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(6)) + \
       "\n\n[CHARTS]\n[Chart Data]\n# Figure 1: Sales\n| Year | Sales |\n|---|---|\n| 2020 | 5 |"


@pytest.mark.parametrize("method", ["recursive", "structure", "semantic"])
def test_every_chunk_has_title_and_fits(method):
    docs = [Doc(TEXT, {"page": 0, "source": "x.pdf", "doc_title": "Sales Review"})]
    chunks = chunker.chunk_documents(docs, 300, 30, method, add_title=True)
    assert chunks
    for c in chunks:
        assert c["content"].startswith("Document: Sales Review\n")
        assert c["metadata"]["doc_title"] == "Sales Review"
        if method != "semantic":                    # semantic falls back to 500
            assert len(c["content"]) <= 300
    fig = [c for c in chunks if c["metadata"].get("is_figure")]
    assert fig and fig[0]["content"].splitlines()[1].startswith("[Chart Data] Figure 1")


def test_filename_fallback_and_toggle(monkeypatch):
    docs = [Doc("[TEXT]\nhello world", {"page": 0, "source": "/x/q3_board_pack.pdf"})]
    assert chunker.chunk_documents(docs, 300, 30)[0]["content"].startswith("Document: q3 board pack\n")
    monkeypatch.setenv("CHUNK_DOC_TITLE", "false")
    assert chunker.chunk_documents(docs, 300, 30)[0]["content"] == "[TEXT]\nhello world"
    monkeypatch.delenv("CHUNK_DOC_TITLE")
    assert chunker.chunk_documents(docs, 300, 30, add_title=False)[0]["content"] == "[TEXT]\nhello world"


def test_no_title_no_source_no_prefix():
    docs = [Doc("[TEXT]\nhello", {"page": 0})]
    assert chunker.chunk_documents(docs, 300, 30)[0]["content"] == "[TEXT]\nhello"
