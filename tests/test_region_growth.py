"""Step 7: chart regions grow over their title / axis labels, not over the
neighbouring text column. Synthetic PDF (reportlab), offline."""
import io
import pytest

pdfplumber = pytest.importorskip("pdfplumber")
pytest.importorskip("reportlab")
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.ingestion import parser


def _pdf():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    H = letter[1]
    # left column: chart with title above, axis labels below and left
    c.setFont("Helvetica-Bold", 10)
    c.drawString(60, H - 100, "FIGURE 4")
    c.drawString(60, H - 113, "Funds by Theme")
    base = H - 300
    for i, v in enumerate([58, 85, 21, 22, 33, 19, 11]):
        c.rect(80 + i * 32, base, 18, v * 1.5, fill=1)
    c.line(75, base, 300, base)
    c.line(75, base, 75, base + 180)          # y axis up to the top tick
    c.setFont("Helvetica", 6)
    for i, lab in enumerate(["Access", "Access", "Green", "Environ-", "Sustain", "Employ", "Other"]):
        c.drawString(80 + i * 32, base - 10, lab)
    for i, lab in enumerate(["to Basic", "to Finance", "Tech", "mental", "Consumer", "Gen", ""]):
        if lab:
            c.drawString(80 + i * 32, base - 17, lab)
    for k, y in enumerate([0, 40, 80, 120]):
        c.drawString(62, base + y * 1.5, str(y))
    c.drawString(60, base - 30, "Source: Some database, 2014.")
    # body paragraph above the figure, same column
    c.setFont("Helvetica", 10)
    for k in range(4):
        c.drawString(60, H - 60 - k * 12 + 36, f"Intro paragraph line {k} above.")
    # right column: body text, close to the chart
    c.setFont("Helvetica", 10)
    for k in range(30):
        c.drawString(320, H - 100 - k * 12, f"Body text line {k} of the other column here.")
    c.save()
    return buf.getvalue()


def test_region_includes_title_and_labels_not_body():
    with pdfplumber.open(io.BytesIO(_pdf())) as pdf:
        page = pdf.pages[0]
        img = Image.new("RGB", (int(page.width * 2), int(page.height * 2)), "white")
        regions = parser._find_image_regions(page, img, 144)
    charts = [r for r in regions if r[1]]
    assert len(charts) == 1
    crop, known, text = charts[0]
    for s in ("FIGURE 4", "Funds by Theme", "to Finance", "Environ-", "mental",
              "Source: Some database", "80"):
        assert s in text, s
    assert "Body text" not in text
    assert "Intro paragraph" not in text
    assert crop.width < img.width * 0.6               # did not swallow the column


def test_region_context_puts_region_text_first():
    ctx = parser._region_context("FIGURE 4\nFunds", "page text " * 100)
    assert ctx.index("FIGURE 4") < ctx.index("page text")
    assert len(ctx) < 2000
