"""Step 7: figure classification, tag routing, figure-aware chunking,
letter-spaced words, region growth. Offline; no API calls."""
import io
import numpy as np
import pytest
from PIL import Image
from langchain_core.messages import AIMessage

from app.ingestion import image_captioner as cap
from app.ingestion import vision
from app.ingestion import chunker
from app.ingestion import layout
from app.ingestion.chart_interpreter import process_image_region


# ── captioner: KIND line ─────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,is_chart,expected", [
    ("KIND: chart\n# Figure 2 Sales\n| Year | Sales |\n|---|---|\n| 2014 | 5 |", False,
     "[Chart Data]\n# Figure 2 Sales\n| Year | Sales |\n|---|---|\n| 2014 | 5 |"),
    ("KIND: logo", True, cap.SKIP),
    ("**KIND:** decorative\n", True, cap.SKIP),
    ("KIND: photo\nA person in a suit.", True, "[Image Description]\nA person in a suit."),
    ("KIND: diagram\n# Flow\nA -> B", False, "[Chart Data]\n# Flow\nA -> B"),
    ("KIND: text\nQuote here.", True, "[Image Description]\nQuote here."),
    ("no kind line, just data", True, "[Chart Data]\nno kind line, just data"),
    ("no kind line, a picture", False, "[Image Description]\nno kind line, a picture"),
    ("KIND: chart", True, ""),                      # nothing after KIND
])
def test_format_caption(raw, is_chart, expected):
    assert cap.format_caption(raw, is_chart) == expected


class FakeLLM:
    def __init__(self, content): self.content, self.calls = content, []
    def invoke(self, msgs):
        self.calls.append(msgs)
        return AIMessage(content=self.content)


def test_prompt_has_kind_hint_and_long_context(monkeypatch):
    llm = FakeLLM("KIND: chart\n# T\n| a | 1 |")
    monkeypatch.setattr(vision, "get_vision_model", lambda spec=None: llm)
    ctx = "Text printed inside this figure: " + "x" * 1500 + " END"
    out = cap.caption_image_with_vlm(Image.new("RGB", (50, 50)), context=ctx, is_chart=True)
    assert out.startswith("[Chart Data]\n# T")
    text = llm.calls[0][1].content[1]["text"]
    assert "KIND:" in text and "probably a chart" in text
    assert "END" in text                              # context no longer cut at 300


# ── chart_interpreter: model's tag wins over the detector ───────────────────

IMG = (np.ones((200, 300, 3)) * 255).astype(np.uint8)


def _run(result, known_chart):
    return process_image_region(IMG, lambda im: "ocr words", context="",
                                use_vision=True, caption_fn=lambda *a: result,
                                known_chart=known_chart)


def test_skip_is_not_indexed():
    assert _run("[Skip]", True) == ("", "image_ignored")


def test_photo_in_vector_cluster_is_caption_not_chart():
    content, method = _run("[Image Description]\nA headshot.", True)
    assert method == "vlm_caption" and content.count("[Image Description]") == 1


def test_chart_tag_not_doubled():
    content, method = _run("[Chart Data]\n# T\n| a | 1 |", False)
    assert method == "vlm_chart" and content.count("[Chart Data]") == 1


def test_untagged_custom_caption_fn_still_tagged():
    assert _run("plain", True) == ("[Chart Data]\nplain", "vlm_chart")


# ── chunker ──────────────────────────────────────────────────────────────────

class Doc:
    def __init__(self, c, m=None): self.content, self.metadata = c, m or {"page": 0}


ROWS = "\n".join(f"| {2000 + i} | {i * 10} |" for i in range(40))
PAGE = ("[TEXT]\nBody paragraph one about something.\n\nFIGURE 1\nGrowth\n\n"
        "[CHARTS]\n[Chart Data]\n# Figure 1: Growth of things\nBar chart.\n\n"
        "| Year | Value |\n|---|---|\n" + ROWS + "\n\nSource: somewhere.\n\n"
        "[Chart Data]\n# Figure 2: Shares\n| Theme | Funds |\n|---|---|\n| A | 85 |")


@pytest.mark.parametrize("method", ["recursive", "structure"])
def test_figure_chunks_carry_title(method):
    chunks = chunker.chunk_documents([Doc(PAGE)], 300, 30, method)
    figs = [c for c in chunks if c["metadata"].get("is_figure")]
    text = [c for c in chunks if not c["metadata"].get("is_figure")]
    assert len(figs) >= 4
    assert all(c["content"].startswith("[Chart Data] Figure") for c in figs)
    assert all(len(c["content"]) <= 300 for c in figs)
    f1 = [c for c in figs if "Figure 1" in c["content"].splitlines()[0]]
    assert len(f1) >= 3
    for c in f1[1:]:
        lines = c["content"].splitlines()
        assert lines[0].endswith("(cont.)")
        if any(l.startswith("| 20") for l in lines):
            assert "| Year | Value |" in lines        # table header repeated
    # every data row survives exactly once, whole
    for i in range(40):
        assert sum(f"| {2000 + i} | {i * 10} |" in c["content"].splitlines()
                   for c in figs) == 1
    # figure text is not also in the page-text chunks
    assert not any("| 2005 |" in c["content"] for c in text)
    assert not any(c["content"].strip() in ("[CHARTS]", "") for c in chunks)


def test_old_untitled_output_still_chunks():
    page = "[TEXT]\nx\n\n[CHARTS]\n[Chart Data]\n## 1. Chart type\nBar.\n| a | 1 |"
    figs = [c for c in chunker.chunk_documents([Doc(page)], 500, 50)
            if c["metadata"].get("is_figure")]
    assert figs and figs[0]["content"].startswith("[Chart Data]\n")


def test_image_description_in_ocr_section():
    page = "[TEXT]\nx\n\n[OCR]\n[Image Description]\nA photo of a bridge."
    chunks = chunker.chunk_documents([Doc(page)], 500, 50, "structure")
    fig = [c for c in chunks if c["metadata"].get("is_figure")][0]
    assert fig["metadata"]["section_type"] == "OCR"
    assert fig["content"] == "[Image Description]\nA photo of a bridge."


# ── layout: letter-spaced display words ──────────────────────────────────────

def W(t, x0, x1, top=100, h=60):
    return {"text": t, "x0": x0, "x1": x1, "top": top, "bottom": top + h}


def test_letterspaced_word_joined():
    out = layout._merge_spaced_words([W("Responsib", 72, 499), W("le", 504.2, 581)])
    assert [w["text"] for w in out] == ["Responsible"]


def test_space_glyph_keeps_words_apart():
    space = {"text": " ", "x0": 443, "x1": 452, "top": 100, "bottom": 160}
    out = layout._merge_spaced_words([W("Impact", 325, 442.5), W("Through", 449.3, 590)], [space])
    assert [w["text"] for w in out] == ["Impact", "Through"]


def test_normal_word_gap_not_joined():
    # body text: 10 pt words, 2.5 pt space and no space glyph -> kept apart
    out = layout._merge_spaced_words([W("net", 72, 86, h=10), W("income", 88.5, 118, h=10)])
    assert len(out) == 2
