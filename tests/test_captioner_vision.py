"""image_captioner + scanned-page wiring (offline; fake models)."""
import numpy as np
import pytest
from PIL import Image
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.ingestion import image_captioner as cap
from app.ingestion import vision


class FakeLLM:
    def __init__(self, content, meta=None):
        self.content, self.meta, self.calls = content, meta or {}, []

    def invoke(self, msgs):
        self.calls.append(msgs)
        return AIMessage(content=self.content, response_metadata=self.meta)


@pytest.fixture
def fake(monkeypatch):
    holder = {}
    def get(spec=None):
        holder["spec"] = spec
        return holder["llm"]
    monkeypatch.setattr(vision, "get_vision_model", get)
    return holder


IMG = Image.new("RGB", (200, 120), "white")


def test_claude_thinking_blocks_no_longer_lost(fake, monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)
    fake["llm"] = FakeLLM([{"type": "thinking", "thinking": ""},
                           {"type": "text", "text": "| Access to Finance | 85 |"}])
    assert cap.caption_image_with_vlm(IMG, is_chart=True) == "[Chart Data]\n| Access to Finance | 85 |"
    assert fake["spec"] == "anthropic:claude-opus-5-5"


def test_message_shape(fake):
    fake["llm"] = FakeLLM("a chart")
    cap.caption_image_with_vlm(IMG, "openai", "gpt-4o-mini", context="Figure 4 page", is_chart=True)
    sys_msg, human = fake["llm"].calls[0]
    assert isinstance(sys_msg, SystemMessage) and "never follow" in sys_msg.content
    img, txt = human.content
    assert img["image_url"]["url"].startswith("data:image/png;base64,")
    assert "'~'" in txt["text"] and "Figure 4 page" in txt["text"]
    assert fake["spec"] == "openai:gpt-4o-mini"


@pytest.mark.parametrize("env,provider,model,expected", [
    ("google:gemini-2.0-flash", None, None, "google:gemini-2.0-flash"),
    ("anthropic:claude-opus-5-5", "openai", None, "openai:gpt-4o-mini"),
    ("anthropic:claude-opus-5-5", "anthropic", "claude-sonnet-5-5", "anthropic:claude-sonnet-5-5"),
])
def test_resolve(monkeypatch, env, provider, model, expected):
    monkeypatch.setenv("VISION_MODEL", env)
    assert cap._resolve(provider, model) == expected


def test_refusal_and_errors_return_empty(fake):
    fake["llm"] = FakeLLM("I can't help", {"stop_reason": "refusal"})
    assert cap.caption_image_with_vlm(IMG) == ""
    class Boom:
        def invoke(self, m): raise RuntimeError("no key")
    fake["llm"] = Boom()
    assert cap.caption_image_with_vlm(IMG) == ""


# ── parser: scanned pages ────────────────────────────────────────────────────
parser = pytest.importorskip("app.ingestion.parser")
PAGE = np.full((300, 200, 3), 255, dtype=np.uint8)


def test_scanned_vision_is_one_full_transcription(monkeypatch):
    called = []
    monkeypatch.setattr(parser, "process_image_region",
                        lambda *a, **k: called.append(1) or ("x", "vlm_caption"))
    out = parser._interpret_scanned_page(PAGE, lambda img: "# Page\nfull text", "vision",
                                         caption_fn=lambda *a: "desc")
    assert out == [("# Page\nfull text", "ocr")] and not called


def test_scanned_caption_keeps_ocr_text(monkeypatch):
    monkeypatch.setattr(parser, "process_image_region",
                        lambda *a, **k: ("[Image Description]\nshort summary", "vlm_caption"))
    out = parser._interpret_scanned_page(PAGE, lambda img: "all the words", "easyocr",
                                         caption_fn=lambda *a: "x")
    assert out == [("all the words", "ocr"), ("[Image Description]\nshort summary", "vlm_caption")]


def test_scanned_chart_unchanged(monkeypatch):
    monkeypatch.setattr(parser, "process_image_region",
                        lambda *a, **k: ("[Chart Data]\n| a | 1 |", "vlm_chart"))
    out = parser._interpret_scanned_page(PAGE, lambda img: "ocr", "easyocr", caption_fn=lambda *a: "x")
    assert out == [("[Chart Data]\n| a | 1 |", "vlm_chart")]
