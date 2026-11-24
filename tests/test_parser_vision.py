"""parser.py vision wiring (offline; no model calls)."""
import numpy as np
import pytest

parser = pytest.importorskip("app.ingestion.parser")


def test_vision_ocr_skips_binarization(monkeypatch):
    seen = {}
    def fake_extract(img, method):
        seen["method"], seen["shape"] = method, img.shape
        seen["colour_kept"] = bool((img[..., 0] != img[..., 2]).any())
        return "| Access to Finance | 85 |"
    monkeypatch.setattr(parser, "extract_text", fake_extract)
    img = np.zeros((50, 40, 3), dtype=np.uint8); img[..., 2] = 200   # red
    out = parser._make_ocr_fn("claude")(img)
    assert out == "| Access to Finance | 85 |"
    assert seen["method"] == "vision" and seen["colour_kept"]


def test_classic_ocr_still_preprocessed(monkeypatch):
    called = {}
    monkeypatch.setattr(parser, "preprocess_for_ocr", lambda img: called.setdefault("pre", img))
    monkeypatch.setattr(parser, "extract_text", lambda img, method: "x")
    monkeypatch.setattr(parser, "clean_ocr_text", lambda t: t)
    parser._make_ocr_fn("tesseract")(np.zeros((5, 5, 3), np.uint8))
    assert "pre" in called


def test_caption_model_defaults_to_vision_model(monkeypatch):
    monkeypatch.delenv("VISION_MODEL", raising=False)
    assert parser._resolve_caption_model() == ("anthropic", "claude-opus-5-5")
    monkeypatch.setenv("VISION_MODEL", "google:gemini-2.0-flash")
    assert parser._resolve_caption_model() == ("google", "gemini-2.0-flash")
    assert parser._resolve_caption_model("openai", "gpt-4o-mini") == ("openai", "gpt-4o-mini")


def test_no_caption_fn_when_off():
    assert parser._make_caption_fn(False) is None
