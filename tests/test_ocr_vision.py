"""OCR router -> vision model (offline: vision call is monkeypatched)."""
import numpy as np

from app.ingestion import ocr, vision


def test_aliases_route_to_vision():
    for name in ["vision", "vlm", "claude", "claude_vision", "VISION"]:
        assert ocr.normalize_method(name) == "vision"
    assert "vision" in ocr.available_methods()


def test_vision_text_is_returned(monkeypatch):
    monkeypatch.setattr(vision, "extract_text_vision", lambda img: "| Access to Finance | 85 |")
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert ocr.extract_text(img, "vision") == "| Access to Finance | 85 |"


def test_empty_vision_falls_back_to_easyocr(monkeypatch):
    monkeypatch.setattr(vision, "extract_text_vision", lambda img: "")
    monkeypatch.setattr(ocr, "extract_text_easyocr", lambda img: "easyocr text")
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert ocr.extract_text(img, "claude") == "easyocr text"
