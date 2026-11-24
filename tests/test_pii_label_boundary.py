"""Context labels match whole words only (regression: 'Image:' was masked as AGE)."""
from app.security.pii import mask_pii


def test_label_inside_word_is_not_masked(monkeypatch):
    monkeypatch.delenv("PII_MASKING_ENABLED", raising=False)
    for line in ["[Image: photo of the authors]", "Percentage: 27%",
                 "Hotel: Hilton Chicago", "Filename: report.pdf", "Spin: 3"]:
        assert mask_pii(line) == line, line


def test_real_labels_still_masked(monkeypatch):
    monkeypatch.delenv("PII_MASKING_ENABLED", raising=False)
    monkeypatch.delenv("PII_SKIP_CATEGORIES", raising=False)
    assert "[AGE]" in mask_pii("Age: 34")
    assert "[NAME]" in mask_pii("Full name: Jane Doe")
    assert "[PHONE]" in mask_pii("Tel: 555 0100")
    assert "[CREDENTIAL]" in mask_pii("PIN: 4321")
