"""Configurable PII masking (PII_MASKING_ENABLED / PII_SKIP_CATEGORIES)."""
from app.security.pii import mask_pii, detect_pii

SAMPLE = (
    "Date of joining: 15/07/2024\n"
    "CTC: Rs. 3,60,000 per annum\n"
    "THE COMPANY PROVIDED BENEFITS\n"
    "SWIFT code: HDFCINBB\n"
    "Aadhaar: 1234 5678 9012\n"
)


def test_default_keeps_dates_and_salary(monkeypatch):
    monkeypatch.delenv("PII_SKIP_CATEGORIES", raising=False)
    monkeypatch.delenv("PII_MASKING_ENABLED", raising=False)
    out = mask_pii(SAMPLE)
    assert "15/07/2024" in out
    assert "3,60,000" in out
    assert "[AADHAAR]" in out


def test_capitalised_words_are_not_swift(monkeypatch):
    monkeypatch.delenv("PII_SKIP_CATEGORIES", raising=False)
    out = mask_pii(SAMPLE)
    assert "THE COMPANY PROVIDED BENEFITS" in out
    assert "SWIFT code: [SWIFT]" in out


def test_empty_skip_list_masks_everything(monkeypatch):
    monkeypatch.setenv("PII_SKIP_CATEGORIES", "")
    out = mask_pii(SAMPLE)
    assert "[DATE]" in out and "[SALARY]" in out


def test_master_switch_off(monkeypatch):
    monkeypatch.setenv("PII_MASKING_ENABLED", "false")
    assert mask_pii(SAMPLE) == SAMPLE


def test_detection_unaffected_by_skip_list(monkeypatch):
    monkeypatch.delenv("PII_SKIP_CATEGORIES", raising=False)
    assert "SALARY" in detect_pii(SAMPLE)
