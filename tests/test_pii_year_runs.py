"""Chart axes made of years are not masked as card / Aadhaar numbers."""
from app.security.pii import detect_pii, mask_pii


def test_year_axis_kept(monkeypatch):
    monkeypatch.delenv("PII_MASKING_ENABLED", raising=False)
    axis = "1995 1997 1999 2001 2003 2005 2007 2010 2012 2014"
    assert mask_pii(axis) == axis
    assert "CARD_NUMBER" not in detect_pii(axis) and "AADHAAR" not in detect_pii(axis)


def test_real_ids_still_masked(monkeypatch):
    monkeypatch.delenv("PII_MASKING_ENABLED", raising=False)
    assert "[CARD_NUMBER]" in mask_pii("Card 4111 1111 1111 1111 on file")
    assert "[AADHAAR]" in mask_pii("Aadhaar 2345 6789 0123")
