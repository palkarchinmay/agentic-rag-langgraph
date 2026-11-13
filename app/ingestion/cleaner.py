"""cleaner.py — Post-OCR text cleaning."""
import re

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[^\x20-\x7E\n\t\u0900-\u097F]', ' ', text)
    text = re.sub(r'[ \t]{3,}', '  ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return text.strip()
