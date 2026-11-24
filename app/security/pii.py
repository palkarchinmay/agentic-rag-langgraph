"""
pii.py — Generic PII detection and masking.

Pattern ordering rules (critical):
  1. Most specific patterns first (SSN before phone, CARD before AADHAAR)
  2. AADHAAR uses space-only separator to avoid matching credit card hyphens
  3. CARD_EXPIRY restricted to MM/YY (2-digit year) only — not full dates
  4. DATE_DMY kept broad but ordered after CARD_EXPIRY so CARD_EXPIRY wins first
  5. Context-label patterns mask only the value after the colon — never free text
"""

import os
import re
from typing import Dict, FrozenSet, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (read from the environment on every call, so a restart is
# enough to change it and tests can override it with monkeypatch)
#
#   PII_MASKING_ENABLED   true/false   master switch for mask_pii (default true)
#   PII_SKIP_CATEGORIES   comma list   tags that mask_pii leaves visible
#                                      (default: dates and salary, because
#                                      financial questions need them)
#
# detect_pii() is NOT affected: the security node still sees every category.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SKIP_CATEGORIES = "DATE_ISO,DATE_DMY,DATE,SALARY"


def masking_enabled() -> bool:
    return os.getenv("PII_MASKING_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off")


def skip_categories() -> FrozenSet[str]:
    raw = os.getenv("PII_SKIP_CATEGORIES", DEFAULT_SKIP_CATEGORIES)
    return frozenset(t.strip().upper() for t in raw.split(",") if t.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Structural patterns — ordered most-specific → least-specific
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURAL_PATTERNS: Dict[str, str] = {

    # ── Email (before everything — @ sign is unambiguous) ─────────────────
    "EMAIL":       r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",

    # ── SSN (before phone — both use digit groups with dashes) ────────────
    "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",

    # ── US EIN (before account numbers) ──────────────────────────────────
    "EIN":         r"\b\d{2}-\d{7}\b",

    # ── Credit/debit card: 13–19 digits, groups of 4 with spaces/hyphens ─
    # Must come BEFORE AADHAAR — both match 12 digit patterns
    "CARD_NUMBER": r"\b(?:\d{4}[\s\-]){3}\d{1,4}\b",

    # ── Card expiry: MM/YY only (2-digit year) ────────────────────────────
    # Restricted to 2-digit year so it does NOT match 15/08/1990 style DOBs
    "CARD_EXPIRY": r"\b(?:0[1-9]|1[0-2])/\d{2}\b",

    # ── Indian PAN: strict AAAAA9999A ─────────────────────────────────────
    "PAN":         r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",

    # ── Aadhaar: 12 digits in 4-4-4 groups separated by SPACES only ──────
    # Using space-only separator prevents matching credit card hyphen groups
    "AADHAAR":     r"\b\d{4} \d{4} \d{4}\b",

    # ── IFSC: 4 letters + 0 + 6 alphanumeric ─────────────────────────────
    "IFSC":        r"\b[A-Z]{4}0[A-Z0-9]{6}\b",

    # ── Indian GST ────────────────────────────────────────────────────────
    "GST":         r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z0-9]\b",

    # ── Indian vehicle reg: MH12AB1234 ────────────────────────────────────
    "VEHICLE_REG": r"\b[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}\b",

    # ── UK NI number: AB 12 34 56 C ──────────────────────────────────────
    "UK_NI":       r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b",

    # ── Passport: 1-2 letters + 6-8 digits ───────────────────────────────
    "PASSPORT":    r"\b[A-Z]{1,2}\d{6,8}\b",

    # ── IBAN: 2 letters + 2 digits + 11-30 alphanum ───────────────────────
    "IBAN":        r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",

    # ── SWIFT/BIC: no structural pattern on purpose. "[A-Z]{6}[A-Z0-9]{2}"
    #    matched ordinary capitalised words (COMPANY, PROVIDED, ...), so SWIFT
    #    is masked only after a "SWIFT:" / "BIC:" label (see context patterns).

    # ── Bank account: 10-18 digits (after card to avoid overlap) ─────────
    "ACCOUNT_NUM": r"\b\d{10,18}\b",

    # ── Phone: +91 NNNNN NNNNN or 10-digit with separators ───────────────
    "PHONE":       r"\b(?:\+\d{1,3}\s?)?\d{5}\s\d{5}\b"
                   r"|\b(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}\b",

    # ── Dates (after CARD_EXPIRY so MM/YY already consumed) ───────────────
    "DATE_ISO":    r"\b\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])\b",
    "DATE_DMY":    r"\b(?:0[1-9]|[12]\d|3[01])[/\-.]\d{2}[/\-.]\d{2,4}\b",

    # ── Network ───────────────────────────────────────────────────────────
    "IP_ADDRESS":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "COORDINATES": r"\b[-+]?\d{1,3}\.\d{4,}\s*,\s*[-+]?\d{1,3}\.\d{4,}\b",
}


# ─────────────────────────────────────────────────────────────────────────────
# Context-label patterns — mask VALUE after label colon only
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT_LABEL_PATTERNS: List[Tuple[str, str]] = [

    # Personal identity
    (r"(?:full[\s\-]?name|first[\s\-]?name|last[\s\-]?name|sur[\s\-]?name"
     r"|middle[\s\-]?name|applicant[\s\-]?name|candidate[\s\-]?name"
     r"|employee[\s\-]?name|customer[\s\-]?name|client[\s\-]?name"
     r"|member[\s\-]?name|patient[\s\-]?name|name)\s*[:\-=]",            "NAME"),

    (r"(?:father[''s]*[\s\-]?name|mother[''s]*[\s\-]?name"
     r"|spouse[''s]*[\s\-]?name|husband[''s]*[\s\-]?name"
     r"|wife[''s]*[\s\-]?name|guardian[''s]*[\s\-]?name"
     r"|nominee[''s]*[\s\-]?name|next[\s\-]?of[\s\-]?kin)\s*[:\-=]",    "NAME"),

    (r"(?:date[\s\-]?of[\s\-]?birth|dob|birth[\s\-]?date|born)\s*[:\-=]","DOB"),
    (r"(?:gender|sex)\s*[:\-=]",                                          "GENDER"),
    (r"(?:nationality|citizenship)\s*[:\-=]",                             "NATIONALITY"),
    (r"(?:marital[\s\-]?status|civil[\s\-]?status)\s*[:\-=]",            "MARITAL_STATUS"),
    (r"(?:religion|caste|ethnicity|race)\s*[:\-=]",                      "SENSITIVE"),
    (r"(?:blood[\s\-]?group|blood[\s\-]?type)\s*[:\-=]",                 "HEALTH"),
    (r"(?:age)\s*[:\-=]",                                                 "AGE"),

    # Contact & location
    (r"(?:residential[\s\-]?address|permanent[\s\-]?address"
     r"|current[\s\-]?address|home[\s\-]?address|mailing[\s\-]?address"
     r"|correspondence[\s\-]?address|office[\s\-]?address"
     r"|address)\s*[:\-=]",                                               "ADDRESS"),
    (r"(?:pin[\s\-]?code|zip[\s\-]?code|postal[\s\-]?code"
     r"|post[\s\-]?code)\s*[:\-=]",                                       "PINCODE"),
    (r"(?:mobile[\s\-]?(?:no|number)|cell(?:ular)?[\s\-]?(?:no|number)"
     r"|phone[\s\-]?(?:no|number)|telephone[\s\-]?(?:no|number)"
     r"|contact[\s\-]?(?:no|number)|tel)\s*[:\-=]",                      "PHONE"),
    (r"(?:email[\s\-]?(?:id|address)|e[\s\-]?mail)\s*[:\-=]",            "EMAIL"),

    # Financial
    (r"(?:(?:annual|monthly|gross|net|take[\s\-]?home)[\s\-]?"
     r"(?:salary|pay|income|compensation|ctc|wage|remuneration|stipend)"
     r"|salary|ctc|compensation|remuneration|stipend)\s*[:\-=]",          "SALARY"),
    (r"(?:account[\s\-]?(?:no|number|num)"
     r"|a\/c[\s\-]?(?:no|number)|bank[\s\-]?account)\s*[:\-=]",          "ACCOUNT_NUM"),
    (r"(?:credit[\s\-]?card|debit[\s\-]?card"
     r"|card[\s\-]?(?:no|number))\s*[:\-=]",                             "CARD_NUMBER"),
    (r"(?:cvv|cvc|security[\s\-]?code|card[\s\-]?verification)\s*[:\-=]","CVV"),
    (r"(?:atm[\s\-]?pin|card[\s\-]?pin|pin)\s*[:\-=]",                   "CREDENTIAL"),
    (r"(?:password|passcode|secret)\s*[:\-=]",                            "CREDENTIAL"),
    (r"(?:routing[\s\-]?(?:no|number)|aba[\s\-]?(?:no|number))\s*[:\-=]","ROUTING_NUM"),
    (r"(?:ifsc[\s\-]?(?:code|no)?)\s*[:\-=]",                            "IFSC"),

    # Employment
    (r"(?:employee[\s\-]?(?:id|no|code|number)"
     r"|emp[\s\-]?(?:id|no|code)|staff[\s\-]?(?:id|no))\s*[:\-=]",      "EMPLOYEE_ID"),
    (r"(?:joining[\s\-]?date|date[\s\-]?of[\s\-]?joining"
     r"|start[\s\-]?date|hire[\s\-]?date)\s*[:\-=]",                     "DATE"),

    # Medical / Insurance
    (r"(?:diagnosis|medical[\s\-]?condition|illness|disease"
     r"|prescription|medication|treatment)\s*[:\-=]",                     "HEALTH"),
    (r"(?:policy[\s\-]?(?:no|number)|insurance[\s\-]?(?:id|no)"
     r"|claim[\s\-]?(?:no|number))\s*[:\-=]",                            "POLICY_NUM"),

    # National IDs
    (r"(?:pan[\s\-]?(?:no|number|card[\s\-]?no)?)\s*[:\-=]",             "PAN"),
    (r"(?:aadhaar|aadhar|uid[\s\-]?(?:no|number))\s*[:\-=]",             "AADHAAR"),
    (r"(?:passport[\s\-]?(?:no|number))\s*[:\-=]",                       "PASSPORT"),
    (r"(?:driving[\s\-]?licen[sc]e[\s\-]?(?:no|number)?"
     r"|dl[\s\-]?(?:no|number)|license[\s\-]?(?:no|number))\s*[:\-=]",  "DL_NUM"),
    (r"(?:swift|bic)(?:[\s\-]?code)?\s*[:\-=]",                       "SWIFT"),
    (r"(?:voter[\s\-]?(?:id|card)|epic[\s\-]?(?:no|number))\s*[:\-=]",  "VOTER_ID"),
]

_CONTEXT_VALUE_RE = r"[ \t]*([^\n\r]{1,80}?)[ \t]*(?:\n|\r|$)"


# ─────────────────────────────────────────────────────────────────────────────
# OCR-robust secondary patterns (noise-tolerant)
# Applied AFTER structural pass — catches garbled IDs from scanned docs
# ─────────────────────────────────────────────────────────────────────────────

_N = r"[\s.\-_]{0,2}"

_OCR_ROBUST_PATTERNS: Dict[str, str] = {
    # PAN with OCR noise (strict: must start with 4+ caps to reduce false positives)
    "PAN":         rf"\b[A-Z]{{4,5}}{_N}\d{{3,4}}{_N}[A-Z]\b",
    # Aadhaar with noise between groups (still requires 3 groups of 4 digits)
    "AADHAAR":     rf"\b\d{{4}}{_N}\d{{4}}{_N}\d{{4}}\b",
    # Card with noise
    "CARD_NUMBER": rf"\b\d{{4}}{_N}\d{{4}}{_N}\d{{4}}{_N}\d{{1,4}}\b",
}


# ─────────────────────────────────────────────────────────────────────────────
# Compiled caches
# ─────────────────────────────────────────────────────────────────────────────

_compiled_structural = {k: re.compile(v) for k, v in STRUCTURAL_PATTERNS.items()}

# Labels must start at a word boundary: without \b, "Image:" matched the AGE
# label, "Percentage:"/"Page:" too, "Hotel:" matched PHONE (tel), "Spin:" PIN
# and "Filename:" NAME -- each wiping the rest of the line. Claude vision
# output ("[Image: ...]") hit this on every page with a photo.
_compiled_context = [
    (
        re.compile(r"\b" + label, re.IGNORECASE),
        tag,
        re.compile(r"\b" + label + _CONTEXT_VALUE_RE, re.IGNORECASE),
    )
    for label, tag in CONTEXT_LABEL_PATTERNS
]

_compiled_ocr = {k: re.compile(v) for k, v in _OCR_ROBUST_PATTERNS.items()}


# Digit-group IDs (card, Aadhaar) also match runs of years, e.g. a chart's
# x-axis "1995 1997 1999 2001" or "2003 2005 2007 2010". When every 4-digit
# group is a plausible year (1900-2099) the match is left alone. A real card or
# Aadhaar made only of year-like groups is rare; that's the trade-off.
_YEAR_RUN_KEYS = {"CARD_NUMBER", "AADHAAR"}
_YEAR = re.compile(r"\d{4}")


def _is_year_run(match_text: str) -> bool:
    groups = _YEAR.findall(match_text)
    return len(groups) >= 3 and all(1900 <= int(g) <= 2099 for g in groups)


def _sub(pattern, key: str, text: str) -> str:
    if key not in _YEAR_RUN_KEYS:
        return pattern.sub(f"[{key}]", text)
    return pattern.sub(lambda m: m.group(0) if _is_year_run(m.group(0)) else f"[{key}]", text)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_pii(text: str) -> Dict[str, List[str]]:
    findings: Dict[str, List[str]] = {}

    for key, pattern in _compiled_structural.items():
        matches = pattern.findall(text)
        flat = [m if isinstance(m, str) else m[0] for m in matches]
        flat = [m.strip() for m in flat if m.strip()
                and not (key in _YEAR_RUN_KEYS and _is_year_run(m))]
        if flat:
            findings[key] = list(dict.fromkeys(flat))

    for label_re, tag, full_re in _compiled_context:
        if label_re.search(text):
            matches = full_re.findall(text)
            flat = [m.strip() for m in matches if m.strip()]
            if flat:
                existing = findings.get(tag, [])
                existing.extend(flat)
                findings[tag] = list(dict.fromkeys(existing))

    return findings


def mask_pii(text: str) -> str:
    """
    Three-pass masking:
      Pass 1 — Structural regex (ordered most-specific first)
      Pass 2 — Context-label (label: value → label: [TAG])
      Pass 3 — OCR-robust (noise-tolerant patterns on remaining text)
    """
    if not text or not masking_enabled():
        return text
    skip = skip_categories()

    # Pass 1
    for key, pattern in _compiled_structural.items():
        if key in skip:
            continue
        text = _sub(pattern, key, text)

    # Pass 2
    for label_re, tag, full_re in _compiled_context:
        if tag in skip:
            continue
        if label_re.search(text):
            def _replace(m, _tag=tag):
                return m.group(0).replace(m.group(1), f"[{_tag}]", 1)
            text = full_re.sub(_replace, text)

    # Pass 3
    for key, pattern in _compiled_ocr.items():
        if key in skip:
            continue
        text = _sub(pattern, key, text)

    return text


def pii_risk_score(text: str) -> Dict[str, int]:
    findings = detect_pii(text)
    report   = {k: len(v) for k, v in findings.items()}
    report["total_types"]     = len(findings)
    report["total_instances"] = sum(len(v) for v in findings.values())
    return report


def integrate_ner(text: str, ner_fn=None) -> str:
    """Optional NER hook — plug in spaCy/BERT for free-text name masking."""
    if ner_fn is None:
        return text
    entities = sorted(ner_fn(text), key=lambda x: x[0], reverse=True)
    for start, end, label in entities:
        tag  = {"PERSON":"NAME","ORG":"ORG","GPE":"LOCATION",
                "LOC":"LOCATION","DATE":"DATE","MONEY":"AMOUNT"}.get(label, label)
        text = text[:start] + f"[{tag}]" + text[end:]
    return text
