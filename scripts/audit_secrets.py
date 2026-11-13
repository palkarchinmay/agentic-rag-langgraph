"""
scripts/audit_secrets.py
Run this locally to find potential secrets before setting up vault.
Usage: python scripts/audit_secrets.py
"""

import os
import re
from pathlib import Path

# Patterns that indicate secrets
PATTERNS = {
    "OpenAI API key":        re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "Anthropic API key":     re.compile(r"sk-ant-[A-Za-z0-9\-]{20,}"),
    "AWS Access Key":        re.compile(r"AKIA[A-Z0-9]{16}"),
    "AWS Secret Key":        re.compile(r"(?i)aws.{0,20}secret.{0,20}['\"][A-Za-z0-9/+]{40}"),
    "Google API key":        re.compile(r"AIza[A-Za-z0-9\-_]{35}"),
    "Generic password":      re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{8,}"),
    "Generic secret":        re.compile(r"(?i)(secret|token|api_key)\s*[=:]\s*['\"][^'\"]{8,}"),
    "JWT secret":            re.compile(r"(?i)jwt.{0,10}secret.{0,10}['\"][^'\"]{16,}"),
    "Database URL":          re.compile(r"(postgresql|mysql|mongodb)://[^'\"\s]{10,}"),
    "Private key header":    re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
}

# Files/dirs to skip
SKIP_DIRS  = {".git", ".venv", "venv", "__pycache__", "node_modules",
              ".mypy_cache", ".pytest_cache", "chroma_db", ".cache"}
SKIP_FILES = {".env", ".env.example", "audit_secrets.py"}
SKIP_EXTS  = {".pyc", ".pyo", ".png", ".jpg", ".pdf", ".docx",
              ".pkl", ".bin", ".lock"}

findings = []

root = Path(".")
for path in root.rglob("*"):
    # Skip directories and binary files
    if any(skip in path.parts for skip in SKIP_DIRS):
        continue
    if path.name in SKIP_FILES:
        continue
    if path.suffix in SKIP_EXTS:
        continue
    if not path.is_file():
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue

    for pattern_name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            line_num = text[:match.start()].count("\n") + 1
            findings.append({
                "file":    str(path),
                "line":    line_num,
                "pattern": pattern_name,
                "snippet": match.group()[:40] + "...",
            })

if findings:
    print(f"\n⚠️  Found {len(findings)} potential secret(s):\n")
    for f in findings:
        print(f"  {f['file']}:{f['line']}")
        print(f"  Pattern: {f['pattern']}")
        print(f"  Snippet: {f['snippet']}")
        print()
    print("ACTION: Move these to vault before proceeding.\n")
else:
    print("\n✅ No secrets found in source code.\n")
    print("Make sure to also check:")
    print("  git log --all -p | grep -E 'sk-|AKIA|password'")
    print("  docker inspect <container> | grep -i secret\n")