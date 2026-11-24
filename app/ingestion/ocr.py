"""
ocr.py
------
OCR backends — tiered from local to cloud.

Tier 1 — Local (no cloud cost, works offline, good baseline):
  pytesseract : Tesseract OCR — requires system binary but well-supported
                in all cloud Docker images. Good for clean scans.
  easyocr     : Deep-learning OCR, better on noisy/rotated images.
                No system binary needed.

Tier 2 — AWS:
  textract    : Best for structured financial docs (tables, forms).
                Understands document layout, not just pixel text.

Tier 3 — GCP:
  docai       : Google Document AI — excellent on mixed-script docs,
                handwriting, stamps. Best overall for Indian financial docs.

Tier 4 — Azure:
  azure_di    : Azure Document Intelligence — strong on invoices, forms.
                Good EU/compliance story.

Tier 5 — Vision LLM (any provider):
  vision      : a vision-language model reads the page image and returns
                Markdown: text in reading order plus charts as label/value
                tables and diagrams as box/arrow lists. Model is chosen by
                VISION_MODEL=provider:model (default anthropic:claude-opus-5-5;
                also openai:, google:, vertex:, azure:, bedrock:, ollama:).
                See vision.py.

All imports are LAZY — only loaded when the backend is actually called.
This means the app starts fast and doesn't crash if a cloud SDK isn't configured.

Selection order in production (configurable via OCR_METHOD env var):
  docai > textract > azure_di > easyocr > pytesseract
"""

import os
import io
import logging

logger = logging.getLogger("financial_rag.ocr")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1a — pytesseract  (local, requires tesseract binary)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_pytesseract(image_np) -> str:
    """
    Tesseract OCR via pytesseract.
    Good on clean, high-contrast scans. Fast on CPU.

    Requires tesseract binary:
      Linux/Docker : apt install tesseract-ocr tesseract-ocr-eng
      macOS        : brew install tesseract
      Windows      : https://github.com/UB-Mannheim/tesseract/wiki

    Recommended config:
      --oem 3   = LSTM + legacy engine (best accuracy)
      --psm 3   = fully automatic page segmentation (best for mixed layouts)
    """
    try:
        import pytesseract
        from PIL import Image
        import numpy as np

        # Windows installers usually don't add tesseract.exe to PATH.
        # Point TESSERACT_CMD at it, e.g.
        #   TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
        tesseract_cmd = os.getenv("TESSERACT_CMD", "")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Tesseract expects PIL image or numpy array
        # image_np is BGR — convert to RGB for PIL
        import cv2
        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        config = "--oem 3 --psm 3"
        text   = pytesseract.image_to_string(pil_image, config=config)
        return text.strip()

    except ImportError:
        raise ImportError(
            "pytesseract not installed. Run: pip install pytesseract\n"
            "Also install system binary: apt install tesseract-ocr"
        )
    except Exception as e:
        if type(e).__name__ == "TesseractNotFoundError":
            # Binary missing: fall back instead of silently returning no text.
            logger.warning(
                "Tesseract binary not found (install it or set TESSERACT_CMD) "
                "-- falling back to easyocr"
            )
            return extract_text_easyocr(image_np)
        logger.warning("pytesseract failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1b — easyocr  (local, no system binary, better on noisy images)
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except ImportError:
            raise ImportError("Run: pip install easyocr")
    return _easyocr_reader


def extract_text_easyocr(image_np) -> str:
    """EasyOCR — best local option for noisy, rotated, or low-quality scans."""
    reader  = _get_easyocr_reader()
    results = reader.readtext(image_np, detail=0)
    return " ".join(results)


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — AWS Textract
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_textract(image_np) -> str:
    """
    AWS Textract — understands document layout (tables, forms, key-value pairs).
    Best for structured financial documents on AWS infra.

    Required env vars:
      AWS_ACCESS_KEY_ID
      AWS_SECRET_ACCESS_KEY
      AWS_DEFAULT_REGION     (e.g. us-east-1)

    Or use IAM role on EC2/ECS/Lambda (no keys needed if role is attached).
    """
    try:
        import boto3
        import cv2
    except ImportError:
        raise ImportError("Run: pip install boto3")

    try:
        ok, enc       = cv2.imencode(".png", image_np)
        if not ok:
            return ""
        image_bytes   = enc.tobytes()

        client = boto3.client("textract", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        resp   = client.detect_document_text(Document={"Bytes": image_bytes})

        lines  = [
            block["Text"]
            for block in resp.get("Blocks", [])
            if block.get("BlockType") == "LINE"
        ]
        return "\n".join(lines)

    except Exception as e:
        logger.warning("Textract failed: %s", e)
        return ""


def extract_tables_textract(image_np) -> str:
    """
    AWS Textract AnalyzeDocument — extracts tables with cell structure preserved.
    More expensive than detect_document_text but much better for financial tables.
    Use this specifically for pages known to contain tables.
    """
    try:
        import boto3
        import cv2
    except ImportError:
        raise ImportError("Run: pip install boto3")

    try:
        ok, enc     = cv2.imencode(".png", image_np)
        if not ok:
            return ""
        image_bytes = enc.tobytes()

        client = boto3.client("textract", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        resp   = client.analyze_document(
            Document={"Bytes": image_bytes},
            FeatureTypes=["TABLES", "FORMS"],
        )

        # Reconstruct table rows from block relationships
        blocks    = {b["Id"]: b for b in resp.get("Blocks", [])}
        rows      = {}
        table_out = []

        for block in resp.get("Blocks", []):
            if block["BlockType"] != "CELL":
                continue
            row = block.get("RowIndex", 0)
            col = block.get("ColumnIndex", 0)
            cell_text = " ".join(
                blocks[r["Id"]]["Text"]
                for r in block.get("Relationships", [])
                if r["Type"] == "CHILD"
                for r_id in [r["Id"]]
                if r_id in blocks and blocks[r_id]["BlockType"] == "WORD"
            )
            rows.setdefault(row, {})[col] = cell_text

        for row_idx in sorted(rows):
            cols = rows[row_idx]
            table_out.append(" | ".join(cols.get(c, "") for c in sorted(cols)))

        return "\n".join(table_out)

    except Exception as e:
        logger.warning("Textract table extraction failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — GCP Document AI
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_docai(image_np) -> str:
    """
    GCP Document AI — best overall accuracy for financial docs.
    Handles handwriting, stamps, mixed scripts (including Hindi/Devanagari).

    Required env vars:
      GCP_PROJECT_ID
      GCP_LOCATION       (e.g. us or eu)
      GCP_PROCESSOR_ID   (create at console.cloud.google.com/documentai)
      GOOGLE_APPLICATION_CREDENTIALS  (path to service account JSON)
         OR use Workload Identity on GKE

    Processor types to use:
      DOCUMENT_OCR_PROCESSOR       — general documents
      FORM_PARSER_PROCESSOR        — structured forms
      INVOICE_PROCESSOR            — invoices
      BANK_STATEMENT_PROCESSOR     — bank statements (if available in your region)
    """
    try:
        from google.cloud import documentai
        import cv2
    except ImportError:
        raise ImportError("Run: pip install google-cloud-documentai")

    try:
        ok, enc     = cv2.imencode(".png", image_np)
        if not ok:
            return ""
        image_bytes = enc.tobytes()

        project_id   = os.getenv("GCP_PROJECT_ID")
        location     = os.getenv("GCP_LOCATION", "us")
        processor_id = os.getenv("GCP_PROCESSOR_ID")

        if not project_id or not processor_id:
            raise ValueError(
                "Set GCP_PROJECT_ID and GCP_PROCESSOR_ID in .env\n"
                "Create a processor at: console.cloud.google.com/documentai"
            )

        client  = documentai.DocumentProcessorServiceClient()
        name    = client.processor_path(project_id, location, processor_id)

        raw_doc = documentai.RawDocument(content=image_bytes, mime_type="image/png")
        request = documentai.ProcessRequest(name=name, raw_document=raw_doc)
        result  = client.process_document(request=request)

        return result.document.text or ""

    except Exception as e:
        logger.warning("DocAI failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tier 4 — Azure Document Intelligence
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_azure_di(image_np) -> str:
    """
    Azure Document Intelligence (formerly Form Recognizer).
    Strong on invoices, receipts, identity documents. Good EU compliance story.

    Required env vars:
      AZURE_DI_ENDPOINT   (e.g. https://<name>.cognitiveservices.azure.com)
      AZURE_DI_KEY        (or use managed identity — see azure-identity)

    Model options (set AZURE_DI_MODEL env var):
      prebuilt-read         — general OCR (default)
      prebuilt-layout       — layout + tables
      prebuilt-invoice      — invoice extraction
      prebuilt-idDocument   — passports, driving licences, PAN-like IDs
      prebuilt-businessCard — business cards
    """
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
        import cv2
    except ImportError:
        raise ImportError("Run: pip install azure-ai-documentintelligence")

    try:
        ok, enc     = cv2.imencode(".png", image_np)
        if not ok:
            return ""
        image_bytes = enc.tobytes()

        endpoint = os.getenv("AZURE_DI_ENDPOINT")
        key      = os.getenv("AZURE_DI_KEY")
        model    = os.getenv("AZURE_DI_MODEL", "prebuilt-read")

        if not endpoint or not key:
            raise ValueError(
                "Set AZURE_DI_ENDPOINT and AZURE_DI_KEY in .env\n"
                "Create resource at: portal.azure.com → Document Intelligence"
            )

        client  = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
        poller  = client.begin_analyze_document(
            model,
            analyze_request=io.BytesIO(image_bytes),
            content_type="image/png",
        )
        result  = poller.result()

        lines = []
        for page in result.pages or []:
            for line in page.lines or []:
                lines.append(line.content)

        return "\n".join(lines)

    except Exception as e:
        logger.warning("Azure DI failed: %s", e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Tier 5 — Vision LLM (provider-agnostic, VISION_MODEL)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_vision(image_np) -> str:
    """Full-page transcription with the configured vision model. Falls back
    to easyocr if the model returns nothing (no key, refusal, error)."""
    from app.ingestion.vision import extract_text_vision as _v
    text = _v(image_np)
    if not text:
        logger.warning("Vision OCR returned no text -- falling back to easyocr")
        return extract_text_easyocr(image_np)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Unified router — called from parser.py
# ─────────────────────────────────────────────────────────────────────────────

# Quality ranking (higher = better for financial docs)
_OCR_QUALITY_RANK = {
    "vision":     4,   # layout + charts + diagrams, not just characters
    "docai":      4,   # best overall
    "textract":   3,   # best for structured tables on AWS
    "azure_di":   3,   # best for forms/invoices on Azure
    "easyocr":    2,   # best local
    "pytesseract":1,   # baseline local
}

_OCR_DISPATCH = {
    "pytesseract": extract_text_pytesseract,
    "easyocr":     extract_text_easyocr,
    "textract":    extract_text_textract,
    "docai":       extract_text_docai,
    "azure_di":    extract_text_azure_di,
    "vision":      extract_text_vision,
}


# Names the UI dropdowns use -> internal dispatch keys.
# The Chainlit settings offer "tesseract" / "google_docai" / "none", which
# previously didn't match any key here and silently fell back to easyocr.
_OCR_ALIASES = {
    "tesseract":    "pytesseract",
    "google_docai": "docai",
    "document_ai":  "docai",
    "azure":        "azure_di",
    "aws_textract": "textract",
    "vlm":          "vision",
    "vision_llm":   "vision",
    "claude":       "vision",          # VISION_MODEL picks the provider
    "claude_vision": "vision",         # name used by the earlier patch
}


def normalize_method(method: str) -> str:
    """Map UI/alias names (e.g. 'tesseract', 'google_docai') to dispatch keys."""
    method = (method or os.getenv("OCR_METHOD", "easyocr")).strip().lower()
    return _OCR_ALIASES.get(method, method)


def extract_text(image_np, method: str = "easyocr") -> str:
    """
    Unified OCR entry point.

    Args:
        image_np : BGR numpy array
        method   : 'pytesseract' | 'easyocr' | 'textract' | 'docai' | 'azure_di'
                   | 'vision'
                   Defaults to OCR_METHOD env var, else 'easyocr'

    Returns: extracted text string
    """
    method = normalize_method(method)

    if method == "none":
        # OCR explicitly disabled in settings: scanned pages yield no text.
        logger.debug("OCR disabled (method=none)")
        return ""

    if method not in _OCR_DISPATCH:
        logger.warning("Unknown OCR method '%s', falling back to easyocr", method)
        method = "easyocr"

    logger.debug("OCR method: %s", method)
    fn = _OCR_DISPATCH[method]
    return fn(image_np)


def available_methods() -> list:
    """Return list of all OCR method keys for UI dropdowns."""
    return list(_OCR_DISPATCH.keys())


def quality_rank(method: str) -> int:
    """Return quality rank for a given OCR method (higher = better)."""
    return _OCR_QUALITY_RANK.get(method, 0)
