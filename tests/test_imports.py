"""
test_imports.py
---------------
Run this FIRST to verify your environment is set up correctly.

    python test_imports.py

Each line prints ✅ OK or ❌ with the exact error.
Does NOT call any API or need documents.
"""

import sys


def check(label, fn):
    try:
        fn()
        print(f"  ✅  {label}")
    except Exception as e:
        print(f"  ❌  {label}")
        print(f"       └─ {e}")


print("\n── Python & Env ─────────────────────────────")

def check_python():
    assert sys.version_info >= (3, 10), f"Need Python 3.10+, got {sys.version}"
check("Python 3.10+", check_python)

def check_dotenv():
    from dotenv import load_dotenv
    load_dotenv()
check("python-dotenv", check_dotenv)

def check_openai_key():
    import os
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "")
    assert key and key.startswith("sk-"), "OPENAI_API_KEY missing or not set in .env"
check("OPENAI_API_KEY in .env", check_openai_key)


print("\n── Core Libraries ───────────────────────────")
check("langchain",                lambda: __import__("langchain"))
check("langchain-openai",         lambda: __import__("langchain_openai"))
check("langchain-chroma",         lambda: __import__("langchain_chroma"))
check("langchain-text-splitters", lambda: __import__("langchain_text_splitters"))
check("langgraph",                lambda: __import__("langgraph"))
check("chromadb",                 lambda: __import__("chromadb"))
check("rank-bm25",                lambda: __import__("rank_bm25"))
check("sentence-transformers",    lambda: __import__("sentence_transformers"))
check("openai",                   lambda: __import__("openai"))


print("\n── Document Parsing ─────────────────────────")
check("pdfplumber",               lambda: __import__("pdfplumber"))
check("pymupdf (fitz)",           lambda: __import__("fitz"))
check("python-docx",              lambda: __import__("docx"))
check("python-pptx",              lambda: __import__("pptx"))
check("pillow",                   lambda: __import__("PIL"))
check("opencv-headless",          lambda: __import__("cv2"))
check("easyocr",                  lambda: __import__("easyocr"))
check("numpy",                    lambda: __import__("numpy"))


print("\n── UI & Auth ────────────────────────────────")
check("streamlit",                lambda: __import__("streamlit"))
check("bcrypt",                   lambda: __import__("bcrypt"))


print("\n── RAGAS ────────────────────────────────────")
check("ragas",                    lambda: __import__("ragas"))
check("datasets",                 lambda: __import__("datasets"))


print("\n── App Modules ──────────────────────────────")
check("app.ingestion.schema",     lambda: __import__("app.ingestion.schema",    fromlist=["Document"]))
check("app.ingestion.cleaner",    lambda: __import__("app.ingestion.cleaner"))
check("app.ingestion.ocr",        lambda: __import__("app.ingestion.ocr"))
check("app.ingestion.parser",     lambda: __import__("app.ingestion.parser"))
check("app.ingestion.loader",     lambda: __import__("app.ingestion.loader"))
check("app.ingestion.chunker",    lambda: __import__("app.ingestion.chunker"))
check("app.security.pii",         lambda: __import__("app.security.pii"))
check("app.security.prompt_guard",lambda: __import__("app.security.prompt_guard"))
check("app.retrieval.bm25",       lambda: __import__("app.retrieval.bm25"))
check("app.retrieval.vector_store",lambda: __import__("app.retrieval.vector_store"))
check("app.retrieval.retriever",  lambda: __import__("app.retrieval.retriever"))
check("app.retrieval.hybrid_retriever", lambda: __import__("app.retrieval.hybrid_retriever"))
check("app.retrieval.reranker",   lambda: __import__("app.retrieval.reranker"))
check("app.monitoring.cost_tracker", lambda: __import__("app.monitoring.cost_tracker"))
check("app.logging.logger",       lambda: __import__("app.logging.logger"))
check("app.graph.state",          lambda: __import__("app.graph.state"))
check("app.graph.workflow",       lambda: __import__("app.graph.workflow"))


print("\n── Loader extension coverage ─────────────────")
def check_extensions():
    from app.ingestion.loader import supported_extensions
    exts = supported_extensions()
    required = [".pdf", ".docx", ".pptx", ".ppt", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    missing  = [e for e in required if e not in exts]
    assert not missing, f"Missing extensions: {missing}"
    print(f"       Supported: {', '.join(exts)}")
check("All expected extensions registered", check_extensions)


print("\n─────────────────────────────────────────────")
print("If all ✅ above — ready to test document ingestion!\n")
