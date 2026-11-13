"""logger.py â€” Structured logging setup."""
import os
import logging
from pathlib import Path

LOG_DIR   = os.getenv("LOG_DIR", "/tmp/logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

Path(LOG_DIR).mkdir(exist_ok=True)


def setup_logging():
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{LOG_DIR}/rag_app.log", encoding="utf-8"),
        ],
    )

setup_logging()

