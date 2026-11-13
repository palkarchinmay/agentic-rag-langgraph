"""
chunker.py — Document chunking with multiple strategies.

Methods:
  recursive : LangChain RecursiveCharacterTextSplitter (default)
  structure : splits on [TEXT]/[TABLES]/[OCR] section boundaries first
  semantic  : embedding-based semantic chunking (requires embedding model)
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("financial_rag.chunker")


def chunk_documents(
    docs,
    chunk_size:    int = 500,
    chunk_overlap: int = 50,
    method:        str = "recursive",
    embedding_model=None,
) -> List[Dict]:
    """
    Chunk a list of Document objects.

    Args:
        docs           : list of Document objects
        chunk_size     : max chars per chunk
        chunk_overlap  : overlap between chunks
        method         : 'recursive' | 'structure' | 'semantic'
        embedding_model: required for method='semantic'

    Returns:
        list of {content: str, metadata: dict}
    """
    if not docs:
        return []

    if method == "recursive":
        return _recursive_chunk(docs, chunk_size, chunk_overlap)
    elif method == "structure":
        return _structure_chunk(docs, chunk_size, chunk_overlap)
    elif method == "semantic":
        return _semantic_chunk(docs, embedding_model)
    else:
        logger.warning("Unknown method '%s', defaulting to recursive", method)
        return _recursive_chunk(docs, chunk_size, chunk_overlap)


def _recursive_chunk(docs, chunk_size, chunk_overlap):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc.content)
        for split in splits:
            if split.strip():
                chunks.append({"content": split, "metadata": dict(doc.metadata)})
    return chunks


def _structure_chunk(docs, chunk_size, chunk_overlap):
    """Split on section tags first, then apply recursive within each section."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter  = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    tag_pattern = re.compile(r'\[(TEXT|TABLES|OCR|CHARTS|NOTES)\]')
    chunks = []

    for doc in docs:
        sections = tag_pattern.split(doc.content)
        current_tag = "TEXT"
        for part in sections:
            if tag_pattern.match(f"[{part}]"):
                current_tag = part
                continue
            if not part.strip():
                continue
            splits = splitter.split_text(part)
            for split in splits:
                if split.strip():
                    meta = {**doc.metadata, "section_type": current_tag}
                    chunks.append({"content": split, "metadata": meta})
    return chunks


def _semantic_chunk(docs, embedding_model):
    """Semantic chunking using embedding similarity."""
    if embedding_model is None:
        logger.warning("No embedding model for semantic chunking, falling back to recursive")
        return _recursive_chunk(docs, 500, 50)

    try:
        from langchain_experimental.text_splitter import SemanticChunker
        splitter = SemanticChunker(embeddings=embedding_model)
        chunks   = []
        for doc in docs:
            splits = splitter.split_text(doc.content)
            for split in splits:
                if split.strip():
                    chunks.append({"content": split, "metadata": dict(doc.metadata)})
        return chunks
    except ImportError:
        logger.warning("langchain-experimental not installed, falling back to recursive")
        return _recursive_chunk(docs, 500, 50)
