from io import BytesIO
from typing import List, Dict

import numpy as np
from pypdf import PdfReader
from docx import Document

from ollama_client import embed_texts


def extract_text_from_file(filename: str, data: bytes) -> str:
    """Extract text from PDF, DOCX or TXT."""
    suffix = filename.lower().rsplit(".", 1)[-1]

    if suffix == "pdf":
        reader = PdfReader(BytesIO(data))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()

    if suffix == "docx":
        document = Document(BytesIO(data))
        paragraphs = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs).strip()

    if suffix == "txt":
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError(
        f"Unsupported file type: .{suffix}. "
        "Use PDF, TXT or DOCX."
    )


def normalize_whitespace(text: str) -> str:
    """Keep text readable before chunking."""
    lines = [
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    ]
    return "\n".join(lines)


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[str]:
    """
    Simple character-based chunker for teaching.

    A production system may use token-aware or structure-aware chunking.
    This version is intentionally easy to understand.
    """
    text = normalize_whitespace(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def build_embeddings(texts: List[str]) -> np.ndarray:
    """Embed all chunks and return a NumPy matrix."""
    embeddings = embed_texts(texts)
    matrix = np.array(embeddings, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {matrix.shape}"
        )

    return normalize_rows(matrix)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each vector for cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def retrieve(
    query: str,
    chunks: List[Dict],
    vectors: np.ndarray,
    top_k: int = 3,
) -> List[Dict]:
    """Return the most similar chunks for a query."""
    query_vector = np.array(
        embed_texts([query])[0],
        dtype=np.float32,
    )

    query_vector = query_vector / max(
        float(np.linalg.norm(query_vector)),
        1e-12,
    )

    scores = vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for index in top_indices:
        item = dict(chunks[int(index)])
        item["score"] = float(scores[int(index)])
        results.append(item)

    return results
