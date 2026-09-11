"""Stage 2: split documents into fixed-size, overlapping chunks."""

from typing import TypedDict

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.ingestion import Document


class Chunk(TypedDict):
    id: str
    text: str
    source: str
    chunk_index: int


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text by character count while preserving nearby context."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = end - overlap
    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Split loaded documents and attach stable provenance metadata."""
    chunks: list[Chunk] = []
    for document in documents:
        for index, text in enumerate(chunk_text(document["text"])):
            chunks.append(
                {
                    "id": f'{document["source"]}::{index}',
                    "text": text,
                    "source": document["source"],
                    "chunk_index": index,
                }
            )
    return chunks
