"""Stage 3: turn document text and questions into vectors with Ollama."""

from collections.abc import Sequence

import ollama

from app.config import EMBED_MODEL


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of strings, preserving their order."""
    if not texts:
        return []
    response = ollama.embed(model=EMBED_MODEL, input=list(texts))
    return [list(embedding) for embedding in response.embeddings]


def embed_query(text: str) -> list[float]:
    """Embed one user question."""
    return embed_texts([text])[0]
