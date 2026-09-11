"""Persist chunk vectors and search them with ChromaDB."""

from collections.abc import Sequence
from typing import Any

import chromadb

from app.chunking import Chunk
from app.config import CHROMA_DB_DIR, COLLECTION_NAME


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def get_collection():
    return get_client().get_or_create_collection(name=COLLECTION_NAME)


def collection_count() -> int:
    return get_collection().count()


def replace_chunks(chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
    """Replace the entire collection with the supplied chunks and vectors."""
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have exactly one embedding")

    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Chroma versions use different not-found exception classes.
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if chunks:
        collection.add(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=[
                {"source": chunk["source"], "chunk_index": chunk["chunk_index"]}
                for chunk in chunks
            ],
        )
    return len(chunks)


def search(query_embedding: Sequence[float], top_k: int) -> dict[str, Any]:
    """Return documents, metadata, and distances for one query vector."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    return get_collection().query(
        query_embeddings=[list(query_embedding)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
