"""Stage 4: retrieve the chunks most relevant to a question."""

from typing import TypedDict, cast

from app.config import TOP_K
from app.embeddings import embed_query
from app.vector_store import search


class RetrievedChunk(TypedDict):
    text: str
    source: str
    chunk_index: int
    distance: float


def retrieve(question: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    if not question.strip():
        raise ValueError("question must not be empty")

    results = search(embed_query(question), top_k)
    documents = cast(list[list[str]], results.get("documents") or [[]])[0]
    metadatas = cast(list[list[dict]], results.get("metadatas") or [[]])[0]
    distances = cast(list[list[float]], results.get("distances") or [[]])[0]

    return [
        {
            "text": text,
            "source": str(metadata.get("source", "unknown")),
            "chunk_index": int(metadata.get("chunk_index", -1)),
            "distance": float(distance),
        }
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]
