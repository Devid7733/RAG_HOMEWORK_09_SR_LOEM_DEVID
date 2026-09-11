"""Connect the offline indexing flow and online question-answering flow."""

from pathlib import Path
from typing import TypedDict

from app.chunking import chunk_documents
from app.config import DATA_DIR, TOP_K
from app.embeddings import embed_texts
from app.generator import generate_answer
from app.ingestion import load_documents
from app.retriever import RetrievedChunk, retrieve
from app.vector_store import collection_count, replace_chunks


class RAGResult(TypedDict):
    question: str
    answer: str
    chunks: list[RetrievedChunk]


def build_index(data_dir: str | Path = DATA_DIR) -> int:
    documents = load_documents(data_dir)
    if not documents:
        raise FileNotFoundError(f"No .txt, .md, or .pdf files found in {data_dir}")
    chunks = chunk_documents(documents)
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    return replace_chunks(chunks, embeddings)


def answer_question(question: str, top_k: int = TOP_K) -> RAGResult:
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("question must not be empty")
    if collection_count() == 0:
        raise RuntimeError("The vector index is empty. Rebuild it before asking questions.")

    chunks = retrieve(cleaned, top_k=top_k)
    return {
        "question": cleaned,
        "answer": generate_answer(cleaned, chunks),
        "chunks": chunks,
    }
