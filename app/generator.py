"""Stage 5: generate a grounded answer from retrieved context."""

import ollama

from app.config import GEN_MODEL, SYSTEM_PROMPT
from app.retriever import RetrievedChunk

NO_ANSWER = "I don't have enough information in the documents to answer that."


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if chunks:
        context = "\n\n".join(
            f"Passage {index}\nSource filename: {chunk['source']}\n{chunk['text']}"
            for index, chunk in enumerate(chunks, start=1)
        )
    else:
        context = "(no relevant context was found)"

    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer only from the context. If it does not contain the answer, "
        "say you do not have enough information in the documents. For a "
        "supported answer, finish with the exact source filename."
    )


def normalize_answer(answer: str) -> str:
    """Keep the grounded refusal exact even if the model adds a citation."""
    cleaned = answer.strip()
    if "don't have enough information in the documents" in cleaned.lower():
        return NO_ANSWER
    return cleaned


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, chunks)},
        ],
        options={"temperature": 0},
    )
    return normalize_answer(response.message.content or "")
