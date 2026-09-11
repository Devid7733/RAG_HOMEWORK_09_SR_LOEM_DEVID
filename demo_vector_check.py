"""Standalone verification that vector search returns useful chunks."""

from app.pipeline import build_index
from app.retriever import retrieve
QUESTION = "What is Step 1 for resetting a jammed printer?"


def main() -> None:
    print(f"Indexed {build_index()} chunks before running the check.\n")

    print(f"Question: {QUESTION}\n")
    for index, chunk in enumerate(retrieve(QUESTION, top_k=3), start=1):
        print(
            f"Result {index}: {chunk['source']} #{chunk['chunk_index']} "
            f"(distance={chunk['distance']:.4f})\n{chunk['text']}\n"
        )


if __name__ == "__main__":
    main()
