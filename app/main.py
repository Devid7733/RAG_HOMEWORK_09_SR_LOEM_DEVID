"""Terminal chat interface for the local RAG application."""

import argparse

from app.pipeline import answer_question, build_index
from app.vector_store import collection_count


def _print_chunks(chunks: list[dict]) -> None:
    print("\nRetrieved chunks:")
    for index, chunk in enumerate(chunks, start=1):
        print(
            f"\n[{index}] {chunk['source']} #{chunk['chunk_index']} "
            f"(distance={chunk['distance']:.4f})\n{chunk['text']}"
        )


def run_chat(reindex: bool = False) -> None:
    if reindex or collection_count() == 0:
        print("Building the document index...")
        count = build_index()
        print(f"Indexed {count} chunks.")

    print("\nLocal document chat is ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not question:
            print("Please enter a question.")
            continue

        try:
            result = answer_question(question)
        except Exception as error:
            print(f"Error: {error}")
            continue
        _print_chunks(result["chunks"])
        print(f"\nAssistant: {result['answer']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the local documents.")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="rebuild the Chroma index before starting the chat",
    )
    args = parser.parse_args()
    run_chat(reindex=args.reindex)


if __name__ == "__main__":
    main()
