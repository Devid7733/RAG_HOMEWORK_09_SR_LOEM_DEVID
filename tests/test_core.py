"""Fast tests for the stages that do not require Ollama."""

import tempfile
import unittest
from pathlib import Path

from app.chunking import chunk_documents, chunk_text
from app.generator import NO_ANSWER, build_prompt, normalize_answer
from app.ingestion import load_documents
from app.pipeline import answer_question


class IngestionTests(unittest.TestCase):
    def test_loads_supported_files_and_skips_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            (directory / "note.txt").write_text("useful text", encoding="utf-8")
            (directory / "empty.md").write_text("   ", encoding="utf-8")
            (directory / "ignored.csv").write_text("ignored", encoding="utf-8")

            self.assertEqual(
                load_documents(directory),
                [{"source": "note.txt", "text": "useful text"}],
            )


class ChunkingTests(unittest.TestCase):
    def test_chunk_text_uses_overlap(self) -> None:
        self.assertEqual(chunk_text("abcdefghij", 6, 2), ["abcdef", "efghij"])

    def test_chunk_documents_keeps_provenance(self) -> None:
        chunks = chunk_documents([{"source": "note.txt", "text": "hello"}])
        self.assertEqual(chunks[0]["id"], "note.txt::0")
        self.assertEqual(chunks[0]["source"], "note.txt")


class GenerationTests(unittest.TestCase):
    def test_prompt_contains_question_context_and_source(self) -> None:
        chunk = {
            "text": "The stipend is $400.",
            "source": "policy.txt",
            "chunk_index": 0,
            "distance": 0.2,
        }
        prompt = build_prompt("What is the stipend?", [chunk])
        self.assertIn("policy.txt", prompt)
        self.assertIn("The stipend is $400.", prompt)
        self.assertIn("What is the stipend?", prompt)

    def test_empty_question_is_rejected_before_external_calls(self) -> None:
        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            answer_question("   ")

    def test_no_answer_removes_an_irrelevant_model_citation(self) -> None:
        self.assertEqual(
            normalize_answer(f"{NO_ANSWER}\n\nSource: unrelated.txt"),
            NO_ANSWER,
        )


if __name__ == "__main__":
    unittest.main()
