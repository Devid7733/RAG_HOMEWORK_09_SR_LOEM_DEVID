"""Configuration shared by every stage of the local RAG pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "llama3.2"

DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4

SYSTEM_PROMPT = (
    "You answer questions using ONLY the supplied document context. Answer "
    "directly and concisely. For a supported answer, end with a new line in "
    "the exact format 'Source: filename.txt' using the supplied source filename. "
    "Never use numeric citations or add claims that are not in the context. "
    "If the answer is not contained in the context, "
    "say \"I don't have enough information in the documents to answer that.\" "
    "Do not use outside knowledge."
)
