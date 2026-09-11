"""Stage 1: load supported source documents from disk."""

from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

from app.config import DATA_DIR


class Document(TypedDict):
    source: str
    text: str


def _read_pdf(path: Path) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(data_dir: str | Path = DATA_DIR) -> list[Document]:
    """Load non-empty .txt, .md, and .pdf files in deterministic order."""
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        elif path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            continue
        if text.strip():
            documents.append({"source": path.name, "text": text})
    return documents
