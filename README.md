# Local Naive RAG: Chat with Documents

This project is an end-to-end local Retrieval-Augmented Generation (RAG)
prototype. It indexes three IT support guides, retrieves relevant
passages with semantic search, and uses a local Ollama model to answer only
from the retrieved context.

## Five-stage pipeline

1. **Ingest** — load `.txt`, `.md`, and `.pdf` files from `data/`.
2. **Chunk** — split each document into overlapping text passages.
3. **Embed and store** — create vectors with Ollama and persist them in ChromaDB.
4. **Retrieve** — embed a question and find the closest document chunks.
5. **Generate** — give the question and retrieved context to the local LLM.

The offline flow performs stages 1–3 when the index is built. The online flow
performs stages 4–5 for every question.

## Technology and model choices

- Python 3.11 or 3.12 with Poetry
- ChromaDB 1.5.9 in persistent local mode
- `nomic-embed-text` for embeddings
- `llama3.2` for answer generation

Both models run locally through Ollama, so document content does not need to be
sent to a hosted model.

## Chunking rationale

The app uses fixed-size chunks of 800 characters with 120 characters of
overlap. The documents are short step-by-step guides, so 800 characters usually
keep a complete troubleshooting step or closely related steps together. The overlap preserves context when a
sentence or section crosses a chunk boundary. This simple and transparent
strategy is appropriate for a Naive RAG baseline and gives clear parameters to
adjust in later experiments.

## Project structure

```text
app/
  ingestion.py      load source documents
  chunking.py       split documents and attach metadata
  embeddings.py     create vectors with Ollama
  vector_store.py   persist and query ChromaDB
  retriever.py      retrieve relevant chunks
  generator.py      create grounded answers
  pipeline.py       connect offline and online flows
  main.py           terminal chat loop
  api.py            optional FastAPI interface
data/                three source documents
demo_vector_check.py standalone top-three retrieval check
scripts/test_queries.py five-question test-log generator
TEST_LOG.md          recorded questions, chunks, and answers
REFLECTION.md        assignment reflection
```

## Setup

Install and start [Ollama](https://ollama.com), then pull the models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Install the Python dependencies from the project directory:

```bash
poetry install
```

## Run the chat app

The first run builds the index automatically if it is empty:

```bash
poetry run python main.py
```

To force a fresh index after changing files in `data/`:

```bash
poetry run python main.py --reindex
```

Type a question and press Enter. The app prints the retrieved chunks before
the answer. Type `exit` or `quit` to close the chat.

## Verify retrieval and generation

Run the standalone vector check:

```bash
poetry run python demo_vector_check.py
```

Run the five assignment questions and regenerate the Markdown test log:

```bash
poetry run python -m scripts.test_queries --reindex --output TEST_LOG.md
```

Run the fast unit tests, which do not require an LLM response:

```bash
poetry run python -m unittest discover -s tests -v
```

## Optional REST API

The required interface is the terminal chat loop. A FastAPI interface is also
included as an extra:

```bash
poetry run uvicorn app.api:app --reload
```

Use `GET /health`, `POST /ingest`, or `POST /chat`. Interactive API
documentation is available at `http://127.0.0.1:8000/docs`.
