"""Optional FastAPI interface backed by the same pipeline as the CLI."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import TOP_K
from app.pipeline import answer_question, build_index

app = FastAPI(title="Baseline Chat-with-Documents API")


class ChatRequest(BaseModel):
    question: str
    top_k: int = Field(default=TOP_K, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest() -> dict[str, int]:
    try:
        return {"chunks_indexed": build_index()}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = answer_question(request.question, request.top_k)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    sources = sorted({chunk["source"] for chunk in result["chunks"]})
    return ChatResponse(answer=result["answer"], sources=sources)
