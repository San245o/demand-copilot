from __future__ import annotations

from io import BytesIO
from collections.abc import AsyncIterator

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.adapters.registry import (
    active_adapter_name,
    delete_dataset,
    get_adapter,
    get_active_profile,
    list_datasets,
    set_active_dataset,
)
from app.adapters.uploaded import save_uploaded_dataset
from app.agents.chat_agent import AVAILABLE_MODELS, chat_stream
from app.core.config import settings
from app.services.profiler import profile_dataset

app = FastAPI(title="Demand Forecasting Co-Pilot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    images: list[str] = Field(default_factory=list)  # base64 data URIs for vision


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "adapter": active_adapter_name(),
        "integrations": {
            "llm": settings.has_llm,
            "tavily": settings.has_tavily,
            "fred": settings.has_fred,
            "e2b": settings.has_e2b,
        },
    }


@app.get("/models")
def models() -> dict[str, object]:
    return {"models": AVAILABLE_MODELS, "default": AVAILABLE_MODELS[0]}


@app.get("/entities")
def entities() -> dict[str, object]:
    adapter = get_adapter()
    return {"adapter": adapter.name, "entities": adapter.list_entities()}


@app.get("/datasets")
def datasets() -> dict[str, object]:
    active = get_active_profile()
    return {"datasets": list_datasets(), "active_id": active.id}


@app.post("/datasets/upload")
async def upload_datasets(files: list[UploadFile] = File(...)) -> dict[str, object]:
    profiles = []
    for file in files:
        raw = await file.read()
        try:
            df = _read_upload(raw, file.filename or "upload.csv")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read {file.filename}: {exc}") from exc
        df.columns = [str(c).strip() for c in df.columns]
        profile = profile_dataset(df, file.filename or "upload.csv")
        save_uploaded_dataset(df, profile)
        profiles.append(profile)
    return {"profiles": profiles, "active_id": get_active_profile().id}


@app.post("/datasets/{dataset_id}/activate")
def activate_dataset(dataset_id: str) -> dict[str, object]:
    try:
        profile = set_active_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"active_id": profile.id, "profile": profile}


@app.delete("/datasets/{dataset_id}")
def remove_dataset(dataset_id: str) -> dict[str, object]:
    try:
        delete_dataset(dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "active_id": get_active_profile().id}


@app.post("/chat/stream")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """Stream a ReAct agent's reasoning, tool calls, and answer as SSE."""
    history = [m.model_dump() for m in req.history]

    async def gen() -> AsyncIterator[dict[str, str]]:
        async for event in chat_stream(req.message, history, req.model, req.images):
            yield event.to_sse()

    return EventSourceResponse(gen())


def _read_upload(raw: bytes, filename: str) -> pd.DataFrame:
    suffix = filename.lower().rsplit(".", 1)[-1]
    buf = BytesIO(raw)
    if suffix == "csv":
        return pd.read_csv(buf)
    if suffix == "xlsx":
        return pd.read_excel(buf)
    raise ValueError("Only .csv and .xlsx files are supported")
