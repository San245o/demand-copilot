from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.adapters.registry import active_adapter_name, get_adapter
from app.agents.chat_agent import AVAILABLE_MODELS, chat_stream
from app.core.config import settings

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


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "adapter": active_adapter_name(),
        "integrations": {
            "llm": settings.has_llm,
            "tavily": settings.has_tavily,
            "fred": settings.has_fred,
        },
    }


@app.get("/models")
def models() -> dict[str, object]:
    return {"models": AVAILABLE_MODELS, "default": AVAILABLE_MODELS[0]}


@app.get("/entities")
def entities() -> dict[str, object]:
    adapter = get_adapter()
    return {"adapter": adapter.name, "entities": adapter.list_entities()}


@app.post("/chat/stream")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """Stream a ReAct agent's reasoning, tool calls, and answer as SSE."""
    history = [m.model_dump() for m in req.history]

    async def gen() -> AsyncIterator[dict[str, str]]:
        async for event in chat_stream(req.message, history, req.model):
            yield event.to_sse()

    return EventSourceResponse(gen())
