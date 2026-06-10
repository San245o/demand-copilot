from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.adapters.registry import active_adapter_name, get_adapter
from app.agents.runner import resume_crew_stream, run_crew_stream
from app.core.config import settings

app = FastAPI(title="Demand Forecasting Co-Pilot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Decision(BaseModel):
    thread_id: str
    action: str = "approve"  # approve | reject | edit
    feedback: str | None = None


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


@app.get("/entities")
def entities() -> dict[str, object]:
    adapter = get_adapter()
    return {"adapter": adapter.name, "entities": adapter.list_entities()}


@app.get("/forecast/stream")
async def forecast_stream(
    entity_id: str = "store_1", horizon: int = 7
) -> EventSourceResponse:
    """Run the crew until the human-approval interrupt, streaming each step."""
    horizon = max(1, min(horizon, 90))
    thread_id = uuid.uuid4().hex

    async def gen() -> AsyncIterator[dict[str, str]]:
        async for event in run_crew_stream(entity_id, horizon, thread_id):
            yield event.to_sse()

    return EventSourceResponse(gen())


@app.post("/forecast/resume")
async def forecast_resume(decision: Decision) -> EventSourceResponse:
    """Resume an interrupted run with the human decision, streaming to completion."""
    payload = {"action": decision.action, "feedback": decision.feedback}

    async def gen() -> AsyncIterator[dict[str, str]]:
        async for event in resume_crew_stream(decision.thread_id, payload):
            yield event.to_sse()

    return EventSourceResponse(gen())
