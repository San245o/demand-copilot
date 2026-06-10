from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app.agents.mock_crew import run_mock_crew
from app.core.config import settings

app = FastAPI(title="Demand Forecasting Co-Pilot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    entity_id: str = "store_1"
    horizon: int = Field(default=7, ge=1, le=90)


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness + which integrations are live vs mocked."""
    return {
        "status": "ok",
        "integrations": {
            "llm": settings.has_llm,
            "tavily": settings.has_tavily,
            "fred": settings.has_fred,
        },
    }


@app.get("/forecast/stream")
async def forecast_stream(
    entity_id: str = "store_1", horizon: int = 7
) -> EventSourceResponse:
    """Stream the crew's reasoning steps as Server-Sent Events.

    GET (not POST) so it works with the browser EventSource API and simple curl.
    """
    horizon = max(1, min(horizon, 90))

    async def event_gen() -> AsyncIterator[dict[str, str]]:
        async for event in run_mock_crew(entity_id, horizon):
            yield event.to_sse()

    return EventSourceResponse(event_gen())
