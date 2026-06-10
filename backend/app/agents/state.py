from __future__ import annotations

from typing import Any, TypedDict


class CrewState(TypedDict, total=False):
    """Shared state threaded through the LangGraph crew.

    Kept JSON-serializable (plain dicts/lists) so the checkpointer can persist it
    across the human-in-the-loop interrupt.
    """

    entity_id: str
    horizon: int

    # produced by sensing
    series: dict[str, Any]  # serialized TimeSeries
    context: dict[str, Any]  # weather/holidays/macro/search enrichments

    # produced by forecast
    forecast: dict[str, Any]  # serialized Forecast

    # produced by validation
    validation: dict[str, Any]

    # human decision (filled on resume)
    decision: dict[str, Any]

    # produced by planning
    brief: dict[str, Any]
