from __future__ import annotations

from collections.abc import AsyncIterator

from langgraph.types import Command

from app.agents.graph import build_graph
from app.core.events import EventType, StreamEvent


async def run_crew_stream(
    entity_id: str, horizon: int, thread_id: str
) -> AsyncIterator[StreamEvent]:
    """Run the crew until it interrupts for approval, streaming each step.

    Stops after emitting INTERRUPT. The client then calls the resume endpoint, which
    drives `resume_crew_stream` with the human decision.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    yield StreamEvent(
        type=EventType.RUN_START,
        message=f"Forecasting {entity_id} for the next {horizon} days",
        data={"entity_id": entity_id, "horizon": horizon, "thread_id": thread_id},
    )

    inputs = {"entity_id": entity_id, "horizon": horizon}
    async for mode, chunk in graph.astream(
        inputs, config=config, stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            yield StreamEvent.model_validate(chunk)
        # 'updates' chunks carry node returns / interrupt markers; the custom
        # events already cover what the UI renders, so we don't re-emit them.

    # If we reach here without hitting planning, an interrupt is pending — the
    # INTERRUPT event was already emitted by approval_node. The client resumes next.


async def resume_crew_stream(
    thread_id: str, decision: dict
) -> AsyncIterator[StreamEvent]:
    """Resume an interrupted run with the human decision and stream to completion."""
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    async for mode, chunk in graph.astream(
        Command(resume=decision), config=config, stream_mode=["custom", "updates"]
    ):
        if mode == "custom":
            event = StreamEvent.model_validate(chunk)
            # approval_node re-runs from its start on resume and re-emits INTERRUPT;
            # the client already handled it, so suppress the duplicate.
            if event.type == EventType.INTERRUPT:
                continue
            yield event

    yield StreamEvent(type=EventType.RUN_END, message="Done")
