from __future__ import annotations

import warnings

import pytest

from app.agents.runner import resume_crew_stream, run_crew_stream
from app.core.events import EventType

warnings.simplefilter("ignore")


@pytest.mark.asyncio
async def test_crew_runs_to_interrupt_then_resumes():
    tid = "pytest-thread"
    # Phase 1: run until human-approval interrupt
    run_events = [e async for e in run_crew_stream("store_1", 7, tid)]
    run_types = [e.type for e in run_events]

    assert run_types[0] == EventType.RUN_START
    assert EventType.FORECAST in run_types
    assert run_types[-1] == EventType.INTERRUPT  # pauses for human

    agents = {e.agent for e in run_events if e.agent}
    assert {"sensing", "forecast", "validation", "approval"} <= agents

    # forecast payload is well-formed
    fc_event = next(e for e in run_events if e.type == EventType.FORECAST)
    assert fc_event.data["points"]
    assert fc_event.data["level"] == 95

    # Phase 2: resume with approval → planning + brief + end
    resume_events = [e async for e in resume_crew_stream(tid, {"action": "approve"})]
    resume_types = [e.type for e in resume_events]
    assert EventType.BRIEF in resume_types
    assert resume_types[-1] == EventType.RUN_END
    # no duplicate interrupt leaked on resume
    assert EventType.INTERRUPT not in resume_types


@pytest.mark.asyncio
async def test_reject_produces_no_brief_recommendation():
    tid = "pytest-thread-reject"
    _ = [e async for e in run_crew_stream("store_2", 5, tid)]
    resume = [
        e
        async for e in resume_crew_stream(
            tid, {"action": "reject", "feedback": "numbers look off"}
        )
    ]
    brief = next(e for e in resume if e.type == EventType.BRIEF)
    assert "reject" in brief.data["headline"].lower()
