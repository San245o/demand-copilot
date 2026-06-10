from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.core.events import EventType, StreamEvent

# P0 stand-in for the LangGraph crew. It emits the SAME event vocabulary the real
# crew will emit in P1, so the frontend is built against the final contract now.
# Replaced node-by-node in P1; the event shapes do not change.

_STEP_DELAY = 0.5  # seconds between steps, so streaming is visible in the UI


async def run_mock_crew(
    entity_id: str, horizon: int
) -> AsyncIterator[StreamEvent]:
    yield StreamEvent(
        type=EventType.RUN_START,
        message=f"Forecasting {entity_id} for the next {horizon} days",
        data={"entity_id": entity_id, "horizon": horizon},
    )

    # 1. Sensing
    yield StreamEvent(type=EventType.AGENT_START, agent="sensing",
                      message="Gathering demand signals")
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(type=EventType.THOUGHT, agent="sensing",
                      message="I need recent sales history plus any context signals.")
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(type=EventType.TOOL_CALL, agent="sensing",
                      message="load_series(entity_id, lookback=180d)",
                      data={"tool": "load_series"})
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(type=EventType.TOOL_RESULT, agent="sensing",
                      message="180 daily observations; signals: promo, holiday",
                      data={"n_points": 180, "signals": ["promo", "holiday"]})
    yield StreamEvent(type=EventType.AGENT_END, agent="sensing")

    # 2. Forecast (the tool does the math)
    yield StreamEvent(type=EventType.AGENT_START, agent="forecast",
                      message="Running the statistical forecast model")
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(type=EventType.THOUGHT, agent="forecast",
                      message="Series is daily with weekly seasonality. Calling AutoETS.")
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(type=EventType.TOOL_CALL, agent="forecast",
                      message=f"forecast(model=AutoETS, horizon={horizon}, level=95)",
                      data={"tool": "forecast", "model": "AutoETS"})
    await asyncio.sleep(_STEP_DELAY)
    mock_points = [
        {"date": f"2015-08-{d:02d}", "mean": 5400 + d * 20,
         "lower": 4800 + d * 18, "upper": 6000 + d * 22}
        for d in range(1, min(horizon, 7) + 1)
    ]
    yield StreamEvent(type=EventType.TOOL_RESULT, agent="forecast",
                      message="Forecast produced with 95% confidence interval")
    yield StreamEvent(type=EventType.FORECAST, agent="forecast",
                      message="Point forecast + interval",
                      data={"model": "AutoETS", "level": 95, "points": mock_points})
    yield StreamEvent(type=EventType.AGENT_END, agent="forecast")

    # 3. Validation
    yield StreamEvent(type=EventType.AGENT_START, agent="validation",
                      message="Checking the forecast against business rules")
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(type=EventType.THOUGHT, agent="validation",
                      message="CI width ~22% of mean — within tolerance. No negative values.")
    yield StreamEvent(type=EventType.AGENT_END, agent="validation",
                      message="Passed: forecast is plausible",
                      data={"ci_width_pct": 22, "passed": True})

    # 4. Planning + narrative
    yield StreamEvent(type=EventType.AGENT_START, agent="planning",
                      message="Drafting inventory recommendation and brief")
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(type=EventType.THOUGHT, agent="planning",
                      message="Reorder point should cover the upper CI to avoid stockouts.")
    await asyncio.sleep(_STEP_DELAY)
    yield StreamEvent(
        type=EventType.BRIEF, agent="planning",
        message="Forecast brief ready",
        data={
            "headline": f"{entity_id}: demand trending up ~+8% next week",
            "recommendation": "Raise reorder point to 6,000 units to cover the upper CI.",
            "confidence": "medium",
            "drivers": ["weekly seasonality", "active promo", "no holidays in window"],
        },
    )
    yield StreamEvent(type=EventType.AGENT_END, agent="planning")

    yield StreamEvent(type=EventType.RUN_END, message="Done")
