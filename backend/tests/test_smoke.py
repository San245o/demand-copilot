from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.events import EventType
from app.core.schema import DataPoint, TimeSeries
from app.main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "integrations" in body


def test_signal_discovery_is_not_hardcoded():
    """Signals are discovered from the data, never assumed."""
    ts = TimeSeries(
        entity_id="store_1",
        points=[
            DataPoint(date="2015-01-01", target=100.0, signals={"promo": 1.0}),
            DataPoint(date="2015-01-02", target=120.0, signals={"holiday": 1.0}),
        ],
    )
    assert ts.signal_names == ["holiday", "promo"]


def test_bare_series_has_no_signals():
    ts = TimeSeries(
        entity_id="sku_9",
        points=[DataPoint(date="2015-01-01", target=5.0)],
    )
    assert ts.signal_names == []


@pytest.mark.asyncio
async def test_mock_crew_emits_full_lifecycle():
    """The mock crew must emit a coherent run: start → forecast → brief → end."""
    from app.agents.mock_crew import run_mock_crew

    types = [e.type async for e in run_mock_crew("store_1", 7)]
    assert types[0] == EventType.RUN_START
    assert types[-1] == EventType.RUN_END
    assert EventType.FORECAST in types
    assert EventType.BRIEF in types
    # all four agents must appear
    agents = {
        e.agent
        async for e in run_mock_crew("store_1", 7)
        if e.agent is not None
    }
    assert {"sensing", "forecast", "validation", "planning"} <= agents
