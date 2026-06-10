from __future__ import annotations

import warnings

import pytest
from fastapi.testclient import TestClient

from app.core.events import EventType
from app.core.schema import DataPoint, TimeSeries
from app.main import app

warnings.simplefilter("ignore")
client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "integrations" in body
    assert "adapter" in body


def test_signal_discovery_is_not_hardcoded():
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
        entity_id="sku_9", points=[DataPoint(date="2015-01-01", target=5.0)]
    )
    assert ts.signal_names == []
