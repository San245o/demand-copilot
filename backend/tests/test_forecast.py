from __future__ import annotations

import warnings

from app.adapters.synthetic import SyntheticAdapter
from app.tools.forecast_tool import run_forecast

warnings.simplefilter("ignore")


def test_forecast_shape_and_intervals():
    ts = SyntheticAdapter().load("store_1", lookback=120)
    fc = run_forecast(ts, horizon=7, level=95)
    assert fc.horizon == 7
    assert len(fc.points) == 7
    for p in fc.points:
        assert p.lower <= p.mean <= p.upper  # interval contains the mean
        assert p.lower >= 0  # sales can't be negative


def test_short_series_uses_fallback():
    ts = SyntheticAdapter().load("store_2", lookback=5)
    fc = run_forecast(ts, horizon=3, level=95)
    assert "fallback" in fc.model.lower()
    assert len(fc.points) == 3


def test_synthetic_is_deterministic():
    a = SyntheticAdapter().load("store_3", lookback=30)
    b = SyntheticAdapter().load("store_3", lookback=30)
    assert [p.target for p in a.points] == [p.target for p in b.points]
