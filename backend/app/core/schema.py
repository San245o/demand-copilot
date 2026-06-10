from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field


class DataPoint(BaseModel):
    """One observation in a series, normalized across all datasets.

    `target` is the quantity we forecast (units sold, revenue, …). `signals` holds
    optional, dataset-specific covariates (promo flag, weather, holiday, macro index).
    Signals are sparse and discovered — never assumed — so a bare time series and a
    rich one share the same shape.
    """

    date: Date
    target: float
    signals: dict[str, float] = Field(default_factory=dict)


class TimeSeries(BaseModel):
    """A canonical, dataset-agnostic series for a single forecastable entity."""

    entity_id: str  # e.g. "store_123" or "sku_456"
    entity_kind: str = "entity"  # "store" | "sku" | "store_sku" | …
    freq: str = "D"  # pandas offset alias: D, W, M
    target_name: str = "sales"
    unit: str | None = None
    points: list[DataPoint] = Field(default_factory=list)

    @property
    def signal_names(self) -> list[str]:
        """Union of signal keys present anywhere in the series (discovered, not assumed)."""
        names: set[str] = set()
        for p in self.points:
            names.update(p.signals.keys())
        return sorted(names)


class ForecastPoint(BaseModel):
    date: Date
    mean: float
    lower: float  # lower bound of the interval
    upper: float  # upper bound of the interval


class Forecast(BaseModel):
    entity_id: str
    horizon: int
    level: int = 95  # confidence level, percent
    model: str = "mock"
    points: list[ForecastPoint] = Field(default_factory=list)
