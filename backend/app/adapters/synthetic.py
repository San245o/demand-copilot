from __future__ import annotations

import math
from datetime import date, timedelta

from app.core.schema import DataPoint, TimeSeries

# Universal fallback adapter: deterministic synthetic sales with weekly seasonality,
# trend, promo spikes, and holiday dips. Used when no real dataset is present so the
# whole pipeline always runs. Deterministic (seeded by entity_id) — no RNG, so tests
# and demos are reproducible.


class SyntheticAdapter:
    name = "synthetic"

    def __init__(self, n_days: int = 365):
        self.n_days = n_days

    def list_entities(self) -> list[str]:
        return [f"store_{i}" for i in range(1, 6)]

    def load(self, entity_id: str, lookback: int | None = None) -> TimeSeries:
        seed = sum(ord(c) for c in entity_id)
        base = 3000 + (seed % 7) * 400
        start = date(2015, 1, 1)
        n = lookback or self.n_days

        points: list[DataPoint] = []
        for i in range(n):
            d = start + timedelta(days=i)
            weekly = 1.0 + 0.25 * math.sin(2 * math.pi * (d.weekday()) / 7.0)
            trend = 1.0 + 0.0003 * i
            # deterministic "promo" every ~14 days offset by seed
            promo = 1.0 if (i + seed) % 14 < 2 else 0.0
            # deterministic "holiday" dips
            holiday = 1.0 if d.month == 12 and 24 <= d.day <= 26 else 0.0
            value = base * weekly * trend
            value *= 1.30 if promo else 1.0
            value *= 0.40 if holiday else 1.0

            signals: dict[str, float] = {"promo": promo}
            if holiday:
                signals["holiday"] = 1.0
            points.append(
                DataPoint(date=d, target=round(value, 1), signals=signals)
            )

        return TimeSeries(
            entity_id=entity_id,
            entity_kind="store",
            freq="D",
            target_name="sales",
            unit="units",
            points=points,
        )
