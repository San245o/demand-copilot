from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.core.schema import DataPoint, TimeSeries

# Rossmann Store Sales adapter. Maps the dataset's columns into canonical signals.
# This is the only place Rossmann-specific column names appear.

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


@lru_cache(maxsize=1)
def _load_train() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_DIR / "train.csv",
        dtype={"StateHoliday": str},
        parse_dates=["Date"],
    )
    return df


class RossmannAdapter:
    name = "rossmann"

    def __init__(self) -> None:
        self._available = (DATA_DIR / "train.csv").exists()

    @property
    def available(self) -> bool:
        return self._available

    def list_entities(self) -> list[str]:
        if not self._available:
            return []
        df = _load_train()
        stores = sorted(df["Store"].unique().tolist())
        return [f"store_{s}" for s in stores[:50]]  # cap for a snappy demo

    def load(self, entity_id: str, lookback: int | None = None) -> TimeSeries:
        store_num = int(entity_id.replace("store_", ""))
        df = _load_train()
        sub = df[df["Store"] == store_num].sort_values("Date")
        # Only model open days with positive sales; closed days are structural zeros.
        sub = sub[(sub["Open"] == 1) & (sub["Sales"] > 0)]
        if lookback:
            sub = sub.tail(lookback)

        points: list[DataPoint] = []
        for row in sub.itertuples(index=False):
            signals: dict[str, float] = {
                "promo": float(row.Promo),
                "school_holiday": float(row.SchoolHoliday),
            }
            # StateHoliday is "0" (none) or a/b/c (public/Easter/Christmas).
            if str(row.StateHoliday) not in ("0", "0.0", ""):
                signals["state_holiday"] = 1.0
            points.append(
                DataPoint(
                    date=row.Date.date(),
                    target=float(row.Sales),
                    signals=signals,
                )
            )

        return TimeSeries(
            entity_id=entity_id,
            entity_kind="store",
            freq="D",
            target_name="sales",
            unit="euros",
            points=points,
        )
