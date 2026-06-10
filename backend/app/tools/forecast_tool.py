from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

from app.core.schema import Forecast, ForecastPoint, TimeSeries

# The forecasting TOOL. The LLM never does this arithmetic — it decides to call this,
# and explains the result. statsforecast gives point forecasts + prediction intervals.


def run_forecast(
    series: TimeSeries, horizon: int, level: int = 95
) -> Forecast:
    """Fit a statistical model and return a point forecast + interval.

    Uses AutoETS (handles trend + weekly seasonality) with a SeasonalNaive fallback
    for very short series. Daily data → season_length=7.
    """
    if len(series.points) < 14:
        return _seasonal_naive_fallback(series, horizon, level)

    df = pd.DataFrame(
        {
            "unique_id": series.entity_id,
            "ds": [p.date for p in series.points],
            "y": [p.target for p in series.points],
        }
    )
    df["ds"] = pd.to_datetime(df["ds"])

    season = 7 if series.freq == "D" else 1
    sf = StatsForecast(
        models=[AutoETS(season_length=season)],
        freq="D" if series.freq == "D" else series.freq,
        n_jobs=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf.fit(df)
        fc = sf.predict(h=horizon, level=[level])

    model_col = "AutoETS"
    lo_col, hi_col = f"{model_col}-lo-{level}", f"{model_col}-hi-{level}"

    # Access columns by label (interval columns have hyphens, which itertuples mangles).
    points: list[ForecastPoint] = []
    for i in range(len(fc)):
        points.append(
            ForecastPoint(
                date=pd.Timestamp(fc["ds"].iloc[i]).date(),
                mean=max(0.0, float(fc[model_col].iloc[i])),
                lower=max(0.0, float(fc[lo_col].iloc[i])),
                upper=max(0.0, float(fc[hi_col].iloc[i])),
            )
        )

    return Forecast(
        entity_id=series.entity_id,
        horizon=horizon,
        level=level,
        model="AutoETS",
        points=points,
    )


def _seasonal_naive_fallback(
    series: TimeSeries, horizon: int, level: int
) -> Forecast:
    """For very short series: repeat last value with a wide heuristic interval."""
    last_date = series.points[-1].date if series.points else None
    last_val = series.points[-1].target if series.points else 0.0
    vals = np.array([p.target for p in series.points]) if series.points else np.array([0.0])
    spread = float(np.std(vals)) * 1.96 if len(vals) > 1 else last_val * 0.3

    points: list[ForecastPoint] = []
    for h in range(1, horizon + 1):
        d = (last_date + timedelta(days=h)) if last_date else None
        points.append(
            ForecastPoint(
                date=d,
                mean=max(0.0, last_val),
                lower=max(0.0, last_val - spread),
                upper=max(0.0, last_val + spread),
            )
        )
    return Forecast(
        entity_id=series.entity_id,
        horizon=horizon,
        level=level,
        model="SeasonalNaive(fallback)",
        points=points,
    )
