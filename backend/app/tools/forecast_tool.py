from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive

from app.core.schema import Forecast, ForecastPoint, TimeSeries

# The forecasting TOOL. The LLM never does this arithmetic — it decides to call this,
# and explains the result. statsforecast gives point forecasts + prediction intervals.


def run_forecast(
    series: TimeSeries,
    horizon: int,
    level: int = 95,
    model: str = "auto",
    season_length: int | None = None,
) -> Forecast:
    """Fit a statistical model and return a point forecast + interval.

    Supports auto/ets/arima/theta/naive, with a SeasonalNaive fallback for very
    short series. Default seasonality: D→7, W→52, M→12.
    """
    level = max(50, min(int(level), 99))
    model_key = model.lower().strip()
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

    season = season_length or _default_season_length(series.freq)
    model_obj, model_col, model_label = _model(model_key, season)
    sf = StatsForecast(
        models=[model_obj],
        freq="D" if series.freq == "D" else series.freq,
        n_jobs=1,
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sf.fit(df)
            fc = sf.predict(h=horizon, level=[level])
    except Exception:
        return _seasonal_naive_fallback(series, horizon, level)

    lo_col, hi_col = f"{model_col}-lo-{level}", f"{model_col}-hi-{level}"

    # Access columns by label (interval columns have hyphens, which itertuples mangles).
    points: list[ForecastPoint] = []
    for i in range(len(fc)):
        points.append(
            ForecastPoint(
                date=pd.Timestamp(fc["ds"].iloc[i]).date(),
                mean=max(0.0, float(fc[model_col].iloc[i])),
                lower=max(0.0, float(fc[lo_col].iloc[i])) if lo_col in fc else 0.0,
                upper=max(0.0, float(fc[hi_col].iloc[i])) if hi_col in fc else max(0.0, float(fc[model_col].iloc[i])),
            )
        )

    return Forecast(
        entity_id=series.entity_id,
        horizon=horizon,
        level=level,
        model=model_label,
        points=points,
    )


def _default_season_length(freq: str) -> int:
    return {"D": 7, "W": 52, "M": 12}.get(freq, 1)


def _model(model: str, season: int):
    if model == "arima":
        return AutoARIMA(season_length=season), "AutoARIMA", "AutoARIMA"
    if model == "theta":
        return AutoTheta(season_length=season), "AutoTheta", "AutoTheta"
    if model == "naive":
        return SeasonalNaive(season_length=season), "SeasonalNaive", "SeasonalNaive"
    return AutoETS(season_length=season), "AutoETS", "AutoETS"


def _seasonal_naive_fallback(
    series: TimeSeries, horizon: int, level: int
) -> Forecast:
    """For very short series: repeat last value with a wide heuristic interval."""
    last_date = series.points[-1].date if series.points else None
    last_val = series.points[-1].target if series.points else 0.0
    vals = np.array([p.target for p in series.points]) if series.points else np.array([0.0])
    spread = float(np.std(vals)) * 1.96 if len(vals) > 1 else last_val * 0.3
    # Cap the band so a tiny sample doesn't produce an interval that swallows the
    # whole chart. It's a rough baseline; keep it readable rather than dramatic.
    spread = min(spread, max(abs(last_val) * 0.35, 1.0))

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
