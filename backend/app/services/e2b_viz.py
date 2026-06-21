from __future__ import annotations

import json
import os
import textwrap
from io import BytesIO
from typing import Literal

from pydantic import BaseModel

from app.core.config import settings
from app.core.schema import Forecast, TimeSeries


ChartType = Literal["history", "forecast_history", "signals"]


class Visualization(BaseModel):
    title: str
    image: str
    mime: str = "image/png"
    source: str = "e2b"


def create_visualization(
    series: TimeSeries,
    chart_type: str = "forecast_history",
    forecast: Forecast | None = None,
) -> Visualization:
    """Generate a matplotlib PNG in E2B, falling back locally if E2B is unreachable."""
    rows = [
        {
            "date": p.date.isoformat(),
            "target": p.target,
            **{f"signal_{k}": v for k, v in p.signals.items()},
        }
        for p in series.points[-2000:]
    ]
    forecast_rows = [p.model_dump(mode="json") for p in forecast.points] if forecast else []
    safe_chart = chart_type if chart_type in {"history", "forecast_history", "signals"} else "forecast_history"
    title = _title(series, safe_chart)

    if not settings.has_e2b:
        return _local_visualization(rows, forecast_rows, title, series.target_name, safe_chart, "local: E2B_API_KEY not set")

    code = _matplotlib_code(rows, forecast_rows, title, series.target_name, safe_chart)

    try:
        from e2b_code_interpreter import Sandbox
    except Exception as exc:
        return _local_visualization(rows, forecast_rows, title, series.target_name, safe_chart, f"local: e2b package unavailable ({exc})")

    try:
        os.environ["E2B_API_KEY"] = settings.e2b_api_key or ""
        with Sandbox.create() as sandbox:
            execution = sandbox.run_code(code)
        image_b64 = _execution_text(execution).strip()
        if not image_b64:
            raise RuntimeError("E2B did not return a chart image.")
        return Visualization(title=title, image=f"data:image/png;base64,{image_b64}", source="e2b")
    except Exception as exc:
        return _local_visualization(rows, forecast_rows, title, series.target_name, safe_chart, f"local fallback: E2B unreachable ({exc})")


def render_forecast_chart(
    series: TimeSeries,
    forecast: Forecast,
    model_label: str | None = None,
) -> Visualization:
    """Render a clear, legible history+forecast chart for a `forecast_demand` call.

    Renders locally (no E2B round-trip). To keep the prediction readable, only the
    recent history leading into the forecast is shown (so the forecast isn't a tiny
    sliver against months of history). The method is stated in plain words as a
    subtitle, so a flat short-history baseline is self-explanatory.
    """
    horizon = max(1, len(forecast.points))
    window = max(horizon * 3, 30)  # zoom: ~3x the forecast length of recent history
    rows = [
        {
            "date": p.date.isoformat(),
            "target": p.target,
            **{f"signal_{k}": v for k, v in p.signals.items()},
        }
        for p in series.points[-window:]
    ]
    forecast_rows = [p.model_dump(mode="json") for p in forecast.points]
    title = f"Forecast: {_pretty_entity(series.entity_id)} · next {horizon} days"
    subtitle = _friendly_method(model_label or forecast.model)
    return _local_visualization(
        rows, forecast_rows, title, series.target_name, "forecast_history", "local",
        subtitle=subtitle,
    )


def _pretty_entity(entity_id: str) -> str:
    """Turn raw entity ids into something readable: 'store_1' -> 'Store 1',
    '1000 | 3 | 9' -> '1000 · 3 · 9'."""
    parts = [p.strip() for p in str(entity_id).split("|")]
    pretty = [p.replace("_", " ").title() if "_" in p else p for p in parts]
    return " · ".join(pretty)


def _friendly_method(model_label: str) -> str:
    """Plain-English description of the forecasting method (no model jargon)."""
    m = (model_label or "").lower()
    if "fallback" in m:
        return "Limited history — rough baseline estimate"
    if "ets" in m:
        return "ETS — captures trend & weekly seasonality"
    if "arima" in m:
        return "ARIMA — captures autocorrelation"
    if "theta" in m:
        return "Theta — trend-friendly forecast"
    if "naive" in m or "seasonal" in m:
        return "Seasonal baseline"
    return model_label or "forecast"


def _title(series: TimeSeries, chart_type: str) -> str:
    entity = _pretty_entity(series.entity_id)
    if chart_type == "history":
        return f"{series.target_name.title()} history · {entity}"
    if chart_type == "signals":
        return f"{series.target_name.title()} vs signals · {entity}"
    return f"Forecast · {entity}"


def _execution_text(execution) -> str:
    text = getattr(execution, "text", None)
    if isinstance(text, str):
        return text
    logs = getattr(execution, "logs", None)
    stdout = getattr(logs, "stdout", None)
    if isinstance(stdout, list):
        return "".join(str(x) for x in stdout)
    if isinstance(stdout, str):
        return stdout
    return ""


def _local_visualization(
    rows: list[dict],
    forecast_rows: list[dict],
    title: str,
    target_name: str,
    chart_type: str,
    source: str,
    subtitle: str | None = None,
) -> Visualization:
    import base64

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    fig = _plot(rows, forecast_rows, title, target_name, chart_type, plt, pd, subtitle)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return Visualization(title=title, image=f"data:image/png;base64,{image_b64}", source=source)


def _plot(rows, forecast_rows, title, target_name, chart_type, plt, pd, subtitle=None):
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    if chart_type == "signals":
        signal_cols = [c for c in df.columns if c.startswith("signal_")]
        ax.plot(df["date"], df["target"], color="#2563eb", linewidth=2.0, label=target_name)
        if signal_cols:
            ax2 = ax.twinx()
            for col in signal_cols[:4]:
                ax2.plot(df["date"], df[col], linewidth=1.2, alpha=0.65, label=col.replace("signal_", ""))
            ax2.set_ylabel("Signals")
            ax2.legend(loc="upper right", frameon=False, fontsize=8)
    elif chart_type == "forecast_history" and forecast_rows:
        # Past: a calm, thin line. Future: bold line + markers over a shaded band,
        # separated by a dashed "now" divider so it's obvious what's predicted.
        fc = pd.DataFrame(forecast_rows)
        fc["date"] = pd.to_datetime(fc["date"])
        split = fc["date"].iloc[0]

        ax.axvspan(split, fc["date"].iloc[-1], color="#eff6ff", zorder=0)
        ax.plot(df["date"], df["target"], color="#94a3b8", linewidth=1.6, label="history")

        # Bridge the last actual point into the forecast so the line is continuous.
        if len(df):
            bridge_x = [df["date"].iloc[-1], fc["date"].iloc[0]]
            bridge_y = [df["target"].iloc[-1], fc["mean"].iloc[0]]
            ax.plot(bridge_x, bridge_y, color="#2563eb", linewidth=2.4, alpha=0.5)
            ax.scatter([df["date"].iloc[-1]], [df["target"].iloc[-1]], color="#0f172a", s=22, zorder=5)

        ax.fill_between(fc["date"], fc["lower"], fc["upper"], color="#bfdbfe", alpha=0.5, label="likely range")
        ax.plot(fc["date"], fc["mean"], color="#2563eb", linewidth=2.6, marker="o", markersize=3.5, label="forecast")
        ax.axvline(split, color="#94a3b8", linestyle="--", linewidth=1.1)
        ax.text(split, ax.get_ylim()[1], "  now", color="#64748b", fontsize=9, va="top", ha="left")
    else:
        ax.plot(df["date"], df["target"], color="#2563eb", linewidth=1.8, label=target_name)

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#111827", pad=18 if subtitle else 8)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=9.5, color="#64748b", va="bottom")
    ax.set_xlabel("")
    ax.set_ylabel(target_name)
    ax.margins(x=0.01)
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#e5e7eb")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _matplotlib_code(
    rows: list[dict],
    forecast_rows: list[dict],
    title: str,
    target_name: str,
    chart_type: str,
) -> str:
    rows_json = json.dumps(rows)
    forecast_json = json.dumps(forecast_rows)
    return textwrap.dedent(
        f"""
        import base64
        from io import BytesIO
        import json
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        rows = json.loads({rows_json!r})
        forecast_rows = json.loads({forecast_json!r})
        chart_type = {chart_type!r}
        title = {title!r}
        target_name = {target_name!r}

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=160)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8fafc")

        if chart_type == "signals":
            signal_cols = [c for c in df.columns if c.startswith("signal_")]
            ax.plot(df["date"], df["target"], color="#2563eb", linewidth=2.0, label=target_name)
            ax2 = ax.twinx()
            for col in signal_cols[:4]:
                ax2.plot(df["date"], df[col], linewidth=1.2, alpha=0.65, label=col.replace("signal_", ""))
            ax2.set_ylabel("Signals")
            ax2.legend(loc="upper right", frameon=False, fontsize=8)
        else:
            ax.plot(df["date"], df["target"], color="#0f172a", linewidth=1.8, label="history")
            if chart_type == "forecast_history" and forecast_rows:
                fc = pd.DataFrame(forecast_rows)
                fc["date"] = pd.to_datetime(fc["date"])
                ax.plot(fc["date"], fc["mean"], color="#2563eb", linewidth=2.2, label="forecast")
                ax.fill_between(fc["date"], fc["lower"], fc["upper"], color="#93c5fd", alpha=0.35, label="confidence interval")

        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#111827")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
        for spine in ax.spines.values():
            spine.set_color("#e5e7eb")
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        image_b64
        """
    )
