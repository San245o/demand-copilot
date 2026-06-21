from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from app.adapters.registry import get_adapter, get_active_profile
from app.core.schema import Forecast, TimeSeries
from app.rag.knowledge import get_knowledge_base
from app.services.e2b_viz import Visualization, create_visualization, render_forecast_chart
from app.tools.enrichers import (
    fetch_holidays,
    fetch_macro_signal,
    fetch_weather_summary,
)
from app.tools.forecast_tool import run_forecast
from app.tools.search_tool import web_search

# The ReAct agent's toolbox. Each is a plain, well-documented function the LLM can
# choose to call. Docstrings matter — the model reads them to decide when to use each.

_LAST_FORECAST: Forecast | None = None
_LAST_FORECAST_VIZ: Visualization | None = None
_LAST_VISUALIZATION: Visualization | None = None


def _run(coro):
    """Run an async enricher from inside a sync tool."""
    return asyncio.run(coro)


@tool
def list_entities(contains: str = "") -> str:
    """List the exact entity ids available to forecast (e.g. store_1, store_2).

    Use this to find the EXACT id to pass to forecast_demand/get_recent_sales — ids
    must match exactly and may be unusual (e.g. "1000 | 3 | 9"). When there are many
    entities, pass `contains` to filter to ones whose id contains that text (e.g.
    contains="500" to find store_500). Returns the total count so you know the range.
    """
    adapter = get_adapter()
    ents = adapter.list_entities()
    total = len(ents)
    needle = contains.strip().lower()
    if needle:
        matches = [e for e in ents if needle in e.lower()]
        if not matches:
            sample = ", ".join(ents[:15])
            return (
                f"adapter={adapter.name}; {total} entities total; none contain "
                f"'{contains}'. Examples: {sample}{' …' if total > 15 else ''}"
            )
        shown = matches[:50]
        suffix = f" (showing 50 of {len(matches)} matches)" if len(matches) > 50 else ""
        return f"adapter={adapter.name}; {len(matches)} of {total} match '{contains}'{suffix}: {', '.join(shown)}"
    shown = ents[:50]
    suffix = (
        f" (showing first 50 of {total}; use list_entities(contains=...) to find a specific one)"
        if total > 50
        else ""
    )
    return f"adapter={adapter.name}; {total} entities total{suffix}: {', '.join(shown)}"


@tool
def describe_dataset() -> str:
    """Describe the active dataset profile, relevance, columns, coverage, signals, and region.
    Use this when the user asks what data is loaded or whether it is suitable.
    """
    p = get_active_profile()
    region = (
        f"{p.region.country_name} ({p.region.country_code}), lat={p.region.lat}, lon={p.region.lon}"
        if p.region
        else "unknown"
    )
    return (
        f"Dataset {p.name} ({p.id}): {p.description}\n"
        f"Relevant={p.relevant}: {p.relevance_reason}\n"
        f"Rows={p.row_count}; coverage={p.date_min} to {p.date_max}; freq={p.freq}\n"
        f"Date column={p.date_column}; target={p.target_column} ({p.target_name}, unit={p.unit or 'unknown'}); "
        f"entities={', '.join(p.entity_columns) or 'none'}; signals={', '.join(p.signal_columns) or 'none'}; "
        f"region={region}."
    )


@tool
def forecast_demand(
    entity_id: str,
    horizon: int = 7,
    model: str = "auto",
    level: int = 95,
    lookback: int = 180,
) -> str:
    """Forecast future demand for one entity using a statistical model.

    Use for 'how much will we sell / demand next week' questions. horizon is 1-90.
    model is one of auto/ets/arima/theta/naive: ets is stable for seasonal retail,
    arima is good for autocorrelation, theta is simple/trend-friendly, naive is a
    baseline. level is confidence percent 50-99. lookback is the history window.
    """
    global _LAST_FORECAST, _LAST_FORECAST_VIZ
    _LAST_FORECAST = None
    _LAST_FORECAST_VIZ = None
    horizon = max(1, min(int(horizon), 90))
    level = max(50, min(int(level), 99))
    lookback = max(14, min(int(lookback), 2000))
    model = model if model in {"auto", "ets", "arima", "theta", "naive"} else "auto"
    adapter = get_adapter()
    try:
        series = adapter.load(entity_id, lookback=lookback)
    except Exception:
        return f"Unknown entity '{entity_id}'. Call list_entities first."
    if not series.points:
        return f"No data for '{entity_id}'."
    fc = run_forecast(series, horizon=horizon, level=level, model=model)
    _LAST_FORECAST = fc
    # Render a legible history+forecast chart (locally) so the UI can show a real
    # matplotlib chart instead of a context-free forecast-only sparkline.
    try:
        _LAST_FORECAST_VIZ = render_forecast_chart(series, fc, fc.model)
    except Exception:
        _LAST_FORECAST_VIZ = None
    total = round(sum(p.mean for p in fc.points))
    lines = [
        f"{p.date}: mean={p.mean:.0f} ({level}% CI {p.lower:.0f}-{p.upper:.0f})"
        for p in fc.points
    ]
    return (
        f"Forecast for {entity_id} ({fc.model}), next {horizon}d, {level}% CI, total≈{total}:\n"
        + "\n".join(lines)
    )


def consume_last_forecast() -> Forecast | None:
    global _LAST_FORECAST
    fc = _LAST_FORECAST
    _LAST_FORECAST = None
    return fc


def consume_last_forecast_viz() -> Visualization | None:
    global _LAST_FORECAST_VIZ
    viz = _LAST_FORECAST_VIZ
    _LAST_FORECAST_VIZ = None
    return viz


@tool
def generate_visualization(
    entity_id: str,
    chart_type: str = "forecast_history",
    horizon: int = 30,
    lookback: int = 180,
    model: str = "auto",
    level: int = 95,
) -> str:
    """Generate a finished matplotlib visualization for the active dataset using E2B.

    Use this when the user asks to plot, visualize, graph, compare history vs forecast,
    show signals, or create a better chart. chart_type is one of history,
    forecast_history, or signals. The chart image is attached to the UI automatically.
    """
    global _LAST_VISUALIZATION
    _LAST_VISUALIZATION = None
    chart_type = chart_type if chart_type in {"history", "forecast_history", "signals"} else "forecast_history"
    horizon = max(1, min(int(horizon), 90))
    lookback = max(14, min(int(lookback), 2000))
    level = max(50, min(int(level), 99))
    model = model if model in {"auto", "ets", "arima", "theta", "naive"} else "auto"

    adapter = get_adapter()
    try:
        series = adapter.load(entity_id, lookback=lookback)
    except Exception:
        return f"Unknown entity '{entity_id}'. Call list_entities first."
    if not series.points:
        return f"No data for '{entity_id}'."

    forecast = None
    if chart_type == "forecast_history":
        forecast = run_forecast(series, horizon=horizon, level=level, model=model)
    try:
        _LAST_VISUALIZATION = create_visualization(series, chart_type=chart_type, forecast=forecast)
    except Exception as exc:
        return f"Could not generate visualization: {exc}"
    return f"Visualization generated: {_LAST_VISUALIZATION.title} ({_LAST_VISUALIZATION.source}). The chart is attached in the UI."


def consume_last_visualization() -> Visualization | None:
    global _LAST_VISUALIZATION
    viz = _LAST_VISUALIZATION
    _LAST_VISUALIZATION = None
    return viz


@tool
def get_recent_sales(entity_id: str, days: int = 14) -> str:
    """Get recent historical sales for an entity, plus which signals (promo, holiday)
    are present. Use to understand recent trend before forecasting or explaining."""
    days = max(1, min(int(days), 90))
    adapter = get_adapter()
    try:
        series: TimeSeries = adapter.load(entity_id, lookback=days)
    except Exception:
        return f"Unknown entity '{entity_id}'. Call list_entities first."
    if not series.points:
        return f"No data for '{entity_id}'."
    pts = ", ".join(f"{p.date}:{p.target:.0f}" for p in series.points[-days:])
    return (
        f"{entity_id} last {len(series.points)}d (signals: "
        f"{', '.join(series.signal_names) or 'none'}): {pts}"
    )


@tool
def search_web(query: str) -> str:
    """Search the web for current context (market trends, events, news, macro).
    Use when the question needs information not in the sales data."""
    results = _run(web_search(query, max_results=3))
    if not results:
        return "No results."
    return "\n".join(
        f"- {r['title']}: {(r['content'] or '')[:200]} ({r['url']})" for r in results
    )


@tool
def get_weather(start: str, end: str, lat: float | None = None, lon: float | None = None) -> str:
    """Get a weather summary (avg/min/max temp) for a date range (YYYY-MM-DD).
    Useful to assess weather impact on demand. Defaults to the active dataset region when known."""
    from datetime import date

    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
    except ValueError:
        return "Dates must be YYYY-MM-DD."
    region = get_active_profile().region
    if lat is None:
        lat = region.lat if region else 51.0
    if lon is None:
        lon = region.lon if region else 9.0
    w = _run(fetch_weather_summary(s, e, lat, lon))
    return (
        f"Weather {start}→{end}: avg {w['avg_temp_c']}°C "
        f"(min {w['min_temp_c']}, max {w['max_temp_c']}), source={w['source']}"
    )


@tool
def get_holidays(year: int, country: str | None = None) -> str:
    """List public holidays for a year and country code.
    Useful to explain demand dips/spikes around holidays."""
    if not country:
        region = get_active_profile().region
        country = region.country_code if region else "DE"
    hs = _run(fetch_holidays(int(year), country))
    return f"{len(hs)} holidays in {year} ({country}): " + ", ".join(
        f"{h['date']} {h['name']}" for h in hs[:12]
    )


@tool
def get_macro_signal() -> str:
    """Get a macroeconomic context signal (consumer sentiment). Optional context."""
    m = _run(fetch_macro_signal())
    return f"Macro {m['series']}={m['value']} (source={m['source']})"


@tool
def search_playbooks(query: str) -> str:
    """Search internal demand-planning playbooks and post-mortems (promotions,
    holidays, inventory policy, seasonality) to ground recommendations."""
    kb = get_knowledge_base()
    hits = kb.query(query, k=3)
    if not hits:
        return "No playbook entries found."
    return "\n".join(f"- ({h['meta'].get('topic')}) {h['text']}" for h in hits)


ALL_TOOLS = [
    list_entities,
    describe_dataset,
    forecast_demand,
    generate_visualization,
    get_recent_sales,
    search_web,
    get_weather,
    get_holidays,
    get_macro_signal,
    search_playbooks,
]
