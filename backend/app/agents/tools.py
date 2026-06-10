from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from app.adapters.registry import get_adapter
from app.core.schema import TimeSeries
from app.rag.knowledge import get_knowledge_base
from app.tools.enrichers import (
    fetch_holidays,
    fetch_macro_signal,
    fetch_weather_summary,
)
from app.tools.forecast_tool import run_forecast
from app.tools.search_tool import web_search

# The ReAct agent's toolbox. Each is a plain, well-documented function the LLM can
# choose to call. Docstrings matter — the model reads them to decide when to use each.


def _run(coro):
    """Run an async enricher from inside a sync tool."""
    return asyncio.run(coro)


@tool
def list_entities() -> str:
    """List the store/entity ids available to forecast (e.g. store_1, store_2)."""
    adapter = get_adapter()
    ents = adapter.list_entities()
    return f"adapter={adapter.name}; entities={', '.join(ents[:25])}" + (
        " …" if len(ents) > 25 else ""
    )


@tool
def forecast_demand(entity_id: str, horizon: int = 7) -> str:
    """Forecast future demand for one entity using a statistical model (AutoETS).

    Returns the point forecast with a 95% confidence interval per day. Use this for
    any 'how much will we sell / demand next week' question. horizon is in days (1-90).
    """
    horizon = max(1, min(int(horizon), 90))
    adapter = get_adapter()
    try:
        series = adapter.load(entity_id, lookback=180)
    except Exception:
        return f"Unknown entity '{entity_id}'. Call list_entities first."
    if not series.points:
        return f"No data for '{entity_id}'."
    fc = run_forecast(series, horizon=horizon, level=95)
    total = round(sum(p.mean for p in fc.points))
    lines = [
        f"{p.date}: mean={p.mean:.0f} (95% CI {p.lower:.0f}–{p.upper:.0f})"
        for p in fc.points
    ]
    return (
        f"Forecast for {entity_id} ({fc.model}), next {horizon}d, total≈{total}:\n"
        + "\n".join(lines)
    )


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
def get_weather(start: str, end: str, lat: float = 51.0, lon: float = 9.0) -> str:
    """Get a weather summary (avg/min/max temp) for a date range (YYYY-MM-DD).
    Useful to assess weather impact on demand. Defaults to central Germany."""
    from datetime import date

    try:
        s = date.fromisoformat(start)
        e = date.fromisoformat(end)
    except ValueError:
        return "Dates must be YYYY-MM-DD."
    w = _run(fetch_weather_summary(s, e, lat, lon))
    return (
        f"Weather {start}→{end}: avg {w['avg_temp_c']}°C "
        f"(min {w['min_temp_c']}, max {w['max_temp_c']}), source={w['source']}"
    )


@tool
def get_holidays(year: int, country: str = "DE") -> str:
    """List public holidays for a year and country code (default DE for Germany).
    Useful to explain demand dips/spikes around holidays."""
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
    forecast_demand,
    get_recent_sales,
    search_web,
    get_weather,
    get_holidays,
    get_macro_signal,
    search_playbooks,
]
