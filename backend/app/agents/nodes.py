from __future__ import annotations

from datetime import date

from langgraph.config import get_stream_writer
from langgraph.types import interrupt

from app.agents.llm import think
from app.agents.state import CrewState
from app.adapters.registry import get_adapter
from app.core.events import EventType, StreamEvent
from app.core.schema import Forecast, TimeSeries
from app.rag.knowledge import get_knowledge_base
from app.tools.enrichers import (
    fetch_holidays,
    fetch_macro_signal,
    fetch_weather_summary,
)
from app.tools.forecast_tool import run_forecast
from app.tools.search_tool import web_search


def _emit(ev: StreamEvent) -> None:
    """Push a StreamEvent to the custom stream (consumed by the SSE runner)."""
    writer = get_stream_writer()
    if writer is not None:
        writer(ev.model_dump())


# ----------------------------------------------------------------------------
# 1. Sensing — gather the canonical series + optional context signals
# ----------------------------------------------------------------------------
async def sensing_node(state: CrewState) -> dict:
    entity_id = state["entity_id"]
    horizon = state["horizon"]
    _emit(StreamEvent(type=EventType.AGENT_START, agent="sensing",
                      message="Gathering demand signals"))

    adapter = get_adapter()
    _emit(StreamEvent(type=EventType.TOOL_CALL, agent="sensing",
                      message=f"load_series({entity_id}, lookback=180) via {adapter.name}",
                      data={"tool": "load_series", "adapter": adapter.name}))
    series = adapter.load(entity_id, lookback=180)
    _emit(StreamEvent(type=EventType.TOOL_RESULT, agent="sensing",
                      message=f"{len(series.points)} observations; "
                              f"signals: {', '.join(series.signal_names) or 'none'}",
                      data={"n_points": len(series.points),
                            "signals": series.signal_names}))

    # Enrich with context (each degrades to mock). Window = around series end.
    last = series.points[-1].date if series.points else date(2015, 8, 1)
    _emit(StreamEvent(type=EventType.TOOL_CALL, agent="sensing",
                      message="enrich: weather + holidays + macro",
                      data={"tool": "enrichers"}))
    weather = await fetch_weather_summary(
        start=last.replace(day=1), end=last
    )
    holidays = await fetch_holidays(last.year)
    macro = await fetch_macro_signal()
    context = {
        "weather": weather,
        "holidays_count": len(holidays),
        "macro": macro,
    }
    _emit(StreamEvent(type=EventType.TOOL_RESULT, agent="sensing",
                      message=f"weather avg {weather['avg_temp_c']}°C "
                              f"({weather['source']}); {len(holidays)} holidays; "
                              f"macro {macro['series']}={macro['value']} ({macro['source']})",
                      data=context))

    thought = await think(
        f"In one sentence, summarize what demand signals we have for {entity_id}: "
        f"{len(series.points)} days of sales, signals {series.signal_names}, "
        f"weather {weather}, {len(holidays)} holidays.",
        mock=f"Loaded {len(series.points)} days for {entity_id} with signals "
             f"{series.signal_names}; context enriched.",
    )
    _emit(StreamEvent(type=EventType.THOUGHT, agent="sensing", message=thought))
    _emit(StreamEvent(type=EventType.AGENT_END, agent="sensing"))

    return {"series": series.model_dump(mode="json"), "context": context}


# ----------------------------------------------------------------------------
# 2. Forecast — call the statistical tool (LLM does NOT do math)
# ----------------------------------------------------------------------------
async def forecast_node(state: CrewState) -> dict:
    _emit(StreamEvent(type=EventType.AGENT_START, agent="forecast",
                      message="Running the statistical forecast model"))
    series = TimeSeries.model_validate(state["series"])
    horizon = state["horizon"]

    thought = await think(
        f"The series for {series.entity_id} is {series.freq} with "
        f"{len(series.points)} points. In one sentence, say which model fits and why.",
        mock="Daily series with weekly seasonality — AutoETS is appropriate.",
    )
    _emit(StreamEvent(type=EventType.THOUGHT, agent="forecast", message=thought))
    _emit(StreamEvent(type=EventType.TOOL_CALL, agent="forecast",
                      message=f"forecast(horizon={horizon}, level=95)",
                      data={"tool": "forecast"}))

    fc: Forecast = run_forecast(series, horizon=horizon, level=95)

    _emit(StreamEvent(type=EventType.TOOL_RESULT, agent="forecast",
                      message=f"{fc.model}: {len(fc.points)} steps with 95% CI"))
    _emit(StreamEvent(type=EventType.FORECAST, agent="forecast",
                      message="Point forecast + interval",
                      data=fc.model_dump(mode="json")))
    _emit(StreamEvent(type=EventType.AGENT_END, agent="forecast"))
    return {"forecast": fc.model_dump(mode="json")}


# ----------------------------------------------------------------------------
# 3. Validation — business rules + CI sanity
# ----------------------------------------------------------------------------
async def validation_node(state: CrewState) -> dict:
    _emit(StreamEvent(type=EventType.AGENT_START, agent="validation",
                      message="Checking the forecast against business rules"))
    fc = Forecast.model_validate(state["forecast"])

    means = [p.mean for p in fc.points] or [0.0]
    avg_mean = sum(means) / len(means)
    avg_width = (
        sum(p.upper - p.lower for p in fc.points) / len(fc.points)
        if fc.points else 0.0
    )
    ci_width_pct = round(100 * avg_width / avg_mean, 1) if avg_mean else 0.0
    has_negative = any(p.lower < 0 for p in fc.points)
    # Wide CI on a meaningful series → flag for human review.
    needs_review = ci_width_pct > 60 or avg_mean > 8000
    passed = not has_negative

    checks = {
        "ci_width_pct": ci_width_pct,
        "has_negative": has_negative,
        "avg_mean": round(avg_mean, 1),
        "needs_review": needs_review,
        "passed": passed,
    }
    _emit(StreamEvent(type=EventType.THOUGHT, agent="validation",
                      message=f"CI width ≈ {ci_width_pct}% of mean; "
                              f"{'flagged for review' if needs_review else 'within tolerance'}."))
    _emit(StreamEvent(type=EventType.AGENT_END, agent="validation",
                      message="Passed" if passed else "Failed", data=checks))
    return {"validation": checks}


# ----------------------------------------------------------------------------
# 4. Human-in-the-loop gate — pause for approval
# ----------------------------------------------------------------------------
def approval_node(state: CrewState) -> dict:
    """Pause and surface the forecast for a human decision.

    interrupt() raises GraphInterrupt (do NOT wrap in try/except). The value passed
    to Command(resume=...) becomes this call's return value on resume.
    """
    fc = state["forecast"]
    val = state["validation"]
    _emit(StreamEvent(type=EventType.INTERRUPT, agent="approval",
                      message="Awaiting human approval of the forecast",
                      data={"forecast": fc, "validation": val}))

    decision = interrupt({
        "kind": "approval",
        "entity_id": state["entity_id"],
        "needs_review": val.get("needs_review", False),
        "summary": f"{len(fc.get('points', []))}-step forecast, "
                   f"CI width {val.get('ci_width_pct')}%",
    })
    # decision is whatever the client sent via Command(resume=decision)
    return {"decision": decision or {"action": "approve"}}


# ----------------------------------------------------------------------------
# 5. Planning — inventory rec + RAG-grounded narrative brief
# ----------------------------------------------------------------------------
async def planning_node(state: CrewState) -> dict:
    _emit(StreamEvent(type=EventType.AGENT_START, agent="planning",
                      message="Drafting inventory recommendation and brief"))
    fc = Forecast.model_validate(state["forecast"])
    context = state.get("context", {})
    decision = state.get("decision", {"action": "approve"})

    if decision.get("action") == "reject":
        rejected = {
            "headline": "Forecast rejected by reviewer",
            "recommendation": decision.get("feedback", "Revise and rerun."),
            "confidence": "n/a",
            "drivers": [],
        }
        _emit(StreamEvent(type=EventType.BRIEF, agent="planning",
                          message="Forecast rejected", data=rejected))
        _emit(StreamEvent(type=EventType.AGENT_END, agent="planning",
                          message="Rejected by reviewer"))
        return {"brief": rejected}

    # RAG: retrieve grounding knowledge.
    kb = get_knowledge_base()
    means = [p.mean for p in fc.points] or [0.0]
    upper = [p.upper for p in fc.points] or [0.0]
    query = f"inventory reorder point promotion forecast confidence interval"
    retrieved = kb.query(query, k=3)
    _emit(StreamEvent(type=EventType.TOOL_CALL, agent="planning",
                      message=f"rag.query (backend={kb.backend})",
                      data={"tool": "rag", "hits": len(retrieved)}))
    _emit(StreamEvent(type=EventType.TOOL_RESULT, agent="planning",
                      message="; ".join(r["meta"].get("topic", "?") for r in retrieved)))

    reorder = round(max(upper))
    total_mean = round(sum(means))
    grounding = "\n".join(f"- {r['text']}" for r in retrieved)

    narrative = await think(
        "You are a demand planner. Write a 2-3 sentence forecast brief.\n"
        f"Entity: {fc.entity_id}. Next {fc.horizon} days total demand ~{total_mean} "
        f"({fc.model}, 95% CI). Context: {context}.\n"
        f"Grounding knowledge:\n{grounding}\n"
        "Be specific and actionable about the reorder point.",
        kind="narrative",
        mock=f"{fc.entity_id}: total demand over the next {fc.horizon} days is "
             f"~{total_mean} units. Set the reorder point near {reorder} to cover the "
             f"upper confidence bound and avoid stockouts. Promotions and weekly "
             f"seasonality are the main drivers.",
    )

    drivers = []
    series_signals = state.get("series", {}).get("points", [])
    if any("promo" in p.get("signals", {}) for p in series_signals):
        drivers.append("active promotions")
    drivers.append("weekly seasonality")
    if context.get("weather", {}).get("avg_temp_c", 99) < 8:
        drivers.append("cold weather")

    confidence = "low" if state.get("validation", {}).get("needs_review") else "medium"
    brief = {
        "headline": narrative.split(".")[0].strip() + ".",
        "recommendation": narrative,
        "reorder_point": reorder,
        "horizon_total": total_mean,
        "confidence": confidence,
        "drivers": drivers,
        "grounding": [r["meta"] for r in retrieved],
    }
    _emit(StreamEvent(type=EventType.BRIEF, agent="planning",
                      message="Forecast brief ready", data=brief))
    _emit(StreamEvent(type=EventType.AGENT_END, agent="planning"))
    return {"brief": brief}
