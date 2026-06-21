from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from app.adapters.registry import get_active_profile
from app.agents.tools import (
    ALL_TOOLS,
    consume_last_forecast,
    consume_last_forecast_viz,
    consume_last_visualization,
)
from app.core.config import settings
from app.core.events import EventType, StreamEvent
from app.core.schema import DatasetProfile


def build_system_prompt(profile: DatasetProfile) -> str:
    region = (
        f"{profile.region.country_name} ({profile.region.country_code}), lat/lon {profile.region.lat}/{profile.region.lon}"
        if profile.region
        else "unknown; ask for location before relying on weather or country-specific holidays"
    )
    caveat = ""
    if not profile.relevant:
        caveat = (
            f"\n- WARNING: this dataset was flagged as possibly NOT demand data "
            f"({profile.relevance_reason}). Caveat every answer and don't overstate forecasts."
        )
    unit = profile.unit or "units (unit unknown)"
    entity_cols = ", ".join(profile.entity_columns) or "none defined"
    signals = ", ".join(profile.signal_columns) or "none"
    # Only state coverage facts we actually have — never assert "0 rows / None→None",
    # which is false for bundled datasets and invites the model to fabricate.
    coverage_bits = []
    if profile.date_min and profile.date_max:
        coverage_bits.append(f"{profile.date_min} → {profile.date_max}")
    if profile.row_count:
        coverage_bits.append(f"{profile.row_count} rows")
    coverage_bits.append(f"{profile.freq} frequency")
    coverage = " · ".join(coverage_bits)
    if not (profile.date_min and profile.row_count):
        coverage += " (exact range unknown here — use describe_dataset / get_recent_sales to check)"
    return f"""You are a demand-forecasting co-pilot for retail/FMCG planners. You answer
questions about the user's sales data — how much they will sell, recent trends, and what
is driving demand. Every number you give MUST come from a tool result; you never invent
data, entity ids, or figures.

ACTIVE DATASET
- name: {profile.name} (id: {profile.id})
- what it is: {profile.description}
- coverage: {coverage}
- target you forecast: {profile.target_name}, measured in {unit}
- entities you can forecast are built from these columns: {entity_cols}
- signals available: {signals}
- region for weather/holidays: {region}{caveat}

HOW TO ANSWER
1. Identify the entity. Entity ids must be EXACTLY what list_entities returns — never
   invent or reformat them (an id may combine several columns, e.g. "1000 | 3 | 9").
   If the user names a store/SKU and you don't already know its exact id, call
   list_entities FIRST and match it. If nothing matches, say so and show the options.
2. Get data, then answer. Never state a number, trend, or forecast that did not come
   from a tool result. If you lack the data, say what's missing instead of guessing.
3. To forecast: call forecast_demand with the exact entity id and a horizon. Report the
   predicted total and the confidence range in {unit}, in plain language. If the result
   mentions a "fallback" / limited-history baseline, say plainly that the history is too
   short for a reliable forecast and treat the figure as a rough estimate.
4. To explain a driver or make a recommendation, ground it in a real signal, recent
   sales, or a playbook (search_playbooks) — do not speculate.

TOOL DISCIPLINE — use the fewest tools needed
- Core: list_entities, describe_dataset, get_recent_sales, forecast_demand.
- generate_visualization: ONLY when the user explicitly wants a chart of history or
  signals, or a comparison. A forecast ALREADY gets a chart attached automatically —
  do NOT call generate_visualization just to display a forecast.
- search_web, get_weather, get_holidays, get_macro_signal: only when the question is
  actually about those topics. Never call them reflexively.
- Batch independent tool calls into a single step instead of one at a time.

CHARTS & IMAGES
- The UI renders charts from tool output. Never redraw a chart in markdown or describe
  it pixel by pixel — just give the takeaway in words.
- If the user attaches an image, look at it and answer about it directly; combine it
  with the data tools when that helps.

STYLE
- Lead with the answer (the key number or finding), then 1–3 short sentences of grounded
  explanation. Be concise and concrete; use {unit} and readable entity names.
- If a question is unrelated to demand or this dataset, just answer briefly, no tools.
Today is {date.today().isoformat()}."""

# Models the user can pick in the UI. First is the default. These are verified
# against the live ListModels API (names differ from the public docs).
# Default is flash-lite: lightest + highest rate limit on the free tier.
AVAILABLE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash",
]


def _make_agent(model_name: str):
    from langchain_google_genai import ChatGoogleGenerativeAI

    base_kwargs = dict(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=0.3,
        max_retries=1,   # don't silently retry forever on 429 — surface it fast
        timeout=60,
    )
    # Ask Gemini to surface its reasoning summary so the UI can show a real
    # "thinking" trace. Older/lighter models may reject the flag — fall back.
    try:
        llm = ChatGoogleGenerativeAI(include_thoughts=True, **base_kwargs)
    except Exception:
        llm = ChatGoogleGenerativeAI(**base_kwargs)
    return create_react_agent(llm, ALL_TOOLS, prompt=build_system_prompt(get_active_profile()))


def _friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "429" in msg or "resource_exhausted" in low or "quota" in low or "rate" in low:
        return ("Gemini rate limit hit (free tier allows only a few requests per "
                "minute, and one chat uses several). Wait a minute and try again, "
                "or pick a lighter model like gemini-3.1-flash-lite.")
    if "404" in msg or "not_found" in low:
        return f"Model not available: {msg[:200]}"
    return f"Agent error: {msg[:300]}"


async def chat_stream(
    message: str,
    history: list[dict] | None = None,
    model_name: str | None = None,
    images: list[str] | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream a ReAct agent's reasoning + tool use + final answer.

    history is a list of {role, content}. images is a list of base64 data URIs
    (data:image/...;base64,...) attached to the current turn for vision (view-only).
    Without an API key we emit a clear error event (the ReAct agent genuinely needs
    a tool-calling LLM)."""
    model_name = model_name or AVAILABLE_MODELS[0]

    yield StreamEvent(type=EventType.RUN_START, message="thinking",
                      data={"model": model_name})

    if not settings.has_llm:
        yield StreamEvent(
            type=EventType.ERROR,
            message="No GOOGLE_API_KEY set. Add it to backend/.env to enable the "
                    "chat agent (it needs a tool-calling LLM).",
        )
        yield StreamEvent(type=EventType.RUN_END)
        return

    # Build message list from history + new message.
    msgs: list = []
    for h in history or []:
        if h.get("role") == "user":
            msgs.append(HumanMessage(content=h["content"]))
        elif h.get("role") == "assistant":
            msgs.append(AIMessage(content=h["content"]))

    if images:
        # Multimodal turn: text + image parts. Gemini reads the data-URI images
        # directly (view-only — no image tools, the model just looks at them).
        parts: list[dict] = [
            {"type": "text", "text": message or "Analyze the attached image(s)."}
        ]
        parts.extend({"type": "image_url", "image_url": img} for img in images)
        msgs.append(HumanMessage(content=parts))
    else:
        msgs.append(HumanMessage(content=message))

    try:
        agent = _make_agent(model_name)
    except Exception as e:
        yield StreamEvent(type=EventType.ERROR,
                          message=f"Could not init model '{model_name}': {e}")
        yield StreamEvent(type=EventType.RUN_END)
        return

    answer_parts: list[str] = []
    final_answer = ""
    try:
        async for event in agent.astream_events(
            {"messages": msgs}, version="v2"
        ):
            kind = event["event"]

            if kind == "on_tool_start":
                args = _event_args(event)
                yield StreamEvent(
                    type=EventType.TOOL_CALL, agent="agent",
                    message=f"{event['name']}({_fmt_args(event)})",
                    data={"id": str(event.get("run_id", "")), "tool": event["name"], "args": args},
                )
            elif kind == "on_tool_end":
                out = event["data"].get("output")
                text = _tool_output_text(out)
                run_id = str(event.get("run_id", ""))
                yield StreamEvent(
                    type=EventType.TOOL_RESULT, agent="agent",
                    message=text[:1200], data={"id": run_id, "tool": event["name"]},
                )
                if event["name"] == "forecast_demand":
                    forecast = consume_last_forecast()
                    if forecast is not None:
                        yield StreamEvent(
                            type=EventType.FORECAST,
                            agent="agent",
                            message="forecast",
                            data={"id": run_id, "forecast": forecast.model_dump(mode="json")},
                        )
                    forecast_viz = consume_last_forecast_viz()
                    if forecast_viz is not None:
                        yield StreamEvent(
                            type=EventType.VISUALIZATION,
                            agent="agent",
                            message="visualization",
                            data={"id": run_id, "visualization": forecast_viz.model_dump(mode="json")},
                        )
                if event["name"] == "generate_visualization":
                    visualization = consume_last_visualization()
                    if visualization is not None:
                        yield StreamEvent(
                            type=EventType.VISUALIZATION,
                            agent="agent",
                            message="visualization",
                            data={"id": run_id, "visualization": visualization.model_dump(mode="json")},
                        )
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                content = getattr(chunk, "content", "")
                thought = _extract_thinking(content)
                if thought:
                    yield StreamEvent(type=EventType.THOUGHT, agent="agent",
                                      message=thought)
                token = _extract_text(content)
                if token:
                    answer_parts.append(token)
                    yield StreamEvent(type=EventType.ANSWER, agent="agent",
                                      message=token)
            elif kind == "on_chat_model_end":
                # Fallback: capture the final answer even if token streaming
                # didn't surface (Gemini returns content as list-of-parts).
                out = event["data"].get("output")
                if out is not None and not getattr(out, "tool_calls", None):
                    t = _extract_text(getattr(out, "content", ""))
                    if t:
                        final_answer = t
    except Exception as e:
        yield StreamEvent(type=EventType.ERROR, message=_friendly_error(e))
        yield StreamEvent(type=EventType.RUN_END)
        return

    answer = "".join(answer_parts).strip() or final_answer
    yield StreamEvent(type=EventType.BRIEF, agent="agent",
                      message="answer", data={"answer": answer})
    yield StreamEvent(type=EventType.RUN_END)


def _fmt_args(event: dict) -> str:
    inp = _event_args(event)
    if isinstance(inp, dict):
        return ", ".join(f"{k}={v}" for k, v in inp.items())
    return str(inp)[:80]


def _event_args(event: dict):
    data = event.get("data", {})
    return data.get("input", {})


def _tool_output_text(out) -> str:
    if out is None:
        return ""
    content = getattr(out, "content", None)
    return content if isinstance(content, str) else str(out)


def _extract_thinking(content) -> str:
    """Pull Gemini reasoning/thinking text from streamed message content.

    With include_thoughts=True, thinking arrives as parts shaped like
    {'type': 'thinking', 'thinking': '…'} (or 'reasoning'). Plain text answer parts
    are ignored here — they go through _extract_text.
    """
    if not isinstance(content, list):
        return ""
    out: list[str] = []
    for part in content:
        if isinstance(part, dict):
            ptype = part.get("type")
            if ptype == "thinking" and isinstance(part.get("thinking"), str):
                out.append(part["thinking"])
            elif ptype == "reasoning" and isinstance(part.get("reasoning"), str):
                out.append(part["reasoning"])
            elif part.get("thought") and isinstance(part.get("text"), str):
                out.append(part["text"])
    return "".join(out)


def _extract_text(content) -> str:
    """Pull plain text from message content that may be a str or list-of-parts.

    Gemini returns content as a list like [{'type':'text','text':'…'}] or plain
    strings; tool-call chunks have non-text parts we skip.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                # Skip reasoning parts — those are surfaced via _extract_thinking.
                if part.get("thought") or part.get("type") in {"thinking", "reasoning"}:
                    continue
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
        return "".join(out)
    return ""
