from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from app.agents.tools import ALL_TOOLS
from app.core.config import settings
from app.core.events import EventType, StreamEvent

SYSTEM_PROMPT = """You are a demand-forecasting co-pilot for retail/FMCG planning.

You help planners understand and predict demand. You have tools to:
- list entities, get recent sales, and forecast demand (a real statistical model)
- search the web for current market context
- check weather, holidays, and a macro signal
- search internal planning playbooks

EFFICIENCY RULES (important — the model has a tight request budget):
1. Call ALL the tools you need in ONE step, in parallel, not one at a time. If a
   question needs a forecast AND playbooks, request both in the same turn.
2. Only call tools that are actually needed to answer. Do NOT reflexively call
   weather, holidays, or macro unless the question is about them.
3. Prefer the fewest round-trips: gather data once, then answer.

When you forecast, always mention the confidence interval. When you recommend
something, ground it (cite a playbook or signal). Be concise and concrete. If the
user asks something unrelated to demand, just answer directly without tools.
Today's date is 2026-06-10."""

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

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=settings.google_api_key,
        temperature=0.3,
        max_retries=1,   # don't silently retry forever on 429 — surface it fast
        timeout=60,
    )
    return create_react_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)


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
) -> AsyncIterator[StreamEvent]:
    """Stream a ReAct agent's reasoning + tool use + final answer.

    history is a list of {role, content}. Without an API key we emit a clear error
    event (the ReAct agent genuinely needs a tool-calling LLM)."""
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
                yield StreamEvent(
                    type=EventType.TOOL_CALL, agent="agent",
                    message=f"{event['name']}({_fmt_args(event)})",
                    data={"tool": event["name"]},
                )
            elif kind == "on_tool_end":
                out = event["data"].get("output")
                text = _tool_output_text(out)
                yield StreamEvent(
                    type=EventType.TOOL_RESULT, agent="agent",
                    message=text[:400], data={"tool": event["name"]},
                )
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                token = _extract_text(getattr(chunk, "content", ""))
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
    data = event.get("data", {})
    inp = data.get("input", {})
    if isinstance(inp, dict):
        return ", ".join(f"{k}={v}" for k, v in inp.items())
    return str(inp)[:80]


def _tool_output_text(out) -> str:
    if out is None:
        return ""
    content = getattr(out, "content", None)
    return content if isinstance(content, str) else str(out)


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
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "".join(out)
    return ""
