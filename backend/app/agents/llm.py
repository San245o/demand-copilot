from __future__ import annotations

from app.core.config import settings

# Thin LLM accessor. Returns a Gemini chat model when a key is present, else None.
# Callers must handle None by emitting a deterministic mock reasoning string, so the
# crew runs identically (minus real language) without any API key.


def get_chat_model(kind: str = "reasoning"):
    """kind: 'reasoning' (fast) or 'narrative' (stronger). None if no key."""
    if not settings.has_llm:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = (
            settings.model_narrative
            if kind == "narrative"
            else settings.model_reasoning
        )
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.google_api_key,
            temperature=0.2 if kind == "reasoning" else 0.4,
        )
    except Exception:
        return None


async def think(prompt: str, kind: str = "reasoning", mock: str = "") -> str:
    """Single-shot reasoning helper. Falls back to `mock` text when no LLM."""
    model = get_chat_model(kind)
    if model is None:
        return mock or "(mock reasoning) proceeding with deterministic heuristics."
    try:
        resp = await model.ainvoke(prompt)
        return resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        return mock or f"(LLM error, using heuristic) {e}"
