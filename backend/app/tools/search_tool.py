from __future__ import annotations

import httpx

from app.core.config import settings

# Agent web search via Tavily (LLM-native search API). Mocked without a key.


async def web_search(query: str, max_results: int = 3) -> list[dict]:
    if not settings.has_tavily:
        return _mock_results(query)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
            )
            r.raise_for_status()
            return [
                {"title": x.get("title"), "url": x.get("url"),
                 "content": x.get("content")}
                for x in r.json().get("results", [])
            ]
    except Exception:
        return _mock_results(query)


def _mock_results(query: str) -> list[dict]:
    return [
        {
            "title": f"(mock) Market context for: {query}",
            "url": "https://example.com/mock",
            "content": "Demand context unavailable (no Tavily key); using mock signal.",
        }
    ]
