# Demand Forecasting Co-Pilot

A **chat co-pilot** for demand planning: you ask a question in natural language, and a
single **ReAct agent** (Gemini) reasons step by step, decides which tools to call
(forecast, web search, weather, holidays, playbooks), and answers — streaming its
reasoning and tool use live, with a collapsible thought trail.

> The LLM never does the arithmetic. Forecasting is a **tool** (`statsforecast`,
> producing point forecasts + confidence intervals). The agent decides *what data to
> pull*, *when to forecast*, and *how to explain it*.

## Architecture

```
                                  ┌──────── tools the agent can call ────────┐
                                  │ forecast_demand (statsforecast)          │
 you ──chat──► ReAct agent ◄────► │ get_recent_sales / list_entities         │
   (Next.js)   (Gemini, LangGraph │ search_web (Tavily)                      │
               create_react_agent)│ get_weather / get_holidays / macro       │
      ▲                           │ search_playbooks (vector RAG)            │
      └──── SSE: thoughts, tool   └──────────────────────────────────────────┘
            calls, streamed answer        each tool → canonical TimeSeries via an Adapter
```

- **Chat + ReAct.** One agent, many tools. It calls tools in parallel and only when
  needed (tuned to minimize requests on rate-limited free tiers).
- **Dataset-agnostic core.** Tools see a canonical `{date, entity_id, target, signals{…}}`
  record. Each dataset (Rossmann, Favorita, M5, a real CSV/warehouse) plugs in behind a
  thin **adapter**. Signals are optional and discovered, never hardcoded.
- **Forecast is a tool, not the LLM.** `statsforecast` AutoETS → point forecast + 95% CI.
- **Plain vector RAG** over promo playbooks / post-mortems. No GraphRAG. Small
  self-contained numpy cosine index (Gemini embeddings when a key is present, hash
  embeddings offline) — same add/query interface, swappable for Chroma later.

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + LangGraph `create_react_agent` (Python 3.12, `uv`) |
| LLM | Gemini via `langchain-google-genai`; model selectable in the UI (default `gemini-3.1-flash-lite`) |
| Forecast tool | `statsforecast` (AutoETS) |
| RAG | self-contained numpy vector store (Gemini / hash embeddings) |
| Signals | Open-Meteo (weather), Nager.Date (holidays), Tavily (search), FRED (macro, optional) |
| Frontend | Next.js (App Router) + Tailwind — chat UI with streamed thought trail |

**Every external dependency has a mock fallback**, so the app runs end-to-end even with no
API keys and no dataset present.

## Quickstart

```bash
# Backend
cd backend
uv sync
cp .env.example .env        # fill in keys you have; everything missing falls back to mock
uv run uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                 # http://localhost:3000
```

## Status

Built in phases — see the project task list. P0: streaming skeleton. P1: full agent crew (mocked).
P2: real Rossmann data + forecast. P3: real Gemini + enrichers + RAG. P4: HITL + search + polish.
