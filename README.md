# Demand Forecasting Co-Pilot

An agentic demand-forecasting assistant: a crew of LLM agents pulls demand signals,
calls a real statistical forecasting model as a tool, validates the result against
business rules, and produces an inventory recommendation + narrative brief — with a
human-in-the-loop approval gate before anything is committed.

> The LLM never does the arithmetic. Forecasting is a **tool** (`statsforecast`,
> producing point forecasts + confidence intervals). The agents decide *what data to
> pull*, *when to forecast*, and *how to explain it*.

## Architecture

```
dataset / API ─► Adapter ─► canonical TimeSeries ─► [ Sensing ─► Forecast ─► Validation ─► Planning ] ─► HITL approve ─► UI
                            (date, entity_id,         LangGraph crew (nodes), streamed step-by-step over SSE
                             target, signals{…})
```

- **Dataset-agnostic core.** Agents only ever see a canonical `{date, entity_id, target, signals{…}}`
  record. Each dataset (Rossmann, Favorita, M5, a real CSV/warehouse) plugs in behind a thin
  **adapter**. Signals are optional and discovered, never hardcoded — so a bare time series and a
  rich promo/weather dataset run through the identical code path.
- **Forecast is a tool, not the LLM.** `statsforecast` (AutoARIMA / AutoETS) → point forecast + 95% CI.
- **Plain vector RAG** over past forecast reports / promo playbooks. No GraphRAG. The
  store is a small self-contained numpy cosine index (Gemini embeddings when a key is
  present, deterministic hash embeddings offline) — same add/query interface, swappable
  for Chroma later, zero extra heavy deps.
- **Human-in-the-loop** via LangGraph `interrupt()` + `Command(resume=…)`.

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + LangGraph (Python 3.12, managed by `uv`) |
| LLM | Gemini (`gemini-3.5-flash` reasoning, `gemini-3.1-pro` narrative) via `langchain-google-genai` |
| Forecast tool | `statsforecast` (AutoARIMA / AutoETS) |
| RAG | self-contained numpy vector store (Gemini / hash embeddings) |
| Signals | Open-Meteo (weather), Nager.Date (holidays), Tavily (search), FRED (macro, optional) |
| Frontend | Next.js (App Router) + Tailwind, SSE streaming |

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
