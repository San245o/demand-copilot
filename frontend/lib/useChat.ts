"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface ToolStep {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  status: "running" | "done";
  result?: string;
  forecast?: ForecastPayload;
  visualization?: VisualizationPayload;
}

export interface ForecastPayload {
  entity_id: string;
  horizon: number;
  level: number;
  model: string;
  points: { date: string; mean: number; lower: number; upper: number }[];
}

export interface VisualizationPayload {
  title: string;
  image: string;
  mime: string;
  source?: string;
}

export interface DatasetRegion {
  country_code: string;
  country_name: string;
  lat: number;
  lon: number;
}

export interface DatasetProfile {
  id: string;
  name: string;
  source_file: string;
  relevant: boolean;
  relevance_reason: string;
  description: string;
  date_column?: string | null;
  target_column?: string | null;
  target_name: string;
  unit?: string | null;
  entity_columns: string[];
  signal_columns: string[];
  freq: string;
  region?: DatasetRegion | null;
  row_count: number;
  date_min?: string | null;
  date_max?: string | null;
  suggested_questions: string[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  steps: ToolStep[];
  thoughts?: string;       // streamed Gemini reasoning ("thinking")
  images?: string[];       // attached image data URIs (user messages)
  model?: string;
  error?: string;
  streaming?: boolean;
}

interface ChatState {
  messages: ChatMessage[];
  models: string[];
  model: string;
  sending: boolean;
  datasets: DatasetProfile[];
  activeDataset: DatasetProfile | null;
  uploading: boolean;
}

/** Drives the ReAct chat agent: POST a message, stream tool calls + answer back. */
export function useChat() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    models: [],
    model: "",
    sending: false,
    datasets: [],
    activeDataset: null,
    uploading: false,
  });
  const abortRef = useRef<AbortController | null>(null);

  // Load available models once.
  useEffect(() => {
    fetch(`${API_BASE}/models`)
      .then((r) => r.json())
      .then((d) =>
        setState((s) => ({
          ...s,
          models: d.models ?? [],
          model: s.model || d.default || (d.models?.[0] ?? ""),
        })),
      )
      .catch(() => {});
  }, []);

  const loadDatasets = useCallback(async () => {
    const res = await fetch(`${API_BASE}/datasets`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const datasets = (data.datasets ?? []) as DatasetProfile[];
    setState((s) => ({
      ...s,
      datasets,
      activeDataset: datasets.find((d) => d.id === data.active_id) ?? datasets[0] ?? null,
    }));
    return datasets;
  }, []);

  useEffect(() => {
    loadDatasets().catch(() => {});
  }, [loadDatasets]);

  const setModel = useCallback((model: string) => {
    setState((s) => ({ ...s, model }));
  }, []);

  const uploadDatasets = useCallback(
    async (files: FileList | File[]) => {
      const form = new FormData();
      Array.from(files).forEach((file) => form.append("files", file));
      setState((s) => ({ ...s, uploading: true }));
      try {
        const res = await fetch(`${API_BASE}/datasets/upload`, { method: "POST", body: form });
        if (!res.ok) throw new Error(await responseText(res));
        const data = await res.json();
        await loadDatasets();
        return (data.profiles ?? []) as DatasetProfile[];
      } finally {
        setState((s) => ({ ...s, uploading: false }));
      }
    },
    [loadDatasets],
  );

  const activateDataset = useCallback(
    async (id: string) => {
      const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(id)}/activate`, { method: "POST" });
      if (!res.ok) throw new Error(await responseText(res));
      await loadDatasets();
    },
    [loadDatasets],
  );

  const deleteDataset = useCallback(
    async (id: string) => {
      const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(await responseText(res));
      await loadDatasets();
    },
    [loadDatasets],
  );

  const patchLast = useCallback(
    (fn: (m: ChatMessage) => ChatMessage) => {
      setState((s) => {
        const msgs = [...s.messages];
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === "assistant") {
            msgs[i] = fn(msgs[i]);
            break;
          }
        }
        return { ...s, messages: msgs };
      });
    },
    [],
  );

  const send = useCallback(
    async (text: string, images: string[] = []) => {
      if ((!text.trim() && images.length === 0) || state.sending) return;

      const history = state.messages
        .filter((m) => !m.error)
        .map((m) => ({ role: m.role, content: m.content }));

      setState((s) => ({
        ...s,
        sending: true,
        messages: [
          ...s.messages,
          { role: "user", content: text, steps: [], images: images.length ? images : undefined },
          { role: "assistant", content: "", steps: [], streaming: true },
        ],
      }));

      const ctrl = new AbortController();
      abortRef.current = ctrl;

      try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify({ message: text, history, model: state.model, images }),
          signal: ctrl.signal,
        });
        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split(/\r?\n\r?\n/);
          buffer = frames.pop() ?? "";
          for (const frame of frames) {
            const evt = parseFrame(frame);
            if (evt) applyEvent(evt, patchLast);
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          patchLast((m) => ({
            ...m,
            error: `Couldn't reach the agent (${(err as Error).message}). Is the backend on :8000?`,
            streaming: false,
          }));
        }
      } finally {
        patchLast((m) => ({ ...m, streaming: false }));
        setState((s) => ({ ...s, sending: false }));
      }
    },
    [state.messages, state.model, state.sending, patchLast],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, sending: false }));
  }, []);

  return { ...state, send, stop, setModel, loadDatasets, uploadDatasets, activateDataset, deleteDataset };
}

function applyEvent(
  evt: { type: string; message: string | null; data: Record<string, unknown> },
  patchLast: (fn: (m: ChatMessage) => ChatMessage) => void,
) {
  switch (evt.type) {
    case "tool_call": {
      const tool = (evt.data.tool as string) ?? "tool";
      patchLast((m) => ({
        ...m,
        steps: [
          ...m.steps,
          {
            id: eventId(evt) || fallbackId(tool),
            tool,
            args: (evt.data.args as Record<string, unknown>) ?? {},
            status: "running",
          },
        ],
      }));
      break;
    }
    case "tool_result":
      patchLast((m) => ({
        ...m,
        steps: completeStep(m.steps, eventId(evt), (evt.data.tool as string) ?? "tool", evt.message ?? ""),
      }));
      break;
    case "forecast":
      patchLast((m) => ({
        ...m,
        steps: attachForecast(m.steps, eventId(evt), evt.data.forecast as ForecastPayload),
      }));
      break;
    case "visualization":
      patchLast((m) => ({
        ...m,
        steps: attachVisualization(m.steps, eventId(evt), evt.data.visualization as VisualizationPayload),
      }));
      break;
    case "thought":
      patchLast((m) => ({ ...m, thoughts: (m.thoughts ?? "") + (evt.message ?? "") }));
      break;
    case "answer":
      patchLast((m) => ({ ...m, content: m.content + (evt.message ?? "") }));
      break;
    case "brief":
      patchLast((m) => {
        const full = (evt.data.answer as string) ?? "";
        return full && full.length > m.content.length ? { ...m, content: full } : m;
      });
      break;
    case "error":
      patchLast((m) => ({ ...m, error: evt.message ?? "Agent error", streaming: false }));
      break;
  }
}

function completeStep(steps: ToolStep[], id: string, tool: string, result: string): ToolStep[] {
  const idx = steps.findIndex((s) => s.id === id || (!id && s.tool === tool && s.status === "running"));
  if (idx === -1) return [...steps, { id: id || fallbackId(tool), tool, args: {}, status: "done", result }];
  return steps.map((s, i) => (i === idx ? { ...s, status: "done" as const, result } : s));
}

function attachForecast(steps: ToolStep[], id: string, forecast: ForecastPayload): ToolStep[] {
  const idx = steps.findIndex((s) => s.id === id || s.tool === "forecast_demand");
  if (idx === -1) return steps;
  return steps.map((s, i) => (i === idx ? { ...s, forecast } : s));
}

function attachVisualization(steps: ToolStep[], id: string, visualization: VisualizationPayload): ToolStep[] {
  const idx = steps.findIndex((s) => s.id === id || s.tool === "generate_visualization");
  if (idx === -1) return steps;
  return steps.map((s, i) => (i === idx ? { ...s, visualization } : s));
}

function eventId(evt: { data: Record<string, unknown> }) {
  return (evt.data.id as string) ?? "";
}

function fallbackId(tool: string) {
  return `${tool}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function responseText(res: Response) {
  try {
    const data = await res.json();
    return data.detail ?? `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

function parseFrame(frame: string) {
  let eventName: string | null = null;
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  try {
    const payload = JSON.parse(dataLines.join("\n"));
    return {
      type: eventName ?? "",
      message: payload.message ?? null,
      data: payload.data ?? {},
    };
  } catch {
    return null;
  }
}
