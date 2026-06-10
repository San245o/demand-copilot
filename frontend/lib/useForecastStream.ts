"use client";

import { useCallback, useRef, useState } from "react";
import type { EventType, StreamEvent } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

interface StreamState {
  events: StreamEvent[];
  running: boolean;
  error: string | null;
}

/**
 * Streams the crew's reasoning over SSE.
 *
 * We use fetch + a manual SSE parser (not the native EventSource) because the
 * backend sends *named* events and we want them reconstructed into a single typed
 * StreamEvent list with the event name folded into `type`.
 */
export function useForecastStream() {
  const [state, setState] = useState<StreamState>({
    events: [],
    running: false,
    error: null,
  });
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(async (entityId: string, horizon: number) => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setState({ events: [], running: true, error: null });

    const url = `${API_BASE}/forecast/stream?entity_id=${encodeURIComponent(
      entityId,
    )}&horizon=${horizon}`;

    try {
      const res = await fetch(url, {
        headers: { Accept: "text/event-stream" },
        signal: ctrl.signal,
      });
      if (!res.ok || !res.body) {
        throw new Error(`Stream failed: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const evt = parseFrame(frame);
          if (evt) {
            setState((s) => ({ ...s, events: [...s.events, evt] }));
            if (evt.type === "run_end") {
              setState((s) => ({ ...s, running: false }));
            }
          }
        }
      }
      setState((s) => ({ ...s, running: false }));
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setState((s) => ({
        ...s,
        running: false,
        error: (err as Error).message,
      }));
    }
  }, []);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, running: false }));
  }, []);

  return { ...state, start, stop };
}

function parseFrame(frame: string): StreamEvent | null {
  let eventName: EventType | null = null;
  const dataLines: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim() as EventType;
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (dataLines.length === 0) return null;
  try {
    const payload = JSON.parse(dataLines.join("\n"));
    return {
      type: (eventName ?? "thought") as EventType,
      agent: payload.agent ?? null,
      message: payload.message ?? null,
      data: payload.data ?? {},
    };
  } catch {
    return null;
  }
}
