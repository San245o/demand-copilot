"use client";

import { useCallback, useRef, useState } from "react";
import type { EventType, StreamEvent } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

interface StreamState {
  events: StreamEvent[];
  running: boolean;
  awaitingApproval: boolean;
  threadId: string | null;
  error: string | null;
}

const INITIAL: StreamState = {
  events: [],
  running: false,
  awaitingApproval: false,
  threadId: null,
  error: null,
};

/**
 * Drives the crew: streams reasoning over SSE, pauses at the human-approval
 * interrupt, and resumes via POST when the user approves/rejects.
 *
 * Uses fetch + a manual SSE parser (not native EventSource) because (a) the backend
 * sends named events and (b) resume is a POST, which EventSource can't do.
 */
export function useForecastStream() {
  const [state, setState] = useState<StreamState>(INITIAL);
  const abortRef = useRef<AbortController | null>(null);

  const consume = useCallback(async (res: Response) => {
    if (!res.ok || !res.body) throw new Error(`Stream failed: ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        const evt = parseFrame(frame);
        if (!evt) continue;
        setState((s) => applyEvent(s, evt));
      }
    }
  }, []);

  const start = useCallback(
    async (entityId: string, horizon: number) => {
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      setState({ ...INITIAL, running: true });

      const url = `${API_BASE}/forecast/stream?entity_id=${encodeURIComponent(
        entityId,
      )}&horizon=${horizon}`;
      try {
        await consume(
          await fetch(url, {
            headers: { Accept: "text/event-stream" },
            signal: ctrl.signal,
          }),
        );
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setState((s) => ({ ...s, running: false, error: (err as Error).message }));
      }
    },
    [consume],
  );

  const resume = useCallback(
    async (action: "approve" | "reject", feedback?: string) => {
      const threadId = state.threadId;
      if (!threadId) return;
      setState((s) => ({ ...s, awaitingApproval: false, running: true }));
      try {
        await consume(
          await fetch(`${API_BASE}/forecast/resume`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "text/event-stream",
            },
            body: JSON.stringify({ thread_id: threadId, action, feedback }),
          }),
        );
      } catch (err) {
        setState((s) => ({ ...s, running: false, error: (err as Error).message }));
      }
    },
    [consume, state.threadId],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, running: false }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL);
  }, []);

  return { ...state, start, resume, stop, reset };
}

function applyEvent(s: StreamState, evt: StreamEvent): StreamState {
  const next: StreamState = { ...s, events: [...s.events, evt] };
  if (evt.type === "run_start") {
    const tid = (evt.data?.thread_id as string) ?? null;
    if (tid) next.threadId = tid;
  }
  if (evt.type === "interrupt") {
    next.awaitingApproval = true;
    next.running = false;
  }
  if (evt.type === "run_end") {
    next.running = false;
    next.awaitingApproval = false;
  }
  return next;
}

function parseFrame(frame: string): StreamEvent | null {
  let eventName: EventType | null = null;
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim() as EventType;
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
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
