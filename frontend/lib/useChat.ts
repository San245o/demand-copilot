"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export interface ToolStep {
  kind: "tool_call" | "tool_result";
  tool?: string;
  text: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  steps: ToolStep[];
  model?: string;
  error?: string;
  streaming?: boolean;
}

interface ChatState {
  messages: ChatMessage[];
  models: string[];
  model: string;
  sending: boolean;
}

/** Drives the ReAct chat agent: POST a message, stream tool calls + answer back. */
export function useChat() {
  const [state, setState] = useState<ChatState>({
    messages: [],
    models: [],
    model: "",
    sending: false,
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

  const setModel = useCallback((model: string) => {
    setState((s) => ({ ...s, model }));
  }, []);

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
    async (text: string) => {
      if (!text.trim() || state.sending) return;

      const history = state.messages
        .filter((m) => !m.error)
        .map((m) => ({ role: m.role, content: m.content }));

      setState((s) => ({
        ...s,
        sending: true,
        messages: [
          ...s.messages,
          { role: "user", content: text, steps: [] },
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
          body: JSON.stringify({ message: text, history, model: state.model }),
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
          const frames = buffer.split("\n\n");
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

  return { ...state, send, stop, setModel };
}

function applyEvent(
  evt: { type: string; message: string | null; data: Record<string, unknown> },
  patchLast: (fn: (m: ChatMessage) => ChatMessage) => void,
) {
  switch (evt.type) {
    case "tool_call":
      patchLast((m) => ({
        ...m,
        steps: [
          ...m.steps,
          { kind: "tool_call", tool: evt.data.tool as string, text: evt.message ?? "" },
        ],
      }));
      break;
    case "tool_result":
      patchLast((m) => ({
        ...m,
        steps: [
          ...m.steps,
          { kind: "tool_result", tool: evt.data.tool as string, text: evt.message ?? "" },
        ],
      }));
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
