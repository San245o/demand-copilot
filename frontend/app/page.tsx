"use client";

import { useEffect, useRef, useState } from "react";
import { useChat, type ChatMessage, type ToolStep } from "@/lib/useChat";

const SUGGESTIONS = [
  "Forecast demand for store_1 over the next 7 days and explain the drivers.",
  "Which stores can I forecast?",
  "How do promotions affect demand? Check the playbooks.",
  "What's the recent sales trend for store_3?",
];

export default function Home() {
  const { messages, models, model, sending, send, stop, setModel } = useChat();
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const submit = (text: string) => {
    if (!text.trim()) return;
    send(text);
    setInput("");
  };

  return (
    <div className="flex h-screen flex-col bg-neutral-50">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-neutral-200 bg-white px-6 py-3">
        <div>
          <h1 className="text-sm font-semibold text-neutral-900">
            Demand Forecasting Co-Pilot
          </h1>
          <p className="text-xs text-neutral-500">
            Ask anything. The agent reasons, calls tools, and answers.
          </p>
        </div>
        <label className="flex items-center gap-2 text-xs text-neutral-600">
          Model
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="rounded-md border border-neutral-300 bg-white px-2 py-1 text-xs text-neutral-900"
          >
            {models.length === 0 && <option>loading…</option>}
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      </header>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-3xl space-y-5">
          {messages.length === 0 && (
            <Welcome onPick={submit} />
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}
        </div>
      </div>

      {/* Composer */}
      <div className="border-t border-neutral-200 bg-white px-4 py-3">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
          className="mx-auto flex max-w-3xl items-end gap-2"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(input);
              }
            }}
            rows={1}
            placeholder="Ask about demand, forecasts, drivers…"
            className="max-h-40 flex-1 resize-none rounded-xl border border-neutral-300 px-4 py-2.5 text-sm text-neutral-900 outline-none focus:border-neutral-500"
          />
          {sending ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-xl border border-neutral-300 px-4 py-2.5 text-sm font-medium text-neutral-700"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="rounded-xl bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-30"
            >
              Send
            </button>
          )}
        </form>
      </div>
    </div>
  );
}

function Welcome({ onPick }: { onPick: (t: string) => void }) {
  return (
    <div className="mt-10 text-center">
      <h2 className="text-lg font-semibold text-neutral-800">
        What would you like to know?
      </h2>
      <p className="mt-1 text-sm text-neutral-500">
        The agent can forecast demand, search the web, check weather/holidays, and
        consult planning playbooks.
      </p>
      <div className="mx-auto mt-6 grid max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="rounded-xl border border-neutral-200 bg-white px-4 py-3 text-left text-sm text-neutral-700 hover:border-neutral-400"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl bg-neutral-900 px-4 py-2.5 text-sm text-white">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-[90%] space-y-2">
        {message.steps.length > 0 && <ThoughtTrail steps={message.steps} streaming={!!message.streaming} />}
        {message.content && (
          <div className="whitespace-pre-wrap rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-800">
            {message.content}
            {message.streaming && <Cursor />}
          </div>
        )}
        {!message.content && message.streaming && message.steps.length === 0 && (
          <div className="rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-400">
            thinking<Cursor />
          </div>
        )}
        {message.error && (
          <div className="rounded-xl bg-red-50 px-4 py-2.5 text-sm text-red-700">
            {message.error}
          </div>
        )}
      </div>
    </div>
  );
}

function ThoughtTrail({ steps, streaming }: { steps: ToolStep[]; streaming: boolean }) {
  const [open, setOpen] = useState(true);
  const calls = steps.filter((s) => s.kind === "tool_call").length;
  return (
    <div className="rounded-xl border border-neutral-200 bg-neutral-100/60">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs font-medium text-neutral-600"
      >
        <span className={`transition ${open ? "rotate-90" : ""}`}>▸</span>
        {streaming ? "Thinking…" : "Reasoning"} · {calls} tool
        {calls === 1 ? "" : "s"} used
      </button>
      {open && (
        <ul className="space-y-1.5 px-3 pb-3">
          {steps.map((s, i) => (
            <li key={i} className="text-xs leading-snug">
              {s.kind === "tool_call" ? (
                <span className="font-medium text-blue-700">→ {s.text}</span>
              ) : (
                <span className="text-green-700">
                  ✓ <span className="text-neutral-500">{s.tool}:</span>{" "}
                  {truncate(s.text, 180)}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Cursor() {
  return <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-neutral-400 align-middle" />;
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
