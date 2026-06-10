"use client";

import { useMemo, useState } from "react";
import { useForecastStream } from "@/lib/useForecastStream";
import {
  AGENTS,
  AGENT_LABELS,
  type BriefData,
  type ForecastData,
  type StreamEvent,
  type ValidationData,
} from "@/lib/types";

export default function Home() {
  const [entityId, setEntityId] = useState("store_1");
  const [horizon, setHorizon] = useState(7);
  const {
    events,
    running,
    awaitingApproval,
    error,
    start,
    resume,
    stop,
  } = useForecastStream();

  const forecast = useMemo(
    () => latestData<ForecastData>(events, "forecast"),
    [events],
  );
  const brief = useMemo(() => latestData<BriefData>(events, "brief"), [events]);
  const validation = useMemo(() => {
    const e = [...events].reverse().find((x) => x.agent === "validation" && x.type === "agent_end");
    return e ? (e.data as unknown as ValidationData) : null;
  }, [events]);

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight">
          Demand Forecasting Co-Pilot
        </h1>
        <p className="mt-1 text-sm text-neutral-500">
          A crew of agents senses demand, calls a real forecast model, validates
          it, then waits for your approval before drafting the inventory brief.
        </p>
      </header>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 rounded-xl border border-neutral-200 bg-white p-4">
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-600">Entity</span>
          <input
            value={entityId}
            onChange={(e) => setEntityId(e.target.value)}
            className="w-40 rounded-md border border-neutral-300 px-3 py-1.5"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-neutral-600">Horizon (days)</span>
          <input
            type="number"
            min={1}
            max={90}
            value={horizon}
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="w-28 rounded-md border border-neutral-300 px-3 py-1.5"
          />
        </label>
        <button
          onClick={() => start(entityId, horizon)}
          disabled={running}
          className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {running ? "Running…" : "Run forecast"}
        </button>
        {running && (
          <button
            onClick={stop}
            className="rounded-md border border-neutral-300 px-4 py-2 text-sm"
          >
            Stop
          </button>
        )}
      </div>

      {error && (
        <p className="mt-4 rounded-md bg-red-50 px-4 py-2 text-sm text-red-700">
          {error} — is the backend running on :8000?
        </p>
      )}

      {/* Agent pipeline */}
      <section className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-4">
        {AGENTS.map((agent) => (
          <AgentColumn
            key={agent}
            agent={agent}
            events={events.filter((e) => e.agent === agent)}
          />
        ))}
      </section>

      {/* Human-in-the-loop approval gate */}
      {awaitingApproval && (
        <ApprovalGate
          forecast={forecast}
          validation={validation}
          onApprove={() => resume("approve")}
          onReject={(fb) => resume("reject", fb)}
        />
      )}

      {/* Results */}
      <section className="mt-8 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {forecast && <ForecastPanel forecast={forecast} />}
        {brief && <BriefPanel brief={brief} />}
      </section>

      {events.length === 0 && !running && (
        <p className="mt-10 text-center text-sm text-neutral-400">
          Run a forecast to watch the crew work.
        </p>
      )}
    </main>
  );
}

function ApprovalGate({
  forecast,
  validation,
  onApprove,
  onReject,
}: {
  forecast: ForecastData | null;
  validation: ValidationData | null;
  onApprove: () => void;
  onReject: (feedback: string) => void;
}) {
  const [feedback, setFeedback] = useState("");
  const flagged = validation?.needs_review;
  return (
    <section
      className={`mt-8 rounded-xl border-2 p-5 ${
        flagged ? "border-amber-400 bg-amber-50" : "border-blue-300 bg-blue-50"
      }`}
    >
      <h3 className="text-sm font-semibold">Human approval required</h3>
      <p className="mt-1 text-sm text-neutral-700">
        The crew produced a {forecast?.points.length ?? 0}-step forecast
        {validation && (
          <>
            {" "}
            with a CI width of{" "}
            <strong>{validation.ci_width_pct}%</strong> of the mean.
          </>
        )}{" "}
        {flagged ? (
          <span className="font-medium text-amber-700">
            Flagged for review (wide interval or high value).
          </span>
        ) : (
          "Within tolerance."
        )}
      </p>
      <textarea
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        placeholder="Optional feedback if rejecting…"
        className="mt-3 w-full rounded-md border border-neutral-300 px-3 py-2 text-sm"
        rows={2}
      />
      <div className="mt-3 flex gap-3">
        <button
          onClick={onApprove}
          className="rounded-md bg-green-600 px-4 py-2 text-sm font-medium text-white"
        >
          Approve → draft brief
        </button>
        <button
          onClick={() => onReject(feedback || "Rejected by reviewer")}
          className="rounded-md border border-red-300 bg-white px-4 py-2 text-sm font-medium text-red-700"
        >
          Reject
        </button>
      </div>
    </section>
  );
}

function AgentColumn({ agent, events }: { agent: string; events: StreamEvent[] }) {
  const active = events.some((e) => e.type === "agent_start");
  const done = events.some((e) => e.type === "agent_end");
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold">{AGENT_LABELS[agent] ?? agent}</h3>
        <StatusDot active={active} done={done} />
      </div>
      <ul className="space-y-1.5">
        {events
          .filter((e) => ["thought", "tool_call", "tool_result"].includes(e.type))
          .map((e, i) => (
            <li key={i} className="text-xs leading-snug">
              <span
                className={
                  e.type === "thought"
                    ? "text-neutral-600"
                    : e.type === "tool_call"
                      ? "font-medium text-blue-700"
                      : "text-green-700"
                }
              >
                {e.type === "tool_call" ? "→ " : e.type === "tool_result" ? "✓ " : "• "}
                {e.message}
              </span>
            </li>
          ))}
      </ul>
    </div>
  );
}

function StatusDot({ active, done }: { active: boolean; done: boolean }) {
  const color = done
    ? "bg-green-500"
    : active
      ? "bg-amber-400 animate-pulse"
      : "bg-neutral-300";
  return <span className={`h-2.5 w-2.5 rounded-full ${color}`} />;
}

function ForecastPanel({ forecast }: { forecast: ForecastData }) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-4">
      <h3 className="mb-3 text-sm font-semibold">
        Forecast · {forecast.model} · {forecast.level}% CI
      </h3>
      <table className="w-full text-xs">
        <thead className="text-neutral-500">
          <tr className="text-left">
            <th className="py-1">Date</th>
            <th className="py-1 text-right">Lower</th>
            <th className="py-1 text-right">Mean</th>
            <th className="py-1 text-right">Upper</th>
          </tr>
        </thead>
        <tbody>
          {forecast.points.map((p) => (
            <tr key={p.date} className="border-t border-neutral-100">
              <td className="py-1">{p.date}</td>
              <td className="py-1 text-right text-neutral-500">
                {Math.round(p.lower).toLocaleString()}
              </td>
              <td className="py-1 text-right font-medium">
                {Math.round(p.mean).toLocaleString()}
              </td>
              <td className="py-1 text-right text-neutral-500">
                {Math.round(p.upper).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BriefPanel({ brief }: { brief: BriefData }) {
  return (
    <div className="rounded-xl border border-neutral-200 bg-white p-4">
      <h3 className="mb-2 text-sm font-semibold">Forecast brief</h3>
      <p className="text-sm font-medium">{brief.headline}</p>
      <p className="mt-2 text-sm text-neutral-700">{brief.recommendation}</p>
      {(brief.reorder_point || brief.horizon_total) && (
        <div className="mt-3 flex gap-4 text-sm">
          {brief.horizon_total != null && (
            <div>
              <span className="text-neutral-500">Horizon total: </span>
              <strong>{brief.horizon_total.toLocaleString()}</strong>
            </div>
          )}
          {brief.reorder_point != null && (
            <div>
              <span className="text-neutral-500">Reorder point: </span>
              <strong>{brief.reorder_point.toLocaleString()}</strong>
            </div>
          )}
        </div>
      )}
      <div className="mt-3 flex flex-wrap gap-1.5">
        <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs">
          confidence: {brief.confidence}
        </span>
        {brief.drivers?.map((d) => (
          <span
            key={d}
            className="rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-700"
          >
            {d}
          </span>
        ))}
      </div>
    </div>
  );
}

function latestData<T>(events: StreamEvent[], type: string): T | null {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].type === type) return events[i].data as T;
  }
  return null;
}
