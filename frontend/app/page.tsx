"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  useChat,
  type ChatMessage,
  type DatasetProfile,
  type ForecastPayload,
  type ToolStep,
  type VisualizationPayload,
} from "@/lib/useChat";


const FALLBACK_SUGGESTIONS = [
  "Forecast demand for store_1 over the next 7 days and explain the drivers.",
  "Which stores can I forecast?",
  "How do promotions affect demand? Check the playbooks.",
  "What's the recent sales trend for store_3?",
];

export default function Home() {
  const {
    messages,
    models,
    model,
    sending,
    datasets,
    activeDataset,
    uploading,
    send,
    stop,
    setModel,
    uploadDatasets,
    activateDataset,
    deleteDataset,
  } = useChat();
  const [input, setInput] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [datasetOpen, setDatasetOpen] = useState(false);
  const [modelOpen, setModelOpen] = useState(false);
  const [uploadProfiles, setUploadProfiles] = useState<DatasetProfile[]>([]);
  const [uploadError, setUploadError] = useState("");
  const [imageError, setImageError] = useState("");
  const [dragImage, setDragImage] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const imageRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  const autoGrow = (el: HTMLTextAreaElement | null) => {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const submit = (text: string) => {
    if (!text.trim() && images.length === 0) return;
    send(text, images);
    setInput("");
    setImages([]);
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const handleUpload = async (files: FileList | File[]) => {
    setUploadError("");
    try {
      const profiles = await uploadDatasets(files);
      setUploadProfiles(profiles);
    } catch (err) {
      setUploadError((err as Error).message);
    }
  };

  const addImages = async (files: FileList | File[]) => {
    setImageError("");
    const picked = Array.from(files).filter((f) => f.type.startsWith("image/"));
    if (!picked.length) return;
    try {
      const urls = await Promise.all(picked.slice(0, 6).map((f) => fileToDataUrl(f)));
      setImages((prev) => [...prev, ...urls].slice(0, 6));
    } catch {
      setImageError("Couldn't read that image.");
    }
  };

  return (
    <div className="flex h-screen flex-col bg-neutral-50">
      <header className="relative border-b border-neutral-200 bg-white px-4 py-3 text-center sm:px-6">
        <h1 className="text-sm font-semibold text-neutral-900">Demand Forecasting Co-Pilot</h1>
        <p className="mt-0.5 text-xs text-neutral-500">Ask about your sales data in plain English — it predicts demand and explains what&apos;s driving it.</p>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-6 pb-44">
        <div className="mx-auto max-w-3xl space-y-5">
          {messages.length === 0 && (
            <Welcome dataset={activeDataset} onPick={submit} />
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} profile={activeDataset} />
          ))}
        </div>
      </div>

      <div className="fixed inset-x-0 bottom-4 z-20 px-4">
        <div className="relative mx-auto max-w-3xl">
          {datasetOpen && (
            <DatasetPopover
              datasets={datasets}
              activeId={activeDataset?.id}
              uploading={uploading}
              uploadProfiles={uploadProfiles}
              uploadError={uploadError}
              onUpload={handleUpload}
              onActivate={activateDataset}
              onDelete={deleteDataset}
              className="absolute bottom-full right-0 mb-3"
              compactUpload
            />
          )}
          <input
            ref={uploadRef}
            type="file"
            accept=".csv,.xlsx"
            multiple
            hidden
            onChange={(e) => e.target.files && handleUpload(e.target.files)}
          />
          <input
            ref={imageRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) addImages(e.target.files);
              e.target.value = "";
            }}
          />
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
          onDragOver={(e) => {
            if (e.dataTransfer.types.includes("Files")) {
              e.preventDefault();
              setDragImage(true);
            }
          }}
          onDragLeave={() => setDragImage(false)}
          onDrop={(e) => {
            if (e.dataTransfer.files.length) {
              e.preventDefault();
              setDragImage(false);
              addImages(e.dataTransfer.files);
            }
          }}
          className={`chat-composer rounded-3xl bg-white/95 p-2 shadow-2xl shadow-neutral-900/10 backdrop-blur ${dragImage ? "ring-2 ring-blue-400" : ""}`}
        >
          {(images.length > 0 || imageError) && (
            <div className="flex flex-wrap items-center gap-2 px-2 pb-2 pt-1">
              {images.map((src, i) => (
                <div key={i} className="group relative h-16 w-16 overflow-hidden rounded-xl border border-neutral-200">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={src} alt="attachment" className="h-full w-full object-cover" />
                  <button
                    type="button"
                    onClick={() => setImages((prev) => prev.filter((_, j) => j !== i))}
                    className="absolute right-0.5 top-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-neutral-900/70 text-xs text-white opacity-0 transition group-hover:opacity-100"
                    title="Remove"
                  >
                    ×
                  </button>
                </div>
              ))}
              {imageError && <span className="text-xs text-red-600">{imageError}</span>}
            </div>
          )}
          <div className="flex items-end gap-2">
          <button
            type="button"
            onClick={() => imageRef.current?.click()}
            className="mb-1 rounded-full p-2 text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900"
            title="Attach image"
          >
            <ImageIcon />
          </button>
          <button
            type="button"
            onClick={() => uploadRef.current?.click()}
            className="mb-1 rounded-full p-2 text-neutral-500 hover:bg-neutral-100 hover:text-neutral-900 disabled:opacity-40"
            disabled={uploading}
            title="Upload CSV/XLSX dataset"
          >
            <PaperclipIcon />
          </button>
          <textarea
            ref={taRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              autoGrow(e.target);
            }}
            onPaste={(e) => {
              const files = Array.from(e.clipboardData.files).filter((f) => f.type.startsWith("image/"));
              if (files.length) {
                e.preventDefault();
                addImages(files);
              }
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(input);
              }
            }}
            rows={1}
            placeholder="Ask about demand, forecasts, drivers... (paste or drop an image)"
            className="flex-1 resize-none overflow-y-auto rounded-2xl bg-transparent px-2 py-2.5 text-sm text-neutral-900 outline-none"
            style={{ maxHeight: 200 }}
          />
          <Tooltip label={`Dataset: ${activeDataset?.name ?? "—"}`}>
            <button
              type="button"
              onClick={() => { setDatasetOpen((v) => !v); setModelOpen(false); }}
              className={`mb-1 rounded-full p-2 hover:bg-neutral-100 ${datasetOpen ? "bg-neutral-100 text-neutral-900" : "text-neutral-500 hover:text-neutral-900"}`}
              aria-label="Switch dataset"
            >
              <DatasetIcon />
            </button>
          </Tooltip>
          <div className="relative mb-1">
            <Tooltip label={`Model: ${model || "—"}`}>
              <button
                type="button"
                onClick={() => { setModelOpen((v) => !v); setDatasetOpen(false); }}
                className={`rounded-full p-2 hover:bg-neutral-100 ${modelOpen ? "bg-neutral-100 text-neutral-900" : "text-neutral-500 hover:text-neutral-900"}`}
                aria-label="Choose model"
              >
                <ModelIcon />
              </button>
            </Tooltip>
            {modelOpen && (
              <ModelMenu
                models={models}
                model={model}
                onPick={(m) => { setModel(m); setModelOpen(false); }}
                onClose={() => setModelOpen(false)}
              />
            )}
          </div>
          {sending ? (
            <button type="button" onClick={stop} className="rounded-2xl bg-neutral-100 px-4 py-2.5 text-sm font-medium text-neutral-700">Stop</button>
          ) : (
            <button type="submit" disabled={!input.trim() && images.length === 0} className="rounded-2xl bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-30">Send</button>
          )}
          </div>
        </form>
        </div>
      </div>
    </div>
  );
}

function DatasetPopover({
  datasets,
  activeId,
  uploading,
  uploadProfiles,
  uploadError,
  onUpload,
  onActivate,
  onDelete,
  className = "",
  compactUpload = false,
}: {
  datasets: DatasetProfile[];
  activeId?: string;
  uploading: boolean;
  uploadProfiles: DatasetProfile[];
  uploadError: string;
  onUpload: (files: FileList | File[]) => void;
  onActivate: (id: string) => void;
  onDelete: (id: string) => void;
  className?: string;
  compactUpload?: boolean;
}) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  return (
    <div className={`${className} z-30 w-[min(92vw,520px)] rounded-2xl border border-neutral-200 bg-white p-3 shadow-xl`}>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-neutral-900">Datasets</h2>
        <span className="text-xs text-neutral-500">Each file is standalone</span>
      </div>
      <div className="max-h-64 space-y-2 overflow-y-auto pr-1">
        {datasets.map((d) => (
          <button
            key={d.id}
            type="button"
            onClick={() => onActivate(d.id)}
            className={`w-full rounded-xl border p-3 text-left transition ${d.id === activeId ? "border-neutral-900 bg-neutral-50" : "border-neutral-200 hover:border-neutral-400"}`}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-medium text-neutral-900">{d.name}</div>
                <div className="mt-0.5 text-xs text-neutral-500">{d.row_count.toLocaleString()} rows · {d.region?.country_code ?? "region ?"}</div>
              </div>
              <div className="flex items-center gap-2">
                <span className={`rounded-full px-2 py-0.5 text-[11px] ${d.relevant ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
                  {d.relevant ? "looks like demand data" : "flagged"}
                </span>
                {d.id.startsWith("upload-") && (
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      onDelete(d.id);
                    }}
                    className="rounded-md px-2 py-1 text-xs text-red-600 hover:bg-red-50"
                  >
                    delete
                  </span>
                )}
              </div>
            </div>
            <p className="mt-2 line-clamp-2 text-xs text-neutral-600">{d.relevance_reason}</p>
          </button>
        ))}
      </div>
      <input
        ref={fileRef}
        type="file"
        accept=".csv,.xlsx"
        multiple
        hidden
        onChange={(e) => e.target.files && onUpload(e.target.files)}
      />

      {!compactUpload && <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files.length) onUpload(e.dataTransfer.files);
        }}
        className={`mt-3 rounded-xl border border-dashed p-4 text-center ${dragging ? "border-neutral-900 bg-neutral-50" : "border-neutral-300"}`}
      >
        <p className="text-sm font-medium text-neutral-800">Drop CSV/XLSX files here</p>
        <p className="mt-1 text-xs text-neutral-500">Gemini profiles headers and sample rows; heuristic fallback works offline.</p>
        <button type="button" onClick={() => fileRef.current?.click()} className="mt-3 rounded-lg bg-neutral-900 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40" disabled={uploading}>
          {uploading ? "Uploading..." : "Choose files"}
        </button>
      </div>}
      {compactUpload && (
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="mt-3 w-full rounded-xl bg-neutral-900 px-3 py-2 text-xs font-medium text-white disabled:opacity-40"
          disabled={uploading}
        >
          {uploading ? "Uploading..." : "Upload CSV/XLSX"}
        </button>
      )}
      {uploadError && <div className="mt-2 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">{uploadError}</div>}
      {uploadProfiles.length > 0 && (
        <div className="mt-2 space-y-1">
          {uploadProfiles.map((p) => (
            <div key={p.id} className={`rounded-lg px-3 py-2 text-xs ${p.relevant ? "bg-emerald-50 text-emerald-800" : "bg-amber-50 text-amber-800"}`}>
              {p.source_file}: {p.relevance_reason}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Welcome({ dataset, onPick }: { dataset: DatasetProfile | null; onPick: (t: string) => void }) {
  const suggestions = dataset?.suggested_questions?.length ? dataset.suggested_questions : FALLBACK_SUGGESTIONS;
  return (
    <div className="mt-10 text-center">
      <div className="mx-auto mb-4 inline-flex rounded-full border border-neutral-200 bg-white px-3 py-1 text-xs text-neutral-600">
        Using: {dataset?.name ?? "loading"}
      </div>
      <h2 className="text-lg font-semibold text-neutral-800">Ask about your sales data</h2>
      <p className="mx-auto mt-2 max-w-xl text-sm text-neutral-500">
        Type a question in plain English. It predicts future demand, shows it on a chart, and
        explains the drivers — using your data plus signals like promotions, weather, and holidays.
        You can also attach an image (e.g. a chart or shelf photo) for it to read.
      </p>
      <p className="mt-5 text-xs font-medium uppercase tracking-wide text-neutral-400">Try one of these</p>
      <div className="mx-auto mt-2 grid max-w-2xl grid-cols-1 gap-2 sm:grid-cols-2">
        {suggestions.map((s) => (
          <button key={s} onClick={() => onPick(s)} className="rounded-xl border border-neutral-200 bg-white px-4 py-3 text-left text-sm text-neutral-700 hover:border-neutral-400">
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ message, profile }: { message: ChatMessage; profile: DatasetProfile | null }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] space-y-2">
          {message.images && message.images.length > 0 && (
            <div className="flex flex-wrap justify-end gap-2">
              {message.images.map((src, i) => (
                <Lightbox key={i} src={src} className="h-32 w-32 rounded-2xl border border-neutral-200 object-cover" />
              ))}
            </div>
          )}
          {message.content && (
            <div className="rounded-2xl bg-neutral-900 px-4 py-2.5 text-sm text-white">{message.content}</div>
          )}
        </div>
      </div>
    );
  }

  const hasReasoning = message.steps.length > 0 || !!message.thoughts;
  return (
    <div className="flex justify-start">
      <div className="w-full max-w-[92%] space-y-2">
        {hasReasoning && (
          <ReasoningPanel thoughts={message.thoughts} steps={message.steps} streaming={!!message.streaming} profile={profile} />
        )}
        {message.content && (
          <div className="rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-800 shadow-sm markdown-content">
            <div className="mb-2 inline-flex rounded-full bg-neutral-100 px-2 py-0.5 text-[11px] font-medium text-neutral-500">Gemini agent</div>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                table: ({ children }) => <div className="markdown-table-wrap"><table>{children}</table></div>,
              }}
            >
              {message.content}
            </ReactMarkdown>
            {message.streaming && <Cursor />}
          </div>
        )}
        {message.steps.some((s) => s.visualization) && (
          <div className="space-y-2">
            {message.steps.filter((s) => s.visualization).map((s) => (
              <VisualizationImage key={`${s.id}-final-viz`} visualization={s.visualization!} />
            ))}
          </div>
        )}
        {!message.content && message.streaming && !hasReasoning && (
          <div className="flex items-center gap-2 rounded-2xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-400">
            <Spinner /> Thinking…
          </div>
        )}
        {message.error && <div className="rounded-xl bg-red-50 px-4 py-2.5 text-sm text-red-700">{message.error}</div>}
      </div>
    </div>
  );
}

/** Claude/GPT-style reasoning panel: a calm, collapsible "thinking" trace with the
 *  model's streamed reasoning followed by humanized tool steps. Auto-collapses once
 *  the final answer starts streaming. */
function ReasoningPanel({ thoughts, steps, streaming, profile }: { thoughts?: string; steps: ToolStep[]; streaming: boolean; profile: DatasetProfile | null }) {
  const [open, setOpen] = useState(true);
  const wasStreaming = useRef(streaming);
  useEffect(() => {
    if (wasStreaming.current && !streaming) setOpen(false);
    wasStreaming.current = streaming;
  }, [streaming]);

  return (
    <div className="reasoning-panel rounded-xl border border-neutral-200 bg-neutral-50/80">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-xs font-medium text-neutral-600"
      >
        <Chevron open={open} />
        {streaming ? (
          <span className="shimmer-text">Thinking…</span>
        ) : (
          <span>Thought process</span>
        )}
        {steps.length > 0 && (
          <span className="text-neutral-400">· {steps.length} step{steps.length === 1 ? "" : "s"}</span>
        )}
      </button>
      {open && (
        <div className="space-y-3 px-3 pb-3">
          {thoughts && (
            <div className="reasoning-thoughts markdown-content text-[13px] leading-relaxed text-neutral-500">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{thoughts}</ReactMarkdown>
            </div>
          )}
          {steps.length > 0 && (
            <ol className="reasoning-steps space-y-1.5">
              {steps.map((step) => <StepRow key={step.id} step={step} profile={profile} />)}
            </ol>
          )}
        </div>
      )}
    </div>
  );
}

function StepRow({ step, profile }: { step: ToolStep; profile: DatasetProfile | null }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <li className="reasoning-step">
      <div className="flex items-start gap-2.5 rounded-lg px-2 py-1.5 hover:bg-white">
        <span className={`step-dot ${step.status === "running" ? "is-running" : "is-done"}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-[13px] text-neutral-700">
            <span className="text-neutral-400">{toolIcon(step.tool)}</span>
            <span className="font-medium">{describeStep(step.tool, step.args)}</span>
          </div>
          {step.forecast && <ForecastCaption forecast={step.forecast} profile={profile} />}
          {/* The full chart renders once, below the answer — not duplicated here. */}
          {step.result && (
            <div className="mt-1">
              <button onClick={() => setExpanded((v) => !v)} className="text-[11px] text-neutral-400 hover:text-neutral-700">
                {expanded ? "Hide" : "Show"} details
              </button>
              {expanded && (
                <pre className="mt-1.5 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg bg-neutral-900 p-3 text-[11px] text-neutral-100">{step.result}</pre>
              )}
            </div>
          )}
        </div>
        <span className="mt-0.5 shrink-0 text-neutral-400">
          {step.status === "running" ? <Spinner /> : <span className="text-emerald-500">✓</span>}
        </span>
      </div>
    </li>
  );
}

/** A short, human-readable summary of a tool call — reads like Claude/GPT's
 *  "Searched for…", "Analyzed…" lines instead of raw function(args). */
function prettyEntity(id: string): string {
  if (!id) return "this item";
  return id
    .split("|")
    .map((p) => p.trim())
    .map((p) => (p.includes("_") ? p.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) : p))
    .join(" · ");
}

function friendlyModel(model: string): string {
  const m = (model || "").toLowerCase();
  if (m.includes("fallback")) return "rough baseline (limited history)";
  if (m.includes("ets")) return "ETS model";
  if (m.includes("arima")) return "ARIMA model";
  if (m.includes("theta")) return "Theta model";
  if (m.includes("naive") || m.includes("seasonal")) return "seasonal baseline";
  return model;
}

function describeStep(tool: string, args: Record<string, unknown>): string {
  const a = args ?? {};
  const s = (k: string) => (a[k] != null ? String(a[k]) : "");
  switch (tool) {
    case "forecast_demand":
      return `Forecasting demand for ${a.entity_id ? prettyEntity(s("entity_id")) : "an entity"}${a.horizon ? ` over ${s("horizon")} days` : ""}`;
    case "generate_visualization":
      return `Creating a ${(s("chart_type") || "chart").replace(/_/g, " ")} chart for ${a.entity_id ? prettyEntity(s("entity_id")) : "the dataset"}`;
    case "get_recent_sales":
      return `Reading recent sales for ${a.entity_id ? prettyEntity(s("entity_id")) : "an entity"}`;
    case "list_entities":
      return "Listing forecastable entities";
    case "describe_dataset":
      return "Inspecting the active dataset";
    case "search_web":
      return a.query ? `Searching the web for “${s("query")}”` : "Searching the web";
    case "search_playbooks":
      return a.query ? `Consulting playbooks for “${s("query")}”` : "Consulting planning playbooks";
    case "get_weather":
      return a.start ? `Checking weather (${s("start")} → ${s("end")})` : "Checking weather";
    case "get_holidays":
      return a.year ? `Looking up holidays for ${s("year")}` : "Looking up holidays";
    case "get_macro_signal":
      return "Checking a macro signal";
    default:
      return tool;
  }
}

function ForecastCaption({ forecast, profile }: { forecast: ForecastPayload; profile: DatasetProfile | null }) {
  const round = (n: number) => Math.round(n).toLocaleString();
  const total = forecast.points.reduce((s, p) => s + p.mean, 0);
  const low = forecast.points.reduce((s, p) => s + p.lower, 0);
  const high = forecast.points.reduce((s, p) => s + p.upper, 0);
  const noun = profile?.unit || profile?.target_name || "units";
  return (
    <div className="mt-1.5">
      <div className="text-[13px] font-semibold text-neutral-800">
        ≈ {round(total)} {noun} over the next {forecast.horizon} days
      </div>
      <div className="text-[12px] text-neutral-500">
        likely {round(low)}–{round(high)} · {friendlyModel(forecast.model)}, {forecast.level}% confidence
      </div>
    </div>
  );
}

function VisualizationImage({ visualization, compact = false }: { visualization: VisualizationPayload; compact?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={`group overflow-hidden rounded-2xl border border-neutral-200 bg-white text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-lg ${compact ? "mt-1 w-full max-w-md" : "w-full max-w-xl p-2"}`}
      >
        <div className="flex items-center justify-between gap-2 px-2 py-1 text-[11px] font-semibold text-neutral-700">
          <span className="truncate">{visualization.title}</span>
          <span className="shrink-0 rounded-full bg-neutral-100 px-2 py-0.5 font-normal text-neutral-500">open</span>
        </div>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={visualization.image} alt={visualization.title} className="w-full rounded-xl bg-white object-contain" />
      </button>
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-neutral-950/60 p-4 backdrop-blur-sm" onClick={() => setOpen(false)}>
          <figure className="max-h-[90vh] w-full max-w-5xl overflow-auto rounded-3xl bg-white p-4 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <figcaption className="mb-3 flex items-center justify-between gap-3 text-sm font-semibold text-neutral-800">
              <span>{visualization.title}</span>
              <div className="flex items-center gap-3">
                {visualization.source && <span className="text-xs font-normal text-neutral-400">{visualization.source}</span>}
                <button type="button" onClick={() => setOpen(false)} className="rounded-full bg-neutral-100 px-3 py-1 text-xs font-medium text-neutral-600 hover:bg-neutral-200">Close</button>
              </div>
            </figcaption>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={visualization.image} alt={visualization.title} className="w-full rounded-2xl bg-white" />
          </figure>
        </div>
      )}
    </>
  );
}

/** A clickable image thumbnail that opens a full-size lightbox (used for the
 *  images a user attaches to a message). */
function Lightbox({ src, className = "" }: { src: string; className?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt="attachment" className={`cursor-zoom-in ${className}`} onClick={() => setOpen(true)} />
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-neutral-950/70 p-4 backdrop-blur-sm" onClick={() => setOpen(false)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={src} alt="attachment" className="max-h-[90vh] max-w-[90vw] rounded-2xl shadow-2xl" onClick={(e) => e.stopPropagation()} />
        </div>
      )}
    </>
  );
}

function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-800" />;
}

function Chevron({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 24 24" className={`h-3.5 w-3.5 transition-transform ${open ? "rotate-90" : ""}`} fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}

function PaperclipIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.4 11.6 12 21a6 6 0 0 1-8.5-8.5l9.9-9.9a4 4 0 0 1 5.7 5.7l-10 10a2 2 0 0 1-2.8-2.8l9.4-9.4" />
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <circle cx="9" cy="9" r="2" />
      <path d="m21 15-3.6-3.6a2 2 0 0 0-2.8 0L6 20" />
    </svg>
  );
}

function DatasetIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5" />
      <path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" />
    </svg>
  );
}

function ModelIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 3l1.9 4.6L18.5 9l-4.6 1.9L12 15.5l-1.9-4.6L5.5 9l4.6-1.4z" />
      <path d="M19 14l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z" />
    </svg>
  );
}

/** Lightweight hover tooltip (no deps) used for the compact composer icons. */
function Tooltip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="group/tt relative inline-flex">
      {children}
      <span className="pointer-events-none absolute bottom-full left-1/2 z-40 mb-2 -translate-x-1/2 whitespace-nowrap rounded-md bg-neutral-900 px-2 py-1 text-[11px] font-medium text-white opacity-0 shadow-lg transition group-hover/tt:opacity-100">
        {label}
      </span>
    </span>
  );
}

/** Popover menu for picking the model, opened from the compact model icon. */
function ModelMenu({
  models,
  model,
  onPick,
  onClose,
}: {
  models: string[];
  model: string;
  onPick: (m: string) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [onClose]);
  return (
    <div ref={ref} className="absolute bottom-full right-0 z-40 mb-2 w-56 rounded-xl border border-neutral-200 bg-white p-1 shadow-xl">
      <div className="px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-neutral-400">Model</div>
      {models.length === 0 && <div className="px-2 py-2 text-xs text-neutral-400">loading…</div>}
      {models.map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onPick(m)}
          className={`flex w-full items-center justify-between gap-2 rounded-lg px-2 py-2 text-left text-xs hover:bg-neutral-100 ${m === model ? "font-semibold text-neutral-900" : "text-neutral-600"}`}
        >
          <span className="truncate">{m}</span>
          {m === model && <span className="text-emerald-500">✓</span>}
        </button>
      ))}
    </div>
  );
}

function Cursor() {
  return <span className="ml-0.5 inline-block h-3.5 w-1.5 animate-pulse bg-neutral-400 align-middle" />;
}

function toolIcon(tool: string) {
  if (tool.includes("forecast")) return "◇";
  if (tool.includes("weather")) return "☁";
  if (tool.includes("holiday")) return "□";
  if (tool.includes("search")) return "⌕";
  if (tool.includes("visualization")) return "▤";
  if (tool.includes("dataset") || tool.includes("describe")) return "▤";
  if (tool.includes("sales") || tool.includes("entit")) return "▦";
  return "•";
}

/** Read an image File into a (downscaled) data URI so vision payloads stay small. */
async function fileToDataUrl(file: File, maxDim = 1024): Promise<string> {
  const original = await new Promise<string>((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result as string);
    fr.onerror = () => reject(new Error("read failed"));
    fr.readAsDataURL(file);
  });
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = () => reject(new Error("decode failed"));
      im.src = original;
    });
    const longest = Math.max(img.width, img.height);
    if (longest <= maxDim) return original;
    const scale = maxDim / longest;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return original;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.85);
  } catch {
    return original;
  }
}
