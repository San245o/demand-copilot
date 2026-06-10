// Mirrors backend app/core/events.py — keep in sync.

export type EventType =
  | "run_start"
  | "agent_start"
  | "thought"
  | "tool_call"
  | "tool_result"
  | "agent_end"
  | "forecast"
  | "interrupt"
  | "brief"
  | "run_end"
  | "error";

export interface StreamEvent {
  type: EventType;
  agent: string | null;
  message: string | null;
  data: Record<string, unknown>;
}

export interface ForecastPoint {
  date: string;
  mean: number;
  lower: number;
  upper: number;
}

export interface ForecastData {
  model: string;
  level: number;
  points: ForecastPoint[];
}

export interface BriefData {
  headline: string;
  recommendation: string;
  confidence: string;
  drivers: string[];
}

export const AGENTS = ["sensing", "forecast", "validation", "planning"] as const;
export type AgentName = (typeof AGENTS)[number];

export const AGENT_LABELS: Record<string, string> = {
  sensing: "Sensing",
  forecast: "Forecast",
  validation: "Validation",
  planning: "Planning",
};
