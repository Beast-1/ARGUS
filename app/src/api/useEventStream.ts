import { useEffect, useReducer, useRef } from "react";
import { apiUrl } from "./client";

export interface StageEvent {
  number: number;
  title: string;
  label: string;
}

export interface ScoreEvent {
  value: number;
  kind: "ok" | "run" | "fail";
  source: string;
}

export interface ApprovalRequest {
  prompt: string | null;
  asset_name: string | null;
  run_id: string | null;
  blueprint: string | null;
  preview_path: string | null;
  asset_path: string | null;
}

export type RunStatus =
  | "idle"
  | "running"
  | "awaiting_approval"
  | "complete"
  | "failed"
  | "rejected"
  | "error"
  | "cancelled";

export interface CompletePayload {
  run_id: string | null;
  asset_name: string | null;
  glb_path: string | null;
  preview_path: string | null;
  visual_score: number | null;
}

export interface RunState {
  status: RunStatus;
  prompt: string | null;
  stages: StageEvent[];
  currentStage: StageEvent | null;
  metrics: Record<string, string>;
  scores: ScoreEvent[];
  latestScore: ScoreEvent | null;
  parts: string[];
  previewPath: string | null;
  logLines: string[];
  approval: ApprovalRequest | null;
  completed: CompletePayload | null;
  message: string | null;
  startedAt: number | null;
}

const INITIAL: RunState = {
  status: "idle",
  prompt: null,
  stages: [],
  currentStage: null,
  metrics: {},
  scores: [],
  latestScore: null,
  parts: [],
  previewPath: null,
  logLines: [],
  approval: null,
  completed: null,
  message: null,
  startedAt: null,
};

const MAX_LOG_LINES = 500;

type Action = { type: string; payload: any };

function reducer(state: RunState, action: Action): RunState {
  const p = action.payload ?? {};
  switch (action.type) {
    case "__reset":
      return INITIAL;
    case "run_started":
      return {
        ...INITIAL,
        status: "running",
        prompt: p.prompt ?? null,
        startedAt: p.started_at ? p.started_at * 1000 : Date.now(),
      };
    case "stage": {
      const stage: StageEvent = { number: p.number, title: p.title, label: p.label };
      const seen = state.stages.some((s) => s.number === stage.number);
      return {
        ...state,
        status: state.status === "idle" ? "running" : state.status,
        currentStage: stage,
        stages: seen ? state.stages : [...state.stages, stage],
      };
    }
    case "log_value":
      return { ...state, metrics: { ...state.metrics, [p.key]: p.value } };
    case "score": {
      const score: ScoreEvent = { value: p.value, kind: p.kind, source: p.source };
      return { ...state, scores: [...state.scores, score], latestScore: score };
    }
    case "graph_parts":
      return { ...state, parts: Array.isArray(p.parts) ? p.parts : [] };
    case "preview_ready":
      return { ...state, previewPath: p.path ?? state.previewPath };
    case "log_line": {
      const lines = [...state.logLines, p.text ?? ""];
      return {
        ...state,
        logLines: lines.length > MAX_LOG_LINES ? lines.slice(-MAX_LOG_LINES) : lines,
      };
    }
    case "memory_approval_pending":
      return { ...state, status: "awaiting_approval", approval: p as ApprovalRequest };
    case "memory_approval_resolved":
      return { ...state, status: "running", approval: null };
    case "complete":
      return { ...state, status: "complete", completed: p as CompletePayload, approval: null };
    case "cancel_requested":
      return { ...state, message: "Cancelling…" };
    case "cancelled":
      // A cancelled run may still carry a real asset (main.py's loops stop
      // cooperatively, then the pipeline finishes export/validation as usual)
      // — reuse the complete payload shape when run_id is present so Workbench
      // can still offer "View result" for it, same as a normal completion.
      return {
        ...state,
        status: "cancelled",
        completed: p.run_id ? (p as CompletePayload) : null,
        approval: null,
        message: p.reason ?? null,
      };
    case "rejected":
      return { ...state, status: "rejected", message: p.reason ?? "Request rejected" };
    case "failed":
      return { ...state, status: "failed", message: p.reason ?? "Pipeline failed" };
    case "error":
      return { ...state, status: "error", message: p.message ?? "Unexpected error" };
    default:
      return state;
  }
}

const EVENT_TYPES = [
  "run_started",
  "stage",
  "log_value",
  "score",
  "graph_parts",
  "preview_ready",
  "log_line",
  "memory_approval_pending",
  "memory_approval_resolved",
  "complete",
  "cancel_requested",
  "cancelled",
  "rejected",
  "failed",
  "error",
];

export function useEventStream() {
  const [state, dispatch] = useReducer(reducer, INITIAL);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource(apiUrl("/api/generate/stream"));
    sourceRef.current = es;

    const handlers = EVENT_TYPES.map((type) => {
      const handler = (ev: MessageEvent) => {
        let payload: unknown = {};
        try {
          payload = JSON.parse(ev.data);
        } catch {
          payload = {};
        }
        dispatch({ type, payload });
      };
      es.addEventListener(type, handler as EventListener);
      return { type, handler };
    });

    return () => {
      handlers.forEach(({ type, handler }) =>
        es.removeEventListener(type, handler as EventListener),
      );
      es.close();
    };
  }, []);

  const reset = () => dispatch({ type: "__reset", payload: {} });

  return { state, reset };
}
