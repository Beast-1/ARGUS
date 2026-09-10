import { useEffect, useReducer, useRef } from "react";
import { apiUrlWithToken } from "./client";

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

export const INITIAL: RunState = {
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

// The exact event shapes service/pipeline_events.py (parsing) and
// service/run_manager.py (terminal states) emit. The SSE boundary itself stays
// `unknown` (JSON.parse of untrusted external data can't be proven to match these
// at compile time — see the `as AppEvent` cast where events are dispatched below),
// but everything downstream of that one boundary — this reducer included — is now
// checked, instead of `payload: any` silently accepting any shape.
interface RunStartedPayload {
  prompt: string | null;
  poly_budget: string | null;
  mcp_mode: boolean;
  use_concept_pipeline: boolean;
  started_at: number;
}
interface LogValuePayload { key: string; value: string }
interface GraphPartsPayload { parts: unknown }
interface PreviewReadyPayload { kind: string; path: string }
interface LogLinePayload { text: string }
interface CancelledPayload {
  run_id?: string;
  asset_name?: string;
  glb_path?: string;
  preview_path?: string;
  visual_score?: number | null;
  reason?: string;
}
interface RejectedPayload { reason: string }
interface FailedPayload { reason: string }
interface ErrorPayload { message: string }
interface MemoryApprovalResolvedPayload { approved: boolean }
type EmptyPayload = Record<string, never>;

export type AppEvent =
  | { type: "__reset"; payload: EmptyPayload }
  | { type: "run_started"; payload: RunStartedPayload }
  | { type: "stage"; payload: StageEvent }
  | { type: "log_value"; payload: LogValuePayload }
  | { type: "score"; payload: ScoreEvent }
  | { type: "graph_parts"; payload: GraphPartsPayload }
  | { type: "preview_ready"; payload: PreviewReadyPayload }
  | { type: "log_line"; payload: LogLinePayload }
  | { type: "memory_approval_pending"; payload: ApprovalRequest }
  | { type: "memory_approval_resolved"; payload: MemoryApprovalResolvedPayload }
  | { type: "complete"; payload: CompletePayload }
  | { type: "cancel_requested"; payload: EmptyPayload }
  | { type: "cancelled"; payload: CancelledPayload }
  | { type: "rejected"; payload: RejectedPayload }
  | { type: "failed"; payload: FailedPayload }
  | { type: "error"; payload: ErrorPayload };

export function reducer(state: RunState, action: AppEvent): RunState {
  switch (action.type) {
    case "__reset":
      return INITIAL;
    case "run_started": {
      const p = action.payload;
      return {
        ...INITIAL,
        status: "running",
        prompt: p.prompt ?? null,
        startedAt: p.started_at ? p.started_at * 1000 : Date.now(),
      };
    }
    case "stage": {
      const stage = action.payload;
      const seen = state.stages.some((s) => s.number === stage.number);
      return {
        ...state,
        status: state.status === "idle" ? "running" : state.status,
        currentStage: stage,
        stages: seen ? state.stages : [...state.stages, stage],
      };
    }
    case "log_value": {
      const { key, value } = action.payload;
      return { ...state, metrics: { ...state.metrics, [key]: value } };
    }
    case "score": {
      const score = action.payload;
      return { ...state, scores: [...state.scores, score], latestScore: score };
    }
    case "graph_parts": {
      const { parts } = action.payload;
      return { ...state, parts: Array.isArray(parts) ? (parts as string[]) : [] };
    }
    case "preview_ready":
      return { ...state, previewPath: action.payload.path ?? state.previewPath };
    case "log_line": {
      const lines = [...state.logLines, action.payload.text ?? ""];
      return {
        ...state,
        logLines: lines.length > MAX_LOG_LINES ? lines.slice(-MAX_LOG_LINES) : lines,
      };
    }
    case "memory_approval_pending":
      return { ...state, status: "awaiting_approval", approval: action.payload };
    case "memory_approval_resolved":
      return { ...state, status: "running", approval: null };
    case "complete":
      return { ...state, status: "complete", completed: action.payload, approval: null };
    case "cancel_requested":
      return { ...state, message: "Cancelling…" };
    case "cancelled": {
      // A cancelled run may still carry a real asset (main.py's loops stop
      // cooperatively, then the pipeline finishes export/validation as usual) —
      // reuse the complete payload shape when run_id is present so Workbench can
      // still offer "View result" for it, same as a normal completion.
      const p = action.payload;
      return {
        ...state,
        status: "cancelled",
        completed: p.run_id
          ? {
              run_id: p.run_id,
              asset_name: p.asset_name ?? null,
              glb_path: p.glb_path ?? null,
              preview_path: p.preview_path ?? null,
              visual_score: p.visual_score ?? null,
            }
          : null,
        approval: null,
        message: p.reason ?? null,
      };
    }
    case "rejected":
      return { ...state, status: "rejected", message: action.payload.reason ?? "Request rejected" };
    case "failed":
      return { ...state, status: "failed", message: action.payload.reason ?? "Pipeline failed" };
    case "error":
      return { ...state, status: "error", message: action.payload.message ?? "Unexpected error" };
    default:
      return state;
  }
}

const EVENT_TYPES: AppEvent["type"][] = [
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
    let cancelled = false;
    let es: EventSource | null = null;
    let cleanupHandlers: (() => void) | null = null;

    (async () => {
      // The token comes from the Tauri host (Bucket-1 backend-auth fix) — SSE via
      // EventSource can't set an Authorization header, so it rides in the URL
      // instead, unlike every other request in client.ts.
      const url = await apiUrlWithToken("/api/generate/stream");
      if (cancelled) return;

      es = new EventSource(url);
      sourceRef.current = es;

      const handlers = EVENT_TYPES.map((type) => {
        const handler = (ev: MessageEvent) => {
          let payload: unknown = {};
          try {
            payload = JSON.parse(ev.data);
          } catch {
            payload = {};
          }
          dispatch({ type, payload } as AppEvent);
        };
        es!.addEventListener(type, handler as EventListener);
        return { type, handler };
      });

      cleanupHandlers = () => {
        handlers.forEach(({ type, handler }) =>
          es!.removeEventListener(type, handler as EventListener),
        );
      };
    })();

    return () => {
      cancelled = true;
      cleanupHandlers?.();
      es?.close();
    };
  }, []);

  const reset = () => dispatch({ type: "__reset", payload: {} } as AppEvent);

  return { state, reset };
}
