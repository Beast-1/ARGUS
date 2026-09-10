import { useEffect, useMemo, useRef, useState } from "react";
import { apiPost } from "../../api/client";
import { useEventStream } from "../../api/useEventStream";
import { useProjects } from "../../api/useProjects";
import { Icon, Icons } from "../../components/Icon";
import { Logo } from "../../components/Logo";
import type { Route } from "../../routes";
import { ApprovalDialog } from "./ApprovalDialog";
import { Inspector } from "./Inspector";
import { PromptDock } from "./PromptDock";
import { Viewport } from "./Viewport";
import "./Workbench.css";

interface WorkbenchProps {
  onNavigate: (route: Route) => void;
  onOpenProject: (name: string) => void;
}

const STATUS_TEXT: Record<string, string> = {
  idle: "Ready",
  running: "Generating",
  awaiting_approval: "Waiting for your decision",
  complete: "Complete",
  failed: "Failed",
  rejected: "Rejected",
  error: "Error",
  cancelled: "Cancelled",
};

/** Absolute pipeline path -> API file url. main.py prints Windows paths, so normalise
 *  separators before taking the last two segments (<run_id>/<filename>). */
function toFileUrl(rawPath: string | null | undefined): string | null {
  if (!rawPath) return null;
  const parts = rawPath.replace(/\\/g, "/").split("/").filter(Boolean);
  const filename = parts[parts.length - 1];
  const folder = parts[parts.length - 2];
  if (!filename || !folder) return null;
  return `/api/projects/${encodeURIComponent(folder)}/files/${encodeURIComponent(filename)}`;
}

function elapsedLabel(startedAt: number | null, now: number): string {
  if (!startedAt) return "—";
  const secs = Math.max(0, Math.floor((now - startedAt) / 1000));
  return `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
}

export function Workbench({ onNavigate, onOpenProject }: WorkbenchProps) {
  const { state: run } = useEventStream();
  const { projects, error: projectsError, refresh } = useProjects();
  const [error, setError] = useState<string | null>(null);
  const [approvalPending, setApprovalPending] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [now, setNow] = useState(Date.now());
  const logRef = useRef<HTMLDivElement | null>(null);

  const busy = run.status === "running" || run.status === "awaiting_approval";

  useEffect(() => {
    if (!busy) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [busy]);

  // A finished run means out/final may have a new/updated folder — resync the
  // recent list. A cancelled run can still have produced an asset (main.py's
  // loops stop cooperatively, then the pipeline finishes as usual), so it
  // counts too, not just a clean "complete".
  useEffect(() => {
    if (run.status === "complete" || run.status === "cancelled") refresh();
  }, [run.status, refresh]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [run.logLines.length]);

  // Everything lands in out/final/<run_id>/, which the projects file endpoint already
  // serves — no separate "in-progress" storage concept needed.
  const livePreviewUrl = useMemo(
    () => toFileUrl(run.completed?.preview_path ?? run.previewPath),
    [run.previewPath, run.completed],
  );

  // The exported GLB feeds the interactive 3D / wireframe tabs. `GLB` is emitted as a
  // log_value at export (stage 5), so the tabs unlock before the whole run finishes.
  const glbUrl = useMemo(
    () => toFileUrl(run.completed?.glb_path ?? run.metrics["GLB"] ?? null),
    [run.metrics, run.completed],
  );

  async function generate(req: {
    prompt: string;
    poly_budget: string;
    mcp_mode: boolean;
    use_concept_pipeline: boolean;
  }) {
    setError(null);
    try {
      await apiPost("/api/generate", req);
    } catch (err) {
      setError(
        (err as Error).message.includes("409")
          ? "A generation is already running."
          : `Couldn't start: ${(err as Error).message}`,
      );
    }
  }

  async function cancel() {
    setCancelling(true);
    try {
      await apiPost("/api/generate/cancel");
    } catch (err) {
      setError(`Couldn't cancel: ${(err as Error).message}`);
    } finally {
      setCancelling(false);
    }
  }

  async function decideApproval(approved: boolean) {
    setApprovalPending(true);
    try {
      await apiPost("/api/generate/memory-approval", { approved });
    } catch (err) {
      setError(`Approval failed: ${(err as Error).message}`);
    } finally {
      setApprovalPending(false);
    }
  }

  return (
    <div className="theme-workbench wb">
      <div className="wb-menubar">
        <div className="wb-brand">
          <Logo />
          <b>ARGUS</b>
        </div>
        <div className="wb-menu-item wb-menu-active">Workbench</div>
        <div className="wb-menu-item" onClick={() => onNavigate("dashboard")}>
          Dashboard
        </div>
        <div className="wb-fill" />
        <div className="wb-path">
          {run.completed?.run_id ?? run.metrics["Asset name"] ?? "no active run"}
        </div>
      </div>

      <div className="wb-shell">
        <PromptDock run={run} busy={busy} onGenerate={generate} error={error} />

        <div className="wb-center">
          <Viewport run={run} livePreviewUrl={livePreviewUrl} glbUrl={glbUrl} />
          <div className="wb-console" ref={logRef}>
            {run.logLines.length === 0 ? (
              <div className="wb-dim">Pipeline output appears here.</div>
            ) : (
              run.logLines.map((line, i) => <div key={i}>{line}</div>)
            )}
          </div>
        </div>

        <Inspector
          run={run}
          recent={projects}
          recentError={projectsError}
          onOpenProject={onOpenProject}
        />

        <div className="wb-statusbar">
          <span
            className="wb-dot"
            style={{
              background:
                run.status === "complete"
                  ? "var(--success)"
                  : run.status === "running" || run.status === "awaiting_approval"
                    ? "var(--accent)"
                    : run.status === "idle"
                      ? "var(--dim)"
                      : run.status === "cancelled"
                        ? "var(--warning)"
                        : "var(--danger)",
            }}
          />
          <span>
            {STATUS_TEXT[run.status] ?? run.status}
            {run.currentStage && busy
              ? ` — stage ${run.currentStage.number}, ${run.currentStage.title.toLowerCase()}`
              : ""}
            {run.message ? ` — ${run.message}` : ""}
          </span>
          <span className="wb-sp" />
          {busy && (
            <button className="wb-link wb-link-danger" disabled={cancelling} onClick={cancel}>
              <Icon icon={Icons.cancel} size={12} /> {cancelling ? "Cancelling…" : "Cancel"}
            </button>
          )}
          {run.completed && (
            <button
              className="wb-link"
              onClick={() => run.completed?.run_id && onOpenProject(run.completed.run_id)}
            >
              View result <Icon icon={Icons.viewResult} size={12} />
            </button>
          )}
          <span>{elapsedLabel(run.startedAt, now)}</span>
        </div>
      </div>

      {run.approval && (
        <ApprovalDialog
          request={run.approval}
          pending={approvalPending}
          onDecide={decideApproval}
        />
      )}
    </div>
  );
}
