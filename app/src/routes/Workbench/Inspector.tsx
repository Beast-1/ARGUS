import { useEffect, useState } from "react";
import type { CSSProperties } from "react";
import type { ProjectSummary } from "../../api/types";
import type { RunState } from "../../api/useEventStream";
import { STAGE_SEQUENCE, STATION_COUNT, formatDuration } from "./stages";

interface InspectorProps {
  run: RunState;
  recent: ProjectSummary[];
  recentError?: string | null;
  onOpenProject: (name: string) => void;
}

/** Ticks once a second while a run is live, so the header's elapsed clock moves.
 *  A stopwatch that only updates when a log line happens to arrive is worse than
 *  no stopwatch, because it reads as a frozen UI. Idle runs don't schedule it. */
function useElapsed(startedAt: number | null, live: boolean): number | null {
  const [, tick] = useState(0);
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, [live]);
  return startedAt === null ? null : Date.now() - startedAt;
}

export function Inspector({ run, recent, recentError, onOpenProject }: InspectorProps) {
  const live = run.status === "running" || run.status === "awaiting_approval";
  const elapsed = useElapsed(run.startedAt, live);

  const byNumber = new Map(run.stages.map((s) => [s.number, s]));
  const currentNumber = run.currentStage?.number;

  // How far down the spine the brass fill reaches. Counting rows rather than
  // stations keeps the fill aligned with the rows actually drawn.
  const reachedRow = STAGE_SEQUENCE.reduce(
    (acc, spec, i) => (byNumber.has(spec.number) ? i : acc),
    -1,
  );
  const stationsDone = STAGE_SEQUENCE.filter(
    (s) => s.ordinal !== undefined && byNumber.has(s.number) && s.number !== currentNumber,
  ).length;

  return (
    <div className="wb-dock-right">
      <div className="wb-insp">
        <div className="wb-insp-head">
          <p className="wb-field-label">Pipeline</p>
          {run.startedAt !== null && (
            <p className="wb-insp-meta">
              {stationsDone}/{STATION_COUNT}
              {elapsed !== null && <> · {formatDuration(elapsed)}</>}
            </p>
          )}
        </div>

        <div
          className="wb-rail"
          style={{ "--rail-reached": Math.max(0, reachedRow) } as CSSProperties}
        >
          {STAGE_SEQUENCE.map((spec) => {
            const rec = byNumber.get(spec.number);
            const active = currentNumber === spec.number;
            const done = rec !== undefined && !active;
            const state = active ? "run" : done ? "ok" : "wait";
            return (
              <div
                key={spec.number}
                className={`wb-rail-row wb-rail-${state}${spec.ordinal === undefined ? " wb-rail-loop" : ""}`}
              >
                <span className="wb-rail-dot" />
                <span className="wb-rail-ord">{spec.ordinal ?? ""}</span>
                <span className="wb-rail-label">{spec.label}</span>
                <span className="wb-rail-meta">
                  {/* Pass count only when the loop actually went round again —
                      "x1" on every inspection pass would be noise. */}
                  {rec && rec.passes > 1 && <b className="wb-rail-passes">×{rec.passes}</b>}
                  {active
                    ? "running"
                    : rec
                      ? formatDuration(rec.elapsedMs)
                      : ""}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="wb-insp">
        <p className="wb-field-label">Parts</p>
        {run.parts.length === 0 ? (
          <p className="wb-empty">
            {live ? "Waiting for the planner to break the object down." : "Run a build to see how the object gets broken down."}
          </p>
        ) : (
          <div className="wb-parts">
            {run.parts.slice(0, 14).map((part, i) => (
              <div className="wb-part-row" key={`${part}-${i}`}>
                {part}
              </div>
            ))}
            {run.parts.length > 14 && (
              <div className="wb-part-row wb-dim">+{run.parts.length - 14} more</div>
            )}
          </div>
        )}
      </div>

      <div className="wb-insp wb-field-last">
        <p className="wb-field-label">Recent runs</p>
        {recentError ? (
          <p className="wb-empty">Couldn't load recent runs — {recentError}</p>
        ) : recent.length === 0 ? (
          <p className="wb-empty">Finished assets collect here.</p>
        ) : (
          recent.slice(0, 6).map((p) => (
            <button className="wb-hist-row" key={p.name} onClick={() => onOpenProject(p.name)}>
              <span className="wb-hist-tag" style={{ background: p.severity_color }} />
              <span className="wb-hist-name">{p.name}</span>
              <span className="wb-hist-score">
                {p.visual_score === null ? "—" : `${p.visual_score}/10`}
              </span>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
