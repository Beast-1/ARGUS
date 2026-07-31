import type { ProjectSummary } from "../../api/types";
import type { RunState } from "../../api/useEventStream";
import { STAGE_SEQUENCE } from "./stages";

interface InspectorProps {
  run: RunState;
  recent: ProjectSummary[];
  onOpenProject: (name: string) => void;
}

export function Inspector({ run, recent, onOpenProject }: InspectorProps) {
  const seen = new Set(run.stages.map((s) => s.number));
  const currentNumber = run.currentStage?.number;

  return (
    <div className="wb-dock-right">
      <div className="wb-insp">
        <p className="wb-field-label">Parts</p>
        {run.parts.length === 0 ? (
          <p className="wb-empty">Planner hasn't reported parts yet.</p>
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

      <div className="wb-insp">
        <p className="wb-field-label">Pipeline</p>
        <table className="wb-kv">
          <tbody>
            {STAGE_SEQUENCE.map((s) => {
              const done = seen.has(s.number) && currentNumber !== s.number;
              const active = currentNumber === s.number;
              return (
                <tr key={s.number}>
                  <td>
                    {String(s.number).padStart(2, "0")} {s.label.toLowerCase()}
                  </td>
                  <td className={active ? "wb-run" : done ? "wb-ok" : undefined}>
                    {active ? "running" : done ? "done" : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="wb-insp wb-field-last">
        <p className="wb-field-label">Recent runs</p>
        {recent.length === 0 ? (
          <p className="wb-empty">No assets yet.</p>
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
