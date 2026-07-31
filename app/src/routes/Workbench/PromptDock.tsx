import { useState } from "react";
import type { RunState } from "../../api/useEventStream";

interface PromptDockProps {
  run: RunState;
  busy: boolean;
  onGenerate: (req: {
    prompt: string;
    poly_budget: string;
    mcp_mode: boolean;
    use_concept_pipeline: boolean;
  }) => void;
  error: string | null;
}

// Labels worth pinning to the metrics table, in display order. Anything else the
// pipeline emits still reaches the raw console pane.
const METRIC_ORDER = [
  "Asset name",
  "Category",
  "Material",
  "Poly budget",
  "Generation",
  "Blender status",
  "Topology status",
  "Severity",
  "N-gons",
  "Open edges",
  "Grounded",
];

export function PromptDock({ run, busy, onGenerate, error }: PromptDockProps) {
  const [prompt, setPrompt] = useState("");
  const [polyBudget, setPolyBudget] = useState("medium");
  const [useConcept, setUseConcept] = useState(true);
  const [mcpMode, setMcpMode] = useState(false);

  const rows = METRIC_ORDER.filter((k) => run.metrics[k] !== undefined);

  return (
    <div className="wb-dock-left">
      <div className="wb-dock-h">Build parameters</div>

      <div className="wb-field-block">
        <p className="wb-field-label">Prompt</p>
        <textarea
          className="wb-field"
          value={prompt}
          placeholder="a cast iron fire hydrant, weathered red paint"
          onChange={(e) => setPrompt(e.currentTarget.value)}
          disabled={busy}
        />
      </div>

      <div className="wb-field-block">
        <p className="wb-field-label">Path</p>
        <div className="wb-row2">
          <select
            className="wb-field wb-select"
            value={polyBudget}
            onChange={(e) => setPolyBudget(e.currentTarget.value)}
            disabled={busy}
          >
            <option value="low">low poly</option>
            <option value="medium">medium poly</option>
            <option value="high">high poly</option>
          </select>
          <select
            className="wb-field wb-select"
            value={useConcept ? "concept" : "direct"}
            onChange={(e) => setUseConcept(e.currentTarget.value === "concept")}
            disabled={busy}
          >
            <option value="concept">concept: on</option>
            <option value="direct">concept: off</option>
          </select>
        </div>
      </div>

      <div className="wb-field-block">
        <label className="wb-check">
          <input
            type="checkbox"
            checked={mcpMode}
            onChange={(e) => setMcpMode(e.currentTarget.checked)}
            disabled={busy}
          />
          Live Blender (MCP)
        </label>
      </div>

      <div className="wb-field-block">
        <p className="wb-field-label">Live metrics</p>
        {rows.length === 0 && run.status === "idle" ? (
          <p className="wb-empty">No run yet.</p>
        ) : (
          <table className="wb-kv">
            <tbody>
              {run.latestScore && (
                <tr>
                  <td>visual score</td>
                  <td className={`wb-${run.latestScore.kind}`}>{run.latestScore.value} / 10</td>
                </tr>
              )}
              {rows.map((key) => (
                <tr key={key}>
                  <td>{key.toLowerCase()}</td>
                  <td>{run.metrics[key]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="wb-field-block wb-field-last">
        <button
          className="wb-btn-primary"
          disabled={busy || !prompt.trim()}
          onClick={() =>
            onGenerate({
              prompt: prompt.trim(),
              poly_budget: polyBudget,
              mcp_mode: mcpMode,
              use_concept_pipeline: useConcept,
            })
          }
        >
          {busy ? "Generating…" : "Generate"}
        </button>
        {error && <p className="wb-error">{error}</p>}
      </div>
    </div>
  );
}
