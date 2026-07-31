import { useMemo } from "react";
import { TopNav } from "../../components/TopNav";
import type { Route } from "../../routes";
import { StatStrip } from "../../components/StatStrip";
import { AssetCard } from "./AssetCard";
import { useProjects } from "../../api/useProjects";
import "./Dashboard.css";

interface DashboardProps {
  onNavigate: (route: Route) => void;
  onOpenProject: (name: string) => void;
}

function isToday(iso: string): boolean {
  const d = new Date(iso);
  const now = new Date();
  return (
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate()
  );
}

export function Dashboard({ onNavigate, onOpenProject }: DashboardProps) {
  const { projects, loading, error, refresh } = useProjects();

  const stats = useMemo(() => {
    const runsToday = projects.filter((p) => isToday(p.updated_at)).length;
    const scored = projects.filter((p) => p.visual_score !== null);
    const avgScore = scored.length
      ? (scored.reduce((sum, p) => sum + (p.visual_score ?? 0), 0) / scored.length).toFixed(1)
      : "—";
    const withSeverity = projects.filter((p) => p.topology_severity !== null);
    const cleanPct = withSeverity.length
      ? `${Math.round(
          (100 * withSeverity.filter((p) => p.topology_severity === "clean").length) /
            withSeverity.length,
        )}%`
      : "—";
    return [
      { value: runsToday, label: "Runs today" },
      { value: 0, label: "In progress" },
      { value: avgScore, label: "Avg. score" },
      { value: cleanPct, label: "Clean topology" },
    ];
  }, [projects]);

  return (
    <div className="theme-reveal dashboard">
      <TopNav current="dashboard" onNavigate={onNavigate} />

      <div className="dashboard-head">
        <h2 className="dashboard-title">Library</h2>
        <div className="dashboard-sort">
          Sort by <b>Most recent</b>
        </div>
      </div>

      <StatStrip stats={stats} />

      {loading && <div className="dashboard-status">Loading…</div>}
      {error && (
        <div className="dashboard-status dashboard-status-error">
          Couldn't reach the API — is it running? ({error})
          <button onClick={refresh}>Retry</button>
        </div>
      )}
      {!loading && !error && projects.length === 0 && (
        <div className="dashboard-status">No assets yet. Generate one from Workbench.</div>
      )}

      <div className="dashboard-grid">
        {projects.map((p) => (
          <AssetCard key={p.name} project={p} onOpen={onOpenProject} />
        ))}
      </div>
    </div>
  );
}
