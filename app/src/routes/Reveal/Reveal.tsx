import { useState } from "react";
import { apiUrl } from "../../api/client";
import { useProject } from "../../api/useProjects";
import { useProjectActions } from "../../api/useProjectActions";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { StatStrip, type Stat } from "../../components/StatStrip";
import { TopNav } from "../../components/TopNav";
import type { Route } from "../../routes";
import "./Reveal.css";

interface RevealProps {
  name: string;
  onNavigate: (route: Route) => void;
}

function formatBytes(bytes: number): string {
  if (bytes >= 1_000_000) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

/** Splits the display name so the last word can carry the emphasis weight —
 *  the Apple product-page headline convention, applied to a real asset name. */
function headlineParts(displayName: string): [string, string] {
  const words = displayName.trim().split(/\s+/);
  if (words.length <= 1) return ["", displayName];
  return [words.slice(0, -1).join(" "), words[words.length - 1]];
}

export function Reveal({ name, onNavigate }: RevealProps) {
  const { project, loading, error } = useProject(name);
  const { openInBlender, openFolder, deleteProject, busyName, error: actionError } =
    useProjectActions(() => onNavigate("dashboard"));
  const [confirming, setConfirming] = useState(false);
  const busy = busyName === name;

  if (loading) {
    return (
      <div className="theme-reveal reveal">
        <TopNav current="reveal" onNavigate={onNavigate} />
        <p className="reveal-status">Loading…</p>
      </div>
    );
  }

  if (error || !project) {
    return (
      <div className="theme-reveal reveal">
        <TopNav current="reveal" onNavigate={onNavigate} />
        <p className="reveal-status reveal-status-error">
          Couldn't load "{name}"{error ? ` — ${error}` : ""}
        </p>
      </div>
    );
  }

  const [lead, emphasis] = headlineParts(project.display_name);
  const glbFile = project.files.find((f) => f.kind === "GLB");
  const blendFile = project.files.find((f) => f.kind === "BLEND");

  // Manifest-less legacy assets render "—" rather than fabricating values.
  const stats: Stat[] = [
    {
      value: project.visual_score === null ? "—" : `${project.visual_score}`,
      label: "Visual score",
    },
    {
      value: project.topology_severity ?? "—",
      label: "Topology",
      color: project.topology_severity ? project.severity_color : undefined,
    },
    {
      value: project.triangle_count === null ? "—" : project.triangle_count.toLocaleString(),
      label: "Triangles",
    },
    {
      value: project.material_count === null ? "—" : `${project.material_count}`,
      label: "Materials",
    },
  ];

  return (
    <div className="theme-reveal reveal">
      <TopNav current="reveal" onNavigate={onNavigate} />

      <div className="reveal-hero">
        <p className="reveal-eyebrow">{project.updated_label}</p>
        <h1 className="reveal-headline">
          {lead && <>{lead} </>}
          <b>{emphasis}</b>
        </h1>
        {project.prompt && (
          <p className="reveal-sub">
            Generated from <q>{project.prompt}</q>
          </p>
        )}
      </div>

      <div className="reveal-stage">
        <div className="reveal-spot" />
        <div className="reveal-shadow" />
        <div className="reveal-asset-wrap">
          {project.preview_url ? (
            <img
              className="reveal-asset-img"
              src={apiUrl(project.preview_url)}
              alt={project.display_name}
            />
          ) : (
            <p className="reveal-status">No render available</p>
          )}
        </div>
      </div>

      <div className="reveal-cta">
        {glbFile && (
          <a className="reveal-btn" href={apiUrl(glbFile.url)} download>
            Export GLB
          </a>
        )}
        {blendFile && (
          <a className="reveal-link" href={apiUrl(blendFile.url)} download>
            Download .blend &rsaquo;
          </a>
        )}
      </div>

      <StatStrip stats={stats} />

      <div className="reveal-foot">
        <span>
          {project.name}
          {glbFile ? ` · ${formatBytes(glbFile.bytes)}` : ""}
          {project.has_manifest ? "" : " · no manifest (legacy asset)"}
          {actionError ? ` · ${actionError}` : ""}
        </span>
        <span className="reveal-foot-actions">
          <button disabled={busy} onClick={() => openInBlender(name)}>
            Open in Blender
          </button>
          <button disabled={busy} onClick={() => openFolder(name)}>
            Show in Folder
          </button>
          <button
            className="reveal-foot-danger"
            disabled={busy}
            onClick={() => setConfirming(true)}
          >
            Delete
          </button>
        </span>
      </div>

      {confirming && (
        <ConfirmDialog
          title="Delete asset?"
          body={`This removes "${project.display_name}" — the exported files and matching run files. This can't be undone.`}
          confirmLabel="Delete"
          danger
          pending={busy}
          onConfirm={() => {
            deleteProject(name);
            setConfirming(false);
          }}
          onCancel={() => setConfirming(false)}
        />
      )}
    </div>
  );
}
