import { useState } from "react";
import { AssetThumbnail } from "../../components/AssetThumbnail";
import { ConfirmDialog } from "../../components/ConfirmDialog";
import { Icon, Icons } from "../../components/Icon";
import { ScoreBadge } from "../../components/ScoreBadge";
import { useProjectActions } from "../../api/useProjectActions";
import type { ProjectSummary } from "../../api/types";
import "./AssetCard.css";

interface AssetCardProps {
  project: ProjectSummary;
  onOpen: (name: string) => void;
  onDeleted: (name: string) => void;
}

export function AssetCard({ project, onOpen, onDeleted }: AssetCardProps) {
  const { openInBlender, openFolder, deleteProject, busyName, error } =
    useProjectActions(onDeleted);
  const [confirming, setConfirming] = useState(false);
  const busy = busyName === project.name;

  return (
    <div
      className="asset-card"
      role="button"
      tabIndex={0}
      onClick={() => onOpen(project.name)}
      onKeyDown={(e) => e.key === "Enter" && onOpen(project.name)}
    >
      <AssetThumbnail previewUrl={project.preview_url} alt={project.display_name} />

      <div className="asset-card-actions">
        <button
          title="Open in Blender"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            openInBlender(project.name);
          }}
        >
          <Icon icon={Icons.blender} /> Blender
        </button>
        <button
          title="Open folder"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            openFolder(project.name);
          }}
        >
          <Icon icon={Icons.folder} /> Folder
        </button>
        <button
          title="Delete"
          className="asset-card-danger"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            setConfirming(true);
          }}
        >
          <Icon icon={Icons.delete} /> Delete
        </button>
      </div>

      <div className="asset-card-name">{project.display_name}</div>
      <ScoreBadge
        visualScore={project.visual_score}
        severityColor={project.severity_color}
        topologySeverity={project.topology_severity}
      />
      {error && busy === false && (
        <div className="asset-card-error" onClick={(e) => e.stopPropagation()}>
          {error}
        </div>
      )}

      {confirming && (
        <ConfirmDialog
          title="Delete asset?"
          body={`This removes "${project.display_name}" — the exported files and matching run files. This can't be undone.`}
          confirmLabel="Delete"
          danger
          pending={busy}
          onConfirm={() => {
            deleteProject(project.name);
            setConfirming(false);
          }}
          onCancel={() => setConfirming(false)}
        />
      )}
    </div>
  );
}
