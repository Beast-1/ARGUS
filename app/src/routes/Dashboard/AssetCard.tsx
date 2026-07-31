import { AssetThumbnail } from "../../components/AssetThumbnail";
import { ScoreBadge } from "../../components/ScoreBadge";
import type { ProjectSummary } from "../../api/types";
import "./AssetCard.css";

interface AssetCardProps {
  project: ProjectSummary;
  onOpen: (name: string) => void;
}

export function AssetCard({ project, onOpen }: AssetCardProps) {
  return (
    <button className="asset-card" onClick={() => onOpen(project.name)}>
      <AssetThumbnail previewUrl={project.preview_url} alt={project.display_name} />
      <div className="asset-card-name">{project.display_name}</div>
      <ScoreBadge
        visualScore={project.visual_score}
        severityColor={project.severity_color}
        topologySeverity={project.topology_severity}
      />
    </button>
  );
}
