import "./ScoreBadge.css";

interface ScoreBadgeProps {
  visualScore: number | null;
  severityColor: string;
  topologySeverity: string | null;
}

export function ScoreBadge({ visualScore, severityColor, topologySeverity }: ScoreBadgeProps) {
  const scoreText = visualScore === null ? "no score" : `${visualScore}/10`;
  const severityText = topologySeverity ?? "unscanned";
  return (
    <div className="score-badge">
      <i style={{ background: severityColor }} />
      {scoreText} · {severityText}
    </div>
  );
}
