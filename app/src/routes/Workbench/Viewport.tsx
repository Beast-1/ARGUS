import { useState } from "react";
import { apiFileUrl } from "../../api/client";
import { Icon, Icons } from "../../components/Icon";
import { LazyModelViewer } from "../../components/LazyModelViewer";
import type { RunState } from "../../api/useEventStream";

type ViewMode = "render" | "model" | "wireframe";

interface ViewportProps {
  run: RunState;
  /** Served through the API rather than read off disk — the webview never touches the FS. */
  livePreviewUrl: string | null;
  /** GLB for the interactive views; null until a build has exported one. */
  glbUrl: string | null;
}

export function Viewport({ run, livePreviewUrl, glbUrl }: ViewportProps) {
  const [mode, setMode] = useState<ViewMode>("render");
  const stage = run.currentStage;
  const score = run.latestScore;
  const interactive = mode === "model" || mode === "wireframe";

  const tabs: { id: ViewMode; label: string; needsGlb: boolean }[] = [
    { id: "render", label: "render", needsGlb: false },
    { id: "model", label: "3d", needsGlb: true },
    { id: "wireframe", label: "wireframe", needsGlb: true },
  ];

  return (
    <div className="wb-viewport">
      <div className="wb-view-tabs">
        {tabs.map((t) => {
          const disabled = t.needsGlb && !glbUrl;
          return (
            <div
              key={t.id}
              className={[
                "wb-view-tab",
                mode === t.id ? "active" : "",
                disabled ? "wb-view-tab-disabled" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              title={disabled ? "Available once a build has exported a GLB" : undefined}
              onClick={() => !disabled && setMode(t.id)}
            >
              {t.label}
            </div>
          );
        })}
      </div>

      {interactive ? (
        <LazyModelViewer src={glbUrl} wireframe={mode === "wireframe"} />
      ) : (
        <>
          <span className="wb-bracket a" />
          <span className="wb-bracket b" />
          <span className="wb-bracket c" />
          <span className="wb-bracket d" />

          <div className="wb-hud tl">
            STAGE <b>{stage ? `${stage.number} · ${stage.label}` : "—"}</b>
            <br />
            PARTS <b>{run.parts.length || "—"}</b>
          </div>
          <div className="wb-hud tr">
            SCORE <b>{score ? `${score.value}/10` : "—"}</b>
            <br />
            ITER <b>{run.scores.length || "—"}</b>
          </div>

          <div className="wb-asset-wrap">
            {livePreviewUrl ? (
              <>
                <div className="wb-shadow" />
                <img className="wb-asset-img" src={apiFileUrl(livePreviewUrl)} alt="current render" />
              </>
            ) : (
              <div className="wb-viewport-empty">
                {run.status !== "running" && <Icon icon={Icons.blender} size={22} />}
                <span>
                  {run.status === "running"
                    ? stage
                      ? `${stage.title}…`
                      : "Starting…"
                    : "No render yet"}
                </span>
              </div>
            )}
          </div>

          {run.status === "running" && stage && (
            <div className="wb-hud bc">
              <span>{stage.title}</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}
