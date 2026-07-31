import { useCallback, useEffect, useState } from "react";
import { apiGet } from "./client";
import type { ProjectDetail, ProjectListResponse, ProjectSummary } from "./types";

export function useProjects() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    setLoading(true);
    setError(null);
    apiGet<ProjectListResponse>("/api/projects")
      .then((res) => setProjects(res.projects))
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { projects, loading, error, refresh };
}

export function useProject(name: string | null) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!name) {
      setProject(null);
      return;
    }
    setLoading(true);
    setError(null);
    apiGet<ProjectDetail>(`/api/projects/${encodeURIComponent(name)}`)
      .then(setProject)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [name]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { project, loading, error, refresh };
}
