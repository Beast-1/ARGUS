// Mirrors service/schemas.py

export interface ProjectFile {
  kind: string;
  url: string;
  bytes: number;
}

export interface ProjectSummary {
  name: string;
  display_name: string;
  updated_at: string;
  updated_label: string;
  preview_url: string | null;
  glb_url: string | null;
  blend_url: string | null;
  files: ProjectFile[];
  visual_score: number | null;
  score_kind: "ok" | "run" | "fail" | null;
  topology_severity: string | null;
  severity_color: string;
  has_manifest: boolean;
}

export interface ProjectDetail extends ProjectSummary {
  prompt: string | null;
  category: string | null;
  material: string | null;
  poly_budget: string | null;
  topology_grounded: boolean | null;
  triangle_count: number | null;
  material_count: number | null;
}

export interface ProjectListResponse {
  projects: ProjectSummary[];
  count: number;
}
