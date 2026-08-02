"""Pydantic response models for the desktop API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ProjectFile(BaseModel):
    kind: str  # GLB | FBX | BLEND | LOD1 | LOD2 | COLLISION
    url: str
    bytes: int


class ProjectSummary(BaseModel):
    name: str
    display_name: str
    updated_at: str  # ISO 8601
    updated_label: str
    preview_url: Optional[str] = None
    glb_url: Optional[str] = None
    blend_url: Optional[str] = None
    files: list[ProjectFile] = []
    visual_score: Optional[int] = None
    score_kind: Optional[str] = None  # ok | run | fail
    topology_severity: Optional[str] = None
    severity_color: str
    has_manifest: bool


class ProjectDetail(ProjectSummary):
    prompt: Optional[str] = None
    category: Optional[str] = None
    material: Optional[str] = None
    poly_budget: Optional[str] = None
    topology_grounded: Optional[bool] = None
    triangle_count: Optional[int] = None
    material_count: Optional[int] = None


class ProjectListResponse(BaseModel):
    projects: list[ProjectSummary]
    count: int


class GenerateRequest(BaseModel):
    prompt: str
    poly_budget: Optional[str] = None
    mcp_mode: bool = False
    use_concept_pipeline: bool = True


class MemoryApprovalRequest(BaseModel):
    approved: bool


class GenerateStatus(BaseModel):
    status: str  # idle|running|awaiting_approval|complete|failed|rejected|error|cancelled
    run_id: Optional[str] = None
    asset_name: Optional[str] = None
    prompt: Optional[str] = None
