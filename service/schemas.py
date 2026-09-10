"""Pydantic response models for the desktop API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, SecretStr


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


class ProviderKeyPayload(BaseModel):
    """Credentials a browser client supplies for its own run.

    Every field is SecretStr, whose repr is "**********" — so a validation error,
    a logged request model or an accidental model_dump() cannot echo a key. Read
    the real value with .get_secret_value(), which happens in exactly one place
    (service/run_manager.py, when the run's key scope is opened).

    Google, Groq and OpenRouter are lists because core/llm.py pools those and
    rotates on rate limits; a user with several Gemini keys gets the same
    quota-spreading the operator's .env pool provides.
    """

    google: list[SecretStr] = []
    groq: list[SecretStr] = []
    openrouter: list[SecretStr] = []
    deepseek: Optional[SecretStr] = None
    huggingface: Optional[SecretStr] = None
    nvidia: Optional[SecretStr] = None
    cloudflare_account_id: Optional[SecretStr] = None
    cloudflare_api_token: Optional[SecretStr] = None


class GenerateRequest(BaseModel):
    prompt: str
    poly_budget: Optional[str] = None
    mcp_mode: bool = False
    use_concept_pipeline: bool = True
    # Absent for the desktop/CLI case, where the operator's .env supplies the
    # keys. Present when a browser client brings its own, in which case the run
    # uses exactly these providers and never falls back to the operator's.
    provider_keys: Optional[ProviderKeyPayload] = None


class MemoryApprovalRequest(BaseModel):
    approved: bool
