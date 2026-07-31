"""Asset/project discovery — ports desktop_app.py's `_discover_projects()` /
`_format_project_name()` / `_score_kind()` / `_severity_color()` (desktop_app.py:968-992,
1615-1680) into API response shapes, plus GLB-derived triangle/material counts.

OUTPUT_ROOT mirrors main.py:77 exactly (same ARGUS_OUTPUT_ROOT env var, same default) so
this always reads the same directory main.py writes into.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from service.schemas import ProjectDetail, ProjectFile, ProjectSummary

OUTPUT_ROOT = Path(os.getenv("ARGUS_OUTPUT_ROOT", "out")).resolve()
FINAL_ROOT = OUTPUT_ROOT / "final"

_SEVERITY_COLORS = {
    "clean": "#34D399",
    "medium": "#FBBF24",
    "high": "#f59e0b",
    "critical": "#F87171",
}
_SEVERITY_DEFAULT = "#A1A1AA"

_FILE_SUFFIXES = [
    ("_lod1.glb", "LOD1"),
    ("_lod2.glb", "LOD2"),
    ("_collision.glb", "COLLISION"),
    (".glb", "GLB"),
    (".fbx", "FBX"),
    (".blend", "BLEND"),
]

# (glb path, mtime_ns) -> (triangle_count, material_count)
_stats_cache: dict[tuple[str, int], tuple[int, int]] = {}


def format_project_name(raw_name: str) -> str:
    name = re.sub(r"^\d{8}_\d{6}_", "", raw_name)
    match = re.match(r"^(.*?)(\(\d+\))?$", name)
    base = match.group(1) if match else name
    suffix = match.group(2) if match and match.group(2) else ""
    title = base.replace("_", " ").strip().title() or "Generated Asset"
    return f"{title} {suffix}".strip()


def score_kind(score: Optional[int]) -> Optional[str]:
    if score is None:
        return None
    if score >= 7:
        return "ok"
    if score >= 4:
        return "run"
    return "fail"


def severity_color(severity: Optional[str]) -> str:
    if not severity:
        return _SEVERITY_DEFAULT
    return _SEVERITY_COLORS.get(severity.strip().lower(), _SEVERITY_DEFAULT)


def _file_url(name: str, filename: str) -> str:
    return f"/api/projects/{name}/files/{filename}"


def _read_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt manifest shouldn't break the list
        return {}


def _build_summary(run_dir: Path) -> ProjectSummary:
    name = run_dir.name
    previews = sorted(run_dir.glob("*_preview.png"))

    main_glb = ""
    blend = ""
    files: list[ProjectFile] = []
    for f in sorted(run_dir.iterdir()):
        if not f.is_file():
            continue
        low = f.name.lower()
        try:
            size = f.stat().st_size
        except OSError:
            continue
        for suffix, kind in _FILE_SUFFIXES:
            if low.endswith(suffix):
                files.append(ProjectFile(kind=kind, url=_file_url(name, f.name), bytes=size))
                if kind == "GLB":
                    main_glb = main_glb or f.name
                elif kind == "BLEND":
                    blend = blend or f.name
                break

    manifest = _read_manifest(run_dir)
    visual_score = manifest.get("visual_score")
    topology_severity = manifest.get("topology_severity")
    mtime = run_dir.stat().st_mtime

    return ProjectSummary(
        name=name,
        display_name=format_project_name(name),
        updated_at=datetime.fromtimestamp(mtime).isoformat(),
        updated_label=datetime.fromtimestamp(mtime).strftime("%d %b %Y, %I:%M %p"),
        preview_url=_file_url(name, previews[0].name) if previews else None,
        glb_url=_file_url(name, main_glb) if main_glb else None,
        blend_url=_file_url(name, blend) if blend else None,
        files=files,
        visual_score=visual_score,
        score_kind=score_kind(visual_score),
        topology_severity=topology_severity,
        severity_color=severity_color(topology_severity),
        has_manifest=bool(manifest),
    )


def list_projects() -> list[ProjectSummary]:
    if not FINAL_ROOT.exists():
        return []
    projects = [
        _build_summary(run_dir)
        for run_dir in FINAL_ROOT.iterdir()
        if run_dir.is_dir()
    ]
    projects.sort(key=lambda p: p.updated_at, reverse=True)
    return projects


def _glb_stats(glb_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Triangle/material counts read straight from the exported GLB — works uniformly
    for legacy assets with no manifest.json, with zero main.py involvement."""
    try:
        mtime_ns = glb_path.stat().st_mtime_ns
    except OSError:
        return None, None

    cache_key = (str(glb_path), mtime_ns)
    cached = _stats_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        import trimesh

        scene = trimesh.load(glb_path, force="scene")
        tri_count = 0
        material_ids: set[int] = set()
        for geom in scene.geometry.values():
            faces = getattr(geom, "faces", None)
            if faces is not None:
                tri_count += len(faces)
            visual = getattr(geom, "visual", None)
            material = getattr(visual, "material", None)
            if material is not None:
                material_ids.add(id(material))
        result = (tri_count, len(material_ids) or None)
    except Exception:  # noqa: BLE001 — a malformed GLB shouldn't 500 the detail endpoint
        result = (None, None)

    _stats_cache[cache_key] = result
    return result


def get_project_detail(name: str) -> Optional[ProjectDetail]:
    run_dir = FINAL_ROOT / name
    if not run_dir.is_dir():
        return None

    summary = _build_summary(run_dir)
    manifest = _read_manifest(run_dir)

    triangle_count: Optional[int] = None
    material_count: Optional[int] = None
    if summary.glb_url:
        glb_name = summary.glb_url.rsplit("/", 1)[-1]
        triangle_count, material_count = _glb_stats(run_dir / glb_name)

    return ProjectDetail(
        **summary.model_dump(),
        prompt=manifest.get("prompt"),
        category=manifest.get("category"),
        material=manifest.get("material"),
        poly_budget=manifest.get("poly_budget"),
        topology_grounded=manifest.get("topology_grounded"),
        triangle_count=triangle_count,
        material_count=material_count,
    )


def resolve_project_file(name: str, filename: str) -> Optional[Path]:
    """Path-traversal-guarded file lookup for GET /api/projects/{name}/files/{filename}."""
    run_dir = (FINAL_ROOT / name).resolve()
    if not run_dir.is_relative_to(FINAL_ROOT) or not run_dir.is_dir():
        return None
    candidate = (run_dir / filename).resolve()
    if not candidate.is_relative_to(run_dir) or not candidate.is_file():
        return None
    return candidate
