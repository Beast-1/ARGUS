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
import shutil
import subprocess
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

from service.pipeline_events import score_kind
from service.schemas import ProjectDetail, ProjectFile, ProjectSummary

OUTPUT_ROOT = Path(os.getenv("ARGUS_OUTPUT_ROOT", "out")).resolve()
FINAL_ROOT = OUTPUT_ROOT / "final"
RUNS_ROOT = OUTPUT_ROOT / "runs"

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

# (glb path, mtime_ns) -> (triangle_count, material_count). Bounded LRU: this is a
# process-lifetime cache with no eviction otherwise, so viewing enough distinct
# assets across a long-running backend process would leak memory slowly forever.
_STATS_CACHE_MAX = 512
_stats_cache: "OrderedDict[tuple[str, int], tuple[Optional[int], Optional[int]]]" = OrderedDict()


def format_project_name(raw_name: str) -> str:
    name = re.sub(r"^\d{8}_\d{6}_", "", raw_name)
    match = re.match(r"^(.*?)(\(\d+\))?$", name)
    base = match.group(1) if match else name
    suffix = match.group(2) if match and match.group(2) else ""
    title = base.replace("_", " ").strip().title() or "Generated Asset"
    return f"{title} {suffix}".strip()


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
        _stats_cache.move_to_end(cache_key)
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
    _stats_cache.move_to_end(cache_key)
    if len(_stats_cache) > _STATS_CACHE_MAX:
        _stats_cache.popitem(last=False)
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


def _project_dir(name: str) -> Optional[Path]:
    d = (FINAL_ROOT / name).resolve()
    if not d.is_relative_to(FINAL_ROOT) or not d.is_dir():
        return None
    return d


def open_in_blender(name: str) -> tuple[bool, str]:
    """Launch the asset in a full Blender GUI session — ported from
    desktop_app.py's `_open_in_blender`. Prefers the .blend (quads, materials,
    lights intact); falls back to importing the GLB into a cleaned scene."""
    from core.blender import BLENDER_PATH

    project_dir = _project_dir(name)
    if project_dir is None:
        return False, "Project not found"

    blender = str(BLENDER_PATH)
    if not Path(blender).exists():
        return False, f"BLENDER_PATH does not exist: {blender}"

    blend = next(project_dir.glob("*.blend"), None)
    glb = next(project_dir.glob("*.glb"), None)

    try:
        if blend is not None:
            subprocess.Popen([blender, str(blend.resolve())])
            return True, f"Opening {blend.name}"
        if glb is not None:
            expr = (
                "import bpy; "
                "bpy.ops.object.select_all(action='SELECT'); "
                "bpy.ops.object.delete(use_global=False); "
                f"bpy.ops.import_scene.gltf(filepath=r'{glb.resolve()}')"
            )
            subprocess.Popen([blender, "--python-expr", expr])
            return True, f"Importing {glb.name}"
        return False, "This project has no .blend or .glb file"
    except OSError as exc:
        return False, f"Blender launch failed: {exc}"


def open_folder(name: str) -> tuple[bool, str]:
    """Reveal the asset folder in Windows Explorer."""
    project_dir = _project_dir(name)
    if project_dir is None:
        return False, "Project not found"
    try:
        subprocess.Popen(["explorer", str(project_dir)])
        return True, "Opened"
    except OSError as exc:
        return False, f"Could not open folder: {exc}"


def delete_project(name: str) -> tuple[bool, str]:
    """Delete both out/final/<name> and out/runs/<name> — ported from
    desktop_app.py's `_delete_project`, same path-safety checks: every target
    must resolve inside FINAL_ROOT or RUNS_ROOT before anything is removed."""
    targets = [FINAL_ROOT / name, RUNS_ROOT / name]
    allowed_roots = [FINAL_ROOT.resolve(), RUNS_ROOT.resolve()]

    deleted_any = False
    for target in targets:
        resolved = target.resolve()
        if not any(resolved == base or base in resolved.parents for base in allowed_roots):
            return False, f"Unsafe delete path: {resolved}"
        if not resolved.exists():
            continue
        try:
            shutil.rmtree(resolved)
            deleted_any = True
        except OSError as exc:
            return False, f"Delete failed: {exc}"

    if not deleted_any:
        return False, "Project not found"
    return True, "Deleted"
