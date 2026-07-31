"""Deterministic per-asset metrics, computed from artifacts on disk.

No vision calls and no re-rendering: every number here is reproducible from the
exported GLB plus manifest.json, so re-scoring a finished library is free and a
metric can never disagree with itself between runs (a real problem observed in the
pipeline, where two vision-scoring stages graded the same render 2 and 4).

`visual_score` is the one model-judged number and is read back from the manifest
rather than recomputed, for the same reason.
"""
from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Severity ranked so "worse" is unambiguous when aggregating.
SEVERITY_RANK = {"clean": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class AssetMetrics:
    asset: str
    exists: bool = False

    # geometry
    triangles: int = 0
    vertices: int = 0
    meshes: int = 0
    glb_bytes: int = 0

    # materials / texturing
    materials: int = 0
    textured_materials: int = 0
    embedded_images: int = 0

    # pipeline-reported quality
    visual_score: Optional[int] = None
    topology_severity: Optional[str] = None
    topology_grounded: Optional[bool] = None
    has_manifest: bool = False

    # spec fidelity
    spec_parts: Optional[int] = None
    built_meshes: Optional[int] = None

    # shape-vocabulary usage (the taper/bevel work)
    boxes: int = 0
    boxes_tapered: int = 0
    cylinders: int = 0
    cylinders_tapered: int = 0

    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _gltf_json(glb_path: Path) -> dict:
    data = glb_path.read_bytes()
    offset = 12
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        if chunk_type == 0x4E4F534A:  # 'JSON'
            return json.loads(data[offset + 8: offset + 8 + chunk_len].decode("utf-8"))
        offset += 8 + chunk_len + ((4 - chunk_len % 4) % 4)
    return {}


def _geometry(gltf: dict) -> tuple[int, int, int]:
    accessors = gltf.get("accessors", [])
    tris = verts = 0
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "indices" in prim:
                tris += accessors[prim["indices"]]["count"] // 3
            pos = prim.get("attributes", {}).get("POSITION")
            if pos is not None:
                verts += accessors[pos]["count"]
    return tris, verts, len(gltf.get("meshes", []))


def _find_glb(asset_dir: Path) -> Optional[Path]:
    """The GLB filename does not always match the folder name: make_run_id()
    disambiguates a collided FOLDER as e.g. "traffic_cone(2)", but the exported
    FILE inside it keeps the plain sanitized asset name, "traffic_cone.glb". A
    real sweep cell hit exactly this — assuming they match silently reported
    "no GLB exported" for an asset that had, in fact, exported cleanly."""
    exact = asset_dir / f"{asset_dir.name}.glb"
    if exact.exists():
        return exact
    candidates = sorted(asset_dir.glob("*.glb"))
    return candidates[0] if candidates else None


def collect(asset_dir: Path, run_dir: Optional[Path] = None) -> AssetMetrics:
    """asset_dir is out/final/<name>; run_dir is out/runs/<name> when available."""
    name = asset_dir.name
    m = AssetMetrics(asset=name)

    glb = _find_glb(asset_dir)
    if glb is None:
        m.notes.append("no GLB exported")
        return m
    m.exists = True
    m.glb_bytes = glb.stat().st_size

    try:
        gltf = _gltf_json(glb)
    except Exception as exc:  # noqa: BLE001
        m.notes.append(f"GLB parse failed: {exc}")
        return m

    m.triangles, m.vertices, m.meshes = _geometry(gltf)
    m.built_meshes = m.meshes

    materials = gltf.get("materials", [])
    m.materials = len(materials)
    m.textured_materials = sum(
        1 for mat in materials
        if "baseColorTexture" in mat.get("pbrMetallicRoughness", {})
    )
    m.embedded_images = len(gltf.get("images", []))

    manifest_path = asset_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            m.has_manifest = True
            m.visual_score = manifest.get("visual_score")
            m.topology_severity = manifest.get("topology_severity")
            m.topology_grounded = manifest.get("topology_grounded")
        except Exception as exc:  # noqa: BLE001
            m.notes.append(f"manifest unreadable: {exc}")

    if run_dir and run_dir.is_dir():
        _shape_vocabulary(run_dir, m)

    return m


def _shape_vocabulary(run_dir: Path, m: AssetMetrics) -> None:
    """Count primitive usage in the winning compiled script.

    Measures whether the planner actually reaches for the shape controls it has —
    the gap that motivated adding box taper (248 boxes / 0 tapered, 98 cylinders /
    4 tapered before the change).
    """
    scripts = run_dir / "scripts"
    if not scripts.is_dir():
        return
    base = [p for p in scripts.glob("*.py")
            if not any(t in p.stem for t in ("_visual_", "_repair_", "_regen_", "_cand_"))]
    if not base:
        return
    try:
        src = base[0].read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if "_argus_smart_mat" not in src:
        m.notes.append("winning script was not compiler output")
        return

    calls = [ln for ln in src.splitlines() if not ln.lstrip().startswith("def ")]
    box_calls = [ln for ln in calls if "argus_box(" in ln]
    cyl_calls = [ln for ln in calls if "argus_cylinder(" in ln]
    m.boxes = len(box_calls)
    m.boxes_tapered = sum(1 for ln in box_calls if "taper=" in ln)
    m.cylinders = len(cyl_calls)
    m.cylinders_tapered = sum(1 for ln in cyl_calls if "r_top=" in ln)


def summarise(records: list[dict]) -> dict:
    """Aggregate a set of per-asset metric dicts into headline numbers.

    Records with env_failure set (network/provider outage — see runner.py) are
    excluded entirely, not counted as failed attempts: they measure the network,
    not the model. Silently including them previously turned a mid-sweep DNS
    outage into an apparent quality collapse.
    """
    usable = [r for r in records if not r.get("env_failure")]
    n_env_failures = len(records) - len(usable)

    done = [r for r in usable if r.get("exists")]
    scored = [r for r in done if isinstance(r.get("visual_score"), int)]
    sev = [r.get("topology_severity") for r in done if r.get("topology_severity")]

    def mean(values: list[float]) -> Optional[float]:
        return round(sum(values) / len(values), 2) if values else None

    return {
        "n_attempted": len(usable),
        "n_env_failures": n_env_failures,
        "n_built": len(done),
        "build_rate": round(len(done) / len(usable), 3) if usable else None,
        "mean_visual_score": mean([r["visual_score"] for r in scored]),
        "pct_at_or_above_7": (
            round(100 * sum(1 for r in scored if r["visual_score"] >= 7) / len(scored))
            if scored else None
        ),
        "pct_topology_clean": (
            round(100 * sum(1 for s in sev if s == "clean") / len(sev)) if sev else None
        ),
        "pct_topology_critical": (
            round(100 * sum(1 for s in sev if s == "critical") / len(sev)) if sev else None
        ),
        "pct_assets_textured": (
            round(100 * sum(1 for r in done if r.get("textured_materials", 0) > 0) / len(done))
            if done else None
        ),
        "mean_triangles": mean([r["triangles"] for r in done]),
        "mean_elapsed_sec": mean([r["elapsed_sec"] for r in usable if r.get("elapsed_sec")]),
        "taper_uptake_boxes": _uptake(done, "boxes_tapered", "boxes"),
        "taper_uptake_cylinders": _uptake(done, "cylinders_tapered", "cylinders"),
    }


def _uptake(records: list[dict], num_key: str, den_key: str) -> Optional[float]:
    den = sum(r.get(den_key, 0) or 0 for r in records)
    if not den:
        return None
    num = sum(r.get(num_key, 0) or 0 for r in records)
    return round(100 * num / den, 1)
