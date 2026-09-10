
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
import threading as _threading
import time
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from core.llm import (
    generate_part_list,
    generate_blender_script,
    generate_asset_name,
    generate_reference_image,
    critique_blender_script,
    generate_repair_patch,
    key_pool_status,
    local_structural_critique,
    reset_repair_state,
    score_visual_quality,
    generate_visual_repair,
    generate_spec_repair,
    generate_fresh_from_critique,
)

from core.graph_compiler import compile_spec
from core.spec import spec_summary

from core.prompt import (
    build_object_prompt,
    MAX_REPAIR_ATTEMPTS,
)

from core.blender import (
    build_export_script,
    run_blender,
    run_mcp_analysis,
    run_quality_analysis,
    render_multiview_grid,
    ExportResult,
)
from core.mcp_server import (
    execute_blender_code,
    run_argus_mcp_validation,
)
from core.structure_library import (
    evaluate_structure_memory_candidate,
    record_structure_memory,
)
from core.texture_sources import pbi_values_for, resolve_material_assets


VALID_POLY_BUDGETS = {"low", "medium", "high"}
VALID_EXPORT_FORMATS = {"glb", "fbx"}

# ARGUS_OUTPUT_ROOT: absolute output directory for this run_pipeline() call.
# Defaults to CWD-relative "out" (desktop/CLI behavior, unchanged). A
# multi-tenant worker MUST set this to a fresh, isolated dir per job (e.g. a
# tempfile.mkdtemp() result) — concurrent jobs sharing the default "out"
# would otherwise collide on the same run_id paths.
OUTPUT_ROOT = Path(os.getenv("ARGUS_OUTPUT_ROOT", "out")).resolve()

import os as _os

# ARGUS_FAST=1: one knob for "good asset, minimum wall time" — lighter loop
# budgets and no topology pass. Explicit env vars still override each value.
if _os.getenv("ARGUS_FAST", "0") == "1":
    _os.environ.setdefault("ARGUS_VISUAL_ITERS", "1")
    _os.environ.setdefault("ARGUS_OUTER_REGEN", "0")
    _os.environ.setdefault("ARGUS_INITIAL_CANDIDATES", "1")
    _os.environ.setdefault("ARGUS_SPEC_CANDIDATES", "2")

VISUAL_TARGET_SCORE = int(_os.getenv("ARGUS_VISUAL_TARGET", "7"))
VISUAL_MAX_ITERS = int(_os.getenv("ARGUS_VISUAL_ITERS", "5"))
INITIAL_CANDIDATES = int(_os.getenv("ARGUS_INITIAL_CANDIDATES", "3"))
OUTER_REGEN_ATTEMPTS = int(_os.getenv("ARGUS_OUTER_REGEN", "3"))
USE_COMPILER = _os.getenv("ARGUS_COMPILER", "1") != "0"
# A full point above target, not merely at it — "stop spending more candidates"
# should require more confidence than "stop repairing this one" does.
BESTN_EARLY_EXIT_SCORE = int(_os.getenv("ARGUS_BESTN_EARLY_EXIT", str(VISUAL_TARGET_SCORE + 1)))

# ARGUS_MAX_SECONDS: hard wall-clock ceiling on a single run_pipeline() call,
# covering best-of-N + the visual loop + outer full-regen combined. 0/unset
# means unbounded (the desktop/CLI default). Multi-tenant server deployments
# with a single worker MUST set this so one expensive prompt can't block the
# whole job queue.
MAX_SECONDS = int(_os.getenv("ARGUS_MAX_SECONDS", "0"))


def _pipeline_deadline() -> float | None:
    return (time.monotonic() + MAX_SECONDS) if MAX_SECONDS > 0 else None


# User-initiated cancellation, layered onto the same checkpoints ARGUS_MAX_SECONDS
# already gates rather than threading a new parameter through every loop
# (run_visual_improvement_loop, the outer regen loop, run_best_of_n_generation).
#
# The state lives in core/cancel.py as a per-run scope, not a module global here.
# A single process-wide flag was defensible while RunManager's single-flight lock
# meant only one generation ever ran at a time — but core/batch.py runs N
# pipelines concurrently in one process, and since core/blender.py now terminates
# the Blender subprocess on this signal, a shared flag would mean cancelling one
# batch job killed every other job's render mid-write. See core/cancel.py for why
# it's a ContextVar scope rather than a thread-local.
#
# Still cooperative for the loop checkpoints (it takes effect at the same
# iteration boundaries a deadline would), but no longer only cooperative: an
# in-flight Blender build is terminated promptly by core/blender.py's
# _run_cancellable, so cancelling mid-render no longer waits out the timeout.
from core.cancel import (  # noqa: E402
    clear_cancel,
    is_cancelled,
    request_cancel,
)


def _past_deadline(deadline: float | None) -> bool:
    return is_cancelled() or (deadline is not None and time.monotonic() >= deadline)


def _stop_reason() -> str:
    """For the log lines at each _past_deadline() checkpoint — same trip wire,
    different cause, worth telling apart in the console/event stream."""
    return "cancelled by user" if is_cancelled() else "deadline exceeded"


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            LOG_DIR / "argus.log",
            encoding="utf-8",
        ),
    ],
)

# Wrap the two handlers above so every logger's output is redacted on its way to
# stdout and logs/argus.log — not just core/llm.py's. The previous filter was
# attached to the ARGUS.llm logger alone, and a logging.Filter on one logger never
# sees records emitted through its siblings, so ARGUS.blender, ARGUS.run_manager
# and the rest were writing to that file unfiltered. Handler-level also catches
# tracebacks, which are rendered from exc_info at format time and so are invisible
# to any filter.
from core.secrets import install_redaction  # noqa: E402

install_redaction()

logger = logging.getLogger("ARGUS")


def sanitize_asset_name(name: str) -> str:

    name = name.strip().lower()

    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)

    return name.strip("_") or "generated_asset"


def make_run_id(asset_name: str):

    base_name = sanitize_asset_name(asset_name)

    if os.environ.get("ARGUS_KEEP_HISTORY", "0") != "1":
        # One folder per asset: clear previous iterations of this asset so
        # out/final/<name> always holds exactly the latest version instead of
        # accumulating <name>(2), <name>(3), ... forever.
        pattern = re.compile(re.escape(base_name) + r"(\(\d+\))?$")
        for parent in (OUTPUT_ROOT / "runs", OUTPUT_ROOT / "final"):
            if not parent.exists():
                continue
            for child in parent.iterdir():
                if child.is_dir() and pattern.fullmatch(child.name):
                    shutil.rmtree(child, ignore_errors=True)
        return base_name

    candidate = base_name
    index = 2

    while (
        (OUTPUT_ROOT / "runs" / candidate).exists()
        or (OUTPUT_ROOT / "final" / candidate).exists()
    ):
        candidate = f"{base_name}({index})"
        index += 1

    return candidate


def make_run_dirs(run_id: str):

    root = OUTPUT_ROOT / "runs" / run_id

    dirs = {
        "root": root,
        "scripts": root / "scripts",
        "reports": root / "reports",
        "textures": root / "textures",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def save_script(
    script_dir: Path,
    name: str,
    script: str,
    suffix: str = "",
):

    path = script_dir / f"{name}{suffix}.py"

    path.write_text(script, encoding="utf-8")

    return path


def parse_traceback(error_text: str):

    ctx = {
        "error_class": "UnknownError",
        "error_msg": error_text[:500],
        "failed_line": "",
        "raw": error_text,
    }

    lines = error_text.splitlines()

    for line in reversed(lines):

        m = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\s*:?\s*(.*)",
            line.strip(),
        )

        if m:
            ctx["error_class"] = m.group(1)
            ctx["error_msg"] = m.group(2)
            break

    return ctx


def log_stage(number: int, title: str):
    print(f"\n[STAGE {number}] {title}")
    print("-" * 52)


def log_value(label: str, value):
    print(f"{label:<18}: {value}")


def log_list(label: str, values, limit: int = 8):
    if not values:
        return
    if isinstance(values, str):
        values = [values]
    print(f"{label}:")
    for item in list(values)[:limit]:
        print(f"  - {item}")
    if len(values) > limit:
        print(f"  - ... {len(values) - limit} more")


def log_part_plan(part_data: dict):
    log_value("Category", part_data.get("category", "object"))
    log_value("Geometry", part_data.get("geometry_mode", "hard_surface"))
    if part_data.get("structure_blueprint"):
        count = part_data.get("blueprint_count")
        suffix = f" ({count} loaded)" if count else ""
        log_value("Blueprint", f"{part_data['structure_blueprint']}{suffix}")
    if part_data.get("memory_success_count"):
        log_value("Memory", f"{part_data['memory_success_count']} strict success run(s)")
    log_value("Style", part_data.get("style", "generic"))
    log_value("Material", part_data.get("material", "metal"))
    log_value("Scale", f"{part_data.get('asset_scale', 1.0)} m")
    log_value("Poly budget", part_data.get("poly_budget", "medium"))
    log_value("Export", part_data.get("export_format", "glb"))
    log_list("Parts", part_data.get("parts", []))

    if part_data.get("structure_summary"):
        log_value("Structure", part_data["structure_summary"])
    log_list("Primary forms", part_data.get("primary_forms", []), limit=5)
    log_list("Assembly order", part_data.get("assembly_order", []), limit=6)
    log_list("Attachments", part_data.get("attachment_points", []), limit=6)
    log_list("Orientation", part_data.get("orientation_notes", []), limit=6)
    if part_data.get("support_strategy"):
        log_value("Support", part_data["support_strategy"])
    log_list("Texture plan", part_data.get("texture_plan", []), limit=6)
    log_list("Shape plan", part_data.get("shape_refinement_plan", []), limit=6)
    log_list("Constraints", part_data.get("critical_constraints", []), limit=6)


def log_critic_report(critic_report: dict):
    required = bool(critic_report.get("repair_required"))
    log_value("Repair required", "yes" if required else "no")
    if critic_report.get("summary"):
        log_value("Summary", critic_report["summary"])

    issues = critic_report.get("issues") or []
    if issues:
        print("Issues:")
        for issue in issues[:6]:
            if isinstance(issue, dict):
                issue_type = issue.get("type", "issue")
                location = issue.get("location", "unknown")
                description = issue.get("description", "")
                print(f"  - {issue_type} @ {location}: {description}")
            else:
                print(f"  - {issue}")


def log_traceback_summary(traceback_obj: dict | None):
    if not traceback_obj:
        print("No traceback details available.")
        return
    log_value(
        "Error type",
        traceback_obj.get("error_type")
        or traceback_obj.get("error_class")
        or traceback_obj.get("repair_class")
        or "Unknown",
    )
    if traceback_obj.get("message"):
        log_value("Message", traceback_obj["message"])
    elif traceback_obj.get("error_msg"):
        log_value("Message", traceback_obj["error_msg"])
    if traceback_obj.get("file"):
        log_value("File", traceback_obj["file"])
    if traceback_obj.get("line"):
        log_value("Line", traceback_obj["line"])


def log_mcp_summary(mcp_result):
    data = mcp_result.as_dict()
    log_value("Success", data.get("success"))
    log_value("Severity", data.get("severity", data.get("repair_class", "unknown")))
    log_value("Meshes", data.get("mesh_count", 0))
    if "triangle_count" in data:
        log_value("Triangles", data.get("triangle_count", 0))
    else:
        log_value("Tri faces", data.get("tri_faces", 0))
        log_value("Quad faces", data.get("quad_faces", 0))
    log_value("N-gons", data.get("ngon_count", data.get("ngon_faces", 0)))
    log_value("Open edges", data.get("open_edges", 0))
    log_value("Manifold errors", data.get("manifold_errors", 0))
    if "disconnected_components" in data:
        log_value("Components", data.get("disconnected_components", 0))
    if "floating_components" in data:
        log_value("Floating parts", data.get("floating_components", 0))
    log_value("Grounded", data.get("grounded", True))


def log_quality_summary(quality: dict):
    log_value("Success", quality.get("success"))
    for key in ("export_bytes_glb", "export_bytes_fbx", "export_bytes_blend"):
        if key in quality:
            log_value(key.replace("export_bytes_", "").upper(), f"{quality[key]} bytes")


def default_memory_approval(context: dict) -> bool:
    # Headless/batch runs can opt in to saving every strict-gate pass —
    # the gate already requires clean topology and a visual score >= 7.
    if os.environ.get("ARGUS_AUTO_APPROVE_MEMORY", "0") == "1":
        return True
    if not sys.stdin.isatty():
        return False

    print("\n[USER STRUCTURE REVIEW]")
    if context.get("blueprint"):
        log_value("Blueprint", context["blueprint"])
    if context.get("asset_path"):
        log_value("Asset", context["asset_path"])
    if context.get("preview_path"):
        log_value("Preview", context["preview_path"])

    while True:
        try:
            answer = input(
                "Is this structure good enough to save as a future reference? [y/N]: "
            ).strip().lower()
        except (EOFError, OSError):
            # isatty() can report True under redirected/pseudo-console stdin
            # (observed on Windows background shells) and input() then EOFs.
            return False
        if answer in {"y", "yes"}:
            return True
        if answer in {"", "n", "no"}:
            return False
        print("Please answer y or n.")


def attempt_repair(
    broken_script,
    error_text,
    user_prompt,
    asset_name,
    script_dir,
    part_data,
    run_id,
    mcp_mode=False,
    critic_report=None,
    initial_traceback=None,
):

    current_script = broken_script
    current_error = error_text

    for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):

        print(f"\n[REPAIR {attempt}/{MAX_REPAIR_ATTEMPTS}]")

        traceback_obj = (
            initial_traceback
            if attempt == 1 and initial_traceback
            else parse_traceback(current_error)
        )
        log_traceback_summary(traceback_obj)

        repair_class = (
            traceback_obj.get("error_class")
            or traceback_obj.get("error_type")
            or "UnknownError"
        )
        if "error_class" not in traceback_obj:
            traceback_obj["error_class"] = repair_class
        if "error_msg" not in traceback_obj:
            traceback_obj["error_msg"] = (
                traceback_obj.get("message")
                or str(current_error)
                or ""
            )

        repaired_script = generate_repair_patch(
            broken_script=current_script,
            traceback_obj=traceback_obj,
            repair_class=repair_class,
            critic_report=critic_report,
            user_prompt=user_prompt,
            part_data=part_data,
        )

        if not repaired_script:

            logger.warning("[REPAIR] Empty patch")
            print("Repair patch      : empty")
            continue

        repair_path = save_script(
            script_dir,
            asset_name,
            repaired_script,
            suffix=f"_repair_{attempt}",
        )

        if mcp_mode:
            result = run_mcp_live_script(
                script_path=repair_path,
                asset_name=asset_name,
                run_id=run_id,
            )
        else:
            result = run_blender(
                script_path=repair_path,
                name=asset_name,
                run_id=run_id,
                iter_num=attempt,
                poly_max=_poly_max_for(part_data),
            )

        if result.success:

            print("Repair status     : success")

            return repaired_script, result, repair_path

        current_script = repaired_script

        current_error = (
            result.traceback.get("message", "Unknown Blender failure")
            if result.traceback
            else "Unknown Blender failure"
        )

        logger.warning(
            "[REPAIR FAILURE] %s",
            current_error,
        )
        print(f"Repair status     : failed ({current_error})")

    return None, None, None


def validation_pipeline(
    script_path,
    asset_name,
    run_id,
    mcp_mode=False,
    export_result=None,
):

    log_stage(6, "Runtime Topology Validation")

    # The export wrapper now measures topology inside the build process and
    # ships the report on stdout — free, so it's used even in FAST mode.
    inline = getattr(export_result, "mcp", None) if export_result else None
    if inline:
        from core.blender import mcp_result_from_report
        mcp_result = mcp_result_from_report(inline, raw="(inline)")
        print("Topology source   : inline (measured during the build — no extra Blender pass)")
        topology_clean = mcp_result.topology_clean
    elif (_os.environ.get("ARGUS_VALIDATE", "1") == "0"
            or _os.environ.get("ARGUS_FAST", "0") == "1"):
        print("Topology status   : skipped (ARGUS_VALIDATE=0 / ARGUS_FAST=1 — saves one Blender pass)")
        from core.blender import McpResult
        return McpResult(success=False, severity="skipped",
                         warnings=["topology validation skipped for speed"])
    elif mcp_mode:
        mcp_result = run_argus_mcp_validation()
        topology_clean = mcp_result.topology_clean
    else:
        mcp_result = run_mcp_analysis(
            script_path=script_path,
            name=asset_name,
            run_id=run_id,
        )
        topology_clean = mcp_result.topology_clean

    log_mcp_summary(mcp_result)

    if topology_clean:

        print("Topology status   : clean")

    else:

        print("Topology status   : issues detected")

        for err in mcp_result.errors:
            print(f"- {err}")

        for warn in mcp_result.warnings:
            print(f"- {warn}")

    return mcp_result


def run_mcp_live_script(
    script_path: Path,
    asset_name: str,
    run_id: str,
) -> ExportResult:
    script = build_export_script(
        script_path=script_path,
        name=asset_name,
        run_id=run_id,
    )
    response = execute_blender_code(script)

    success = bool(response.get("success")) and "Traceback" not in response.get("stderr", "")
    stdout = response.get("stdout", "")
    stderr = response.get("stderr", "")

    if not success:
        return ExportResult(
            run_id=run_id,
            name=asset_name,
            success=False,
            traceback={
                "error_class": "MCPExecutionError",
                "message": stderr or response.get("error", "MCP execution failed"),
                "raw": response,
            },
        )

    final_dir = (OUTPUT_ROOT / "final" / run_id).resolve()
    glb_path = final_dir / f"{asset_name}.glb"
    fbx_path = final_dir / f"{asset_name}.fbx"
    blend_path = final_dir / f"{asset_name}.blend"
    preview_path = final_dir / f"{asset_name}_preview.png"

    if not glb_path.exists():
        return ExportResult(
            run_id=run_id,
            name=asset_name,
            success=False,
            traceback={
                "error_class": "MCPExportError",
                "message": f"MCP execution completed but no GLB was exported: {glb_path}",
                "raw": response,
            },
        )

    return ExportResult(
        run_id=run_id,
        name=asset_name,
        success=True,
        glb_path=str(glb_path),
        fbx_path=str(fbx_path) if fbx_path.exists() else "",
        blend_path=str(blend_path) if blend_path.exists() else "",
        preview_path=str(preview_path) if preview_path.exists() else "",
        glb_bytes=glb_path.stat().st_size if glb_path.exists() else 0,
        fbx_bytes=fbx_path.stat().st_size if fbx_path.exists() else 0,
        blend_bytes=blend_path.stat().st_size if blend_path.exists() else 0,
        preview_bytes=preview_path.stat().st_size if preview_path.exists() else 0,
        traceback={"stdout": stdout},
    )


class _BestFiles:
    """Byte-level snapshot of the current best asset's exported files.

    Every candidate build overwrites the SAME canonical out/final paths, so
    without this the files on disk are whatever was built last — even a
    rejected candidate. capture() after each adoption, restore() when the
    iteration stops, and the canonical paths hold the winner again."""

    _KEYS = ("glb_path", "fbx_path", "blend_path", "preview_path")

    def __init__(self, result=None):
        import tempfile
        self.dir = Path(tempfile.mkdtemp(prefix="argus_best_"))
        self.files: dict = {}
        if result is not None:
            self.capture(result)

    def capture(self, result):
        import shutil
        self.files = {}
        for key in self._KEYS:
            p = str(getattr(result, key, "") or "")
            if p and Path(p).exists():
                snap = self.dir / Path(p).name
                try:
                    shutil.copy2(p, snap)
                    self.files[p] = snap
                except OSError:
                    pass

    def restore(self):
        import shutil
        for orig, snap in self.files.items():
            try:
                if snap.exists():
                    shutil.copy2(snap, orig)
            except OSError:
                pass

    def cleanup(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)


# The vision rubric grades silhouette and completeness — it cannot see topology
# defects or materials. Left unchecked that selects actively worse assets: a flat,
# untextured crate scored 8 and displaced a correctly textured one that scored 5,
# and a "critical"-topology build shipped at 8/10. These helpers fold the two blind
# spots back into the number the improvement loop actually compares on.
_SEVERITY_PENALTY = {"clean": 0, "medium": 0, "high": 1, "critical": 3}


def _textured_material_count(glb_path) -> int:
    """Materials carrying a real image texture, read straight from the exported GLB.
    Deterministic, dependency-free, and cheap (parses only the glTF JSON chunk)."""
    try:
        import json as _json
        import struct as _struct

        data = Path(glb_path).read_bytes()
        offset = 12
        while offset < len(data):
            chunk_len, chunk_type = _struct.unpack_from("<II", data, offset)
            if chunk_type == 0x4E4F534A:  # 'JSON'
                gltf = _json.loads(data[offset + 8: offset + 8 + chunk_len].decode("utf-8"))
                return sum(
                    1 for m in gltf.get("materials", [])
                    if "baseColorTexture" in m.get("pbrMetallicRoughness", {})
                )
            offset += 8 + chunk_len + ((4 - chunk_len % 4) % 4)
    except Exception:  # noqa: BLE001 — scoring must never fail the pipeline
        pass
    return 0


def _defect_penalty(result) -> tuple[int, str]:
    """Score penalty for topology the vision model can't see. Uses the inline
    report measured during the build (ExportResult.mcp), so it costs nothing."""
    report = getattr(result, "mcp", None) or {}
    severity = str(report.get("severity", "") or "").lower()
    penalty = _SEVERITY_PENALTY.get(severity, 0)
    return penalty, (f"topology {severity}" if penalty else "")


def _texture_bonus(textured_count: int) -> float:
    """Reward for real photo materials the vision model can't judge on its own,
    capped so silhouette quality still dominates. 0.5/material, same cap used as
    the "textures are worth about N points" exchange rate — one economic model,
    not two: this is what makes the loop prefer richer materials on a near-tie,
    and what stops it discarding them for a marginal silhouette win."""
    cap = float(os.environ.get("ARGUS_TEXTURE_WORTH", "2"))
    return min(0.5 * textured_count, cap)


def _poly_max_for(part_data) -> int:
    """Triangle ceiling for this asset's budget, or 0 when enforcement is off.

    The budget was advisory until export-time enforcement existed (see
    core/blender.py): the prompt asked the model to decimate itself and nothing
    checked, so shipped assets ran 1.5x-56x over. Reads the same _POLY_TARGETS
    table the prompt quotes, so the number the model is told and the number
    actually enforced can never drift apart."""
    if os.environ.get("ARGUS_POLY_ENFORCE", "1") != "1":
        return 0
    from core.prompt import _POLY_TARGETS
    budget = str((part_data or {}).get("poly_budget", "medium")).lower()
    if budget not in _POLY_TARGETS:
        budget = "medium"
    return _POLY_TARGETS[budget][1]


def _poly_penalty(result) -> tuple[float, str]:
    """Penalty for a build that only met its budget by being crushed.

    Enforcement guarantees the exported asset fits, which would otherwise make
    over-tessellation invisible to the loop — a 281k-triangle gas pump would
    quietly become a 5k one and still score 8. Needing a heavy reduction means
    the *generation* was wrong (dense spheres where flat panels belonged), and
    the loop should treat that as a defect worth regenerating, not a success.
    Graduated, and deliberately free up to 4x: some reduction is normal and
    healthy, since planar dissolve alone often removes coplanar detail at no
    visual cost at all."""
    report = getattr(result, "poly", None) or {}
    reduction = float(report.get("reduction", 1) or 1)
    if reduction < 4:
        return 0.0, ""
    if reduction < 10:
        return 0.5, f"needed {reduction:.0f}x poly reduction"
    if reduction < 25:
        return 1.0, f"needed {reduction:.0f}x poly reduction"
    return 2.0, f"needed {reduction:.0f}x poly reduction"


def _effective_score(score: int, result) -> tuple[float, str, int]:
    penalty, penalty_why = _defect_penalty(result)
    poly_pen, poly_why = _poly_penalty(result)
    textured = _textured_material_count(result.glb_path)
    effective = score - penalty - poly_pen + _texture_bonus(textured)
    why = "; ".join(w for w in (penalty_why, poly_why) if w)
    return effective, why, textured


def _accept_candidate(cand_score, best_score, cand_result, best_result) -> tuple[bool, str]:
    """Should this candidate replace the current best?

    Raw vision score alone can't be trusted: it grades silhouette only, so a flat
    untextured build can out-score a correctly textured one, and a critical-topology
    build can win outright. Both were observed in real runs. Comparing on the
    effective score (score, minus a topology penalty, plus a capped texture bonus)
    fixes both in one formula — same mechanism prefers more texturing on a near-tie
    as protects existing texturing from a marginal silhouette win.
    """
    cand_eff, cand_why, cand_tex = _effective_score(cand_score, cand_result)
    best_eff, _, best_tex = _effective_score(best_score, best_result)

    if cand_eff <= best_eff:
        reasons = [r for r in (cand_why,) if r]
        if cand_tex < best_tex:
            reasons.append(f"would drop textured materials ({best_tex}→{cand_tex})")
        why = f"effective {cand_eff:.1f} <= {best_eff:.1f}"
        return False, (f"{why} ({', '.join(reasons)})" if reasons else why)

    return True, ""


def _score_render(glb_path, prompt, part_data, spec=None, reference_image=None):
    """Render a 2x2 multi-view grid and score it (one vision call).
    Returns (vqa, view_kind, grid_bytes). grid_bytes is the rendered image so
    the visual-repair step can SEE the asset, not just read a score.
    spec enables the checklist critic; reference_image grounds the judgment."""
    grid = render_multiview_grid(glb_path)
    if grid:
        vqa = score_visual_quality(grid, prompt, part_data, multiview=True,
                                   spec=spec, reference_image=reference_image)
        return vqa, "multiview", grid
    return {"visual_score": -1, "missing_parts": [], "feedback": "", "skipped": True}, "none", None


def run_best_of_n_generation(
    *,
    prompt,
    part_data,
    script,
    script_path,
    result,
    asset_name,
    run_id,
    script_dir,
    mcp_mode,
    reference_image,
    n_candidates,
    deadline=None,
):
    """Generate N candidate scripts, build each, score each, keep the best.

    The first candidate is the existing `script`/`result`. Each subsequent
    candidate regenerates fresh with slightly higher temperature for variety.
    Returns (best_script, best_path, best_result, best_vqa).
    """
    if n_candidates <= 1 or not result.success or not result.glb_path:
        return script, script_path, result, None

    log_stage(54, f"Best-of-{n_candidates} Candidate Generation")

    best_vqa, _, _ = _score_render(result.glb_path, prompt, part_data,
                                   reference_image=reference_image)
    best_score = int(best_vqa.get("visual_score", -1)) if not best_vqa.get("skipped") else -1
    best_script, best_path, best_result = script, script_path, result
    _best_files = _BestFiles(best_result)
    print(f"Candidate 1/{n_candidates}: score {best_score}/10")

    from core.prompt import build_object_prompt as _bop
    for _i in range(2, n_candidates + 1):
        if _past_deadline(deadline):
            print(f"\n[BEST-OF-N {_i}/{n_candidates}] {_stop_reason()} — stopping early")
            break
        # No point spending another full LLM-codegen + Blender build on a candidate
        # that already has clean topology and real materials at a confident score —
        # unlike the outer regen loop (which already breaks at VISUAL_TARGET_SCORE),
        # this loop previously always burned all N candidates regardless. Judged on
        # the same effective score used for candidate selection, not the raw vision
        # score, so a high score hiding critical topology can't trigger this.
        _best_eff, _, _ = _effective_score(best_score, best_result)
        if _best_eff >= BESTN_EARLY_EXIT_SCORE:
            print(f"\n[BEST-OF-N {_i}/{n_candidates}] skipped — candidate 1 already "
                  f"confidently good (effective {_best_eff:.1f} >= {BESTN_EARLY_EXIT_SCORE})")
            break
        print(f"\n[BEST-OF-N {_i}/{n_candidates}] generating fresh candidate")
        gen_prompt_i = _bop(prompt, part_data, mcp_mode=mcp_mode)
        cand_script = generate_blender_script(
            gen_prompt_i, part_data,
            reference_image=reference_image, user_prompt=prompt,
        )
        if not cand_script:
            print(f"Candidate {_i}: generation returned nothing")
            continue
        cand_path = save_script(script_dir, asset_name, cand_script, suffix=f"_cand_{_i}")

        if mcp_mode:
            cand_result = run_mcp_live_script(
                script_path=cand_path, asset_name=asset_name, run_id=run_id,
            )
        else:
            cand_result = run_blender(
                script_path=cand_path, name=asset_name, run_id=run_id, iter_num=_i,
                poly_max=_poly_max_for(part_data),
            )
        if not cand_result.success or not cand_result.glb_path:
            print(f"Candidate {_i}: failed to build")
            continue

        cand_vqa, _, _ = _score_render(cand_result.glb_path, prompt, part_data,
                                       reference_image=reference_image)
        cand_score = int(cand_vqa.get("visual_score", -1)) if not cand_vqa.get("skipped") else -1
        print(f"Candidate {_i}/{n_candidates}: score {cand_score}/10")

        _adopt, _why = _accept_candidate(cand_score, best_score, cand_result, best_result)
        if _adopt:
            print(f"Candidate {_i}: ADOPTED ({cand_score} > {best_score})")
            best_script, best_path, best_result = cand_script, cand_path, cand_result
            best_vqa, best_score = cand_vqa, cand_score
            _best_files.capture(cand_result)
        elif cand_score > best_score:
            print(f"Candidate {_i}: rejected — {_why}")

    _best_files.restore()
    _best_files.cleanup()
    print(f"\nBest-of-N final   : {best_score}/10 (from {n_candidates} candidates)")
    return best_script, best_path, best_result, best_vqa


def run_visual_improvement_loop(
    *,
    prompt,
    part_data,
    script,
    script_path,
    result,
    asset_name,
    run_id,
    script_dir,
    mcp_mode,
    reference_image=None,
    spec=None,
    texture_paths=None,
    deadline=None,
):
    """Render → score → targeted-improve, keeping the best-scoring asset.

    When `spec` is given (compiler path), improvement is done by asking the
    vision model for a SMALL JSON edit list against the build spec, which is
    re-validated/auto-snapped and recompiled — far more reliable than having
    a model rewrite a whole script. Falls back to script rewriting when spec
    edits are unavailable; once a script-rewrite candidate is adopted, the
    spec no longer matches the script and spec mode is abandoned.

    Returns (best_script, best_script_path, best_result, best_vqa).
    Bounded by VISUAL_MAX_ITERS to respect free-tier rate limits.
    """
    log_stage(55, "Visual Feedback Loop")

    if not result.glb_path:
        print("Visual loop       : skipped (no exported asset)")
        return script, script_path, result, {"skipped": True, "visual_score": -1}

    best_script, best_path, best_result = script, script_path, result
    best_vqa, view_kind, best_grid = _score_render(result.glb_path, prompt, part_data,
                                                   spec=spec, reference_image=reference_image)

    if best_vqa.get("skipped"):
        print("Visual loop       : skipped (no vision model / render unavailable)")
        return best_script, best_path, best_result, best_vqa

    best_score = int(best_vqa.get("visual_score", -1))
    print(f"Visual score      : {best_score}/10 ({view_kind})")
    if best_vqa.get("feedback"):
        print(f"Feedback          : {best_vqa['feedback']}")
    if best_vqa.get("missing_parts"):
        log_list("Missing parts", best_vqa["missing_parts"], limit=6)

    current_spec = spec
    _best_files = _BestFiles(best_result)
    _consecutive_no_improvement = 0
    for _it in range(1, VISUAL_MAX_ITERS + 1):
        if best_score >= VISUAL_TARGET_SCORE:
            print(f"Visual target     : met ({best_score} >= {VISUAL_TARGET_SCORE})")
            break
        if _past_deadline(deadline):
            print(f"\n[VISUAL IMPROVE {_it}/{VISUAL_MAX_ITERS}] {_stop_reason()} — stopping early")
            break

        print(f"\n[VISUAL IMPROVE {_it}/{VISUAL_MAX_ITERS}] score {best_score} < target {VISUAL_TARGET_SCORE}")

        improved = ""
        cand_spec = None
        if current_spec is not None:
            cand_spec = generate_spec_repair(
                current_spec, prompt, part_data, best_vqa,
                render_image=best_grid, reference_image=reference_image,
            )
            if cand_spec:
                try:
                    improved = compile_spec(cand_spec, texture_paths)
                    print("Visual improve    : spec edits applied → recompiled")
                except Exception as _spec_exc:
                    logger.warning("[VISUAL] recompile failed: %s", _spec_exc)
                    improved = ""
                    cand_spec = None
            if not improved:
                print("Visual improve    : no usable spec edits — trying script rewrite")

        if not improved:
            cand_spec = None
            improved = generate_visual_repair(
                best_script, prompt, part_data, best_vqa,
                render_image=best_grid, reference_image=reference_image,
            )
        if not improved:
            print("Visual improve    : no improved script produced this round")
            _consecutive_no_improvement += 1
            if _consecutive_no_improvement >= 2:
                print("Visual improve    : 2 consecutive failures — stopping loop")
                break
            continue

        cand_path = save_script(script_dir, asset_name, improved, suffix=f"_visual_{_it}")
        if mcp_mode:
            cand_result = run_mcp_live_script(
                script_path=cand_path, asset_name=asset_name, run_id=run_id,
            )
        else:
            cand_result = run_blender(
                script_path=cand_path, name=asset_name, run_id=run_id, iter_num=_it,
                poly_max=_poly_max_for(part_data),
            )

        if not cand_result.success or not cand_result.glb_path:
            print("Visual improve    : candidate failed to build — attempting runtime repair")
            _err_msg = (cand_result.traceback or {}).get("message", "Unknown Blender error")
            _repaired_s, _repaired_r, _repaired_p = attempt_repair(
                broken_script=improved,
                error_text=_err_msg,
                user_prompt=prompt,
                asset_name=asset_name,
                script_dir=script_dir,
                part_data=part_data,
                run_id=run_id,
                mcp_mode=mcp_mode,
                critic_report=None,
                initial_traceback=cand_result.traceback,
            )
            if _repaired_r and _repaired_r.success and _repaired_r.glb_path:
                print("Visual improve    : runtime repair succeeded — scoring repaired candidate")
                cand_result = _repaired_r
                improved = _repaired_s
                cand_path = _repaired_p
                cand_spec = None
            else:
                print("Visual improve    : runtime repair also failed")
                _consecutive_no_improvement += 1
                if _consecutive_no_improvement >= 2:
                    print("Visual improve    : 2 consecutive failures — stopping loop")
                    break
                continue

        cand_vqa, cand_view, cand_grid = _score_render(
            cand_result.glb_path, prompt, part_data,
            spec=cand_spec or current_spec, reference_image=reference_image)
        cand_score = int(cand_vqa.get("visual_score", -1))
        print(f"Candidate score   : {cand_score}/10 ({cand_view})")
        if cand_vqa.get("feedback"):
            print(f"Feedback          : {cand_vqa['feedback']}")

        _accept, _reject_why = _accept_candidate(
            cand_score, best_score, cand_result, best_result)
        if _accept:
            print("Visual improve    : candidate is better → adopting")
            best_script, best_path, best_result = improved, cand_path, cand_result
            best_vqa, best_score, best_grid = cand_vqa, cand_score, cand_grid
            current_spec = cand_spec
            _consecutive_no_improvement = 0
            _best_files.capture(cand_result)
        else:
            print(f"Visual improve    : rejected — {_reject_why}")
            _consecutive_no_improvement += 1
            if _consecutive_no_improvement >= 2:
                print("Visual improve    : 2 consecutive non-improvements — stopping loop")
                break

    # Candidate builds overwrote the canonical files; put the winner back.
    _best_files.restore()
    _best_files.cleanup()
    print(f"\nVisual loop final : {best_score}/10")
    return best_script, best_path, best_result, best_vqa


def run_pipeline(
    prompt,
    poly_budget=None,
    export_format="glb",
    mcp_mode=False,
    use_concept_pipeline=True,
    memory_approval_callback=None,
):
    reset_repair_state()
    _run_deadline = _pipeline_deadline()

    if mcp_mode:
        from core.mcp_server import ensure_mcp_server
        print("\n[MCP CONNECTION CHECK]")
        ok, detail = ensure_mcp_server()
        if ok:
            log_value("Status", f"connected to {detail}")
        else:
            log_value("Status", f"unavailable: {detail}")
            print("Auto-launch failed — start Blender manually or set BLENDER_PATH, then retry.")
            return False


    log_stage(1, "Planning and Structural Analysis")

    part_data = generate_part_list(prompt)

    if part_data.get("rejected"):
        reason = part_data.get("rejection_reason") or "Asset is too complex to generate reliably."
        print(f"\n[ARGUS] Request rejected: {reason}")
        return None

    if poly_budget and poly_budget.lower() in VALID_POLY_BUDGETS:
        part_data["poly_budget"] = poly_budget.lower()

    log_part_plan(part_data)

    import json as _json
    print(f"[GRAPH_PARTS]{_json.dumps(part_data.get('parts', []))}")


    log_stage(2, "Asset Naming")

    asset_name = sanitize_asset_name(
        generate_asset_name(prompt)
    )

    run_id = make_run_id(asset_name)

    run_dirs = make_run_dirs(run_id)

    log_value("Asset name", asset_name)
    log_value("Project folder", run_id)


    selections: dict[str, str] = part_data.get("poly_haven_textures") or {}
    texture_paths: dict = {}
    value_overrides: dict = {}
    if selections:
        print(f"\n[MATERIALS] Resolving {len(selections)} planner material pick(s)…")
        texture_paths, value_overrides = resolve_material_assets(
            selections, run_dirs["textures"])
        for mat_name, paths in texture_paths.items():
            print(f"  {mat_name}: texture ({', '.join(paths.keys())})")
        for mat_name, vals in value_overrides.items():
            print(f"  {mat_name}: measured values ({vals.get('source', 'pbi')})")
    part_data["poly_haven_texture_paths"] = texture_paths

    # Replace planner-guessed colours with measured values wherever no photo
    # texture resolved — LLM-invented RGB is the last-resort, not the default.
    _spec_mats = (part_data.get("build_spec") or {}).get("materials") or {}
    for _mname, _mdef in _spec_mats.items():
        if not isinstance(_mdef, dict):
            continue
        _key = _mdef.get("texture") or _mname
        if _key in texture_paths or _mname in texture_paths:
            continue
        _vals = (value_overrides.get(_key) or value_overrides.get(_mname)
                 or pbi_values_for(_mname) or pbi_values_for(str(_key)))
        if _vals:
            _mdef["color"] = list(_vals["color"])
            _mdef["metallic"] = _vals["metallic"]
            _mdef["roughness"] = _vals["roughness"]
            if _mname not in value_overrides and _key not in value_overrides:
                print(f"  {_mname}: measured values ({_vals.get('source', 'pbi')}, auto-matched)")


    # The reference image is not needed until LLM codegen (non-compiled mode)
    # or the visual loop (after the first Blender build), so it generates in a
    # background thread and overlaps texture download + compile + build.
    reference_image: bytes | None = None
    _ref_box: dict = {}
    _ref_thread = None
    if use_concept_pipeline:
        print("\n[VISION] Generating reference image in background (Gemini > Pollinations FLUX)...")

        def _gen_reference():
            try:
                _ref_box["img"] = generate_reference_image(part_data, prompt)
            except Exception as _ref_exc:  # noqa: BLE001 — never kill the pipeline
                logger.warning("[VISION] reference image generation failed: %s", _ref_exc)

        _ref_thread = _threading.Thread(target=_gen_reference, daemon=True)
        _ref_thread.start()
    else:
        print("\n[VISION] 2D Concept Pipeline disabled — skipping reference image")

    def _await_reference() -> bytes | None:
        """First caller pays any remaining wait; result is cached and saved."""
        if _ref_thread is not None and _ref_thread.is_alive():
            _ref_thread.join(timeout=120)
        img = _ref_box.get("img")
        if img and not _ref_box.get("saved"):
            img_ext = "jpg" if img[:3] == b"\xff\xd8\xff" else "png"
            ref_path = run_dirs["root"] / f"reference.{img_ext}"
            ref_path.write_bytes(img)
            print(f"[VISION] Reference image saved -> {ref_path.name} ({len(img)//1024}KB)")
            _ref_box["saved"] = True
        elif _ref_thread is not None and not img and not _ref_box.get("warned"):
            print("[VISION] Reference image unavailable — proceeding text-only")
            _ref_box["warned"] = True
        return img


    log_stage(3, "Blender Script Generation")

    current_spec = part_data.get("build_spec") if USE_COMPILER else None
    compiled_mode = False
    script = ""

    if current_spec:
        try:
            script = compile_spec(
                current_spec, part_data.get("poly_haven_texture_paths"),
            )
            compiled_mode = True
            log_value("Generation", "scene-graph compiler (deterministic)")
            log_value("Build spec", spec_summary(current_spec))
            for warn in (part_data.get("build_spec_warnings") or [])[:6]:
                print(f"  spec: {warn}")
        except Exception as exc:  # noqa: BLE001 — fall back to LLM codegen
            logger.warning("[COMPILER] compile failed (%s) — falling back to LLM", exc)
            current_spec = None

    if not compiled_mode:
        log_value("Generation", "LLM codegen (no usable build spec)"
                  if USE_COMPILER else "LLM codegen (compiler disabled)")
        reference_image = _await_reference()  # codegen is vision-grounded
        generation_prompt = build_object_prompt(
            prompt,
            part_data,
            mcp_mode=mcp_mode,
        )

        script = generate_blender_script(
            generation_prompt,
            part_data,
            reference_image=reference_image,
            user_prompt=prompt,
        )

    if not script:

        print("Generation failed : no script returned")
        return False

    script_path = save_script(
        run_dirs["scripts"],
        asset_name,
        script,
    )

    log_value("Script saved", script_path)


    log_stage(4, "Critic Review")

    _MAX_CRITIC_PASSES = 2 if not compiled_mode else 0
    critic_report: dict | None = None

    if compiled_mode:
        print("Critic review     : skipped (deterministic compiler output)")

    for _critic_pass in range(1, _MAX_CRITIC_PASSES + 1):

        critic_report = critique_blender_script(script, prompt, part_data,
                                                reference_image=reference_image)

        if not critic_report:
            print(f"Critic pass {_critic_pass}     : unavailable; continuing")
            break

        log_critic_report(critic_report)

        if not critic_report.get("repair_required"):
            print(f"Critic pass {_critic_pass}     : no repairs needed")
            break

        print(f"Critic pass {_critic_pass}     : issuing repair patch")

        repaired_script = generate_repair_patch(
            broken_script=script,
            traceback_obj={
                "error_class": "CriticFailure",
                "error_msg": critic_report.get("summary", ""),
            },
            repair_class="CriticFailure",
            critic_report=critic_report,
            user_prompt=prompt,
            part_data=part_data,
        )

        if not repaired_script:
            print(f"Critic pass {_critic_pass}     : repair unavailable; proceeding with best available script")
            break

        script = repaired_script
        script_path = save_script(
            run_dirs["scripts"],
            asset_name,
            repaired_script,
            suffix=f"_critic_fix_{_critic_pass}",
        )
        print(f"Critic pass {_critic_pass}     : patch applied → re-reviewing")


    log_stage(5, "Blender Execution and Export")

    if mcp_mode:
        log_value("Execution mode", "live MCP server")
        result = run_mcp_live_script(
            script_path=script_path,
            asset_name=asset_name,
            run_id=run_id,
        )
    else:
        result = run_blender(
            script_path=script_path,
            name=asset_name,
            run_id=run_id,
            poly_max=_poly_max_for(part_data),
        )

    if not result.success:

        print("Blender status    : failed")

        if result.traceback:
            log_traceback_summary(result.traceback)

        error_message = (
            result.traceback.get("message", "Unknown Blender error")
            if result.traceback
            else "Unknown Blender error"
        )

        if mcp_mode and any(
            marker in error_message.lower()
            for marker in ("10061", "actively refused", "connection", "refused")
        ):
            print("\n[MCP CONNECTION ERROR]")
            print("Could not reach live MCP server. Repair loop skipped.")
            print("Open Blender > BlenderMCP tab > Connect to MCP server, then rerun.")
            return False

        repaired_script, repaired_result, repaired_script_path = attempt_repair(
            broken_script=script,
            error_text=error_message,
            user_prompt=prompt,
            asset_name=asset_name,
            script_dir=run_dirs["scripts"],
            part_data=part_data,
            run_id=run_id,
            mcp_mode=mcp_mode,
            critic_report=critic_report,
            initial_traceback=result.traceback,
        )

        if repaired_result is None:

            print("Repair status     : failed")
            return False

        result = repaired_result
        script = repaired_script
        script_path = repaired_script_path

    print("Blender status    : export complete")
    reference_image = _await_reference()  # build is done; loop needs it now
    log_value("GLB", result.glb_path)
    if result.fbx_path:
        log_value("FBX", result.fbx_path)
    if result.blend_path:
        log_value("BLEND", result.blend_path)
    if result.preview_path:
        print("[PREVIEW IMAGE]")
        print(result.preview_path)

    if reference_image and result.glb_path:
        _final_dir = Path(result.glb_path).parent
        _img_ext = "jpg" if reference_image[:3] == b"\xff\xd8\xff" else "png"
        _ref_dest = _final_dir / f"reference.{_img_ext}"
        try:
            _ref_dest.write_bytes(reference_image)
            log_value("Reference image", str(_ref_dest))
        except OSError as _e:
            logger.warning("[VISION] Could not save reference to final folder: %s", _e)

    if compiled_mode:
        print("\nBest-of-N         : skipped (compiler output is deterministic)")
    else:
        bestn_script, bestn_path, bestn_result, bestn_vqa = run_best_of_n_generation(
            prompt=prompt,
            part_data=part_data,
            script=script,
            script_path=script_path,
            result=result,
            asset_name=asset_name,
            run_id=run_id,
            script_dir=run_dirs["scripts"],
            mcp_mode=mcp_mode,
            reference_image=reference_image,
            n_candidates=INITIAL_CANDIDATES,
            deadline=_run_deadline,
        )
        script, script_path, result = bestn_script, bestn_path, bestn_result

    final_visual = run_visual_improvement_loop(
        prompt=prompt,
        part_data=part_data,
        script=script,
        script_path=script_path,
        result=result,
        asset_name=asset_name,
        run_id=run_id,
        script_dir=run_dirs["scripts"],
        mcp_mode=mcp_mode,
        reference_image=reference_image,
        spec=current_spec,
        texture_paths=part_data.get("poly_haven_texture_paths"),
        deadline=_run_deadline,
    )
    script, script_path, result, last_vqa = final_visual

    if OUTER_REGEN_ATTEMPTS > 0 and not _past_deadline(_run_deadline):
        _outer_score = int((last_vqa or {}).get("visual_score", -1))
        _outer_best_score = _outer_score
        _outer_best = (script, script_path, result, last_vqa)
        _outer_files = _BestFiles(result)

        for _outer_it in range(1, OUTER_REGEN_ATTEMPTS + 1):
            if _outer_best_score >= VISUAL_TARGET_SCORE:
                break
            if _past_deadline(_run_deadline):
                print(f"\n[FULL REGEN {_outer_it}/{OUTER_REGEN_ATTEMPTS}] {_stop_reason()} — stopping early")
                break

            _last_render = None
            if _outer_best[2].glb_path:
                _last_render = render_multiview_grid(_outer_best[2].glb_path)

            log_stage(56, f"Full Regen Attempt {_outer_it}/{OUTER_REGEN_ATTEMPTS} (score {_outer_best_score} < target {VISUAL_TARGET_SCORE})")

            from core.prompt import build_object_prompt as _bop
            _regen_prompt = _bop(prompt, part_data, mcp_mode=mcp_mode)
            _fresh = generate_fresh_from_critique(
                user_prompt=prompt,
                part_data=part_data,
                generation_prompt=_regen_prompt,
                visual_feedback=(_outer_best[3] or {}),
                render_image=_last_render,
                reference_image=reference_image,
            )
            if not _fresh:
                print("Full regen        : no script produced; stopping outer loop")
                break

            reset_repair_state()
            _regen_path = save_script(run_dirs["scripts"], asset_name, _fresh,
                                      suffix=f"_regen_{_outer_it}")
            _regen_result = run_blender(script_path=_regen_path, name=asset_name, run_id=run_id,
                                        poly_max=_poly_max_for(part_data))

            if not _regen_result.success or not _regen_result.glb_path:
                _rr_s, _rr_r, _rr_p = attempt_repair(
                    broken_script=_fresh, error_text=((_regen_result.traceback or {}).get("message", "")),
                    user_prompt=prompt, asset_name=asset_name,
                    script_dir=run_dirs["scripts"], part_data=part_data,
                    run_id=run_id, mcp_mode=mcp_mode,
                )
                if _rr_r and _rr_r.success:
                    _regen_result = _rr_r; _fresh = _rr_s; _regen_path = _rr_p
                else:
                    print(f"Full regen {_outer_it}    : build failed — continuing")
                    _outer_files.restore()  # failed builds overwrote the best's files
                    continue

            _regen_vqa, _, _regen_grid = _score_render(_regen_result.glb_path, prompt, part_data,
                                                       reference_image=reference_image)
            _regen_score = int(_regen_vqa.get("visual_score", -1)) if not _regen_vqa.get("skipped") else -1
            print(f"Full regen {_outer_it}     : score {_regen_score}/10")

            _regen_final = run_visual_improvement_loop(
                prompt=prompt, part_data=part_data, script=_fresh,
                script_path=_regen_path, result=_regen_result,
                asset_name=asset_name, run_id=run_id,
                script_dir=run_dirs["scripts"], mcp_mode=mcp_mode,
                reference_image=reference_image,
                deadline=_run_deadline,
            )
            _rs, _rsp, _rr, _rv = _regen_final
            _final_regen_score = int((_rv or {}).get("visual_score", _regen_score))

            print(f"Full regen {_outer_it} final : {_final_regen_score}/10")
            _regen_ok, _regen_why = _accept_candidate(
                _final_regen_score, _outer_best_score, _rr, result)
            if not _regen_ok and _final_regen_score > _outer_best_score:
                print(f"Full regen {_outer_it}      : rejected — {_regen_why}")
            if _regen_ok:
                print(f"Full regen {_outer_it}      : NEW BEST adopted ({_final_regen_score} > {_outer_best_score})")
                _outer_best = (_rs, _rsp, _rr, _rv)
                _outer_best_score = _final_regen_score
                script, script_path, result, last_vqa = _rs, _rsp, _rr, _rv
                _outer_files.capture(_rr)
            else:
                _outer_files.restore()  # rejected regen overwrote the best's files

        _outer_files.restore()
        _outer_files.cleanup()


    mcp_result = validation_pipeline(
        script_path,
        asset_name,
        run_id,
        mcp_mode=mcp_mode,
        export_result=result,
    )


    log_stage(7, "Export Quality Summary")

    quality = run_quality_analysis(
        asset_name,
        run_id,
    )

    log_quality_summary(quality)


    log_stage(75, "Visual Quality Assessment")
    vqa = last_vqa or {}
    if not vqa or vqa.get("skipped"):
        print("Visual QA         : skipped (no vision model / render available)")
    else:
        print(f"Visual score      : {vqa.get('visual_score', -1)}/10")
        if vqa.get("feedback"):
            print(f"Feedback          : {vqa['feedback']}")
        if vqa.get("missing_parts"):
            log_list("Missing parts", vqa["missing_parts"], limit=6)
        if vqa.get("visible_parts"):
            log_list("Visible parts", vqa["visible_parts"], limit=6)


    if (_os.environ.get("ARGUS_PAINT", "1") == "1"
            and _os.environ.get("ARGUS_FAST", "0") != "1"
            and result.glb_path):
        log_stage(77, "Diffusion Texture Paint")
        import shutil as _sh
        _final_dir = Path(result.glb_path).parent
        _pre_score = (int(vqa.get("visual_score", -1))
                      if (vqa and not vqa.get("skipped")) else -1)
        # Back up the un-painted deliverables (incl. the rig, baked before the
        # paint stage) so a repaint that makes things worse can be reverted.
        _paint_baks: dict = {}
        for _nm in (f"{asset_name}.glb", f"{asset_name}.fbx", f"{asset_name}.blend",
                    f"{asset_name}_preview.png"):
            _src = _final_dir / _nm
            if _src.exists():
                _bak = _src.with_name(_src.name + ".prepaint")
                try:
                    _sh.copy2(_src, _bak)
                    _paint_baks[_src] = _bak
                except Exception:  # noqa: BLE001
                    pass
        try:
            from core.painter import paint_asset
            _painted = paint_asset(_final_dir, asset_name, prompt)
        except Exception as _paint_exc:  # noqa: BLE001
            logger.warning("[PAINT] stage failed: %s", _paint_exc)
            _painted = False
        if _painted:
            # Quality gate: keep the paint ONLY if it scores >= the un-painted
            # asset. The img2img repaint can crush albedo to near-black or leave
            # projection gaps — never let it ship a worse asset than we had.
            try:
                _post_vqa, _, _ = _score_render(
                    result.glb_path, prompt, part_data,
                    spec=current_spec, reference_image=reference_image)
                _post_score = (int(_post_vqa.get("visual_score", -1))
                               if not _post_vqa.get("skipped") else -1)
            except Exception:  # noqa: BLE001
                _post_score = -1
            if 0 <= _post_score < _pre_score:
                for _src, _bak in _paint_baks.items():
                    try:
                        _sh.copy2(_bak, _src)
                    except Exception:  # noqa: BLE001
                        pass
                print(f"Texture paint     : reverted — repaint {_post_score}/10 "
                      f"< un-painted {_pre_score}/10 (kept procedural materials)")
            else:
                print(f"Texture paint     : applied (img2img) — "
                      f"{_post_score}/10 vs {_pre_score}/10 un-painted")
        else:
            print("Texture paint     : skipped")
        for _bak in _paint_baks.values():
            try:
                _bak.unlink()
            except Exception:  # noqa: BLE001
                pass


    log_stage(8, "Structure Memory Approval")

    local_report = local_structural_critique(
        script,
        prompt,
        part_data,
    )
    if isinstance(quality, dict) and isinstance(vqa, dict) and not vqa.get("skipped"):
        quality["visual_score"] = vqa.get("visual_score", -1)
    mcp_report = mcp_result.as_dict()
    result_report = result.as_dict()
    candidate_ok, candidate_reason = evaluate_structure_memory_candidate(
        part_data=part_data,
        quality=quality,
        mcp_report=mcp_report,
        local_report=local_report,
    )

    if not candidate_ok:
        log_value("Memory status", "skipped")
        log_value("Reason", candidate_reason)
    else:
        approval_context = {
            "prompt": prompt,
            "asset_name": asset_name,
            "run_id": run_id,
            "blueprint": part_data.get("structure_blueprint", ""),
            "asset_path": result.glb_path,
            "preview_path": result.preview_path,
            "quality": quality,
            "mcp_report": mcp_report,
        }
        approval_fn = memory_approval_callback or default_memory_approval
        approved = bool(approval_fn(approval_context))

        if not approved:
            log_value("Memory status", "skipped")
            log_value("Reason", "user did not approve this structure as a reference")
        else:
            memory_updated, memory_status = record_structure_memory(
                user_prompt=prompt,
                part_data=part_data,
                script=script,
                quality=quality,
                mcp_report=mcp_report,
                local_report=local_report,
                result=result_report,
            )
            log_value("Memory status", "updated" if memory_updated else "skipped")
            log_value("Reason", memory_status)


    try:
        import json as _json
        manifest = {
            "asset_name": asset_name,
            "prompt": prompt,
            "category": part_data.get("category"),
            "material": part_data.get("material"),
            "poly_budget": part_data.get("poly_budget"),
            "visual_score": vqa.get("visual_score") if vqa and not vqa.get("skipped") else None,
            "topology_severity": getattr(mcp_result, "severity", "clean"),
            "topology_grounded": getattr(mcp_result, "grounded", True),
        }
        Path(result.glb_path).with_name("manifest.json").write_text(
            _json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as _manifest_exc:  # noqa: BLE001
        logger.warning("[MANIFEST] write failed: %s", _manifest_exc)

    # Ship the concept reference image alongside the asset so it's visible in
    # out/final/<asset>/ (and survives the one-folder-per-asset cleanup that
    # wipes out/runs). The bytes were cached by _await_reference during the run.
    try:
        _ref_img = _ref_box.get("img")
        if _ref_img and result.glb_path:
            _ext = "jpg" if _ref_img[:3] == b"\xff\xd8\xff" else "png"
            _ref_dest = Path(result.glb_path).with_name(f"reference.{_ext}")
            _ref_dest.write_bytes(_ref_img)
            print(f"Reference   : {_ref_dest.name}")
    except Exception as _ref_save_exc:  # noqa: BLE001
        logger.warning("[VISION] final reference copy failed: %s", _ref_save_exc)

    print("\n[ARGUS COMPLETE]")

    print(f"Final Asset : {result.glb_path}")
    if result.preview_path:
        print(f"Preview     : {result.preview_path}")
    print(f"Project     : {run_id}")

    return True


def audit_library(fix: bool = False) -> int:
    """ARGUS inspects its own asset library with its own validators and, with
    fix=True, regenerates every failing asset — no human triage involved.

    Verdict sources, in order of trust:
      - manifest.json written at generation time (topology severity, visual
        score, groundedness, original prompt)
      - for legacy assets without a manifest: the vision model re-scores the
        existing preview render directly.
    Returns the number of assets still failing afterwards."""
    import json as _json

    threshold = int(os.environ.get("ARGUS_AUDIT_THRESHOLD", "7"))
    final_root = OUTPUT_ROOT / "final"
    if not final_root.exists():
        print("[AUDIT] No asset library at out/final yet.")
        return 0

    failing: list[tuple[Path, str, str]] = []  # (folder, prompt, reason)
    folders = sorted(p for p in final_root.iterdir() if p.is_dir())

    print(f"\n[AUDIT] Inspecting {len(folders)} asset(s) "
          f"(pass = topology clean AND visual >= {threshold})\n")
    for folder in folders:
        name = folder.name
        prompt = name.replace("_", " ")
        reasons: list[str] = []
        manifest_path = folder / "manifest.json"

        if manifest_path.exists():
            try:
                m = _json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                m = {}
            prompt = (m.get("prompt") or "").strip() or prompt
            sev = str(m.get("topology_severity") or "unknown").lower()
            score = m.get("visual_score")
            if sev != "clean":
                reasons.append(f"topology {sev}")
            if isinstance(score, (int, float)) and 0 <= score < threshold:
                reasons.append(f"visual {int(score)}/10")
            if not m.get("topology_grounded", True):
                reasons.append("not grounded")
        else:
            # Legacy asset: let the vision model judge the stored preview.
            preview = folder / f"{name}_preview.png"
            if preview.exists():
                try:
                    from core.llm import score_visual_quality
                    vqa = score_visual_quality(preview.read_bytes(), prompt, {})
                    score = vqa.get("visual_score", -1)
                    if vqa.get("skipped"):
                        reasons.append("no manifest, vision unavailable")
                    elif 0 <= score < threshold:
                        reasons.append(
                            f"vision re-check {int(score)}/10 — "
                            + (vqa.get("feedback") or "")[:90])
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[AUDIT] vision check failed for %s: %s", name, exc)
                    reasons.append("no manifest, vision check failed")
            else:
                reasons.append("no manifest and no preview")

        if not (folder / f"{name}.glb").exists():
            reasons.append("missing GLB")

        status = "FAIL" if reasons else "PASS"
        print(f"[AUDIT] {status:4s}  {name}"
              + (f"  — {'; '.join(reasons)}" if reasons else ""))
        if reasons:
            failing.append((folder, prompt, "; ".join(reasons)))

    print(f"\n[AUDIT] {len(folders) - len(failing)}/{len(folders)} pass; "
          f"{len(failing)} need regeneration.")
    if not fix or not failing:
        return len(failing)

    print(f"\n[AUDIT] Regenerating {len(failing)} asset(s)…")
    still_failing = 0
    for i, (folder, prompt, why) in enumerate(failing, 1):
        print(f"\n{'=' * 60}\n[AUDIT {i}/{len(failing)}] {folder.name} — {why}")
        print(f"[AUDIT] prompt: {prompt}")
        pre_mtime = folder.stat().st_mtime if folder.exists() else None
        try:
            ok = run_pipeline(prompt=prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[AUDIT] regeneration crashed for %s: %s", folder.name, exc)
            ok = False
        if ok and pre_mtime is not None and folder.exists() \
                and folder.stat().st_mtime == pre_mtime:
            # The new run picked a different asset name; the old folder is
            # superseded — keep one folder per asset.
            shutil.rmtree(folder, ignore_errors=True)
            print(f"[AUDIT] removed superseded folder {folder.name}")
        if not ok:
            still_failing += 1
    print(f"\n[AUDIT] Complete: {len(failing) - still_failing}/{len(failing)} "
          "regenerated successfully.")
    return still_failing


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str)
    parser.add_argument("--mcp", action="store_true")
    parser.add_argument(
        "--batch-file",
        type=str,
        metavar="FILE",
        help="Path to a .txt or .json file containing one prompt per line/entry. "
             "Runs all prompts sequentially (or in parallel with --workers).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel generation workers for --batch-file (default: 1).",
    )
    parser.add_argument(
        "--batch-report",
        type=str,
        metavar="FILE",
        help="Optional path to write a JSON batch report (only used with --batch-file).",
    )
    parser.add_argument(
        "--no-concept",
        action="store_true",
        help="Skip reference-image generation and build straight from the prompt. "
             "Exposed for ablation runs (eval/conditions.py); the concept image is "
             "on by default.",
    )
    parser.add_argument(
        "--poly-budget",
        type=str,
        choices=sorted(VALID_POLY_BUDGETS),
        help="Override the planner's polygon budget.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Inspect every asset in out/final with ARGUS's own validators and report pass/fail.",
    )
    parser.add_argument(
        "--audit-fix",
        action="store_true",
        help="Audit the library, then automatically regenerate every failing asset.",
    )

    args = parser.parse_args()

    print("\n[KEY POOL STATUS]")
    print(key_pool_status())

    if args.audit or args.audit_fix:
        sys.exit(0 if audit_library(fix=args.audit_fix) == 0 else 1)

    if args.batch_file:
        from core.batch import load_prompts, run_batch

        prompts = load_prompts(args.batch_file)
        if not prompts:
            print("[ARGUS] No prompts found in batch file.")
            sys.exit(1)

        report_path = Path(args.batch_report) if args.batch_report else None
        results = run_batch(
            prompts,
            workers=max(1, args.workers),
            mcp_mode=args.mcp,
            report_path=report_path,
        )
        passed = sum(1 for r in results if r["success"])
        sys.exit(0 if passed > 0 else 1)

    prompt = args.prompt

    if not prompt:

        prompt = input("\nEnter object prompt:\n> ").strip()

    success = run_pipeline(
        prompt=prompt,
        poly_budget=args.poly_budget,
        mcp_mode=args.mcp,
        use_concept_pipeline=not args.no_concept,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
