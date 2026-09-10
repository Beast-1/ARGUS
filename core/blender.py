
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from core.script_safety import sanitize_generated_code


try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

BLENDER_PATH: str = os.getenv("BLENDER_PATH", "blender")

BLENDER_EXEC_TIMEOUT: int = int(os.getenv("BLENDER_EXEC_TIMEOUT", "300"))
BLENDER_MCP_TIMEOUT: int = int(os.getenv("BLENDER_MCP_TIMEOUT", "120"))
# Prefer ARGUS_OUTPUT_ROOT — the name main.py:77 uses. This module previously read
# only ARGUS_OUT_ROOT, so setting the documented variable moved main.py's output but
# NOT the Blender exports, silently splitting a run across two directories. The legacy
# name still works. Read at import time, so a per-run output directory must be set
# before this module is imported (i.e. one process per run — see eval/runner.py).
ARGUS_OUT_ROOT: Path = Path(
    os.getenv("ARGUS_OUTPUT_ROOT") or os.getenv("ARGUS_OUT_ROOT") or "out"
)
ARGUS_MAX_REPAIRS: int = int(os.getenv("ARGUS_MAX_REPAIRS", "4"))
ARGUS_SEED: int = int(os.getenv("ARGUS_SEED", "42"))

logger = logging.getLogger("ARGUS.blender")
logger.setLevel(logging.WARNING)


class RepairClass(str, Enum):
    SYNTAX_ERROR = "syntax_error"
    ATTRIBUTE_ERROR = "attribute_error"
    RUNTIME_ERROR = "runtime_error"
    MATERIAL_NODE_ERROR = "material_node_error"
    BMESH_ERROR = "bmesh_error"
    TOPOLOGY_ERROR = "topology_error"
    EXPORT_ERROR = "export_error"
    MODULE_ERROR = "module_error"
    KEY_ERROR = "key_error"
    TYPE_ERROR = "type_error"
    UNKNOWN = "unknown"


@dataclass
class ArgusTraceback:
    raw: str
    repair_class: RepairClass = RepairClass.UNKNOWN
    error_type: str = ""
    message: str = ""
    file: str = ""
    line: int = 0
    snippet: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["repair_class"] = self.repair_class.value
        return d


@dataclass
class McpResult:
    success: bool

    mesh_count: int = 0

    manifold_errors: int = 0
    non_manifold_faces: int = 0

    ngon_count: int = 0
    isolated_verts: int = 0
    open_edges: int = 0

    tri_faces: int = 0
    quad_faces: int = 0

    severity: str = "clean"

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    raw_stdout: str = ""
    disconnected_components: int = 0
    floating_components: int = 0

    @property
    def topology_clean(self) -> bool:
        return (
            self.manifold_errors == 0
            and self.non_manifold_faces == 0
            and self.isolated_verts == 0
            and self.ngon_count == 0
            and self.disconnected_components <= 1
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExportResult:
    run_id: str
    name: str
    success: bool

    glb_path: str = ""
    fbx_path: str = ""
    blend_path: str = ""
    preview_path: str = ""

    glb_bytes: int = 0
    fbx_bytes: int = 0
    blend_bytes: int = 0
    preview_bytes: int = 0

    traceback: Optional[dict] = None
    mcp: Optional[dict] = None
    # Poly-budget enforcement report (before/after triangles, whether the
    # export landed inside its budget). None when enforcement was off.
    poly: Optional[dict] = None

    repair_class: str = RepairClass.UNKNOWN.value

    iter_num: int = 0
    elapsed_sec: float = 0.0

    seed: int = ARGUS_SEED

    def as_dict(self) -> dict:
        return asdict(self)


# sanitize_script/validate_script/_FORBIDDEN_MODULE_ROOTS used to live here as a
# second, independent (and weaker — validate_script was a bare ast.parse syntax
# check with no safety denylist at all) copy of what core/script_safety.py now
# does once, shared with core/llm.py. validate_script and _FORBIDDEN_MODULE_ROOTS
# were dead code — defined, never called from anywhere in the repo — which made
# it easy to mistake this module for having its own enforcement point when it
# didn't. sanitize_generated_code below is the one real function that had actual
# callers, now imported instead of duplicated with a different regex.

_TRACEBACK_HEADER = re.compile(r"Traceback \(most recent call last\)")
_LOCATION_RE = re.compile(r'File "([^"]+)", line (\d+)')
_ERROR_TYPE_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_]*(?:Error|Exception|Warning)): (.+)$"
)


_REPAIR_CLASS_MAP = {
    "SyntaxError": RepairClass.SYNTAX_ERROR,
    "IndentationError": RepairClass.SYNTAX_ERROR,
    "TabError": RepairClass.SYNTAX_ERROR,
    "AttributeError": RepairClass.ATTRIBUTE_ERROR,
    "RuntimeError": RepairClass.RUNTIME_ERROR,
    "TypeError": RepairClass.TYPE_ERROR,
    "KeyError": RepairClass.KEY_ERROR,
    "ModuleNotFoundError": RepairClass.MODULE_ERROR,
    "ImportError": RepairClass.MODULE_ERROR,
}


def parse_traceback(stderr: str) -> ArgusTraceback:

    if not stderr:
        return ArgusTraceback(raw=stderr)

    lines = stderr.splitlines()

    tb_start = -1

    for i, ln in enumerate(lines):

        if _TRACEBACK_HEADER.search(ln):
            tb_start = i
            break

    tb_block = (
        "\n".join(lines[tb_start:])
        if tb_start >= 0
        else stderr
    )

    locations = _LOCATION_RE.findall(tb_block)

    file_hint, line_hint = ("", 0)

    if locations:
        file_hint = locations[-1][0]
        line_hint = int(locations[-1][1])

    error_type = ""
    message = ""

    for ln in reversed(tb_block.splitlines()):

        m = _ERROR_TYPE_RE.match(ln.strip())

        if m:
            error_type = m.group(1)
            message = m.group(2)
            break

    repair_class = _REPAIR_CLASS_MAP.get(
        error_type,
        RepairClass.UNKNOWN,
    )

    return ArgusTraceback(
        raw=stderr,
        repair_class=repair_class,
        error_type=error_type,
        message=message,
        file=file_hint,
        line=line_hint,
    )


def _embed_script_safely(script: str) -> str:

    import base64

    encoded = base64.b64encode(
        script.encode("utf-8")
    ).decode("ascii")

    return textwrap.dedent(f"""
    import base64 as _b64

    _ARGUS_GENERATED_SOURCE = _b64.b64decode(
        "{encoded}"
    ).decode("utf-8")

    exec(
        compile(
            _ARGUS_GENERATED_SOURCE,
            "<argus_generated>",
            "exec",
        )
    )
    """).strip()


# ── ARGUS auto-rig stage ─────────────────────────────────────────────────────
# Injected as its own block (single braces, NOT inside the f-string footer).
# ARGUS already builds separate, named parts, so rigging is just: drop a bone
# at each articulating part's pivot, bind it rigid (full weight to one bone),
# and bake a demo spin. Exports a real skeleton + animation into the GLB/FBX
# plus a rig.json joint manifest. Guarded + opt-out via ARGUS_RIG=0.
_RIG_STAGE = r'''
def _argus_build_rig(manifest_path=None):
    import bpy, mathutils, math, json
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        return None
    _SPIN = ("wheel", "tyre", "tire", "caster", "castor", "roller")
    _HINGE = ("door", "lid", "hatch", "hood", "trunk", "flap", "gate", "cover")
    def _kind(nm):
        n = nm.lower()
        if any(t in n for t in _SPIN):
            return "spin"
        if any(t in n for t in _HINGE):
            return "hinge"
        return None
    artic = [(o, _kind(o.name)) for o in meshes]
    artic = [(o, k) for o, k in artic if k]
    if not artic:
        return None  # nothing articulates — skip the armature entirely
    arm_data = bpy.data.armatures.new("ARGUS_Rig")
    arm = bpy.data.objects.new("ARGUS_Rig", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    root = eb.new("root")
    root.head = (0.0, 0.0, 0.0)
    root.tail = (0.0, 0.0, 0.25)
    manifest = {"joints": []}
    bone_for = {}
    for o, k in artic:
        # World bbox centre is the robust pivot (glTF bakes the transform into
        # verts, so object.location is (0,0,0) and would orbit the part).
        ws = [o.matrix_world @ mathutils.Vector(c) for c in o.bound_box]
        ctr = sum(ws, mathutils.Vector()) / 8.0
        dims = list(o.dimensions)
        ax = min(range(3), key=lambda i: dims[i])  # axle = thinnest dimension
        axisv = [0.0, 0.0, 0.0]
        axisv[ax] = 1.0
        b = eb.new(o.name + "_bone")
        b.head = ctr
        b.tail = ctr + mathutils.Vector(axisv) * max(dims[ax], 0.08)
        b.parent = root
        bone_for[o.name] = b.name
        manifest["joints"].append({"node": o.name, "type": k,
            "pivot": [round(c, 5) for c in ctr], "axis": axisv})
    bpy.ops.object.mode_set(mode="OBJECT")
    # Rigid skin: every part follows ONE bone at full weight (mechanical parts
    # don't deform). Articulating parts -> their bone; the rest -> root.
    for o in meshes:
        tgt = bone_for.get(o.name, "root")
        vg = o.vertex_groups.get(tgt) or o.vertex_groups.new(name=tgt)
        vg.add([v.index for v in o.data.vertices], 1.0, "REPLACE")
        m = o.modifiers.new("ARGUS_Armature", "ARMATURE")
        m.object = arm
    spin = [o.name for o, k in artic if k == "spin"]
    if spin:
        sc = bpy.context.scene
        sc.frame_start, sc.frame_end = 1, 60
        arm.animation_data_create()
        arm.animation_data.action = bpy.data.actions.new("ARGUS_Spin")
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="POSE")
        for nm in spin:
            pb = arm.pose.bones.get(bone_for[nm])
            if not pb:
                continue
            pb.rotation_mode = "XYZ"
            for fr, ang in ((1, 0.0), (60, 2 * math.pi)):
                sc.frame_set(fr)
                pb.rotation_euler = (0.0, ang, 0.0)  # spin around the axle
                pb.keyframe_insert("rotation_euler", frame=fr)
        bpy.ops.object.mode_set(mode="OBJECT")
        sc.frame_set(1)
    if manifest_path:
        try:
            with open(manifest_path, "w") as _rf:
                json.dump(manifest, _rf, indent=2)
        except Exception:
            pass
    print("[RIG] armature: %d joint(s), %d spin" % (len(artic), len(spin)))
    return arm
'''


def build_export_script(
    script_path,
    name,
    run_id,
    poly_max: int = 0,
):

    script_path = Path(script_path)

    raw_script = script_path.read_text(
        encoding="utf-8"
    )

    raw_script = sanitize_generated_code(raw_script)

    generated_block = _embed_script_safely(raw_script)

    final_dir = (
        ARGUS_OUT_ROOT / "final" / run_id
    ).resolve()

    glb_path = final_dir / f"{name}.glb"
    fbx_path = final_dir / f"{name}.fbx"
    blend_path = final_dir / f"{name}.blend"
    preview_path = final_dir / f"{name}_preview.png"
    lod1_path = final_dir / f"{name}_lod1.glb"
    lod2_path = final_dir / f"{name}_lod2.glb"
    collision_path = final_dir / f"{name}_collision.glb"

    header = textwrap.dedent("""
    import bpy
    import mathutils
    import os

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    """).strip()

    footer = textwrap.dedent(f"""
    import math as _math
    import json as _json
    _argus_meshes = [
        o for o in bpy.data.objects
        if o.type == "MESH"
    ]

    def _argus_tri_total(_objs=None):
        _t = 0
        for _o in (_argus_meshes if _objs is None else _objs):
            for _p in _o.data.polygons:
                _t += max(1, len(_p.vertices) - 2)
        return _t

    # Per-object triangle floor, scaled to how many objects share the budget.
    # A fixed floor is unsatisfiable for part-heavy assets: a 35-part fire
    # hydrant against a flat 200-triangle floor reserves 7,000 triangles for a
    # 5,000 budget, leaving the allocator nothing to distribute -- it gave up
    # and left that asset 31% over. Half the fair share keeps the floor
    # meaningful without exceeding the budget; 24 is the hard minimum below
    # which a part stops reading as a shape at all (a cube is 12).
    def _argus_poly_floor(_n_objects, _poly_max):
        return max(24, min(200, _poly_max // max(1, _n_objects) // 2))

    # Water-fill the budget across objects, honouring the per-object floor.
    # A single scene-wide ratio overshoots: any object small enough that
    # floor/tris exceeds that ratio gets clamped back up to the floor, and
    # those triangles have to come out of everyone else's share. Note the
    # distinction between an object already under the floor (costs its full
    # size, cannot be decimated) and one clamped *to* the floor (costs only
    # the floor) -- charging the latter its original size makes pinning raise
    # the fixed cost, lowering the ratio, pinning more, until everything is
    # pinned and nothing is decimated at all.
    def _argus_allocate(_per_obj, _poly_max, _floor):
        _untouchable = set()
        for _n in _per_obj:
            if _per_obj[_n] <= _floor:
                _untouchable.add(_n)
        _clamped = set()
        _ratio = 1.0
        for _ in range(12):
            _fixed = _floor * len(_clamped)
            for _n in _untouchable:
                _fixed += _per_obj[_n]
            _flexible = 0
            for _n in _per_obj:
                if _n not in _untouchable and _n not in _clamped:
                    _flexible += _per_obj[_n]
            if _flexible <= 0:
                break
            _ratio = max(0.0, _poly_max - _fixed) / _flexible
            _newly = set()
            for _n in _per_obj:
                if _n in _untouchable or _n in _clamped:
                    continue
                if _per_obj[_n] * _ratio < _floor:
                    _newly.add(_n)
            if not _newly:
                break
            _clamped |= _newly
        _out = dict()
        for _n in _per_obj:
            if _n in _untouchable:
                _out[_n] = 1.0
            elif _n in _clamped:
                _out[_n] = float(_floor) / _per_obj[_n]
            else:
                _out[_n] = min(1.0, _ratio)
        return _out, _ratio

    if not _argus_meshes:
        raise RuntimeError(
            "No mesh objects produced"
        )

    # ── ARGUS game-ready finishing pass (T1.6 + T2.10) ────────────────────────
    # Controlled, deterministic post-process the LLM can't skip: weighted normals
    # for clean bevel shading, a cheap dirty-vertex-color AO bake for surface
    # richness, and a guaranteed UV layer per mesh. Every step is guarded so it
    # can never fail the export.
    _argus_ao_bake = os.environ.get("ARGUS_AO_BAKE", "1") != "0"
    for _m in _argus_meshes:
        try:
            bpy.context.view_layer.objects.active = _m
            bpy.ops.object.select_all(action='DESELECT')
            _m.select_set(True)
            if _m.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            # Guarantee a UV layer for texturing / texel density.
            if not _m.data.uv_layers:
                _m.data.uv_layers.new(name="ARGUS_UV")
            # Apply pending subdivision modifiers (argus_smooth) first so the
            # finishing passes below operate on the final smooth surface.
            for _sub in [_sm for _sm in _m.modifiers if _sm.type == 'SUBSURF']:
                try:
                    bpy.ops.object.modifier_apply(modifier=_sub.name)
                except Exception as _sub_exc:
                    print("[FINISH SUBSURF skip] " + str(_sub_exc))
            # Bevel sharp edges (angle-limited, % width so it's scale-safe) so
            # razor edges catch light like real manufactured parts.
            if os.environ.get("ARGUS_BEVEL", "1") != "0":
                try:
                    _bev = _m.modifiers.new("ARGUS_BEVEL", 'BEVEL')
                    _bev.offset_type = 'PERCENT'
                    _bev.width_pct = 5.0
                    _bev.segments = 1   # 1 segment = single chamfer, minimal tris
                    _bev.limit_method = 'ANGLE'
                    _bev.angle_limit = _math.radians(35)
                    bpy.ops.object.modifier_apply(modifier=_bev.name)
                except Exception as _bev_exc:
                    print("[FINISH BEVEL skip] " + str(_bev_exc))
            # Smooth shading + smooth-by-angle: curved surfaces read as smooth,
            # hard edges stay crisp. This is what makes weighted-normals visible.
            try:
                bpy.ops.object.shade_smooth()
                for _sm_op in ("shade_auto_smooth", "shade_smooth_by_angle"):
                    _fn = getattr(bpy.ops.object, _sm_op, None)
                    if _fn is None:
                        continue
                    try:
                        _fn(angle=_math.radians(35)); break
                    except Exception:
                        try:
                            _fn(); break
                        except Exception:
                            continue
            except Exception as _sm_exc:
                print("[FINISH SMOOTH skip] " + str(_sm_exc))
            # Weighted-normal modifier → crisp shading across bevels.
            try:
                _wn = _m.modifiers.new("ARGUS_WN", 'WEIGHTED_NORMAL')
                _wn.keep_sharp = True
                bpy.ops.object.modifier_apply(modifier=_wn.name)
            except Exception as _wn_exc:
                print("[FINISH WN skip] " + str(_wn_exc))
            # Re-quadify: bevel and WN sometimes leave stray tris.
            # bmesh.ops.join_triangles is context-safe (no mode switch needed).
            try:
                import bmesh as _bm_mod
                _bm_qt = _bm_mod.new()
                _bm_qt.from_mesh(_m.data)
                _bm_mod.ops.join_triangles(
                    _bm_qt, faces=_bm_qt.faces[:],
                    angle_face_threshold=0.698,
                    angle_shape_threshold=0.698,
                )
                _bm_qt.to_mesh(_m.data)
                _bm_qt.free()
                _m.data.update()
            except Exception as _qt_exc:
                print("[FINISH QUADIFY skip] " + str(_qt_exc))
            # Cheap AO: dirty vertex colors darken crevices (exported as COLOR_0).
            if _argus_ao_bake:
                try:
                    if not _m.data.color_attributes:
                        _m.data.color_attributes.new(
                            name="ARGUS_AO", type='BYTE_COLOR', domain='CORNER')
                    bpy.ops.paint.vertex_color_dirt(
                        blur_strength=1.0, blur_iterations=1,
                        dirt_angle=0.0, dirt_only=False)
                except Exception as _ao_exc:
                    print("[FINISH AO skip] " + str(_ao_exc))
        except Exception as _fin_exc:
            print("[FINISH skip] " + str(_fin_exc))

    # ── ARGUS smart-material bake: Cycles → per-object PBR maps ─────────────
    # Procedural smart materials use Pointiness/AO nodes that only evaluate in
    # Cycles, so their richness is baked to textures here. Baked maps travel in
    # the GLB/blend and render correctly in EEVEE previews and any engine.
    _bake_tagged = [
        _m for _m in _argus_meshes
        if _m.material_slots and _m.material_slots[0].material
        and _m.material_slots[0].material.get("argus_bake")
    ]
    if (os.environ.get("ARGUS_BAKE", "1") != "0" and _bake_tagged
            and len(_bake_tagged) <= 24):
        try:
            _bake_res = int(os.environ.get("ARGUS_BAKE_RES", "512"))
            _prev_engine = bpy.context.scene.render.engine
            bpy.context.scene.render.engine = 'CYCLES'
            bpy.context.scene.cycles.samples = 16
            try:
                bpy.context.scene.cycles.use_denoising = False
                bpy.context.scene.cycles.device = 'CPU'
            except Exception:
                pass
            for _m in _bake_tagged:
                try:
                    bpy.context.view_layer.objects.active = _m
                    bpy.ops.object.select_all(action='DESELECT')
                    _m.select_set(True)
                    if "ARGUS_BAKE_UV" not in _m.data.uv_layers:
                        _m.data.uv_layers.new(name="ARGUS_BAKE_UV")
                    _m.data.uv_layers.active = _m.data.uv_layers["ARGUS_BAKE_UV"]
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
                    bpy.ops.object.mode_set(mode='OBJECT')

                    _src_mat = _m.material_slots[0].material
                    _nt = _src_mat.node_tree
                    _img_c = bpy.data.images.new(_m.name + "_albedo",
                                                 _bake_res, _bake_res)
                    _img_r = bpy.data.images.new(_m.name + "_rough",
                                                 _bake_res, _bake_res)
                    _img_r.colorspace_settings.name = 'Non-Color'
                    _tex = _nt.nodes.new('ShaderNodeTexImage')
                    _uvn = _nt.nodes.new('ShaderNodeUVMap')
                    _uvn.uv_map = "ARGUS_BAKE_UV"
                    _nt.links.new(_uvn.outputs[0], _tex.inputs[0])

                    _tex.image = _img_c
                    _nt.nodes.active = _tex
                    bpy.ops.object.bake(type='DIFFUSE',
                                        pass_filter={{'COLOR'}},
                                        use_clear=True, margin=4)
                    _tex.image = _img_r
                    _nt.nodes.active = _tex
                    bpy.ops.object.bake(type='ROUGHNESS',
                                        use_clear=True, margin=4)
                    _nt.nodes.remove(_tex)
                    _nt.nodes.remove(_uvn)
                    _img_c.pack()
                    _img_r.pack()

                    _bm = bpy.data.materials.new(_m.name + "_BAKED")
                    _bm.use_nodes = True
                    _bnt = _bm.node_tree
                    _bb = _bnt.nodes.get("Principled BSDF")
                    _bb.inputs["Metallic"].default_value = float(
                        _src_mat.get("argus_metallic", 0.0))
                    _ic = _bnt.nodes.new('ShaderNodeTexImage')
                    _ic.image = _img_c
                    _ir = _bnt.nodes.new('ShaderNodeTexImage')
                    _ir.image = _img_r
                    _ir.image.colorspace_settings.name = 'Non-Color'
                    _buv = _bnt.nodes.new('ShaderNodeUVMap')
                    _buv.uv_map = "ARGUS_BAKE_UV"
                    _bnt.links.new(_buv.outputs[0], _ic.inputs[0])
                    _bnt.links.new(_buv.outputs[0], _ir.inputs[0])
                    _bnt.links.new(_ic.outputs["Color"], _bb.inputs["Base Color"])
                    _bnt.links.new(_ir.outputs["Color"], _bb.inputs["Roughness"])
                    _coat = float(_src_mat.get("argus_coat", 0.0))
                    if _coat > 0:
                        try:
                            _bb.inputs["Coat Weight"].default_value = _coat
                            _bb.inputs["Coat Roughness"].default_value = float(
                                _src_mat.get("argus_coat_rough", 0.1))
                        except KeyError:
                            pass
                    _m.data.materials.clear()
                    _m.data.materials.append(_bm)
                except Exception as _obj_bake_exc:
                    print("[BAKE part skip] " + _m.name + ": " + str(_obj_bake_exc))
            bpy.context.scene.render.engine = _prev_engine
            print("[ARGUS BAKE] " + str(len(_bake_tagged)) + " object(s) baked to "
                  + str(_bake_res) + "px PBR maps")
        except Exception as _bake_exc:
            print("[BAKE skip] " + str(_bake_exc))
    elif _bake_tagged:
        # Bake disabled or too many parts: flatten smart materials so the
        # GLB/FBX exporters don't drop the linked (procedural) colour sockets.
        for _m in _bake_tagged:
            try:
                _sm = _m.material_slots[0].material
                _c = list(_sm.get("argus_color", (0.6, 0.6, 0.6)))
                _fm = bpy.data.materials.new(_sm.name + "_FLAT")
                _fm.use_nodes = True
                _fb = _fm.node_tree.nodes.get("Principled BSDF")
                _fb.inputs["Base Color"].default_value = (_c[0], _c[1], _c[2], 1.0)
                _fb.inputs["Metallic"].default_value = float(_sm.get("argus_metallic", 0.0))
                _fb.inputs["Roughness"].default_value = float(_sm.get("argus_rough", 0.6))
                _coat = float(_sm.get("argus_coat", 0.0))
                if _coat > 0:
                    try:
                        _fb.inputs["Coat Weight"].default_value = _coat
                        _fb.inputs["Coat Roughness"].default_value = float(
                            _sm.get("argus_coat_rough", 0.1))
                    except KeyError:
                        pass
                _m.data.materials.clear()
                _m.data.materials.append(_fm)
            except Exception as _flat_exc:
                print("[FLAT skip] " + str(_flat_exc))

    print("[ARGUS_OK]")

    _final_dir = {json.dumps(str(final_dir))}

    os.makedirs(_final_dir, exist_ok=True)

    _preview_path = {json.dumps(str(preview_path))}

    try:
        _world_points = [
            obj.matrix_world @ mathutils.Vector(corner)
            for obj in _argus_meshes
            for corner in obj.bound_box
        ]
        _min_corner = mathutils.Vector((
            min(point.x for point in _world_points),
            min(point.y for point in _world_points),
            min(point.z for point in _world_points),
        ))
        _max_corner = mathutils.Vector((
            max(point.x for point in _world_points),
            max(point.y for point in _world_points),
            max(point.z for point in _world_points),
        ))
        _center = (_min_corner + _max_corner) * 0.5
        # Guard only against a degenerate zero-size bound — a 1.0 m floor here
        # made every small prop (a 9 cm mug) render as a speck in the frame.
        _span = max((_max_corner - _min_corner).length, 0.05)

        _camera_data = bpy.data.cameras.new("ARGUS_PREVIEW_CAMERA")
        _camera = bpy.data.objects.new("ARGUS_PREVIEW_CAMERA", _camera_data)
        bpy.context.collection.objects.link(_camera)
        _camera.location = _center + mathutils.Vector((_span * 1.55, -_span * 1.9, _span * 1.1))
        _direction = _center - _camera.location
        _camera.rotation_euler = _direction.to_track_quat("-Z", "Y").to_euler()
        _camera_data.lens = 50
        bpy.context.scene.camera = _camera

        # ── Sun-based 3-point rig — reliable in EEVEE headless ───────────────
        def _add_sun(name, rx_deg, ry_deg, energy):
            ld = bpy.data.lights.new(name, type="SUN")
            lo = bpy.data.objects.new(name, ld)
            bpy.context.collection.objects.link(lo)
            lo.rotation_euler = (math.radians(rx_deg), 0, math.radians(ry_deg))
            ld.energy = energy
            ld.angle  = math.radians(8)
            return lo

        _add_sun("ARGUS_KEY",  50,  30, 5.0)   # main key — warm front-left
        _add_sun("ARGUS_FILL", 30, 200, 2.5)   # fill — cool right
        _add_sun("ARGUS_RIM",  15, 130, 2.0)   # rim — top-rear highlight
        _add_sun("ARGUS_BOT",  -25, 10, 1.0)   # soft underlight

        # World: soft vertical gradient (darker floor, lighter sky)
        _world = bpy.context.scene.world
        if _world is None:
            _world = bpy.data.worlds.new("ARGUS_WORLD")
            bpy.context.scene.world = _world
        _world.use_nodes = True
        _wn = _world.node_tree.nodes
        _wl = _world.node_tree.links
        _bg = _wn.get("Background") or _wn.new("ShaderNodeBackground")
        try:
            _wgrad = _wn.new("ShaderNodeTexGradient")
            _wmap = _wn.new("ShaderNodeMapping")
            _wcoord = _wn.new("ShaderNodeTexCoord")
            _wmap.inputs["Rotation"].default_value = (0.0, -1.5708, 0.0)
            _wl.new(_wcoord.outputs["Window"], _wmap.inputs["Vector"])
            _wl.new(_wmap.outputs["Vector"], _wgrad.inputs["Vector"])
            _wramp = _wn.new("ShaderNodeValToRGB")
            _wramp.color_ramp.elements[0].color = (0.045, 0.05, 0.06, 1.0)
            _wramp.color_ramp.elements[1].color = (0.16, 0.18, 0.21, 1.0)
            _wl.new(_wgrad.outputs["Fac"], _wramp.inputs["Fac"])
            _wl.new(_wramp.outputs["Color"], _bg.inputs["Color"])
        except Exception:
            _bg.inputs["Color"].default_value = (0.12, 0.13, 0.15, 1.0)
        _bg.inputs["Strength"].default_value = 1.1

        # Temporary ground plane so the asset reads grounded (removed after).
        _ground = None
        try:
            _gmesh = bpy.data.meshes.new("ARGUS_GROUND")
            _gsize = _span * 6.0
            _gmesh.from_pydata(
                [(-_gsize, -_gsize, 0.0), (_gsize, -_gsize, 0.0),
                 (_gsize, _gsize, 0.0), (-_gsize, _gsize, 0.0)],
                [], [(0, 1, 2, 3)])
            _gmesh.update()
            _ground = bpy.data.objects.new("ARGUS_GROUND", _gmesh)
            bpy.context.collection.objects.link(_ground)
            _gmat = bpy.data.materials.new("ARGUS_GROUND_MAT")
            _gmat.use_nodes = True
            _gb = _gmat.node_tree.nodes.get("Principled BSDF")
            _gb.inputs["Base Color"].default_value = (0.085, 0.09, 0.10, 1.0)
            _gb.inputs["Roughness"].default_value = 0.95
            _gmesh.materials.append(_gmat)
        except Exception as _g_exc:
            print("[PREVIEW ground skip] " + str(_g_exc))

        bpy.context.scene.render.resolution_x = 960
        bpy.context.scene.render.resolution_y = 540
        bpy.context.scene.render.film_transparent = False
        # AgX tonemapping + gentle contrast = beauty-render look for free.
        try:
            bpy.context.scene.view_settings.view_transform = 'AgX'
            bpy.context.scene.view_settings.look = 'AgX - Punchy'
        except Exception:
            try:
                bpy.context.scene.view_settings.view_transform = 'Filmic'
            except Exception:
                pass
        # Use EEVEE — faster, no noise, consistent across all hardware
        for _eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
            try:
                bpy.context.scene.render.engine = _eng
                break
            except TypeError:
                pass
        if hasattr(bpy.context.scene, "eevee"):
            bpy.context.scene.eevee.taa_render_samples = 24
            try:
                bpy.context.scene.eevee.use_shadows = True
            except Exception:
                pass
        bpy.context.scene.render.filepath = _preview_path
        bpy.ops.render.render(write_still=True)
        print("[PREVIEW IMAGE] " + _preview_path)
        if _ground is not None:
            try:
                bpy.data.objects.remove(_ground, do_unlink=True)
            except Exception:
                pass
    except Exception as _preview_exc:
        print("[PREVIEW FAILED] " + str(_preview_exc))

    # ── Auto-rig BEFORE the .blend save (ARGUS_RIG) ──────────────────────────
    # Must run before the save so the .blend carries the skeleton — otherwise
    # the texture painter (Stage 77) reloads the rig-less .blend, repaints, and
    # re-exports a GLB with the armature stripped (the original skins:0 bug).
    # Rigid full-weight skinning binds existing verts; the later export-only
    # triangulation never adds verts, so the weights survive into GLB/FBX.
    if os.environ.get("ARGUS_RIG", "1") != "0":
        try:
            _argus_build_rig({json.dumps(str(glb_path.with_name("rig.json")))})
        except Exception as _rig_exc:
            print("[RIG skip] " + str(_rig_exc))

    try:
        bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
    except Exception:
        pass
    bpy.ops.wm.save_as_mainfile(
        filepath={json.dumps(str(blend_path))}
    )
    print("[BLEND_SAVED] " + {json.dumps(str(blend_path))})

    # ── Controlled triangulation for export only ─────────────────────────────
    # The .blend above keeps clean quads for editing. Engines need triangles,
    # and if we leave it to the exporters glTF uses a naive split (sliver tris)
    # while FBX ships quads — so GLB and FBX disagree. Triangulating here with
    # BEAUTY (balanced diagonals) gives sliver-free, identical topology in both.
    import bmesh as _bm_tri_mod
    for _m in _argus_meshes:
        try:
            _bt = _bm_tri_mod.new()
            _bt.from_mesh(_m.data)
            try:
                _bm_tri_mod.ops.triangulate(
                    _bt, faces=_bt.faces[:],
                    quad_method="BEAUTY", ngon_method="BEAUTY")
            except TypeError:
                _bm_tri_mod.ops.triangulate(_bt, faces=_bt.faces[:])
            _bt.to_mesh(_m.data)
            _bt.free()
            _m.data.update()
        except Exception as _tri_exc:
            print("[EXPORT TRI skip] " + str(_tri_exc))

    # ── Poly-budget enforcement (export only) ────────────────────────────────
    # The .blend above is saved BEFORE this, so the editable source keeps full
    # detail; only the shipped GLB is budgeted.
    #
    # Until this existed the budget was advisory: a prompt asked the model to
    # emit its own dissolve_limit snippet and nothing verified the result, so
    # real output ran 1.5x-56x over the "game-ready" ceiling (a gas pump shipped
    # at 281,182 triangles against a 5,000 budget) and the vision scorer, which
    # never sees a triangle count, happily scored it.
    #
    # Two stages, in this order for a reason. Planar dissolve first, because it
    # removes coplanar detail at essentially no silhouette cost and does most of
    # the work for free (281,182 -> 11,223 on that gas pump). Collapse only for
    # whatever remains: collapse treats a dense sphere exactly like a flat box
    # face, and running it alone at the resulting 56x ratio left the boxy body
    # perfect while crumpling the domed top into jagged garbage. Dissolving
    # first leaves a 0.29 collapse ratio instead of 0.015 — 20x gentler on the
    # curved parts that actually need their triangles.
    _poly_max = {poly_max}
    _poly_before = _argus_tri_total()
    _poly_after_planar = _poly_before
    _poly_ratio = None
    if _poly_max > 0 and _poly_before > _poly_max:
        for _pm in _argus_meshes:
            try:
                bpy.context.view_layer.objects.active = _pm
                _pmod = _pm.modifiers.new("ARGUS_POLY_PLANAR", "DECIMATE")
                _pmod.decimate_type = "DISSOLVE"
                _pmod.angle_limit = _math.radians(5.0)
                bpy.ops.object.modifier_apply(modifier=_pmod.name)
            except Exception as _pl_exc:
                print("[POLY planar skip] " + str(_pl_exc))
        # Dissolve leaves ngons; re-triangulate so the count and the export agree.
        for _pm in _argus_meshes:
            try:
                _pb = _bm_tri_mod.new()
                _pb.from_mesh(_pm.data)
                _bm_tri_mod.ops.triangulate(_pb, faces=_pb.faces[:])
                _pb.to_mesh(_pm.data)
                _pb.free()
                _pm.data.update()
            except Exception:
                pass
        _poly_after_planar = _argus_tri_total()

        # Up to three collapse rounds: DECIMATE lands approximately on its
        # ratio, and per-object rounding across many parts accumulated to +16%
        # in testing, so a corrective round is what actually hits the target.
        for _attempt in range(3):
            if _argus_tri_total() <= _poly_max:
                break
            _per_obj = dict()
            for _pm in _argus_meshes:
                _per_obj[_pm.name] = _argus_tri_total([_pm])
            _floor = _argus_poly_floor(len(_per_obj), _poly_max)
            _ratios, _poly_ratio = _argus_allocate(_per_obj, _poly_max, _floor)
            for _pm in _argus_meshes:
                _r = _ratios.get(_pm.name, 1.0)
                if _r >= 1.0 or _per_obj.get(_pm.name, 0) <= _floor:
                    continue
                try:
                    bpy.context.view_layer.objects.active = _pm
                    _dmod = _pm.modifiers.new("ARGUS_POLY_BUDGET", "DECIMATE")
                    _dmod.decimate_type = "COLLAPSE"
                    _dmod.ratio = _r
                    bpy.ops.object.modifier_apply(modifier=_dmod.name)
                except Exception as _dc_exc:
                    print("[POLY collapse skip] " + str(_dc_exc))

    _poly_final = _argus_tri_total()
    print("ARGUS_POLY_REPORT:" + _json.dumps(dict(
        budget=_poly_max,
        before=_poly_before,
        after_planar=_poly_after_planar,
        after=_poly_final,
        within_budget=bool(_poly_max <= 0 or _poly_final <= _poly_max),
        reduction=round(_poly_before / max(1, _poly_final), 2),
    )))

    bpy.ops.export_scene.gltf(
        filepath={json.dumps(str(glb_path))},
        export_format='GLB',
        use_selection=False,
    )
    print("[GLB_EXPORTED] " + {json.dumps(str(glb_path))})

    try:
        bpy.ops.export_scene.fbx(
            filepath={json.dumps(str(fbx_path))},
            use_selection=False,
        )
        print("[FBX_EXPORTED] " + {json.dumps(str(fbx_path))})
    except Exception as _fbx_exc:
        print("[FBX SKIP] " + str(_fbx_exc))

    # ── LOD + collision exports (game-engine deliverables) ───────────────────
    # LOD1/LOD2 via decimate on duplicates (originals untouched); collision is
    # a single convex hull of all visible geometry, named UCX_* so Unreal (and
    # most pipelines) auto-recognise it. Everything guarded — these extras can
    # never fail the main export. Off by default; enable with ARGUS_LODS=1.
    if os.environ.get("ARGUS_LODS", "0") == "1":
        import bmesh as _bm_lod
        for _lod_path, _ratio in [({json.dumps(str(lod1_path))}, 0.5),
                                  ({json.dumps(str(lod2_path))}, 0.2)]:
            _dups = []
            try:
                for _src in _argus_meshes:
                    _dup = _src.copy()
                    _dup.data = _src.data.copy()
                    bpy.context.collection.objects.link(_dup)
                    bpy.context.view_layer.objects.active = _dup
                    _dec = _dup.modifiers.new("ARGUS_DECIMATE", 'DECIMATE')
                    _dec.ratio = _ratio
                    bpy.ops.object.modifier_apply(modifier=_dec.name)
                    _dups.append(_dup)
                for _o in bpy.data.objects:
                    _o.select_set(False)
                for _d in _dups:
                    _d.select_set(True)
                bpy.ops.export_scene.gltf(
                    filepath=_lod_path, export_format='GLB', use_selection=True)
                print("[LOD_EXPORTED] " + _lod_path)
            except Exception as _lod_exc:
                print("[LOD skip] " + str(_lod_exc))
            finally:
                for _d in _dups:
                    try:
                        bpy.data.objects.remove(_d, do_unlink=True)
                    except Exception:
                        pass
        try:
            _hb = _bm_lod.new()
            for _src in _argus_meshes:
                _mw = _src.matrix_world
                for _v in _src.data.vertices:
                    _hb.verts.new(_mw @ _v.co)
            _res = _bm_lod.ops.convex_hull(_hb, input=_hb.verts[:])
            _junk = list(set(
                _e for _e in (list(_res.get("geom_unused", [])) +
                              list(_res.get("geom_interior", [])))
                if isinstance(_e, _bm_lod.types.BMVert)))
            if _junk:
                _bm_lod.ops.delete(_hb, geom=_junk, context='VERTS')
            _hull_mesh = bpy.data.meshes.new("ARGUS_COLLISION")
            _hb.to_mesh(_hull_mesh)
            _hb.free()
            _hull_obj = bpy.data.objects.new("UCX_" + {json.dumps(name)} + "_01", _hull_mesh)
            bpy.context.collection.objects.link(_hull_obj)
            for _o in bpy.data.objects:
                _o.select_set(False)
            _hull_obj.select_set(True)
            bpy.ops.export_scene.gltf(
                filepath={json.dumps(str(collision_path))},
                export_format='GLB', use_selection=True)
            print("[COLLISION_EXPORTED] " + {json.dumps(str(collision_path))})
            bpy.data.objects.remove(_hull_obj, do_unlink=True)
        except Exception as _col_exc:
            print("[COLLISION skip] " + str(_col_exc))
    """).strip()

    # Topology analysis runs in the SAME Blender process right after export —
    # the report rides home on stdout, so Stage 6 needs no extra Blender boot.
    return "\n\n".join([header, generated_block, _RIG_STAGE, footer, _MCP_ANALYSIS_BODY])


_CANCEL_POLL_SEC = 0.4
_TERMINATE_GRACE_SEC = 5.0


def _run_cancellable(cmd, *, timeout, **popen_kwargs):
    """subprocess.run()-compatible (returns a CompletedProcess, raises
    TimeoutExpired on timeout) — but polls the pipeline's cancel flag every
    ~0.4s and terminates (then kills, after a grace period) the Blender process
    the moment Cancel is requested, instead of leaving it running untouched
    until the full multi-minute timeout elapses.

    Previously every Blender subprocess.run() call here used a hard timeout as
    its only exit path: clicking Cancel mid-render set a flag nothing checked
    until the pipeline's NEXT loop-boundary checkpoint (main.py's
    _past_deadline()), so the UI could show "Cancelling..." for up to
    BLENDER_EXEC_TIMEOUT (5 minutes by default) while the subprocess — the step
    that dominates a run's wall-clock time — kept running untouched.
    """
    from main import is_cancelled  # deferred: main.py imports core.blender at load time

    popen_kwargs.setdefault("stdout", subprocess.PIPE)
    popen_kwargs.setdefault("stderr", subprocess.PIPE)
    popen_kwargs.setdefault("text", True)
    proc = subprocess.Popen(cmd, **popen_kwargs)
    start = time.monotonic()

    while True:
        try:
            stdout, stderr = proc.communicate(timeout=_CANCEL_POLL_SEC)
            return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            pass

        if is_cancelled():
            logger.info("[CANCEL] terminating Blender subprocess (pid %d)", proc.pid)
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(
                cmd, round(time.monotonic() - start, 1), output=stdout, stderr=stderr,
            )

        if time.monotonic() - start >= timeout:
            proc.kill()
            try:
                stdout, stderr = proc.communicate(timeout=_TERMINATE_GRACE_SEC)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)


def run_blender(
    script_path,
    name,
    run_id,
    fix_path=None,
    iter_num=None,
    seed=ARGUS_SEED,
    poly_max: int = 0,
):

    t0 = time.monotonic()

    export_script = build_export_script(
        script_path,
        name,
        run_id,
        poly_max=poly_max,
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".py",
        mode="w",
        encoding="utf-8",
    ) as fh:

        fh.write(export_script)

        temp_path = fh.name

    try:

        result = _run_cancellable(
            [
                BLENDER_PATH,
                "--background",
                "--python",
                temp_path,
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=BLENDER_EXEC_TIMEOUT,
        )

    except subprocess.TimeoutExpired:

        tb = ArgusTraceback(
            raw="Blender timed out",
            repair_class=RepairClass.RUNTIME_ERROR,
            error_type="TimeoutError",
            message="Execution timeout",
        )

        return ExportResult(
            run_id=run_id,
            name=name,
            success=False,
            traceback=tb.as_dict(),
        )

    finally:

        try:
            os.unlink(temp_path)
        except OSError:
            pass

    elapsed = time.monotonic() - t0

    _combined_output = (result.stderr or "") + (result.stdout or "")
    _has_traceback = "Traceback (most recent call last)" in _combined_output

    if (
        result.returncode != 0
        or _has_traceback
        or "[ARGUS_OK]" not in result.stdout
    ):

        tb = parse_traceback(_combined_output)

        return ExportResult(
            run_id=run_id,
            name=name,
            success=False,
            traceback=tb.as_dict(),
            repair_class=tb.repair_class.value,
            elapsed_sec=elapsed,
        )

    final_dir = (
        ARGUS_OUT_ROOT / "final" / run_id
    ).resolve()

    glb_path = final_dir / f"{name}.glb"
    fbx_path = final_dir / f"{name}.fbx"
    blend_path = final_dir / f"{name}.blend"
    preview_path = final_dir / f"{name}_preview.png"

    if "[GLB_EXPORTED]" not in result.stdout:
        _stdout_tail = result.stdout[-600:]
        return ExportResult(
            run_id=run_id,
            name=name,
            success=False,
            traceback={
                "error_class": "ExportOperatorFailed",
                "message": (
                    "Blender reached [ARGUS_OK] (mesh objects were created) but the "
                    "gltf export operator did not complete. This usually means a C-level "
                    "crash or operator context error occurred after mesh creation. "
                    "Simplify the geometry, remove any bpy.ops.paint.* calls, and ensure "
                    "all mesh objects are linked to the scene collection before export."
                ),
                "raw": _stdout_tail,
            },
            elapsed_sec=elapsed,
        )

    if not glb_path.exists() or glb_path.stat().st_size < 5000:
        return ExportResult(
            run_id=run_id,
            name=name,
            success=False,
            traceback={
                "error_class": "EmptyExport",
                "message": (
                    f"GLB is empty or missing ({glb_path.stat().st_size if glb_path.exists() else 0} bytes). "
                    "Script ran without errors and gltf export was called, but the file is "
                    "too small to contain geometry. Ensure bpy.data.objects.new() objects "
                    "are linked to the collection (bpy.context.collection.objects.link(obj)) "
                    "and that no bpy.ops.object.delete() is called after building geometry."
                ),
                "raw": result.stdout[-500:],
            },
            elapsed_sec=elapsed,
        )

    _poly_inline = None
    _poly_match = _POLY_REPORT_RE.search(result.stdout or "")
    if _poly_match:
        try:
            _poly_inline = json.loads(_poly_match.group(1))
        except ValueError:
            _poly_inline = None
    if _poly_inline:
        # Surfaced on the pipeline's own stdout, not just captured into the
        # result: budget enforcement that silently rewrites the shipped mesh is
        # exactly the kind of thing an operator should be able to see happening.
        _pb, _pa = _poly_inline.get("before", 0), _poly_inline.get("after", 0)
        if _pb != _pa:
            print(f"Poly budget       : {_pb:,} -> {_pa:,} tris "
                  f"(budget {_poly_inline.get('budget', 0):,}, "
                  f"{_poly_inline.get('reduction', 1)}x)")
        elif _poly_inline.get("budget"):
            print(f"Poly budget       : {_pa:,} tris (within budget, untouched)")

    _mcp_inline = None
    _mcp_match = _MCP_REPORT_RE.search(result.stdout or "")
    if _mcp_match:
        try:
            _mcp_inline = json.loads(_mcp_match.group(1))
        except ValueError:
            _mcp_inline = None

    return ExportResult(
        run_id=run_id,
        name=name,
        success=True,
        glb_path=str(glb_path),
        fbx_path=str(fbx_path),
        blend_path=str(blend_path),
        preview_path=str(preview_path) if preview_path.exists() else "",
        glb_bytes=glb_path.stat().st_size if glb_path.exists() else 0,
        fbx_bytes=fbx_path.stat().st_size if fbx_path.exists() else 0,
        blend_bytes=blend_path.stat().st_size if blend_path.exists() else 0,
        preview_bytes=preview_path.stat().st_size if preview_path.exists() else 0,
        elapsed_sec=elapsed,
        mcp=_mcp_inline,
        poly=_poly_inline,
    )


_MCP_ANALYSIS_BODY = r"""
import bpy
import bmesh
import json
import mathutils
import sys

_errors = []
_warnings = []

objs = [o for o in bpy.data.objects if o.type == "MESH"]

mesh_count = len(objs)
disconnected_components = 0
floating_components = 0
_bounds = []

manifold_errors = 0
non_manifold_faces = 0
ngon_count = 0
isolated_verts = 0
open_edges = 0

tri_faces = 0
quad_faces = 0

# T1.4 — real connectivity: keep sampled world-space verts per object AND a
# per-object BVH of the full surface. Vertex-to-vertex proximity alone gives
# false "disconnected" verdicts when a part attaches to the smooth SIDE of
# another (e.g. a nozzle on a plain cylinder, whose side has no mid-height
# verts) — so we also test nearest-point-ON-SURFACE via the BVH.
from mathutils.bvhtree import BVHTree as _BVHTree
_obj_world_verts = []
_obj_bvh = []

for obj in objs:
    world_points = [
        obj.matrix_world @ mathutils.Vector(corner)
        for corner in obj.bound_box
    ]
    _mw = obj.matrix_world
    _verts = obj.data.vertices
    _n = len(_verts)
    if _n <= 240:
        _sample = [(_mw @ v.co) for v in _verts]
    else:
        _step = (_n // 240) + 1
        _sample = [(_mw @ _verts[i].co) for i in range(0, _n, _step)]
    _obj_world_verts.append(_sample)
    try:
        _wv = [(_mw @ v.co) for v in _verts]
        _polys = [tuple(p.vertices) for p in obj.data.polygons]
        _obj_bvh.append(_BVHTree.FromPolygons(_wv, _polys) if _polys else None)
    except Exception:
        _obj_bvh.append(None)
    if world_points:
        min_corner = mathutils.Vector((
            min(point.x for point in world_points),
            min(point.y for point in world_points),
            min(point.z for point in world_points),
        ))
        max_corner = mathutils.Vector((
            max(point.x for point in world_points),
            max(point.y for point in world_points),
            max(point.z for point in world_points),
        ))
        _bounds.append((min_corner, max_corner))

    if world_points and min(point.z for point in world_points) > 0.05:
        floating_components += 1

    bm = bmesh.new()

    try:
        bm.from_mesh(obj.data)

        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        for face in bm.faces:

            vcount = len(face.verts)

            if vcount == 3:
                tri_faces += 1

            elif vcount == 4:
                quad_faces += 1

            else:
                ngon_count += 1

            if not face.is_valid:
                non_manifold_faces += 1

        for edge in bm.edges:

            if len(edge.link_faces) != 2:
                manifold_errors += 1

                if len(edge.link_faces) == 0:
                    open_edges += 1

        for vert in bm.verts:

            if not vert.link_edges:
                isolated_verts += 1

    finally:
        bm.free()

if _bounds:
    scene_min = mathutils.Vector((
        min(bounds[0].x for bounds in _bounds),
        min(bounds[0].y for bounds in _bounds),
        min(bounds[0].z for bounds in _bounds),
    ))
    scene_max = mathutils.Vector((
        max(bounds[1].x for bounds in _bounds),
        max(bounds[1].y for bounds in _bounds),
        max(bounds[1].z for bounds in _bounds),
    ))
    connect_threshold = max((scene_max - scene_min).length * 0.03, 0.03)

    parent = list(range(len(_bounds)))

    def _find(idx):
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def _union(a, b):
        root_a = _find(a)
        root_b = _find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    def _axis_gap(a_min, a_max, b_min, b_max):
        if a_max < b_min:
            return b_min - a_max
        if b_max < a_min:
            return a_min - b_max
        return 0.0

    def _surface_touch(i, j):
        # True if object i's sampled points come within connect_threshold of
        # object j's actual SURFACE (BVH nearest-point), or vice-versa. Surface
        # testing catches attachments onto smooth, sparsely-tessellated sides
        # that vertex-to-vertex distance misses.
        for src, bvh in ((i, _obj_bvh[j]), (j, _obj_bvh[i])):
            if bvh is None:
                continue
            for _co in _obj_world_verts[src]:
                _loc, _nrm, _idx, _d = bvh.find_nearest(_co, connect_threshold * 1.5)
                if _loc is not None and _d is not None and _d <= connect_threshold:
                    return True
        return False

    for i, (a_min, a_max) in enumerate(_bounds):
        for j in range(i + 1, len(_bounds)):
            b_min, b_max = _bounds[j]
            # Fast bbox reject — if even bounding boxes are far apart, skip the
            # expensive surface test (loose 2x margin so we never miss a contact).
            gap = max(
                _axis_gap(a_min.x, a_max.x, b_min.x, b_max.x),
                _axis_gap(a_min.y, a_max.y, b_min.y, b_max.y),
                _axis_gap(a_min.z, a_max.z, b_min.z, b_max.z),
            )
            if gap > connect_threshold * 2.0:
                continue
            if _surface_touch(i, j):
                _union(i, j)

    disconnected_components = len({_find(idx) for idx in range(len(_bounds))})

if disconnected_components > 1:
    _warnings.append(f"assembly_has_{disconnected_components}_separate_clusters")
if floating_components > 0:
    _warnings.append(f"{floating_components}_floating_mesh_objects")

severity = "clean"

if isolated_verts > 0:
    severity = "critical"

elif manifold_errors > 0:
    severity = "high"

elif disconnected_components > 1:
    # Assembly check: every part must touch the rest (KD-tree surface
    # proximity). A crate in 6 separate clusters is not a valid prop even
    # when each individual mesh is watertight.
    severity = "high"

elif ngon_count > 0:
    severity = "medium"

report = {
    "success": isolated_verts == 0,

    "mesh_count": mesh_count,

    "manifold_errors": manifold_errors,
    "non_manifold_faces": non_manifold_faces,

    "ngon_count": ngon_count,
    "isolated_verts": isolated_verts,
    "open_edges": open_edges,

    "tri_faces": tri_faces,
    "quad_faces": quad_faces,

    "severity": severity,

    "errors": _errors,
    "warnings": _warnings,
    "disconnected_components": disconnected_components,
    "floating_components": floating_components,
}

print("ARGUS_MCP_REPORT:" + json.dumps(report))

sys.stdout.flush()
"""

_MCP_REPORT_RE = re.compile(
    r"^ARGUS_MCP_REPORT:(.+)$",
    re.MULTILINE,
)

_POLY_REPORT_RE = re.compile(
    r"^ARGUS_POLY_REPORT:(.+)$",
    re.MULTILINE,
)


def run_mcp_analysis(
    script_path,
    name,
    run_id,
    fix_path=None,
):
    glb_path = (ARGUS_OUT_ROOT / "final" / run_id / f"{name}.glb").resolve()

    if not glb_path.exists():
        logger.warning("[MCP] GLB not found for analysis: %s", glb_path)
        return McpResult(success=False, errors=["glb_not_found"])

    glb_path_str = json.dumps(str(glb_path))
    wrapper = textwrap.dedent(f"""
import bpy, json, sys

# Clear default scene
for _o in list(bpy.data.objects):
    bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes):
    bpy.data.meshes.remove(_m)

# Import the exported GLB
bpy.ops.import_scene.gltf(filepath={glb_path_str})
bpy.context.view_layer.update()
""").strip() + "\n\n" + _MCP_ANALYSIS_BODY

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".py",
        mode="w",
        encoding="utf-8",
    ) as fh:

        fh.write(wrapper)

        temp_path = fh.name

    try:

        result = _run_cancellable(
            [
                BLENDER_PATH,
                "--background",
                "--python",
                temp_path,
            ],
            text=True,
            encoding="utf-8",
            timeout=BLENDER_MCP_TIMEOUT,
        )

    finally:

        try:
            os.unlink(temp_path)
        except OSError:
            pass

    match = _MCP_REPORT_RE.search(result.stdout)

    if not match:

        return McpResult(
            success=False,
            errors=["mcp_report_missing"],
        )

    try:
        data = json.loads(match.group(1))
    except ValueError:
        return McpResult(success=False, errors=["mcp_report_unparseable"])

    return mcp_result_from_report(data, raw=result.stdout)


def mcp_result_from_report(data: dict, raw: str = "") -> McpResult:
    """ARGUS_MCP_REPORT dict → McpResult, shared by the inline (in-build),
    headless re-import and live-MCP validation paths."""
    return McpResult(
        success=data.get("success", False),
        mesh_count=data.get("mesh_count", 0),
        manifold_errors=data.get("manifold_errors", 0),
        non_manifold_faces=data.get("non_manifold_faces", 0),
        ngon_count=data.get("ngon_count", 0),
        isolated_verts=data.get("isolated_verts", 0),
        open_edges=data.get("open_edges", 0),
        tri_faces=data.get("tri_faces", 0),
        quad_faces=data.get("quad_faces", 0),
        severity=data.get("severity", "unknown"),
        errors=data.get("errors", []),
        warnings=data.get("warnings", []),
        disconnected_components=data.get("disconnected_components", 0),
        floating_components=data.get("floating_components", 0),
        raw_stdout=raw,
    )


def run_quality_analysis(name, run_id):

    base = (
        ARGUS_OUT_ROOT / "final" / run_id
    ).resolve()

    info = {
        "name": name,
        "run_id": run_id,
    }

    for ext in ("glb", "fbx", "blend"):

        fp = base / f"{name}.{ext}"

        if fp.exists():

            info[f"export_bytes_{ext}"] = fp.stat().st_size
            info[f"export_path_{ext}"] = str(fp)

    info["success"] = "export_bytes_glb" in info

    return info


_MULTIVIEW_RENDER_SCRIPT = r"""
import bpy, math, sys, json, os
import mathutils
import numpy as np

args  = json.loads(sys.argv[sys.argv.index("--") + 1])
GLB   = args["glb_path"]
OUT   = args["out_path"]
W     = int(args.get("tile", 384))
H     = W
VIEWS = args.get("views", [[-0.65, 0.35], [0.9, 0.3], [2.4, 0.35], [-0.65, 1.2]])

# ── clear scene ───────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()
for blk in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
    for item in list(blk):
        blk.remove(item)

bpy.ops.import_scene.gltf(filepath=GLB)
bpy.context.view_layer.update()

pts = []
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    wm = obj.matrix_world
    pts.extend((wm @ v.co)[:] for v in obj.data.vertices)
if not pts:
    sys.exit(1)
arr = np.array(pts)
lo, hi = arr.min(axis=0), arr.max(axis=0)
center = mathutils.Vector(((lo + hi) * 0.5).tolist())
extent = float(np.linalg.norm(hi - lo)) or 1.0

# Orientation normalisation: the fixed view angles below assume the asset's
# LENGTH runs along X. A long asset built along Y (e.g. an 18 m locomotive)
# would otherwise be rendered end-on — the critic then sees a featureless
# block and scores it ~2/10 ("missing wheels") even though it is correct.
# Spin it 90° about Z (render-only) so the side views see the full profile.
if (hi[1] - lo[1]) > (hi[0] - lo[0]) * 1.15:
    _rotM = (mathutils.Matrix.Translation(center)
             @ mathutils.Matrix.Rotation(math.radians(90.0), 4, "Z")
             @ mathutils.Matrix.Translation(center).inverted())
    for _ob in list(bpy.data.objects):
        if _ob.parent is None:
            _ob.matrix_world = _rotM @ _ob.matrix_world
    bpy.context.view_layer.update()
    pts = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        wm = obj.matrix_world
        pts.extend((wm @ v.co)[:] for v in obj.data.vertices)
    arr = np.array(pts)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    center = mathutils.Vector(((lo + hi) * 0.5).tolist())
    extent = float(np.linalg.norm(hi - lo)) or 1.0

cam_d = bpy.data.cameras.new("Cam"); cam_d.lens = 50
cam_o = bpy.data.objects.new("Cam", cam_d)
bpy.context.scene.collection.objects.link(cam_o)
bpy.context.scene.camera = cam_o

def add_sun(name, rx, ry, energy):
    ld = bpy.data.lights.new(name, "SUN"); ld.energy = energy; ld.angle = math.radians(10)
    o = bpy.data.objects.new(name, ld); bpy.context.scene.collection.objects.link(o)
    o.rotation_euler = (math.radians(rx), 0, math.radians(ry))
add_sun("Key", 55, 30, 3.5); add_sun("Fill", 30, 200, 1.5); add_sun("Rim", 15, 130, 1.0)

world = bpy.context.scene.world; world.use_nodes = True
bg = world.node_tree.nodes.get("Background") or world.node_tree.nodes.new("ShaderNodeBackground")
bg.inputs["Color"].default_value = (0.04, 0.055, 0.075, 1.0)
bg.inputs["Strength"].default_value = 0.4

sc = bpy.context.scene
sc.render.resolution_x = W; sc.render.resolution_y = H
sc.render.film_transparent = False
sc.render.image_settings.file_format = "PNG"
for _eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        sc.render.engine = _eng; break
    except TypeError:
        pass
if hasattr(sc, "eevee"):
    sc.eevee.taa_render_samples = 24

def render_view(yaw, pitch, path):
    dist = extent * 1.2
    cam_o.location = (center.x + math.sin(yaw) * math.cos(pitch) * dist,
                      center.y - math.cos(yaw) * math.cos(pitch) * dist,
                      center.z + math.sin(pitch) * dist)
    cam_o.rotation_mode = "XYZ"
    cam_o.rotation_euler = (center - cam_o.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)

tiles = []
tmpdir = os.path.dirname(OUT)
for i, (yaw, pitch) in enumerate(VIEWS[:4]):
    tp = os.path.join(tmpdir, "view_%d.png" % i)
    render_view(float(yaw), float(pitch), tp)
    img = bpy.data.images.load(tp)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], 4)
    tiles.append(px)
    bpy.data.images.remove(img)

# Pad to four tiles so the 2x2 grid is always complete.
while len(tiles) < 4:
    tiles.append(np.zeros((H, W, 4), dtype=np.float32))

top = np.concatenate([tiles[0], tiles[1]], axis=1)
bot = np.concatenate([tiles[2], tiles[3]], axis=1)
grid = np.concatenate([bot, top], axis=0)   # numpy is bottom-up; keep order
gh, gw = grid.shape[0], grid.shape[1]

out_img = bpy.data.images.new("ARGUS_GRID", width=gw, height=gh)
out_img.pixels = grid.reshape(-1).tolist()
out_img.filepath_raw = OUT
out_img.file_format = "PNG"
out_img.save()
print("ARGUS_MULTIVIEW_OK")
"""


def render_multiview_grid(
    glb_path: "str | Path",
    *,
    tile: int = 384,
    timeout: int = 150,
) -> "Optional[bytes]":
    """Render four angles (front-iso, side, rear, top-down) composited into one
    2x2 PNG. A single image keeps multi-view visual QA to ONE vision API call.
    Returns PNG bytes on success, None on failure."""
    glb_path = Path(glb_path)
    if not glb_path.exists():
        return None

    views = [[-0.65, 0.35], [0.9, 0.3], [2.45, 0.35], [-0.65, 1.15]]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "grid.png"
        script_path = Path(tmpdir) / "mv_script.py"
        args_json = json.dumps({
            "glb_path": str(glb_path),
            "out_path": str(out_path),
            "tile": tile,
            "views": views,
        })
        script_path.write_text(_MULTIVIEW_RENDER_SCRIPT, encoding="utf-8")
        try:
            result = _run_cancellable(
                [BLENDER_PATH, "--background", "--python", str(script_path), "--", args_json],
                timeout=timeout,
            )
            if out_path.exists() and out_path.stat().st_size > 512:
                return out_path.read_bytes()
            logger.warning("[MV_RENDER] No output. stderr: %s", result.stderr[-600:])
        except subprocess.TimeoutExpired:
            logger.warning("[MV_RENDER] Timed out after %ds", timeout)
        except Exception as exc:
            logger.warning("[MV_RENDER] Failed: %s", exc)
        return None
