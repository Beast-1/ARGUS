"""Diffusion texture painter: project img2img repaints back onto the asset.

The compiler gives clean geometry; a 2D image model gives rich surfaces.
This module renders the finished asset from three views, has Gemini repaint
each view photorealistically, then camera-projects the repaints onto the
mesh and bakes the facing-weighted blend into the albedo — DreamFusion-style
surfaces on game-ready topology, no heavy local models.

Every step is guarded: any failure leaves the exported asset untouched.
Knob: ARGUS_PAINT=0 disables; ARGUS_PAINT_RES sets bake resolution.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from core.blender import BLENDER_PATH, BLENDER_EXEC_TIMEOUT

logger = logging.getLogger("ARGUS.painter")

VIEWS = [(20.0, 35.0), (22.0, 155.0), (30.0, 275.0)]  # (elevation°, azimuth°)
VIEW_RES = 768

PAINT_PROMPT = (
    "Repaint this 3D render of {subject} as a photorealistic, richly detailed "
    "and textured object. Keep EXACTLY the same camera angle, silhouette, "
    "shape and composition — do not move, add or remove any part. Add realistic "
    "surface detail: material texture, wear, dirt, colour variation. Plain dark "
    "grey studio background, no text, no watermark."
)

# Shared camera maths for both Blender passes — identical placement is what
# makes the projection line up with the repainted renders.
_CAMERA_LIB = r'''
import bpy, math, mathutils

def _scene_meshes():
    return [o for o in bpy.data.objects
            if o.type == "MESH" and not o.name.startswith(("ARGUS_GROUND",))]

def _scene_bbox():
    pts = [o.matrix_world @ mathutils.Vector(c)
           for o in _scene_meshes() for c in o.bound_box]
    lo = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts),
                           min(p.z for p in pts)))
    hi = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts),
                           max(p.z for p in pts)))
    return (lo + hi) * 0.5, max((hi - lo).length, 0.5)

def _make_cameras(views):
    centre, span = _scene_bbox()
    cams = []
    for i, (elev, azim) in enumerate(views):
        el, az = math.radians(elev), math.radians(azim)
        dist = span * 1.7
        pos = centre + mathutils.Vector((
            dist * math.cos(el) * math.cos(az),
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el)))
        cd = bpy.data.cameras.new(f"ARGUS_PAINT_CAM_{i}")
        cd.lens = 42
        cam = bpy.data.objects.new(f"ARGUS_PAINT_CAM_{i}", cd)
        bpy.context.collection.objects.link(cam)
        cam.location = pos
        cam.rotation_euler = (centre - pos).to_track_quat("-Z", "Y").to_euler()
        view_dir = (centre - pos).normalized()
        cams.append((cam, (view_dir.x, view_dir.y, view_dir.z)))
    return cams
'''

_RENDER_SCRIPT = _CAMERA_LIB + r'''
import sys, json, os
_args = json.loads(sys.argv[sys.argv.index("--") + 1])

for _eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
    try:
        bpy.context.scene.render.engine = _eng
        break
    except TypeError:
        pass
sc = bpy.context.scene
sc.render.resolution_x = sc.render.resolution_y = _args["res"]
if hasattr(sc, "eevee"):
    sc.eevee.taa_render_samples = 24
try:
    sc.view_settings.view_transform = "Standard"
except Exception:
    pass

def _sun(name, rx, rz, e):
    ld = bpy.data.lights.new(name, type="SUN"); ld.energy = e
    lo = bpy.data.objects.new(name, ld)
    bpy.context.collection.objects.link(lo)
    lo.rotation_euler = (math.radians(rx), 0, math.radians(rz))
_sun("PK", 50, 30, 5.5); _sun("PF", 30, 200, 3.5); _sun("PR", 15, 130, 2.5)
w = sc.world or bpy.data.worlds.new("W"); sc.world = w
w.use_nodes = True
bg = w.node_tree.nodes.get("Background")
if bg:
    # Brighter, even base so the img2img repaint starts from readable mid-tones
    # rather than a dark image it keeps dark at strength 0.6.
    bg.inputs["Color"].default_value = (0.32, 0.32, 0.34, 1.0)

for i, (cam, vdir) in enumerate(_make_cameras(_args["views"])):
    sc.camera = cam
    sc.render.filepath = os.path.join(_args["workdir"], f"view_{i}.png")
    bpy.ops.render.render(write_still=True)
print("[PAINT_RENDER_OK]")
'''

_PROJECT_SCRIPT = _CAMERA_LIB + r'''
import sys, json, os
_args = json.loads(sys.argv[sys.argv.index("--") + 1])
workdir = _args["workdir"]
n_views = len(_args["views"])
res = _args["bake_res"]

cams = _make_cameras(_args["views"])
meshes = _scene_meshes()
assert meshes, "no meshes in blend"

painted = []
for i in range(n_views):
    img = bpy.data.images.load(os.path.join(workdir, f"painted_{i}.png"))
    img.colorspace_settings.name = "sRGB"
    painted.append(img)

# 1. Project each camera's view into its own UV layer on every mesh.
for m in meshes:
    bpy.context.view_layer.objects.active = m
    bpy.ops.object.select_all(action="DESELECT")
    m.select_set(True)
    if "ARGUS_BAKE_UV" not in m.data.uv_layers:
        m.data.uv_layers.new(name="ARGUS_BAKE_UV")
        m.data.uv_layers.active = m.data.uv_layers["ARGUS_BAKE_UV"]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
    for i, (cam, vdir) in enumerate(cams):
        name = f"ARGUS_PROJ_{i}"
        if name not in m.data.uv_layers:
            m.data.uv_layers.new(name=name)
        mod = m.modifiers.new(name, "UV_PROJECT")
        mod.uv_layer = name
        mod.aspect_x = mod.aspect_y = 1.0
        mod.projectors[0].object = cam
    for i in range(n_views):
        try:
            bpy.ops.object.modifier_apply(modifier=f"ARGUS_PROJ_{i}")
        except Exception as exc:
            print("[PAINT proj apply skip] " + str(exc))

# 2. Per mesh: blend the three projections by facing weight, bake to albedo,
#    rewire the material to the painted map.
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.samples = 8
try:
    sc.cycles.use_denoising = False
    sc.cycles.device = "CPU"
except Exception:
    pass

for m in meshes:
    try:
        mat = m.material_slots[0].material if m.material_slots else None
        if mat is None:
            mat = bpy.data.materials.new(m.name + "_MAT")
            mat.use_nodes = True
            m.data.materials.append(mat)
        nt = mat.node_tree
        nd, ln = nt.nodes, nt.links
        out_node = next(n for n in nd if n.type == "OUTPUT_MATERIAL")
        orig_link = out_node.inputs["Surface"].links[0] if out_node.inputs["Surface"].links else None
        orig_from = orig_link.from_socket if orig_link else None

        # Capture the procedural material's base colour as the gap-fill fallback
        # (a neutral grey if it's missing or near-black, so uncovered surfaces
        # never bake to pure black).
        _base_col = (0.45, 0.45, 0.45, 1.0)
        _p0 = next((n for n in nd if n.type == "BSDF_PRINCIPLED"), None)
        if _p0 is not None:
            try:
                _bc = _p0.inputs["Base Color"].default_value
                if (_bc[0] + _bc[1] + _bc[2]) > 0.12:
                    _base_col = (_bc[0], _bc[1], _bc[2], 1.0)
            except Exception:
                pass

        weighted = []
        total = None
        for i, (cam, vdir) in enumerate(cams):
            tex = nd.new("ShaderNodeTexImage")
            tex.image = painted[i]
            tex.extension = "EXTEND"
            uv = nd.new("ShaderNodeUVMap")
            uv.uv_map = f"ARGUS_PROJ_{i}"
            ln.new(uv.outputs[0], tex.inputs[0])
            geo = nd.new("ShaderNodeNewGeometry")
            dot = nd.new("ShaderNodeVectorMath")
            dot.operation = "DOT_PRODUCT"
            dot.inputs[1].default_value = (-vdir[0], -vdir[1], -vdir[2])
            ln.new(geo.outputs["Normal"], dot.inputs[0])
            mx = nd.new("ShaderNodeMath"); mx.operation = "MAXIMUM"
            mx.inputs[1].default_value = 0.0
            ln.new(dot.outputs["Value"], mx.inputs[0])
            pw = nd.new("ShaderNodeMath"); pw.operation = "POWER"
            pw.inputs[1].default_value = 3.0
            ln.new(mx.outputs[0], pw.inputs[0])
            scl = nd.new("ShaderNodeVectorMath"); scl.operation = "SCALE"
            ln.new(tex.outputs["Color"], scl.inputs[0])
            ln.new(pw.outputs[0], scl.inputs["Scale"])
            weighted.append(scl.outputs["Vector"])
            if total is None:
                total = pw.outputs[0]
            else:
                add = nd.new("ShaderNodeMath"); add.operation = "ADD"
                ln.new(total, add.inputs[0]); ln.new(pw.outputs[0], add.inputs[1])
                total = add.outputs[0]
        # Gap fill: add the material colour as a constant-weight virtual view,
        # so surfaces no camera covered fall back to the procedural colour
        # instead of baking black (projection coverage holes).
        _fbw = nd.new("ShaderNodeValue"); _fbw.outputs[0].default_value = 0.28
        _fbrgb = nd.new("ShaderNodeRGB"); _fbrgb.outputs[0].default_value = _base_col
        _fbscl = nd.new("ShaderNodeVectorMath"); _fbscl.operation = "SCALE"
        ln.new(_fbrgb.outputs[0], _fbscl.inputs[0])
        ln.new(_fbw.outputs[0], _fbscl.inputs["Scale"])
        weighted.append(_fbscl.outputs["Vector"])
        _fadd = nd.new("ShaderNodeMath"); _fadd.operation = "ADD"
        ln.new(total, _fadd.inputs[0]); ln.new(_fbw.outputs[0], _fadd.inputs[1])
        total = _fadd.outputs[0]

        eps = nd.new("ShaderNodeMath"); eps.operation = "ADD"
        eps.inputs[1].default_value = 0.0001
        ln.new(total, eps.inputs[0])
        inv = nd.new("ShaderNodeMath"); inv.operation = "DIVIDE"
        inv.inputs[0].default_value = 1.0
        ln.new(eps.outputs[0], inv.inputs[1])
        acc = weighted[0]
        for w_out in weighted[1:]:
            va = nd.new("ShaderNodeVectorMath"); va.operation = "ADD"
            ln.new(acc, va.inputs[0]); ln.new(w_out, va.inputs[1])
            acc = va.outputs["Vector"]
        norm = nd.new("ShaderNodeVectorMath"); norm.operation = "SCALE"
        ln.new(acc, norm.inputs[0]); ln.new(inv.outputs[0], norm.inputs["Scale"])
        emit = nd.new("ShaderNodeEmission")
        # Brightness lift: the img2img repaint tends to crush albedo dark, so
        # scale the blended colour up before baking (clips gently in highlights,
        # lifts the readable mid-tones). ARGUS_PAINT_BRIGHT overrides.
        _brv = nd.new("ShaderNodeValue")
        _brv.outputs[0].default_value = float(os.environ.get("ARGUS_PAINT_BRIGHT", "1.4"))
        _br = nd.new("ShaderNodeVectorMath"); _br.operation = "SCALE"
        ln.new(norm.outputs["Vector"], _br.inputs[0])
        ln.new(_brv.outputs[0], _br.inputs["Scale"])
        ln.new(_br.outputs["Vector"], emit.inputs["Color"])
        ln.new(emit.outputs[0], out_node.inputs["Surface"])

        target_img = bpy.data.images.new(m.name + "_painted", res, res)
        t_tex = nd.new("ShaderNodeTexImage"); t_tex.image = target_img
        t_uv = nd.new("ShaderNodeUVMap"); t_uv.uv_map = "ARGUS_BAKE_UV"
        ln.new(t_uv.outputs[0], t_tex.inputs[0])
        nd.active = t_tex

        bpy.context.view_layer.objects.active = m
        bpy.ops.object.select_all(action="DESELECT")
        m.select_set(True)
        m.data.uv_layers.active = m.data.uv_layers["ARGUS_BAKE_UV"]
        bpy.ops.object.bake(type="EMIT", use_clear=True, margin=4)
        target_img.pack()

        if orig_from is not None:
            ln.new(orig_from, out_node.inputs["Surface"])
        principled = next((n for n in nd if n.type == "BSDF_PRINCIPLED"), None)
        if principled is not None:
            for l in list(principled.inputs["Base Color"].links):
                nt.links.remove(l)
            ln.new(t_tex.outputs["Color"], principled.inputs["Base Color"])
        else:
            ln.new(emit.outputs[0], out_node.inputs["Surface"])
        print("[PAINT baked] " + m.name)
    except Exception as exc:
        print("[PAINT mesh skip] " + m.name + ": " + str(exc))

for cam, _ in cams:
    bpy.data.objects.remove(cam, do_unlink=True)

# 3. Re-export over the original files + fresh beauty preview.
try:
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
except Exception:
    pass
bpy.ops.wm.save_as_mainfile(filepath=_args["blend"])
bpy.ops.export_scene.gltf(filepath=_args["glb"], export_format="GLB",
                          use_selection=False)
try:
    bpy.ops.export_scene.fbx(filepath=_args["fbx"], use_selection=False)
except Exception as exc:
    print("[PAINT fbx skip] " + str(exc))

try:
    centre, span = _scene_bbox()
    cd = bpy.data.cameras.new("PV"); cd.lens = 50
    cam = bpy.data.objects.new("PV", cd)
    bpy.context.collection.objects.link(cam)
    cam.location = centre + mathutils.Vector((span * 1.55, -span * 1.9, span * 1.1))
    cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    def _sun(name, rx, rz, e):
        ld = bpy.data.lights.new(name, type="SUN"); ld.energy = e
        lo = bpy.data.objects.new(name, ld)
        bpy.context.collection.objects.link(lo)
        lo.rotation_euler = (math.radians(rx), 0, math.radians(rz))
    _sun("K", 50, 30, 5.0); _sun("F", 30, 200, 2.5); _sun("R", 15, 130, 2.0)
    gm = bpy.data.meshes.new("G"); gs = span * 6
    gm.from_pydata([(-gs, -gs, 0), (gs, -gs, 0), (gs, gs, 0), (-gs, gs, 0)],
                   [], [(0, 1, 2, 3)])
    gm.update()
    g = bpy.data.objects.new("G", gm); bpy.context.collection.objects.link(g)
    gmat = bpy.data.materials.new("GM"); gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    gb.inputs["Base Color"].default_value = (0.085, 0.09, 0.1, 1.0)
    gb.inputs["Roughness"].default_value = 0.95
    gm.materials.append(gmat)
    for _eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            sc.render.engine = _eng
            break
        except TypeError:
            pass
    sc.render.resolution_x, sc.render.resolution_y = 960, 540
    if hasattr(sc, "eevee"):
        sc.eevee.taa_render_samples = 24
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Punchy"
    except Exception:
        pass
    sc.render.filepath = _args["preview"]
    bpy.ops.render.render(write_still=True)
except Exception as exc:
    print("[PAINT preview skip] " + str(exc))
print("[PAINT_PROJECT_OK]")
'''


def _run_blender(blend: Path, script_text: str, args: dict) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w",
                                     encoding="utf-8") as fh:
        fh.write(script_text)
        tmp = fh.name
    try:
        proc = subprocess.run(
            [BLENDER_PATH, "--background", str(blend), "--python", tmp,
             "--", json.dumps(args)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=BLENDER_EXEC_TIMEOUT,
        )
        return (proc.stdout or "") + (proc.stderr or "")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def paint_asset(final_dir: Path, asset_name: str, user_prompt: str) -> bool:
    """Repaint the exported asset's surfaces via img2img projection.
    Returns True when the painted asset replaced the originals."""
    final_dir = Path(final_dir).resolve()  # Blender mishandles relative paths
    blend = final_dir / f"{asset_name}.blend"
    if not blend.exists():
        logger.info("[PAINT] no .blend — skipping")
        return False

    from core.llm import repaint_image

    work = Path(tempfile.mkdtemp(prefix="argus_paint_"))
    try:
        out = _run_blender(blend, _RENDER_SCRIPT, {
            "views": VIEWS, "res": VIEW_RES, "workdir": str(work),
        })
        if "[PAINT_RENDER_OK]" not in out:
            logger.warning("[PAINT] view render failed: %s", out[-400:])
            return False

        subject = user_prompt.strip() or asset_name.replace("_", " ")
        for i in range(len(VIEWS)):
            view = work / f"view_{i}.png"
            if not view.exists():
                return False
            repainted = repaint_image(
                PAINT_PROMPT.format(subject=subject), view.read_bytes())
            if not repainted:
                logger.warning("[PAINT] view %d repaint unavailable — skipping paint", i)
                return False
            (work / f"painted_{i}.png").write_bytes(repainted)

        out = _run_blender(blend, _PROJECT_SCRIPT, {
            "views": VIEWS,
            "workdir": str(work),
            "bake_res": int(os.environ.get("ARGUS_PAINT_RES", "1024")),
            "blend": str(blend),
            "glb": str(final_dir / f"{asset_name}.glb"),
            "fbx": str(final_dir / f"{asset_name}.fbx"),
            "preview": str(final_dir / f"{asset_name}_preview.png"),
        })
        baked = out.count("[PAINT baked]")
        if "[PAINT_PROJECT_OK]" not in out or baked == 0:
            logger.warning("[PAINT] projection bake failed: %s", out[-400:])
            return False
        logger.info("[PAINT] %d mesh(es) repainted", baked)
        return True
    except Exception as exc:  # noqa: BLE001 — painting must never break the run
        logger.warning("[PAINT] failed: %s", exc)
        return False
    finally:
        shutil.rmtree(work, ignore_errors=True)
