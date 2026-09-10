
from __future__ import annotations

import json
import re
from typing import Optional

# This module used to define its own third copy of sanitize_generated_code (after
# core/llm.py's and core/blender.py's, each with a different regex). They agreed
# on the cases that matter but not exactly: this one returned a script with no
# trailing newline, and _procedural_blueprint_fallback below concatenates
# `script + _texture_upgrade_epilogue(...)` — so the script's last line and the
# epilogue's first were one edit away from being welded into a syntax error.
# One shared implementation, in the module that owns script handling.
from core.script_safety import sanitize_generated_code


def _steam_locomotive_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 4.1+, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

# ── materials ──────────────────────────────────────────────────────────────────
def mat(name, col, met=0.0, rgh=0.6, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], out.inputs["Surface"])
    p.inputs["Base Color"].default_value = col
    p.inputs["Metallic"].default_value = met
    p.inputs["Roughness"].default_value = rgh
    if emission:
        p.inputs["Emission Color"].default_value = emission
        p.inputs["Emission Strength"].default_value = 2.0
    return m

mat_black  = mat("loco_black",  (0.04, 0.04, 0.04, 1), met=0.6, rgh=0.45)
mat_red    = mat("loco_red",    (0.72, 0.06, 0.04, 1), met=0.0, rgh=0.55)
mat_brass  = mat("loco_brass",  (0.80, 0.55, 0.12, 1), met=0.9, rgh=0.3)
mat_chrome = mat("loco_chrome", (0.85, 0.85, 0.88, 1), met=1.0, rgh=0.15)
mat_dark   = mat("loco_dark",   (0.12, 0.10, 0.08, 1), met=0.3, rgh=0.7)
mat_wood   = mat("loco_wood",   (0.42, 0.28, 0.14, 1), met=0.0, rgh=0.85)
mat_glass  = mat("loco_glass",  (0.6,  0.8,  1.0,  0.3), met=0.0, rgh=0.05)
mat_headlamp = mat("loco_headlamp", (1.0, 0.95, 0.7, 1), emission=(1.0, 0.95, 0.7, 1))

# ── helper functions ───────────────────────────────────────────────────────────
def obj_link(name, mesh, mat_, parent):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat_)
    if parent:
        obj.parent = parent
        obj.matrix_parent_inverse = parent.matrix_world.inverted()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try: bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
    except: pass
    return obj

def vcyl(name, r1, r2, depth, loc, mat_, segs=32, rot=(0,0,0)):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs,
                          radius1=r1, radius2=r2, depth=depth)
    if rot[0]: bmesh.ops.rotate(bm, cent=(0,0,0),
                  matrix=mathutils.Matrix.Rotation(rot[0],4,"X"), verts=bm.verts[:])
    if rot[1]: bmesh.ops.rotate(bm, cent=(0,0,0),
                  matrix=mathutils.Matrix.Rotation(rot[1],4,"Y"), verts=bm.verts[:])
    if rot[2]: bmesh.ops.rotate(bm, cent=(0,0,0),
                  matrix=mathutils.Matrix.Rotation(rot[2],4,"Z"), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    o = obj_link(name, mesh, mat_, root)
    o.location = mathutils.Vector(loc)
    return o

def box(name, sx, sy, sz, loc, mat_, bv=0.01):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx,sy,sz), verts=bm.verts[:],
                    space=mathutils.Matrix.Identity(4))
    if bv > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bv, segments=2, profile=0.7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    o = obj_link(name, mesh, mat_, root)
    o.location = mathutils.Vector(loc)
    return o

def sphere(name, loc, scale, mat_):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=0.5)
    bmesh.ops.scale(bm, vec=scale, verts=bm.verts[:],
                    space=mathutils.Matrix.Identity(4))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    o = obj_link(name, mesh, mat_, root)
    o.location = mathutils.Vector(loc)
    return o

def wheel(name, cx, cz, r=0.52, w=0.18):
    """Upright driving wheel: tyre + rim + spoke detail."""
    # Tyre (torus-like flat cylinder, axis=Y)
    mesh = bpy.data.meshes.new(name+"_tyre")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=36, radius1=r, radius2=r, depth=w)
    rot = mathutils.Matrix.Rotation(math.radians(90),4,"X")
    bmesh.ops.rotate(bm, cent=(0,0,0), matrix=rot, verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    t = obj_link(name+"_tyre", mesh, mat_red, root)
    t.location = (cx, 0.0, cz)

    # Hub
    mesh2 = bpy.data.meshes.new(name+"_hub")
    bm2 = bmesh.new()
    bmesh.ops.create_cone(bm2, cap_ends=True, segments=16, radius1=r*0.22, radius2=r*0.22, depth=w*1.2)
    rot2 = mathutils.Matrix.Rotation(math.radians(90),4,"X")
    bmesh.ops.rotate(bm2, cent=(0,0,0), matrix=rot2, verts=bm2.verts[:])
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces[:])
    bm2.to_mesh(mesh2); bm2.free(); mesh2.update()
    h = obj_link(name+"_hub", mesh2, mat_chrome, root)
    h.location = (cx, 0.0, cz)
    return t

# ── root empty ────────────────────────────────────────────────────────────────
root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"
root.location = (0,0,0)

# ── layout constants (Z=0 at rail level) ──────────────────────────────────────
# Locomotive faces +Y. Total length ~12m.
BOILER_R   = 0.55
BOILER_LEN = 5.5
BOILER_Z   = 1.65   # boiler centreline height
WHEEL_R    = 0.55   # driving wheel radius
WHEEL_Z    = WHEEL_R  # wheel axle height = radius (so bottom at Z=0)

# ── 1. RUNNING PLATE (frame) ──────────────────────────────────────────────────
box("running_plate_L", 0.12, 9.0, 0.18, (-0.72, 1.5, 0.62), mat_black, bv=0.005)
box("running_plate_R", 0.12, 9.0, 0.18, ( 0.72, 1.5, 0.62), mat_black, bv=0.005)

# ── 2. COW CATCHER ────────────────────────────────────────────────────────────
box("cow_catcher_body", 1.5, 0.08, 0.65, (0, 5.8, 0.5), mat_black, bv=0.008)
for i, x in enumerate([-0.55, -0.28, 0.0, 0.28, 0.55]):
    box(f"catcher_bar_{i}", 0.06, 0.55, 0.06, (x, 5.65, 0.3), mat_black, bv=0.004)

# ── 3. SMOKEBOX ───────────────────────────────────────────────────────────────
vcyl("smokebox", BOILER_R+0.08, BOILER_R+0.08, 1.0,
     (0, 5.0, BOILER_Z), mat_black, segs=32,
     rot=(math.radians(90),0,0))
# Smokebox door (circular face)
vcyl("smokebox_door", BOILER_R+0.09, BOILER_R+0.09, 0.04,
     (0, 5.52, BOILER_Z), mat_dark, segs=32,
     rot=(math.radians(90),0,0))
# Door handle
vcyl("door_handle", 0.025, 0.025, 0.35, (0, 5.55, BOILER_Z), mat_chrome, segs=8,
     rot=(0, math.radians(90),0))

# ── 4. BOILER BARREL ─────────────────────────────────────────────────────────
vcyl("boiler_barrel", BOILER_R, BOILER_R, BOILER_LEN,
     (0, 2.25, BOILER_Z), mat_black, segs=36,
     rot=(math.radians(90),0,0))

# Boiler bands (3 rings)
for y_pos in [1.2, 2.4, 3.6]:
    vcyl(f"boiler_band_{int(y_pos*10)}", BOILER_R+0.02, BOILER_R+0.02, 0.08,
         (0, y_pos, BOILER_Z), mat_chrome, segs=36,
         rot=(math.radians(90),0,0))

# ── 5. CHIMNEY / SMOKESTACK ───────────────────────────────────────────────────
vcyl("chimney_base", 0.18, 0.22, 0.55, (0, 4.8, BOILER_Z+BOILER_R+0.28), mat_black, segs=20)
vcyl("chimney_cap",  0.25, 0.20, 0.25, (0, 4.8, BOILER_Z+BOILER_R+0.72), mat_black, segs=20)
vcyl("chimney_lip",  0.28, 0.28, 0.06, (0, 4.8, BOILER_Z+BOILER_R+0.87), mat_black, segs=20)

# ── 6. STEAM DOME ────────────────────────────────────────────────────────────
sphere("steam_dome", (0, 2.8, BOILER_Z+BOILER_R+0.14), (0.48, 0.48, 0.55), mat_brass)
vcyl("steam_dome_base", 0.22, 0.22, 0.12, (0, 2.8, BOILER_Z+BOILER_R+0.04), mat_black, segs=20)

# ── 7. SAND DOME ─────────────────────────────────────────────────────────────
sphere("sand_dome", (0, 1.8, BOILER_Z+BOILER_R+0.10), (0.36, 0.36, 0.42), mat_black)

# ── 8. SAFETY VALVE ──────────────────────────────────────────────────────────
vcyl("safety_valve", 0.07, 0.05, 0.28, (0.18, 2.2, BOILER_Z+BOILER_R+0.22), mat_brass, segs=12)
vcyl("whistle",      0.05, 0.04, 0.22, (-0.18, 2.4, BOILER_Z+BOILER_R+0.22), mat_brass, segs=10)

# ── 9. DRIVING WHEELS (4 large, 2 per side) ──────────────────────────────────
wheel_positions_x = [1.0, -1.0]  # right and left (mirrored on Y axis)
wheel_positions_y = [0.8, 2.2, 3.6, 5.0]  # four along boiler
for wy in [0.8, 2.2, 3.6]:
    for side, sx in ((0, -1), (1, 1)):
        # Full wheel assembly
        WX = sx * (BOILER_R + 0.25)
        wname = f"drive_wheel_{int(wy*10)}_{side}"
        mesh = bpy.data.meshes.new(wname)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=36,
                              radius1=WHEEL_R, radius2=WHEEL_R, depth=0.20)
        rot = mathutils.Matrix.Rotation(math.radians(90),4,"X")
        bmesh.ops.rotate(bm, cent=(0,0,0), matrix=rot, verts=bm.verts[:])
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(mesh); bm.free(); mesh.update()
        wo = obj_link(wname, mesh, mat_red, root)
        wo.location = (WX, wy, WHEEL_Z)

        # Hub
        hub_mesh = bpy.data.meshes.new(wname+"_hub")
        bm3 = bmesh.new()
        bmesh.ops.create_cone(bm3, cap_ends=True, segments=12,
                              radius1=0.12, radius2=0.12, depth=0.22)
        bmesh.ops.rotate(bm3, cent=(0,0,0),
                         matrix=mathutils.Matrix.Rotation(math.radians(90),4,"X"),
                         verts=bm3.verts[:])
        bmesh.ops.recalc_face_normals(bm3, faces=bm3.faces[:])
        bm3.to_mesh(hub_mesh); bm3.free(); hub_mesh.update()
        ho = obj_link(wname+"_hub", hub_mesh, mat_chrome, root)
        ho.location = (WX, wy, WHEEL_Z)

# ── 10. COUPLING RODS (connecting driving wheels each side) ───────────────────
for side, sx in ((-1, -1), (1, 1)):
    RX = sx * (BOILER_R + 0.25 + WHEEL_R * 0.7)
    box(f"coupling_rod_{side}", 0.065, 2.6, 0.085,
        (RX, 2.2, WHEEL_Z), mat_black, bv=0.006)

# ── 11. PISTON & CYLINDER ─────────────────────────────────────────────────────
for side, sx in ((-1, -1), (1, 1)):
    PX = sx * 0.52
    vcyl(f"piston_cylinder_{side}", 0.16, 0.16, 1.1,
         (PX, 4.7, WHEEL_Z+0.2), mat_dark, segs=16,
         rot=(math.radians(90),0,0))
    box(f"piston_rod_{side}", 0.05, 1.1, 0.05,
        (PX, 3.9, WHEEL_Z+0.2), mat_chrome, bv=0.005)

# ── 12. LEADING WHEELS (2 small at front) ────────────────────────────────────
LEAD_R = 0.32
LEAD_Z = LEAD_R
for side, sx in ((-1, -1), (1, 1)):
    LX = sx * (BOILER_R + 0.18)
    lmesh = bpy.data.meshes.new(f"lead_wheel_{side}")
    bm4 = bmesh.new()
    bmesh.ops.create_cone(bm4, cap_ends=True, segments=28,
                          radius1=LEAD_R, radius2=LEAD_R, depth=0.14)
    bmesh.ops.rotate(bm4, cent=(0,0,0),
                     matrix=mathutils.Matrix.Rotation(math.radians(90),4,"X"),
                     verts=bm4.verts[:])
    bmesh.ops.recalc_face_normals(bm4, faces=bm4.faces[:])
    bm4.to_mesh(lmesh); bm4.free(); lmesh.update()
    lo = obj_link(f"lead_wheel_{side}", lmesh, mat_red, root)
    lo.location = (LX, 5.2, LEAD_Z)

# ── 13. RUNNING BOARDS ───────────────────────────────────────────────────────
for side, sx in ((-1, -1), (1, 1)):
    RBX = sx * (BOILER_R + 0.55)
    box(f"running_board_{side}", 0.28, 7.0, 0.06,
        (RBX, 1.5, BOILER_Z - BOILER_R - 0.06), mat_black, bv=0.004)

# ── 14. CAB ───────────────────────────────────────────────────────────────────
CAB_Y  = -1.2
CAB_Z  = 1.25
CAB_H  = 2.0
CAB_W  = 2.0

# Cab walls (4 sides)
box("cab_body",  CAB_W, 1.6, CAB_H, (0, CAB_Y, CAB_Z + CAB_H/2), mat_black, bv=0.015)

# Cab roof
box("cab_roof",  CAB_W+0.08, 1.70, 0.10,
    (0, CAB_Y, CAB_Z + CAB_H + 0.05), mat_dark, bv=0.02)

# Cab windows (dark glass insets — front facing Y-)
for side, sx in ((-1, -1), (1, 1)):
    box(f"cab_window_{side}", 0.52, 0.04, 0.62,
        (sx*0.58, CAB_Y + 0.82, CAB_Z + CAB_H*0.65), mat_glass, bv=0.01)

# Front window (boiler side)
box("cab_front_window", 0.9, 0.04, 0.55,
    (0, CAB_Y - 0.82, CAB_Z + CAB_H*0.65), mat_glass, bv=0.01)

# ── 15. FIREBOX (boiler rear bulge at cab) ────────────────────────────────────
box("firebox", 1.4, 0.9, 1.8, (0, -0.4, 1.3), mat_black, bv=0.02)

# ── 16. HEADLAMP ─────────────────────────────────────────────────────────────
sphere("headlamp_housing", (0, 5.65, BOILER_Z+0.08), (0.26, 0.26, 0.20), mat_black)
sphere("headlamp_lens",    (0, 5.73, BOILER_Z+0.08), (0.18, 0.14, 0.14), mat_headlamp)

# ── 17. TENDER (coal car behind cab) ─────────────────────────────────────────
TY = -4.0  # tender centre Y
# Tender body
box("tender_body",  2.1, 3.4, 1.5, (0, TY, 1.25), mat_black, bv=0.02)
# Coal load (rough top)
box("coal_load",    1.9, 3.0, 0.45, (0, TY, 2.18), mat_dark, bv=0.03)
# Tender frame
box("tender_frame", 2.2, 3.6, 0.18, (0, TY, 0.55), mat_dark, bv=0.01)

# Tender wheels (4 axles x2 sides)
TEND_W_R = 0.32
TEND_W_Z = TEND_W_R
for wy in [-3.0, -4.0, -5.0]:
    for side, sx in ((-1,-1),(1,1)):
        TWX = sx * 1.18
        tmesh = bpy.data.meshes.new(f"tender_wheel_{int(abs(wy)*10)}_{side}")
        bm5 = bmesh.new()
        bmesh.ops.create_cone(bm5, cap_ends=True, segments=24,
                              radius1=TEND_W_R, radius2=TEND_W_R, depth=0.14)
        bmesh.ops.rotate(bm5, cent=(0,0,0),
                         matrix=mathutils.Matrix.Rotation(math.radians(90),4,"X"),
                         verts=bm5.verts[:])
        bmesh.ops.recalc_face_normals(bm5, faces=bm5.faces[:])
        bm5.to_mesh(tmesh); bm5.free(); tmesh.update()
        two = obj_link(f"tender_wheel_{int(abs(wy)*10)}_{side}", tmesh, mat_red, root)
        two.location = (TWX, wy, TEND_W_Z)

# Tender-cab coupling
box("tender_coupling", 0.3, 0.5, 0.18, (0, -2.12, 0.85), mat_chrome, bv=0.01)

# ── export ────────────────────────────────────────────────────────────────────
import os
ARGUS_EXPORT_OBJ  = bpy.data.objects["ARGUS_ROOT"]
ARGUS_EXPORT_PATH = os.path.join(r"out/final/steam_locomotive_handcrafted",
                                  f"ARGUS_{ARGUS_EXPORT_OBJ.name}.glb")
os.makedirs(os.path.dirname(ARGUS_EXPORT_PATH) or ".", exist_ok=True)
bpy.ops.object.select_all(action="DESELECT")
for _o in bpy.data.objects:
    if _o.type == "MESH": _o.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.export_scene.gltf(filepath=ARGUS_EXPORT_PATH, export_format="GLB",
    use_selection=True, export_apply=True, export_materials="EXPORT",
    export_normals=True, export_texcoords=True, export_yup=True,
    export_draco_mesh_compression_enable=False)
print(f"ARGUS_OK:{ARGUS_EXPORT_OBJ.name}")
print(f"ARGUS_PATH:{ARGUS_EXPORT_PATH}")
''')


def _luxury_yacht_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 4.1+, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

def mat(name, col, met=0.0, rgh=0.5, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], out.inputs["Surface"])
    col_full = (col[0], col[1], col[2], alpha)
    p.inputs["Base Color"].default_value = col_full
    p.inputs["Metallic"].default_value = met
    p.inputs["Roughness"].default_value = rgh
    if alpha < 1.0:
        m.blend_method = "BLEND"
    return m

mat_hull_white  = mat("hull_white",   (0.95, 0.95, 0.95), rgh=0.25, met=0.0)
mat_hull_blue   = mat("hull_blue",    (0.04, 0.12, 0.48), rgh=0.3)
mat_hull_bottom = mat("hull_bottom",  (0.25, 0.35, 0.25), rgh=0.7)
mat_deck        = mat("deck_teak",    (0.62, 0.42, 0.22), rgh=0.85)
mat_cabin       = mat("cabin_white",  (0.92, 0.92, 0.92), rgh=0.25)
mat_glass       = mat("glass_tinted", (0.25, 0.35, 0.45), rgh=0.05, alpha=0.35)
mat_chrome      = mat("chrome",       (0.88, 0.88, 0.90), met=1.0, rgh=0.12)
mat_black       = mat("black_rubber", (0.06, 0.06, 0.06), rgh=0.8)
mat_motor       = mat("motor_black",  (0.08, 0.08, 0.08), met=0.3, rgh=0.6)
mat_nav_green   = mat("nav_green",    (0.0, 0.8, 0.15),   rgh=0.3)
mat_nav_red_l   = mat("nav_red",      (0.9, 0.05, 0.05),  rgh=0.3)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"
root.location = (0,0,0)

def obj_link(name, mesh, mat_, parent=root):
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat_)
    if parent:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    try: bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
    except: pass
    return o

def box_obj(name, sx, sy, sz, loc, mat_, bv=0.012):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx,sy,sz), verts=bm.verts[:],
                    space=mathutils.Matrix.Identity(4))
    if bv > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bv, segments=2, profile=0.7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    o = obj_link(name, mesh, mat_)
    o.location = mathutils.Vector(loc)
    return o

def cyl(name, r1, r2, depth, loc, mat_, segs=24, axis="Z"):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r1, radius2=r2, depth=depth)
    if axis == "X":
        bmesh.ops.rotate(bm, cent=(0,0,0),
                         matrix=mathutils.Matrix.Rotation(math.radians(90),4,"Y"),
                         verts=bm.verts[:])
    elif axis == "Y":
        bmesh.ops.rotate(bm, cent=(0,0,0),
                         matrix=mathutils.Matrix.Rotation(math.radians(90),4,"X"),
                         verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    o = obj_link(name, mesh, mat_)
    o.location = mathutils.Vector(loc)
    return o

# ── 1. HULL — tapered bow, wide amidships, flat stern ────────────────────────
# Built from from_pydata with 10 profile points giving the boat shape
hull_verts = []
hull_faces = []

# Hull cross-sections at key stations (Y positions along boat):
# bow tip (Y=4.5), bow shoulder (Y=3.5), mid (Y=0), stern shoulder (Y=-3.0), stern (Y=-4.5)
# Each cross-section: bottom-keel, chine-L, deck-L, deck-R, chine-R
HULL_Y_STATIONS = [4.5, 3.5, 1.5, 0.0, -1.5, -3.0, -4.5]
HULL_WIDTH      = [0.0, 0.8, 1.5, 1.55, 1.5, 1.3, 1.3]   # half-width at each station
HULL_DEPTH      = [0.0, 0.6, 1.1, 1.2,  1.1, 1.0, 0.9]   # depth at each station
HULL_DECK_Z     = 1.4   # deck height

verts_3d = []
for i, (y, hw, d) in enumerate(zip(HULL_Y_STATIONS, HULL_WIDTH, HULL_DEPTH)):
    if hw < 0.02:  # bow tip — single vertex per side group
        verts_3d.append((0.0, y, 0.0))        # keel
        verts_3d.append((0.0, y, HULL_DECK_Z)) # deck
    else:
        verts_3d.append((0.0,  y, -d*0.3))    # keel
        verts_3d.append((-hw,  y, d*0.4))     # chine L
        verts_3d.append((-hw,  y, HULL_DECK_Z))
        verts_3d.append(( hw,  y, HULL_DECK_Z))
        verts_3d.append(( hw,  y, d*0.4))     # chine R

hull_mesh = bpy.data.meshes.new("hull_body")
hull_mesh.from_pydata(verts_3d, [], [])  # faces added via bmesh
hull_mesh.update()

# Simpler approach: use a tapered box and sculpt it
hull_body = box_obj("hull_body", 3.1, 9.0, 1.4, (0, 0, 0.7), mat_hull_white, bv=0.08)
# Bow taper — scale front vertices
bm_hull = bmesh.new()
bm_hull.from_mesh(hull_body.data)
bm_hull.verts.ensure_lookup_table()
# Taper the front (Y>3) and narrow the keel
for v in bm_hull.verts:
    if v.co.y > 3.2:
        taper = max(0.0, 1.0 - (v.co.y - 3.2) / 1.3)
        v.co.x *= taper * 0.15 + 0.01
        if v.co.z < 0.6: v.co.z *= taper * 0.3
    if v.co.y < -3.5:
        taper = max(0.15, 1.0 - abs(v.co.y + 3.5) / 1.2)
        v.co.x *= taper * 0.7 + 0.1
bmesh.ops.recalc_face_normals(bm_hull, faces=bm_hull.faces[:])
bm_hull.to_mesh(hull_body.data); bm_hull.free(); hull_body.data.update()

# Blue waterline stripe
box_obj("waterline_stripe", 3.12, 9.0, 0.22, (0, 0, 0.22), mat_hull_blue, bv=0.02)
# Anti-fouling bottom
box_obj("hull_bottom", 2.9, 8.8, 0.22, (0, 0.1, -0.02), mat_hull_bottom, bv=0.05)

# ── 2. DECK ───────────────────────────────────────────────────────────────────
box_obj("deck_surface", 2.95, 8.6, 0.10, (0, 0, 1.45), mat_deck, bv=0.04)

# ── 3. MAIN CABIN ─────────────────────────────────────────────────────────────
# Forward cabin (main saloon)
box_obj("main_cabin", 2.55, 4.5, 1.35, (0, 0.8, 2.15), mat_cabin, bv=0.03)
# Cabin roof
box_obj("cabin_roof", 2.60, 4.55, 0.12, (0, 0.8, 2.88), mat_cabin, bv=0.04)

# Front windshield (angled)
mesh_ws = bpy.data.meshes.new("windshield")
bm_ws = bmesh.new()
bmesh.ops.create_cube(bm_ws, size=1.0)
bmesh.ops.scale(bm_ws, vec=(2.5, 0.05, 0.85), verts=bm_ws.verts[:],
                space=mathutils.Matrix.Identity(4))
bm_ws.to_mesh(mesh_ws); bm_ws.free(); mesh_ws.update()
ws = obj_link("windshield", mesh_ws, mat_glass)
ws.location = (0, 3.1, 2.2)
ws.rotation_euler = (math.radians(18), 0, 0)

# Side windows
for side, sx in ((-1, -1.28), (1, 1.28)):
    box_obj(f"side_window_{side}", 0.05, 1.8, 0.58, (sx, 1.2, 2.25), mat_glass, bv=0.01)
    box_obj(f"side_window_rear_{side}", 0.05, 0.9, 0.50, (sx, -0.6, 2.22), mat_glass, bv=0.01)

# ── 4. FLYBRIDGE ──────────────────────────────────────────────────────────────
box_obj("flybridge_deck",  2.55, 2.5, 0.10, (0, -1.5, 3.05), mat_deck, bv=0.04)
box_obj("flybridge_helm",  0.95, 0.55, 0.75, (0, -0.45, 3.45), mat_cabin, bv=0.02)
# Helm windshield
box_obj("helm_windshield", 1.8, 0.04, 0.52, (0, -0.05, 3.55), mat_glass, bv=0.01)
# Helm seat
box_obj("helm_seat", 1.1, 0.65, 0.30, (0, -1.5, 3.42), mat_cabin, bv=0.03)
# Radar arch
cyl("radar_arch_L", 0.04, 0.04, 1.85, (-1.15, -2.3, 3.92), mat_chrome, segs=12, axis="Z")
cyl("radar_arch_R", 0.04, 0.04, 1.85, ( 1.15, -2.3, 3.92), mat_chrome, segs=12, axis="Z")
box_obj("radar_arch_top", 2.32, 0.06, 0.06, (0, -2.3, 4.82), mat_chrome, bv=0.01)
# Radar dome
cyl("radar_dome", 0.28, 0.22, 0.18, (0, -2.3, 4.95), mat_cabin, segs=20)

# ── 5. SWIM PLATFORM ─────────────────────────────────────────────────────────
box_obj("swim_platform", 2.6, 1.1, 0.12, (0, -4.85, 1.22), mat_deck, bv=0.02)
# Boarding ladder (3 rungs)
for i, rz in enumerate([0.88, 1.0, 1.12]):
    box_obj(f"ladder_rung_{i}", 0.5, 0.04, 0.03, (0, -5.42, rz), mat_chrome, bv=0.005)
cyl("ladder_side_L", 0.02, 0.02, 0.55, (-0.27, -5.42, 1.0), mat_chrome, segs=8, axis="Z")
cyl("ladder_side_R", 0.02, 0.02, 0.55, ( 0.27, -5.42, 1.0), mat_chrome, segs=8, axis="Z")

# ── 6. OUTBOARD MOTORS (twin at stern) ───────────────────────────────────────
for side, mx in ((-1, -0.72), (1, 0.72)):
    box_obj(f"motor_housing_{side}", 0.42, 0.35, 0.92, (mx, -4.6, 0.88), mat_motor, bv=0.03)
    cyl(f"motor_leg_{side}",   0.1, 0.08, 0.65, (mx, -4.6, 0.28), mat_motor, segs=14)
    cyl(f"motor_prop_{side}",  0.28, 0.24, 0.1,  (mx, -4.85, 0.12), mat_chrome, segs=8, axis="Y")
    box_obj(f"motor_tilt_{side}", 0.08, 0.08, 0.35, (mx, -4.35, 1.42), mat_chrome, bv=0.01)

# ── 7. DECK RAILS & STANCHIONS ────────────────────────────────────────────────
# Bow rail (U-shape)
cyl("bow_rail_L", 0.025, 0.025, 2.8, (-1.35, 2.6, 1.82), mat_chrome, segs=8, axis="Z")
cyl("bow_rail_R", 0.025, 0.025, 2.8, ( 1.35, 2.6, 1.82), mat_chrome, segs=8, axis="Z")
cyl("bow_rail_top", 0.025, 0.025, 2.72, (0, 4.1, 2.22), mat_chrome, segs=8, axis="X")
# Stern rail
cyl("stern_rail_L", 0.025, 0.025, 0.88, (-1.35, -4.35, 1.82), mat_chrome, segs=8, axis="Z")
cyl("stern_rail_R", 0.025, 0.025, 0.88, ( 1.35, -4.35, 1.82), mat_chrome, segs=8, axis="Z")

# Stanchions (8 along sides)
for i, (sy, sx) in enumerate([(2.0,-1.5),(0.5,-1.5),(-1.0,-1.5),(-2.5,-1.5),
                                (2.0, 1.5),(0.5, 1.5),(-1.0, 1.5),(-2.5, 1.5)]):
    cyl(f"stanchion_{i}", 0.025, 0.025, 0.72, (sx, sy, 1.86), mat_chrome, segs=8, axis="Z")

# ── 8. DECK FITTINGS ─────────────────────────────────────────────────────────
# Cleats (4)
for ci, (cx, cy) in enumerate([(-1.35, 3.5), (1.35, 3.5), (-1.35, -3.8), (1.35, -3.8)]):
    box_obj(f"cleat_{ci}", 0.22, 0.08, 0.08, (cx, cy, 1.52), mat_chrome, bv=0.01)

# Bow anchor roller
box_obj("anchor_roller_housing", 0.22, 0.22, 0.18, (0, 4.45, 1.55), mat_chrome, bv=0.02)
cyl("anchor_roller", 0.09, 0.09, 0.24, (0, 4.45, 1.62), mat_chrome, segs=12, axis="X")

# Navigation lights
cyl("nav_light_green", 0.045, 0.045, 0.12, (1.42, 3.75, 1.95), mat_nav_green, segs=10)
cyl("nav_light_red",   0.045, 0.045, 0.12, (-1.42, 3.75, 1.95), mat_nav_red_l, segs=10)
# Masthead light
cyl("masthead_light", 0.035, 0.035, 0.65, (0, 1.2, 2.95), mat_chrome, segs=8)

# ── export ────────────────────────────────────────────────────────────────────
import os
ARGUS_EXPORT_OBJ  = bpy.data.objects["ARGUS_ROOT"]
ARGUS_EXPORT_PATH = os.path.join(r"out/final/luxury_motor_yacht_handcrafted",
                                  f"ARGUS_{ARGUS_EXPORT_OBJ.name}.glb")
os.makedirs(os.path.dirname(ARGUS_EXPORT_PATH) or ".", exist_ok=True)
bpy.ops.object.select_all(action="DESELECT")
for _o in bpy.data.objects:
    if _o.type == "MESH": _o.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.export_scene.gltf(filepath=ARGUS_EXPORT_PATH, export_format="GLB",
    use_selection=True, export_apply=True, export_materials="EXPORT",
    export_normals=True, export_texcoords=True, export_yup=True,
    export_draco_mesh_compression_enable=False)
print(f"ARGUS_OK:{ARGUS_EXPORT_OBJ.name}")
print(f"ARGUS_PATH:{ARGUS_EXPORT_PATH}")
''')


def _semi_truck_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 4.1+, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

def mat(name, col, met=0.0, rgh=0.5, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    out = n.new("ShaderNodeOutputMaterial")
    p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], out.inputs["Surface"])
    p.inputs["Base Color"].default_value = col
    p.inputs["Metallic"].default_value = met
    p.inputs["Roughness"].default_value = rgh
    if emission:
        p.inputs["Emission Color"].default_value = emission
        p.inputs["Emission Strength"].default_value = 2.5
    return m

mat_cab_red    = mat("cab_red",      (0.72, 0.05, 0.04, 1), met=0.1, rgh=0.35)
mat_chrome     = mat("chrome",       (0.90, 0.90, 0.92, 1), met=1.0, rgh=0.12)
mat_glass      = mat("glass_dark",   (0.25, 0.35, 0.42, 1), met=0.0, rgh=0.05)
mat_black      = mat("black_rubber", (0.05, 0.05, 0.05, 1), met=0.0, rgh=0.9)
mat_trailer    = mat("trailer_alu",  (0.72, 0.72, 0.74, 1), met=0.4, rgh=0.55)
mat_dark_grey  = mat("dark_grey",    (0.18, 0.18, 0.18, 1), met=0.2, rgh=0.7)
mat_orange     = mat("marker_light", (1.0, 0.55, 0.0, 1), emission=(1.0, 0.55, 0.0, 1))
mat_red_light  = mat("tail_light",   (0.9, 0.05, 0.05, 1), emission=(1.0, 0.1, 0.0, 1))
mat_white_light= mat("headlight",    (0.95, 0.95, 1.0, 1), emission=(0.95, 0.95, 1.0, 1))

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"
root.location = (0,0,0)

def obj_link(name, mesh, mat_, parent=root):
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat_)
    if parent:
        o.parent = parent
        o.matrix_parent_inverse = parent.matrix_world.inverted()
    bpy.context.view_layer.objects.active = o
    o.select_set(True)
    try: bpy.ops.object.shade_smooth_by_angle(angle=math.radians(28))
    except: pass
    return o

def box(name, sx, sy, sz, loc, mat_, bv=0.015):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(sx,sy,sz), verts=bm.verts[:],
                    space=mathutils.Matrix.Identity(4))
    if bv > 0:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bv, segments=2, profile=0.7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    o = obj_link(name, mesh, mat_)
    o.location = mathutils.Vector(loc)
    return o

def cyl(name, r1, r2, depth, loc, mat_, segs=24, axis="Z"):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r1, radius2=r2, depth=depth)
    if axis == "X":
        bmesh.ops.rotate(bm, cent=(0,0,0),
                         matrix=mathutils.Matrix.Rotation(math.radians(90),4,"Y"),
                         verts=bm.verts[:])
    elif axis == "Y":
        bmesh.ops.rotate(bm, cent=(0,0,0),
                         matrix=mathutils.Matrix.Rotation(math.radians(90),4,"X"),
                         verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(mesh); bm.free(); mesh.update()
    o = obj_link(name, mesh, mat_)
    o.location = mathutils.Vector(loc)
    return o

def wheel(name, loc, r=0.55, w=0.28):
    """Upright tyre (axis X) with separate rim."""
    tyre_mesh = bpy.data.meshes.new(name+"_tyre")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=36, radius1=r, radius2=r, depth=w)
    bmesh.ops.rotate(bm, cent=(0,0,0),
                     matrix=mathutils.Matrix.Rotation(math.radians(90),4,"Y"),
                     verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(tyre_mesh); bm.free(); tyre_mesh.update()
    t = obj_link(name+"_tyre", tyre_mesh, mat_black)
    t.location = mathutils.Vector(loc)

    rim_mesh = bpy.data.meshes.new(name+"_rim")
    bm2 = bmesh.new()
    bmesh.ops.create_cone(bm2, cap_ends=True, segments=12, radius1=r*0.58, radius2=r*0.58, depth=w*0.9)
    bmesh.ops.rotate(bm2, cent=(0,0,0),
                     matrix=mathutils.Matrix.Rotation(math.radians(90),4,"Y"),
                     verts=bm2.verts[:])
    bmesh.ops.recalc_face_normals(bm2, faces=bm2.faces[:])
    bm2.to_mesh(rim_mesh); bm2.free(); rim_mesh.update()
    r_ = obj_link(name+"_rim", rim_mesh, mat_chrome)
    r_.location = mathutils.Vector(loc)
    return t

# ── layout ────────────────────────────────────────────────────────────────────
# Truck faces +Y. Cab at front, trailer behind.
# All wheels: bottom at Z=0, axle at Z=wheel_radius
WHEEL_R = 0.55
WHEEL_Z = WHEEL_R   # axle height

# ── 1. CHASSIS FRAME ─────────────────────────────────────────────────────────
box("chassis_L", 0.18, 22.0, 0.28, (-0.82, -4.0, WHEEL_Z*0.65), mat_dark_grey, bv=0.01)
box("chassis_R", 0.18, 22.0, 0.28, ( 0.82, -4.0, WHEEL_Z*0.65), mat_dark_grey, bv=0.01)
# Cross members
for cy_pos in [3.5, 2.0, 0.5, -1.0, -2.5]:
    box(f"crossmember_{int(abs(cy_pos)*10)}", 1.8, 0.14, 0.18, (0, cy_pos, WHEEL_Z*0.65), mat_dark_grey, bv=0.006)

# ── 2. CAB BODY ───────────────────────────────────────────────────────────────
# Main cab volume
box("cab_main", 2.52, 2.55, 2.85, (0, 3.55, 2.05), mat_cab_red, bv=0.05)
# Hood (lower front taper)
hood_mesh = bpy.data.meshes.new("cab_hood")
bm_h = bmesh.new()
bmesh.ops.create_cube(bm_h, size=1.0)
bmesh.ops.scale(bm_h, vec=(2.5, 1.4, 1.1), verts=bm_h.verts[:],
                space=mathutils.Matrix.Identity(4))
# Taper front of hood down
for v in bm_h.verts:
    if v.co.y > 0:  # front face
        v.co.z -= 0.28 * (v.co.y / 0.7)
bmesh.ops.recalc_face_normals(bm_h, faces=bm_h.faces[:])
bm_h.to_mesh(hood_mesh); bm_h.free(); hood_mesh.update()
hood_o = obj_link("cab_hood", hood_mesh, mat_cab_red)
hood_o.location = (0, 4.95, 1.15)

# Windshield
box("windshield_main", 2.2, 0.06, 1.0, (0, 2.38, 2.85), mat_glass, bv=0.01)
# Side windows
for side, sx in ((-1, -1.27), (1, 1.27)):
    box(f"cab_side_window_{side}", 0.06, 0.85, 0.82, (sx, 3.25, 2.88), mat_glass, bv=0.01)
    box(f"cab_door_window_{side}", 0.06, 0.75, 0.55, (sx, 4.1, 2.82), mat_glass, bv=0.01)

# Sleeper cab roof pod
box("sleeper_pod", 2.45, 1.8, 0.65, (0, 2.3, 3.6), mat_cab_red, bv=0.04)

# Cab roof
box("cab_roof_panel", 2.52, 2.58, 0.12, (0, 3.55, 3.52), mat_cab_red, bv=0.04)

# Air deflector on roof
box("air_deflector",  2.45, 0.12, 1.05, (0, 2.44, 3.9), mat_cab_red, bv=0.03)
box("deflector_top",  2.45, 1.45, 0.12, (0, 3.18, 4.4), mat_cab_red, bv=0.03)

# ── 3. BULL BAR ───────────────────────────────────────────────────────────────
box("bull_bar_main",    2.55, 0.12, 0.55, (0, 5.28, 1.15), mat_chrome, bv=0.02)
box("bull_bar_lower",   2.45, 0.10, 0.28, (0, 5.32, 0.65), mat_chrome, bv=0.02)
cyl("bull_bar_tube_L", 0.045, 0.045, 1.15, (-1.1, 5.28, 1.15), mat_chrome, segs=10)
cyl("bull_bar_tube_R", 0.045, 0.045, 1.15, ( 1.1, 5.28, 1.15), mat_chrome, segs=10)
# Fog lights in bull bar
cyl("fog_light_L", 0.1, 0.1, 0.08, (-0.68, 5.34, 0.92), mat_white_light, segs=14, axis="Y")
cyl("fog_light_R", 0.1, 0.1, 0.08, ( 0.68, 5.34, 0.92), mat_white_light, segs=14, axis="Y")

# ── 4. HEADLIGHTS ─────────────────────────────────────────────────────────────
for side, sx in ((-1, -1.0), (1, 1.0)):
    box(f"headlight_{side}", 0.5, 0.08, 0.32, (sx, 5.0, 1.62), mat_white_light, bv=0.01)

# ── 5. EXHAUST STACKS ─────────────────────────────────────────────────────────
for side, sx in ((-1, -1.38), (1, 1.38)):
    cyl(f"exhaust_stack_{side}", 0.09, 0.09, 3.2, (sx, 2.65, 2.95), mat_chrome, segs=16)
    cyl(f"exhaust_tip_{side}",   0.12, 0.09, 0.22, (sx, 2.65, 4.61), mat_chrome, segs=16)

# ── 6. FUEL TANKS ─────────────────────────────────────────────────────────────
for side, sx in ((-1, -1.45), (1, 1.45)):
    cyl(f"fuel_tank_{side}", 0.42, 0.42, 1.65, (sx, 3.2, WHEEL_Z+0.42), mat_chrome, segs=20, axis="Y")

# ── 7. WEST COAST MIRRORS ─────────────────────────────────────────────────────
for side, sx in ((-1, -1.72), (1, 1.72)):
    box(f"mirror_arm_{side}", 0.55, 0.05, 0.06, (sx, 4.85, 2.78), mat_chrome, bv=0.01)
    box(f"mirror_head_{side}", 0.38, 0.05, 0.60, (sx*1.5, 4.85, 2.78), mat_dark_grey, bv=0.02)

# ── 8. FRONT STEER WHEELS (x2) ────────────────────────────────────────────────
STEER_Y = 4.8
for side, sx in ((-1, -1.28), (1, 1.28)):
    wheel(f"steer_wheel_{side}", (sx, STEER_Y, WHEEL_Z), r=WHEEL_R, w=0.26)

# ── 9. DRIVE WHEELS — tandem axles (x8 = 4 per side, 2 axles) ────────────────
for axle_y in [1.8, 0.4]:
    for side, sx in ((-1, -1.35), (1, 1.35)):
        # Outer wheel
        wheel(f"drive_wheel_outer_{int(axle_y*10)}_{side}",
              (sx, axle_y, WHEEL_Z), r=WHEEL_R, w=0.26)
        # Inner wheel (dual rear)
        INNER_X = sx * (1.35 - 0.30)
        wheel(f"drive_wheel_inner_{int(axle_y*10)}_{side}",
              (INNER_X, axle_y, WHEEL_Z), r=WHEEL_R, w=0.24)

# Axle tubes
for axle_y in [STEER_Y, 1.8, 0.4]:
    box(f"axle_{int(axle_y*10)}", 2.85, 0.12, 0.12, (0, axle_y, WHEEL_Z), mat_dark_grey, bv=0.01)

# ── 10. FIFTH WHEEL COUPLING ─────────────────────────────────────────────────
box("fifth_wheel_plate", 1.6, 1.4, 0.14, (0, 1.1, WHEEL_Z*2+0.55), mat_chrome, bv=0.03)
cyl("king_pin",          0.08, 0.08, 0.22, (0, 0.82, WHEEL_Z*2+0.68), mat_chrome, segs=10)

# ── 11. TRAILER ───────────────────────────────────────────────────────────────
# Trailer body (53-foot = ~16m, scaled)
box("trailer_body",      2.6, 15.5, 2.75, (0, -8.5, WHEEL_Z+WHEEL_R+0.55), mat_trailer, bv=0.025)
# Corrugated side panels (ribs)
for i in range(14):
    ry = -1.0 - i * 1.0
    box(f"trailer_rib_L_{i}", 0.04, 0.08, 2.72, (-1.32, ry, WHEEL_Z+WHEEL_R+0.55), mat_trailer, bv=0.004)
    box(f"trailer_rib_R_{i}", 0.04, 0.08, 2.72, ( 1.32, ry, WHEEL_Z+WHEEL_R+0.55), mat_trailer, bv=0.004)
# Trailer underframe
box("trailer_underframe", 2.45, 15.4, 0.22, (0, -8.5, WHEEL_Z+0.14), mat_dark_grey, bv=0.01)

# Rear swing doors (2 panels)
box("trailer_door_L", 1.27, 0.08, 2.72, (-0.67, -16.3, WHEEL_Z+WHEEL_R+0.55), mat_trailer, bv=0.02)
box("trailer_door_R", 1.27, 0.08, 2.72, ( 0.67, -16.3, WHEEL_Z+WHEEL_R+0.55), mat_trailer, bv=0.02)
# Door latch bars
cyl("door_latch_L", 0.04, 0.04, 2.65, (-1.26, -16.32, WHEEL_Z+WHEEL_R+0.55), mat_chrome, segs=8)
cyl("door_latch_R", 0.04, 0.04, 2.65, ( 1.26, -16.32, WHEEL_Z+WHEEL_R+0.55), mat_chrome, segs=8)

# ── 12. LANDING GEAR ─────────────────────────────────────────────────────────
for side, sx in ((-1, -0.95), (1, 0.95)):
    cyl(f"landing_leg_{side}", 0.07, 0.07, WHEEL_Z+0.55, (sx, -1.5, WHEEL_Z*0.7), mat_dark_grey, segs=10)
    box(f"landing_foot_{side}", 0.28, 0.28, 0.08, (sx, -1.5, 0.04), mat_dark_grey, bv=0.01)
cyl("landing_crank", 0.04, 0.04, 2.0, (0, -1.5, WHEEL_Z+0.55), mat_chrome, segs=8, axis="X")

# ── 13. TRAILER WHEELS (2 tandem axles x2 sides) ─────────────────────────────
for axle_y in [-12.5, -14.0]:
    for side, sx in ((-1, -1.35), (1, 1.35)):
        wheel(f"trailer_wheel_outer_{int(abs(axle_y)*10)}_{side}",
              (sx, axle_y, WHEEL_Z), r=WHEEL_R, w=0.26)
        wheel(f"trailer_wheel_inner_{int(abs(axle_y)*10)}_{side}",
              (sx*0.82, axle_y, WHEEL_Z), r=WHEEL_R, w=0.24)

# Trailer axles
for axle_y in [-12.5, -14.0]:
    box(f"trailer_axle_{int(abs(axle_y)*10)}", 2.85, 0.12, 0.12, (0, axle_y, WHEEL_Z), mat_dark_grey, bv=0.01)

# ── 14. LIGHTS ────────────────────────────────────────────────────────────────
# Cab marker lights (top row)
for i, lx in enumerate([-1.05, -0.55, 0.0, 0.55, 1.05]):
    box(f"marker_light_{i}", 0.12, 0.04, 0.12, (lx, 2.4, 4.48), mat_orange, bv=0.005)
# Tail lights (rear trailer)
for side, sx in ((-1, -1.08), (1, 1.08)):
    box(f"tail_light_upper_{side}", 0.28, 0.06, 0.22, (sx, -16.35, WHEEL_Z+WHEEL_R+1.8), mat_red_light, bv=0.01)
    box(f"tail_light_lower_{side}", 0.22, 0.06, 0.18, (sx, -16.35, WHEEL_Z+WHEEL_R+1.4), mat_orange, bv=0.01)

# ── 15. MUD FLAPS ─────────────────────────────────────────────────────────────
for side, sx in ((-1,-1.42),(1,1.42)):
    box(f"mud_flap_drive_{side}", 0.05, 0.48, 0.72, (sx, -0.3, 0.38), mat_black, bv=0.01)
    box(f"mud_flap_trailer_{side}", 0.05, 0.48, 0.72, (sx, -11.5, 0.38), mat_black, bv=0.01)

# ── export ────────────────────────────────────────────────────────────────────
import os
ARGUS_EXPORT_OBJ  = bpy.data.objects["ARGUS_ROOT"]
ARGUS_EXPORT_PATH = os.path.join(r"out/final/semi_truck_handcrafted",
                                  f"ARGUS_{ARGUS_EXPORT_OBJ.name}.glb")
os.makedirs(os.path.dirname(ARGUS_EXPORT_PATH) or ".", exist_ok=True)
bpy.ops.object.select_all(action="DESELECT")
for _o in bpy.data.objects:
    if _o.type == "MESH": _o.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.export_scene.gltf(filepath=ARGUS_EXPORT_PATH, export_format="GLB",
    use_selection=True, export_apply=True, export_materials="EXPORT",
    export_normals=True, export_texcoords=True, export_yup=True,
    export_draco_mesh_compression_enable=False)
print(f"ARGUS_OK:{ARGUS_EXPORT_OBJ.name}")
print(f"ARGUS_PATH:{ARGUS_EXPORT_PATH}")
''')


# ===========================================================================
# Keyword-matched procedural fallbacks
#
# Moved here from core/llm.py, which had grown to 6,145 lines by holding these
# ~2,300 lines of hardcoded per-object Blender geometry alongside the actual
# LLM orchestration — 35% of that file was static mesh code that never makes a
# network call. This module already held exactly this kind of content for three
# other assets (locomotive, yacht, semi truck), so the convention existed; the
# newer scripts had simply been pasted into llm.py instead.
#
# Pure move: no logic changed. These build a Blender script as a string, keyed
# off the planner's blueprint/category/style, and are the last stop before the
# generic-cube fallback when every LLM provider has failed.
# ===========================================================================

def _sci_fi_crate_fallback_script() -> str:
    return sanitize_generated_code(
        r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(42)

root_obj = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root_obj)
root_obj.empty_display_type = "PLAIN_AXES"

def make_mat(name, color, metallic=0.0, roughness=0.55, emission=None, strength=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    if emission is not None:
        principled.inputs["Emission Color"].default_value = emission
        principled.inputs["Emission Strength"].default_value = strength
    return mat

mat_body = make_mat("ARGUS_DarkPaintedMetal", (0.08, 0.11, 0.14, 1), 0.35, 0.48)
mat_guard = make_mat("ARGUS_RawEdgeMetal", (0.42, 0.43, 0.44, 1), 0.9, 0.32)
mat_rubber = make_mat("ARGUS_BlackRubber", (0.015, 0.014, 0.013, 1), 0.0, 0.88)
mat_glow = make_mat("ARGUS_BlueEmissiveGlass", (0.05, 0.4, 0.9, 1), 0.2, 0.2, (0.0, 0.65, 1.0, 1), 3.0)

def box_obj(name, loc, scale, mat, bevel=0.015):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = root_obj
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector(scale), verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    if bevel:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2, profile=0.55)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    mesh.uv_layers.new(name="ARGUS_UV")
    obj.data.materials.append(mat)
    return obj

box_obj("ARGUS_crate_beveled_rectangular_cargo_body", (0, 0, 0.45), (2.3, 1.24, 0.84), mat_body, 0.035)
box_obj("ARGUS_crate_recessed_front_panel", (0, -0.626, 0.47), (1.64, 0.025, 0.50), mat_body, 0.018)
box_obj("ARGUS_crate_recessed_back_panel", (0, 0.626, 0.47), (1.64, 0.025, 0.50), mat_body, 0.018)
box_obj("ARGUS_crate_top_lid_with_visible_seam", (0, 0, 0.895), (2.08, 1.10, 0.07), mat_body, 0.012)

for x in (-1.08, 1.08):
    for y in (-0.57, 0.57):
        box_obj("ARGUS_crate_reinforced_metal_corner_guard", (x, y, 0.47), (0.095, 0.095, 0.46), mat_guard, 0.02)

for z in (0.12, 0.84):
    box_obj("ARGUS_crate_horizontal_front_edge_rail", (0, -0.655, z), (1.05, 0.045, 0.035), mat_guard, 0.01)
    box_obj("ARGUS_crate_horizontal_back_edge_rail", (0, 0.655, z), (1.05, 0.045, 0.035), mat_guard, 0.01)

for x in (-0.72, -0.48, -0.24, 0.24, 0.48, 0.72):
    box_obj("ARGUS_crate_vent_slat_front", (x, -0.682, 0.52), (0.065, 0.018, 0.16), mat_guard, 0.006)

for x in (-0.38, 0.38):
    box_obj("ARGUS_crate_front_latch_handle", (x, -0.695, 0.33), (0.19, 0.028, 0.045), mat_guard, 0.008)
    box_obj("ARGUS_crate_latch_mount_plate", (x, -0.688, 0.28), (0.11, 0.022, 0.08), mat_guard, 0.006)

for x in (-0.9, -0.55, -0.2, 0.2, 0.55, 0.9):
    box_obj("ARGUS_crate_blue_emissive_strip", (x, -0.704, 0.74), (0.11, 0.012, 0.025), mat_glow, 0.004)

for x in (-0.9, 0.9):
    for y in (-0.48, 0.48):
        box_obj("ARGUS_crate_black_rubber_foot", (x, y, 0.035), (0.13, 0.12, 0.035), mat_rubber, 0.012)

for x in (-0.98, -0.72, 0.72, 0.98):
    for z in (0.22, 0.72):
        box_obj("ARGUS_crate_bolt_row_front", (x, -0.708, z), (0.025, 0.012, 0.025), mat_guard, 0.008)
        box_obj("ARGUS_crate_bolt_row_back", (x, 0.708, z), (0.025, 0.012, 0.025), mat_guard, 0.008)

bpy.context.view_layer.update()
'''
    )


def _banana_tree_fallback_script() -> str:
    return sanitize_generated_code(
        r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(42)

root_obj = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root_obj)
root_obj.empty_display_type = "PLAIN_AXES"

def make_mat(name, color, metallic=0.0, roughness=0.7):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return mat

mat_stem = make_mat("ARGUS_FibrousGreenBrownStem", (0.26, 0.38, 0.16, 1), 0.0, 0.82)
mat_bark = make_mat("ARGUS_DryBarkStrips", (0.34, 0.20, 0.08, 1), 0.0, 0.88)
mat_leaf = make_mat("ARGUS_DarkBananaLeaf", (0.03, 0.24, 0.055, 1), 0.0, 0.66)
mat_leaf_light = make_mat("ARGUS_LightLeafVein", (0.24, 0.58, 0.12, 1), 0.0, 0.62)
mat_banana = make_mat("ARGUS_RipeYellowBanana", (0.95, 0.72, 0.08, 1), 0.0, 0.55)
mat_root = make_mat("ARGUS_RootBase", (0.20, 0.13, 0.07, 1), 0.0, 0.9)

def mesh_obj(name, mesh, mat):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = root_obj
    obj.data.materials.append(mat)
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")
    return obj

def cone_part(name, radius1, radius2, depth, loc, mat, segments=28, rot=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius1, radius2=radius2, depth=depth)
    bmesh.ops.rotate(bm, cent=mathutils.Vector((0, 0, 0)), matrix=mathutils.Euler(rot).to_matrix().to_4x4(), verts=bm.verts[:])
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh_obj(name, mesh, mat)

def closed_leaf_mesh(name, length, max_width, thickness, curve_z):
    segs = 9
    verts = []
    for side_z in (thickness, -thickness):
        for i in range(segs + 1):
            t = i / segs
            y = t * length
            width = max_width * math.sin(math.pi * t) * (0.92 + 0.08 * math.sin(t * math.pi * 3))
            z = curve_z * math.sin(math.pi * t) + side_z
            verts.append((-width, y, z))
            verts.append((width, y, z))
    faces = []
    for i in range(segs):
        a = i * 2
        faces.append((a, a + 1, a + 3, a + 2))
        b = (segs + 1) * 2 + i * 2
        faces.append((b + 2, b + 3, b + 1, b))
        faces.append((a, a + 2, b + 2, b))
        faces.append((a + 1, b + 1, b + 3, a + 3))
    faces.append((0, (segs + 1) * 2, (segs + 1) * 2 + 1, 1))
    tip = segs * 2
    btip = (segs + 1) * 2 + segs * 2
    faces.append((tip, tip + 1, btip + 1, btip))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh

def leaf(name, angle, pitch, roll, length, width, loc, mat):
    mesh = closed_leaf_mesh(name, length, width, 0.006, 0.12)
    obj = mesh_obj(name, mesh, mat)
    obj.location = mathutils.Vector(loc)
    obj.rotation_euler = (math.radians(pitch), 0, math.radians(angle))
    obj.rotation_euler.rotate_axis("Y", math.radians(roll))
    return obj

def leaf_vein(name, angle, pitch, roll, length, loc):
    mesh = bpy.data.meshes.new(name)
    w = 0.012
    t = 0.006
    y0 = 0.08
    y1 = length * 0.88
    verts = [
        (-w, y0, t), (w, y0, t), (w, y1, t), (-w, y1, t),
        (-w, y0, -t), (w, y0, -t), (w, y1, -t), (-w, y1, -t),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (3, 7, 4, 0),
    ]
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = mesh_obj(name, mesh, mat_leaf_light)
    obj.location = mathutils.Vector(loc)
    obj.rotation_euler = (math.radians(pitch), 0, math.radians(angle))
    obj.rotation_euler.rotate_axis("Y", math.radians(roll))
    return obj

def banana_finger(name, base_angle, row, loc):
    for i in range(4):
        angle = math.radians(base_angle)
        offset = mathutils.Vector((math.sin(angle) * (0.035 * i), math.cos(angle) * (0.035 * i), -0.055 * i))
        cone_part(name + "_segment", 0.025 - i * 0.002, 0.022 - i * 0.002, 0.105, mathutils.Vector(loc) + offset, mat_banana, segments=12, rot=(math.radians(18 + row * 4), math.radians(12), angle))

for i in range(8):
    z = 0.18 + i * 0.27
    lean_x = math.sin(i * 0.7) * 0.035
    lean_y = math.cos(i * 0.6) * 0.025
    cone_part("ARGUS_banana_tree_fibrous_pseudostem_section", 0.20 - i * 0.012, 0.185 - i * 0.012, 0.31, (lean_x, lean_y, z), mat_stem, segments=32, rot=(math.radians(i * 1.5), math.radians(i * -1.2), 0))

for angle in range(0, 360, 45):
    rad = math.radians(angle)
    cone_part("ARGUS_banana_tree_root_flare", 0.045, 0.018, 0.52, (math.cos(rad) * 0.22, math.sin(rad) * 0.22, 0.055), mat_root, segments=12, rot=(math.radians(82), 0, rad))

for angle in range(0, 360, 40):
    rad = math.radians(angle)
    cone_part("ARGUS_banana_tree_vertical_bark_strip", 0.012, 0.008, 1.55, (math.cos(rad) * 0.19, math.sin(rad) * 0.19, 1.03), mat_bark, segments=8, rot=(0, 0, 0))

for idx in range(18):
    angle = idx * 20 + random.uniform(-7, 7)
    pitch = 48 + random.uniform(-18, 10)
    roll = random.uniform(-18, 18)
    length = 1.08 + random.uniform(-0.18, 0.16)
    width = 0.19 + random.uniform(-0.035, 0.025)
    z = 2.18 + random.uniform(-0.08, 0.11)
    leaf("ARGUS_banana_leaf_blade_with_center_vein", angle, pitch, roll, length, width, (0, 0, z), mat_leaf)
    leaf_vein("ARGUS_banana_leaf_center_vein_attached_strip", angle, pitch, roll, length, (0, 0, z + 0.01))

cone_part("ARGUS_banana_bunch_hanging_stalk", 0.035, 0.026, 0.62, (0.08, -0.06, 1.86), mat_stem, segments=14, rot=(math.radians(12), math.radians(-12), 0))
for row in range(3):
    for col in range(5):
        angle = -40 + col * 20 + row * 5
        loc = (0.08 + (col - 2) * 0.04, -0.08 - row * 0.04, 1.72 - row * 0.11)
        banana_finger("ARGUS_banana_finger_attached_to_bunch", angle, row, loc)

bpy.context.view_layer.update()
'''
    )


def _banyan_tree_fallback_script() -> str:
    return sanitize_generated_code(
        r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(73)

root_obj = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root_obj)
root_obj.empty_display_type = "PLAIN_AXES"

def make_mat(name, color, metallic=0.0, roughness=0.8):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return mat

mat_bark = make_mat("ARGUS_BanyanGreyBrownBark", (0.30, 0.22, 0.13, 1), 0.0, 0.86)
mat_bark_dark = make_mat("ARGUS_BanyanDarkCreviceBark", (0.12, 0.075, 0.045, 1), 0.0, 0.94)
mat_root = make_mat("ARGUS_BanyanDryAerialRoot", (0.24, 0.15, 0.07, 1), 0.0, 0.9)
mat_leaf = make_mat("ARGUS_BanyanDeepGreenLeaves", (0.025, 0.20, 0.055, 1), 0.0, 0.64)
mat_leaf_light = make_mat("ARGUS_BanyanLightLeafHighlights", (0.12, 0.38, 0.08, 1), 0.0, 0.62)

def mesh_obj(name, mesh, mat):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = root_obj
    obj.data.materials.append(mat)
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")
    return obj

def cone_part(name, radius1, radius2, depth, loc, mat, segments=20, rot=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius1, radius2=radius2, depth=depth)
    bmesh.ops.rotate(bm, cent=mathutils.Vector((0, 0, 0)), matrix=mathutils.Euler(rot).to_matrix().to_4x4(), verts=bm.verts[:])
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh_obj(name, mesh, mat)

def ellipsoid(name, loc, scale, mat, segments=16):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=8, radius=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector(scale), verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh_obj(name, mesh, mat)

def leaf_card(name, loc, yaw, pitch, scale, mat):
    length = 0.34 * scale
    width = 0.11 * scale
    verts = [
        (0, 0, 0.006),
        (-width * 0.55, length * 0.38, 0.018),
        (-width * 0.35, length * 0.78, 0.010),
        (0, length, 0.0),
        (width * 0.35, length * 0.78, 0.010),
        (width * 0.55, length * 0.38, 0.018),
        (0, 0, -0.006),
        (-width * 0.55, length * 0.38, -0.004),
        (-width * 0.35, length * 0.78, -0.008),
        (0, length, -0.012),
        (width * 0.35, length * 0.78, -0.008),
        (width * 0.55, length * 0.38, -0.004),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5),
        (6, 8, 7), (6, 9, 8), (6, 10, 9), (6, 11, 10),
        (0, 6, 7, 1), (1, 7, 8, 2), (2, 8, 9, 3),
        (3, 9, 10, 4), (4, 10, 11, 5), (5, 11, 6, 0),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = mesh_obj(name, mesh, mat)
    obj.location = mathutils.Vector(loc)
    obj.rotation_euler = (math.radians(pitch), 0, math.radians(yaw))
    return obj

for idx, (x, y, r) in enumerate(((0, 0, 0.42), (0.24, -0.12, 0.31), (-0.22, 0.10, 0.28), (0.10, 0.25, 0.25), (-0.05, -0.28, 0.24))):
    cone_part("ARGUS_banyan_massive_trunk_cluster_column", r, r * 0.62, 2.45 + idx * 0.08, (x, y, 1.22), mat_bark, segments=32, rot=(math.radians(idx * 1.4), math.radians(-idx * 1.1), 0))

for angle in range(0, 360, 30):
    rad = math.radians(angle)
    length = 1.05 + 0.18 * math.sin(rad * 3)
    cone_part("ARGUS_banyan_buttress_root_flare_grounded", 0.105, 0.035, length, (math.cos(rad) * 0.55, math.sin(rad) * 0.55, 0.09), mat_root, segments=14, rot=(math.radians(83), 0, rad))

for angle in range(0, 360, 36):
    rad = math.radians(angle)
    cone_part("ARGUS_banyan_horizontal_limb_scaffold", 0.16, 0.08, 1.65, (math.cos(rad) * 0.82, math.sin(rad) * 0.82, 2.30 + 0.16 * math.sin(rad * 2)), mat_bark, segments=18, rot=(math.radians(82), math.radians(5 * math.sin(rad)), rad))

for idx in range(30):
    angle = math.radians(idx * 12 + random.uniform(-4, 4))
    dist = random.uniform(0.65, 1.75)
    top_z = random.uniform(1.85, 2.45)
    depth = max(0.35, top_z - 0.04)
    x = math.cos(angle) * dist
    y = math.sin(angle) * dist
    radius = random.uniform(0.012, 0.032)
    cone_part("ARGUS_banyan_hanging_aerial_root_curtain", radius, max(0.007, radius * 0.55), depth, (x, y, depth * 0.5), mat_root, segments=8, rot=(math.radians(random.uniform(-2, 2)), math.radians(random.uniform(-2, 2)), 0))

for idx in range(34):
    angle = math.radians(idx * 10.6)
    dist = random.uniform(0.45, 2.15)
    x = math.cos(angle) * dist
    y = math.sin(angle) * dist
    z = random.uniform(2.25, 3.25) - dist * 0.08
    sx = random.uniform(0.32, 0.62)
    sy = random.uniform(0.22, 0.50)
    sz = random.uniform(0.10, 0.22)
    mat = mat_leaf if idx % 3 else mat_leaf_light
    ellipsoid("ARGUS_banyan_layered_canopy_leaf_cluster", (x, y, z), (sx, sy, sz), mat, segments=16)

for idx in range(80):
    angle = random.uniform(0, 360)
    dist = random.uniform(0.75, 2.45)
    loc = (
        math.cos(math.radians(angle)) * dist,
        math.sin(math.radians(angle)) * dist,
        random.uniform(2.05, 3.22),
    )
    leaf_card("ARGUS_banyan_outer_oval_leaf_card", loc, angle + random.uniform(-35, 35), random.uniform(20, 58), random.uniform(0.75, 1.25), mat_leaf if idx % 4 else mat_leaf_light)

for angle in range(0, 360, 24):
    rad = math.radians(angle)
    cone_part("ARGUS_banyan_vertical_bark_ridge_on_trunk", 0.018, 0.010, 1.95, (math.cos(rad) * 0.42, math.sin(rad) * 0.42, 1.18), mat_bark_dark, segments=7, rot=(math.radians(1.5 * math.sin(rad)), math.radians(1.5 * math.cos(rad)), 0))

bpy.context.view_layer.update()
'''
    )


def _castle_fallback_script() -> str:
    return sanitize_generated_code(
        r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(137)

root_obj = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root_obj)
root_obj.empty_display_type = "PLAIN_AXES"

def make_mat(name, color, metallic=0.0, roughness=0.75):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    return mat

mat_stone = make_mat("ARGUS_cold_grey_stone_blocks", (0.36, 0.37, 0.39, 1), 0.0, 0.88)
mat_dark = make_mat("ARGUS_dark_mossy_stone_seams", (0.13, 0.16, 0.16, 1), 0.0, 0.95)
mat_wood = make_mat("ARGUS_dark_ironbound_gate_wood", (0.12, 0.07, 0.035, 1), 0.0, 0.82)
mat_iron = make_mat("ARGUS_blackened_iron_hardware", (0.025, 0.027, 0.03, 1), 0.8, 0.42)
mat_flag = make_mat("ARGUS_deep_crimson_flags", (0.42, 0.02, 0.04, 1), 0.0, 0.7)

def mesh_obj(name, mesh, mat):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = root_obj
    obj.data.materials.append(mat)
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")
    return obj

def cube_part(name, loc, scale, mat, bevel=0.012):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector(scale), verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=1, profile=0.55)
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh_obj(name, mesh, mat)

def cylinder_part(name, loc, radius, depth, mat, segments=24):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=depth)
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return mesh_obj(name, mesh, mat)

def tower(x, y, name):
    cylinder_part(f"ARGUS_castle_corner_guard_tower_{name}", (x, y, 1.35), 0.42, 2.7, mat_stone, segments=28)
    cylinder_part(f"ARGUS_castle_tower_top_rail_{name}", (x, y, 2.78), 0.48, 0.16, mat_dark, segments=28)
    for idx in range(8):
        ang = math.radians(idx * 45)
        cube_part(
            f"ARGUS_castle_battlement_guard_block_{name}",
            (x + math.cos(ang) * 0.35, y + math.sin(ang) * 0.35, 2.96),
            (0.10, 0.08, 0.18),
            mat_stone,
            bevel=0.006,
        )
    cylinder_part(f"ARGUS_castle_flag_pole_{name}", (x, y, 3.38), 0.018, 0.72, mat_iron, segments=8)
    cube_part(f"ARGUS_castle_crimson_flag_panel_{name}", (x + 0.16, y, 3.55), (0.22, 0.018, 0.10), mat_flag, bevel=0.003)

for x, y, name in ((-2.2, -1.45, "front_left"), (2.2, -1.45, "front_right"), (-2.2, 1.45, "rear_left"), (2.2, 1.45, "rear_right")):
    tower(x, y, name)

cube_part("ARGUS_castle_front_curtain_wall_panel", (0, -1.45, 0.82), (2.25, 0.18, 0.82), mat_stone, bevel=0.02)
cube_part("ARGUS_castle_rear_curtain_wall_panel", (0, 1.45, 0.82), (2.25, 0.18, 0.82), mat_stone, bevel=0.02)
cube_part("ARGUS_castle_left_curtain_wall_panel", (-2.2, 0, 0.82), (0.18, 1.28, 0.82), mat_stone, bevel=0.02)
cube_part("ARGUS_castle_right_curtain_wall_panel", (2.2, 0, 0.82), (0.18, 1.28, 0.82), mat_stone, bevel=0.02)

for idx, x in enumerate([-1.6, -1.15, -0.70, 0.70, 1.15, 1.6]):
    cube_part("ARGUS_castle_front_wall_battlement_block", (x, -1.45, 1.75), (0.16, 0.20, 0.20), mat_stone, bevel=0.006)
    cube_part("ARGUS_castle_rear_wall_battlement_block", (x, 1.45, 1.75), (0.16, 0.20, 0.20), mat_stone, bevel=0.006)
for y in [-0.95, -0.50, 0.0, 0.50, 0.95]:
    cube_part("ARGUS_castle_side_wall_battlement_block", (-2.2, y, 1.75), (0.20, 0.14, 0.20), mat_stone, bevel=0.006)
    cube_part("ARGUS_castle_side_wall_battlement_block", (2.2, y, 1.75), (0.20, 0.14, 0.20), mat_stone, bevel=0.006)

cube_part("ARGUS_castle_gatehouse_left_tower", (-0.48, -1.68, 1.20), (0.34, 0.30, 1.20), mat_stone, bevel=0.018)
cube_part("ARGUS_castle_gatehouse_right_tower", (0.48, -1.68, 1.20), (0.34, 0.30, 1.20), mat_stone, bevel=0.018)
cube_part("ARGUS_castle_gatehouse_bridge_with_arch_seam", (0, -1.68, 2.05), (0.88, 0.30, 0.24), mat_stone, bevel=0.018)
cube_part("ARGUS_castle_arched_gate_panel_dark_wood", (0, -1.87, 0.62), (0.42, 0.06, 0.62), mat_wood, bevel=0.012)
cube_part("ARGUS_castle_iron_gate_crossbar", (0, -1.93, 0.78), (0.46, 0.025, 0.045), mat_iron, bevel=0.004)
cube_part("ARGUS_castle_iron_gate_vertical_strap", (0, -1.94, 0.62), (0.035, 0.025, 0.56), mat_iron, bevel=0.004)

cube_part("ARGUS_castle_central_keep_main_mass", (0, 0.28, 1.45), (0.82, 0.72, 1.45), mat_stone, bevel=0.024)
cube_part("ARGUS_castle_keep_roof_block", (0, 0.28, 3.05), (0.90, 0.80, 0.16), mat_dark, bevel=0.012)
for x in [-0.55, -0.18, 0.18, 0.55]:
    cube_part("ARGUS_castle_keep_battlement_block", (x, -0.25, 3.25), (0.11, 0.13, 0.18), mat_stone, bevel=0.006)
    cube_part("ARGUS_castle_keep_battlement_block", (x, 0.80, 3.25), (0.11, 0.13, 0.18), mat_stone, bevel=0.006)

for x in [-1.25, -0.85, -0.45, 0.45, 0.85, 1.25]:
    for y in [-1.66, 1.66]:
        cube_part("ARGUS_castle_stone_block_rib_detail", (x, y, 0.98), (0.018, 0.035, 0.28), mat_dark, bevel=0.002)
for x in [-2.42, 2.42]:
    for y in [-0.95, -0.45, 0.05, 0.55, 1.05]:
        cube_part("ARGUS_castle_side_stone_block_rib_detail", (x, y, 0.98), (0.035, 0.018, 0.28), mat_dark, bevel=0.002)

cube_part("ARGUS_castle_ground_foundation_foot", (0, 0, 0.04), (2.75, 1.85, 0.04), mat_dark, bevel=0.008)
bpy.context.view_layer.update()
'''
    )


def _semantic_game_asset_fallback_script(kind: str) -> str:
    kind = re.sub(r"[^a-z0-9_]+", "_", kind.lower()).strip("_") or "prop"
    script = r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(42)

root_obj = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root_obj)
root_obj.empty_display_type = "PLAIN_AXES"
root_obj.location = mathutils.Vector((0.0, 0.0, 0.0))

def make_mat(name, color, metallic=0.0, roughness=0.75, emission=None):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    if emission:
        principled.inputs["Emission Color"].default_value = emission[0]
        principled.inputs["Emission Strength"].default_value = emission[1]
    return mat

paint_red = make_mat("paint_red", (0.62, 0.04, 0.035, 1), 0.25, 0.42)
paint_blue = make_mat("paint_blue", (0.05, 0.10, 0.34, 1), 0.2, 0.45)
paint_dark = make_mat("paint_dark", (0.03, 0.035, 0.045, 1), 0.15, 0.55)
rubber_black = make_mat("rubber_black", (0.012, 0.012, 0.011, 1), 0.0, 0.92)
brushed_metal = make_mat("brushed_metal", (0.62, 0.62, 0.58, 1), 0.88, 0.34)
glass_dark = make_mat("glass_dark", (0.05, 0.09, 0.12, 0.82), 0.0, 0.08)
fabric_canvas = make_mat("fabric_canvas", (0.08, 0.13, 0.18, 1), 0.0, 0.84)
fabric_trim = make_mat("fabric_trim", (0.015, 0.018, 0.02, 1), 0.0, 0.88)
leather_dark = make_mat("leather_dark", (0.035, 0.035, 0.04, 1), 0.0, 0.62)
foam_white = make_mat("foam_white", (0.86, 0.84, 0.78, 1), 0.0, 0.67)
sole_black = make_mat("sole_black", (0.02, 0.02, 0.018, 1), 0.0, 0.89)
emissive_cyan = make_mat("emissive_cyan", (0.02, 0.45, 0.8, 1), 0.0, 0.25, ((0.0, 0.75, 1.0, 1), 2.5))

def finish_obj(obj, mat):
    obj.parent = root_obj
    obj.matrix_parent_inverse = root_obj.matrix_world.inverted()
    if mat:
        obj.data.materials.append(mat)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="ARGUS_UV")
    _tri_count = sum(len(p.vertices) - 2 for p in obj.data.polygons)
    assert _tri_count <= 5000, f"ARGUS poly budget exceeded: {_tri_count} > 5000"
    return obj

def cube_part(name, loc, scale, mat, bevel=0.025, segments=1):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector(scale), verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=segments, profile=0.6)
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return finish_obj(obj, mat)

def cyl_part(name, loc, radius, depth, mat, axis="Z", segments=24, bevel=0.0):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=radius, radius2=radius, depth=depth)
    if axis == "X":
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, "Y")
        bmesh.ops.rotate(bm, cent=mathutils.Vector((0, 0, 0)), matrix=rot, verts=bm.verts[:])
    elif axis == "Y":
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, "X")
        bmesh.ops.rotate(bm, cent=mathutils.Vector((0, 0, 0)), matrix=rot, verts=bm.verts[:])
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=1, profile=0.5)
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return finish_obj(obj, mat)

def sphere_part(name, loc, scale, mat, segments=16):
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=8, radius=0.5)
    bmesh.ops.scale(bm, vec=mathutils.Vector(scale), verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return finish_obj(obj, mat)

def build_gaming_chair():
    sphere_part("chair_seat_cushion", (0, 0, 0.58), (1.12, 0.95, 0.18), leather_dark)
    sphere_part("chair_seat_bolster_left", (-0.46, 0.02, 0.68), (0.18, 0.86, 0.20), leather_dark)
    sphere_part("chair_seat_bolster_right", (0.46, 0.02, 0.68), (0.18, 0.86, 0.20), leather_dark)
    sphere_part("chair_backrest_cushion", (0, 0.42, 1.22), (0.92, 0.16, 1.25), leather_dark)
    sphere_part("chair_backrest_wing_left", (-0.43, 0.40, 1.25), (0.22, 0.17, 1.05), leather_dark)
    sphere_part("chair_backrest_wing_right", (0.43, 0.40, 1.25), (0.22, 0.17, 1.05), leather_dark)
    sphere_part("chair_headrest_pillow", (0, 0.25, 1.72), (0.54, 0.16, 0.20), fabric_canvas)
    sphere_part("chair_lumbar_pillow", (0, 0.25, 1.15), (0.48, 0.16, 0.26), fabric_canvas)
    cube_part("chair_tilt_mechanism", (0, 0, 0.45), (0.50, 0.36, 0.08), brushed_metal, 0.015)
    cyl_part("chair_gas_lift", (0, 0, 0.29), 0.07, 0.42, brushed_metal, "Z", 24, 0.004)
    cyl_part("chair_base_hub", (0, 0, 0.11), 0.16, 0.10, brushed_metal, "Z", 24, 0.004)
    for i in range(5):
        ang = math.radians(i * 72)
        x = math.cos(ang)
        y = math.sin(ang)
        cube_part(f"chair_base_spoke_{i+1}", (x * 0.34, y * 0.34, 0.08), (0.62, 0.09, 0.06), brushed_metal, 0.02)
        spoke = bpy.data.objects[f"chair_base_spoke_{i+1}"]
        spoke.rotation_euler[2] = ang
        cyl_part(f"chair_caster_wheel_{i+1}", (x * 0.68, y * 0.68, 0.055), 0.075, 0.055, rubber_black, "X", 20, 0.003)
        wheel = bpy.data.objects[f"chair_caster_wheel_{i+1}"]
        wheel.rotation_euler[2] = ang
        cube_part(f"chair_caster_fork_{i+1}", (x * 0.62, y * 0.62, 0.12), (0.10, 0.04, 0.13), brushed_metal, 0.006)
    for side, sx in (("left", -1), ("right", 1)):
        cyl_part(f"chair_arm_post_front_{side}", (sx * 0.64, -0.28, 0.80), 0.025, 0.34, brushed_metal, "Z", 12)
        cyl_part(f"chair_arm_post_rear_{side}", (sx * 0.64, 0.26, 0.88), 0.025, 0.46, brushed_metal, "Z", 12)
        sphere_part(f"chair_arm_pad_{side}", (sx * 0.64, -0.02, 1.04), (0.16, 0.66, 0.09), leather_dark)
    for z in (0.67, 1.04, 1.37):
        cube_part(f"chair_upholstery_seam_{int(z*100)}", (0, 0.235, z), (0.70, 0.012, 0.012), fabric_trim, 0.002)

def build_car():
    cube_part("car_body_shell", (0, 0, 0.50), (2.30, 1.00, 0.28), paint_red, 0.09, 2)
    sphere_part("car_cabin_canopy", (0, 0.02, 0.98), (1.08, 0.62, 0.30), paint_red, 14)
    sphere_part("car_front_nose", (0, -0.97, 0.60), (0.72, 0.30, 0.18), paint_red, 12)
    sphere_part("car_rear_deck", (0, 0.90, 0.62), (0.82, 0.34, 0.17), paint_red, 12)
    cube_part("car_hood", (0, -0.72, 0.71), (0.90, 0.56, 0.08), paint_red, 0.032)
    cube_part("car_trunk", (0, 0.74, 0.70), (0.78, 0.44, 0.08), paint_red, 0.032)
    cube_part("car_windshield", (0, -0.18, 1.03), (0.84, 0.028, 0.24), glass_dark, 0.012)
    cube_part("car_rear_window", (0, 0.30, 1.00), (0.78, 0.028, 0.21), glass_dark, 0.012)
    for side, sx in (("left", -1), ("right", 1)):
        cube_part(f"car_side_window_{side}", (sx * 0.53, 0.08, 1.12), (0.035, 0.46, 0.24), glass_dark, 0.01)
        cube_part(f"car_door_panel_{side}", (sx * 1.24, 0.02, 0.65), (0.035, 0.66, 0.32), paint_red, 0.012)
    wheel_layout = (
        ("front_left", -1.02, -0.70),
        ("front_right", 1.02, -0.70),
        ("rear_left", -1.02, 0.70),
        ("rear_right", 1.02, 0.70),
    )
    for corner, x, y in wheel_layout:
        cyl_part(f"car_tire_{corner}", (x, y, 0.31), 0.24, 0.20, rubber_black, "X", 32, 0.006)
        cyl_part(f"car_rim_{corner}", (x, y, 0.31), 0.14, 0.21, brushed_metal, "X", 24, 0.004)
        cyl_part(f"car_hub_{corner}", (x, y, 0.31), 0.055, 0.225, brushed_metal, "X", 16)
        sphere_part(f"car_fender_{corner}", (x, y, 0.57), (0.32, 0.18, 0.10), paint_red, 12)
    for side, y in (("front", -0.70), ("rear", 0.70)):
        cyl_part(f"car_axle_{side}", (0, y, 0.31), 0.028, 2.04, brushed_metal, "X", 16)
    cube_part("car_front_bumper", (0, -1.10, 0.43), (1.04, 0.09, 0.10), paint_dark, 0.02)
    cube_part("car_rear_bumper", (0, 1.08, 0.43), (1.00, 0.09, 0.10), paint_dark, 0.02)
    cube_part("car_front_splitter", (0, -1.18, 0.29), (0.94, 0.05, 0.03), paint_dark, 0.01)
    cube_part("car_side_skirt_left", (-1.12, 0.0, 0.32), (0.03, 0.90, 0.04), paint_dark, 0.006)
    cube_part("car_side_skirt_right", (1.12, 0.0, 0.32), (0.03, 0.90, 0.04), paint_dark, 0.006)
    for sx in (-0.34, 0.34):
        sphere_part(f"car_headlight_{'left' if sx < 0 else 'right'}", (sx, -1.18, 0.66), (0.18, 0.035, 0.08), foam_white, 12)
        sphere_part(f"car_tail_light_{'left' if sx < 0 else 'right'}", (sx, 1.18, 0.62), (0.14, 0.035, 0.07), paint_red, 12)

def build_bike():
    for side, y in (("front", -0.72), ("rear", 0.72)):
        cyl_part(f"bike_tire_{side}", (0, y, 0.42), 0.34, 0.075, rubber_black, "X", 36, 0.005)
        cyl_part(f"bike_rim_{side}", (0, y, 0.42), 0.25, 0.085, brushed_metal, "X", 30)
        cyl_part(f"bike_hub_{side}", (0, y, 0.42), 0.055, 0.14, brushed_metal, "X", 16)
    cyl_part("bike_top_tube", (0, 0.04, 0.90), 0.035, 1.05, brushed_metal, "Y", 16)
    cyl_part("bike_down_tube", (0, -0.16, 0.68), 0.04, 0.94, brushed_metal, "Y", 16)
    cyl_part("bike_seat_tube", (0, 0.36, 0.73), 0.035, 0.68, brushed_metal, "Z", 16)
    cyl_part("bike_front_fork_left", (-0.07, -0.70, 0.72), 0.025, 0.62, brushed_metal, "Z", 12)
    cyl_part("bike_front_fork_right", (0.07, -0.70, 0.72), 0.025, 0.62, brushed_metal, "Z", 12)
    cyl_part("bike_handlebar", (0, -0.92, 1.16), 0.025, 0.62, brushed_metal, "X", 12)
    sphere_part("bike_saddle", (0, 0.45, 1.08), (0.34, 0.18, 0.09), leather_dark)
    cyl_part("bike_crank", (0, 0.18, 0.53), 0.075, 0.10, brushed_metal, "X", 16)
    cube_part("bike_chain_guard", (0, 0.42, 0.48), (0.04, 0.54, 0.035), paint_dark, 0.006)
    for y in (-0.72, 0.72):
        for i in range(8):
            ang = math.radians(i * 45)
            cube_part(f"bike_spoke_{'front' if y < 0 else 'rear'}_{i+1}", (0, y + math.sin(ang) * 0.12, 0.42 + math.cos(ang) * 0.12), (0.012, 0.22, 0.012), brushed_metal, 0.001)

def build_backpack():
    sphere_part("backpack_main_body", (0, 0, 0.70), (0.80, 0.38, 1.15), fabric_canvas)
    sphere_part("backpack_front_pocket", (0, -0.31, 0.58), (0.58, 0.12, 0.46), fabric_canvas)
    cube_part("backpack_zipper_front", (0, -0.39, 0.82), (0.54, 0.018, 0.025), brushed_metal, 0.003)
    cube_part("backpack_zipper_main", (0, -0.22, 1.20), (0.70, 0.018, 0.025), brushed_metal, 0.003)
    for side, sx in (("left", -1), ("right", 1)):
        sphere_part(f"backpack_side_pocket_{side}", (sx * 0.43, -0.04, 0.50), (0.16, 0.22, 0.38), fabric_canvas)
        cube_part(f"backpack_shoulder_strap_{side}", (sx * 0.31, 0.25, 0.76), (0.10, 0.055, 0.92), fabric_trim, 0.025, 2)
        cube_part(f"backpack_buckle_{side}", (sx * 0.31, 0.19, 0.36), (0.14, 0.035, 0.08), brushed_metal, 0.006)
        cube_part(f"backpack_side_seam_{side}", (sx * 0.40, -0.24, 0.76), (0.025, 0.025, 0.78), fabric_trim, 0.004)
    cyl_part("backpack_carry_handle", (0, 0.02, 1.32), 0.035, 0.46, fabric_trim, "X", 16)
    cube_part("backpack_bottom_reinforced_panel", (0, 0, 0.12), (0.68, 0.36, 0.10), fabric_trim, 0.02)

def build_shoe():
    cube_part("shoe_outsole", (0, 0, 0.12), (1.25, 0.42, 0.12), sole_black, 0.06, 2)
    cube_part("shoe_midsole", (0, 0, 0.23), (1.16, 0.38, 0.10), foam_white, 0.05, 2)
    sphere_part("shoe_upper", (-0.04, 0, 0.42), (1.02, 0.38, 0.38), fabric_canvas)
    sphere_part("shoe_toe_box", (-0.48, 0, 0.36), (0.45, 0.36, 0.24), fabric_canvas)
    sphere_part("shoe_heel_counter", (0.48, 0, 0.43), (0.30, 0.35, 0.42), fabric_trim)
    cube_part("shoe_tongue", (0.08, 0, 0.58), (0.46, 0.12, 0.07), leather_dark, 0.025)
    for side, sy in (("left", -1), ("right", 1)):
        for i, x in enumerate([-0.18, -0.02, 0.14, 0.30], start=1):
            cyl_part(f"shoe_eyelet_{side}_{i}", (x, sy * 0.22, 0.53), 0.022, 0.018, brushed_metal, "Y", 12)
            cube_part(f"shoe_lace_{side}_{i}", (x - 0.03, sy * 0.08, 0.57), (0.22, 0.018, 0.018), foam_white, 0.004)
    for i, x in enumerate([-0.48, -0.22, 0.04, 0.30, 0.54], start=1):
        cube_part(f"shoe_tread_block_{i}", (x, 0, 0.025), (0.13, 0.36, 0.035), rubber_black, 0.01)
    cube_part("shoe_collar_padded_rim", (0.34, 0, 0.66), (0.36, 0.32, 0.055), fabric_trim, 0.025)

def build_crate():
    cube_part("crate_body_shell", (0, 0, 0.48), (1.45, 0.90, 0.76), paint_blue, 0.045, 2)
    cube_part("crate_lid_panel", (0, 0, 0.90), (1.34, 0.80, 0.06), paint_dark, 0.018)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cube_part(f"crate_corner_guard_{'left' if sx < 0 else 'right'}_{'front' if sy < 0 else 'rear'}", (sx * 0.76, sy * 0.48, 0.52), (0.10, 0.10, 0.82), brushed_metal, 0.012)
            cube_part(f"crate_rubber_foot_{'left' if sx < 0 else 'right'}_{'front' if sy < 0 else 'rear'}", (sx * 0.56, sy * 0.34, 0.05), (0.16, 0.12, 0.08), rubber_black, 0.012)
    for side, y in (("front", -0.48), ("rear", 0.48)):
        cube_part(f"crate_inset_panel_{side}", (0, y, 0.52), (1.00, 0.035, 0.44), paint_dark, 0.01)
        cube_part(f"crate_latch_{side}", (0, y * 1.04, 0.73), (0.28, 0.035, 0.10), brushed_metal, 0.008)
        cyl_part(f"crate_handle_{side}", (0, y * 1.07, 0.52), 0.035, 0.42, brushed_metal, "X", 16)
        for i, x in enumerate([-0.42, -0.14, 0.14, 0.42], start=1):
            cube_part(f"crate_vent_slat_{side}_{i}", (x, y * 1.04, 0.38), (0.14, 0.025, 0.035), brushed_metal, 0.003)
    for i, x in enumerate([-0.52, -0.26, 0.0, 0.26, 0.52], start=1):
        cyl_part(f"crate_bolt_row_front_{i}", (x, -0.515, 0.83), 0.025, 0.018, brushed_metal, "Y", 12)

kind = "__ARGUS_KIND__"
if kind == "gaming_chair":
    build_gaming_chair()
elif kind == "car":
    build_car()
elif kind in {"bike", "bicycle", "motorcycle"}:
    build_bike()
elif kind == "backpack":
    build_backpack()
elif kind == "shoe":
    build_shoe()
else:
    build_crate()

min_z = min((obj.matrix_world @ v.co).z for obj in bpy.data.objects if obj.type == "MESH" for v in obj.data.vertices)
for obj in bpy.data.objects:
    if obj.type == "MESH":
        obj.location.z -= min_z
bpy.context.view_layer.update()

# ── ARGUS export ─────────────────────────────────
import os
ARGUS_EXPORT_OBJ  = bpy.data.objects["ARGUS_ROOT"]
ARGUS_EXPORT_PATH = os.path.join(r"/tmp", f"ARGUS_{ARGUS_EXPORT_OBJ.name}.glb")

os.makedirs(os.path.dirname(ARGUS_EXPORT_PATH) or ".", exist_ok=True)

bpy.ops.object.select_all(action="DESELECT")
for _o in bpy.data.objects:
    if _o.type == "MESH":
        _o.select_set(True)
bpy.ops.object.transform_apply(
    location=True, rotation=True, scale=True
)

bpy.ops.export_scene.gltf(
    filepath=ARGUS_EXPORT_PATH,
    export_format="GLB",
    use_selection=True,
    export_apply=True,
    export_materials="EXPORT",
    export_normals=True,
    export_texcoords=True,
    export_yup=True,
    export_draco_mesh_compression_enable=False,
)

print(f"ARGUS_OK:{ARGUS_EXPORT_OBJ.name}")
print(f"ARGUS_PATH:{ARGUS_EXPORT_PATH}")
# ─────────────────────────────────────────────────
'''
    return sanitize_generated_code(script.replace("__ARGUS_KIND__", kind))


def _park_bench_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"

def mat(name, col, met=0.0, rgh=0.75):
    m = bpy.data.materials.new(name=name); m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    o = n.new("ShaderNodeOutputMaterial"); p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], o.inputs["Surface"])
    p.inputs["Base Color"].default_value = col; p.inputs["Metallic"].default_value = met
    p.inputs["Roughness"].default_value = rgh; return m

mat_wood  = mat("pine_slat",    (0.58, 0.38, 0.18, 1), 0.0, 0.80)
mat_iron  = mat("cast_iron",    (0.06, 0.05, 0.06, 1), 0.70, 0.52)
mat_bolt  = mat("iron_bolt",    (0.20, 0.20, 0.22, 1), 0.85, 0.35)

def obj(name, bm_, mat_):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces[:])
    ng = [f for f in bm_.faces if len(f.verts) > 4]
    if ng: bmesh.ops.triangulate(bm_, faces=ng)
    bm_.to_mesh(mesh); bm_.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = False
    o_ = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o_); o_.parent = root
    o_.matrix_parent_inverse = root.matrix_world.inverted()
    o_.data.materials.append(mat_); return o_

def box(name, sx, sy, sz, loc, mat_, bv=0.008):
    bm_ = bmesh.new()
    bmesh.ops.create_cube(bm_, size=1.0)
    bmesh.ops.scale(bm_, vec=mathutils.Vector((sx,sy,sz)), verts=bm_.verts[:], space=mathutils.Matrix.Identity(4))
    if bv: bmesh.ops.bevel(bm_, geom=bm_.edges[:], offset=bv, segments=2, profile=0.6)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector(loc))
    return obj(name, bm_, mat_)

def cyl(name, r, h, loc, mat_, segs=16, axis='Z'):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=h, radius1=r, radius2=r)
    if axis == 'X':
        bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)),
                         matrix=mathutils.Matrix.Rotation(math.pi/2,4,'Y'), verts=bm_.verts[:])
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector(loc))
    return obj(name, bm_, mat_)

# ── DIMENSIONS ───────────────────────────────────────────────────────────────
BW = 1.80   # bench width
BD = 0.44   # bench depth
SH = 0.46   # seat height
BH = 0.88   # back height
SW = 0.048  # slat width  (thickness)
SD = 0.110  # slat depth

# ── SEAT SLATS (5 horizontal planks) ─────────────────────────────────────────
slat_ys = [-BD/2 + SD/2 + i*(SD + 0.016) for i in range(4)]
for i, sy in enumerate(slat_ys):
    box(f"seat_slat_{i+1}", BW, SD, SW, (0, sy, SH), mat_wood, bv=0.006)

# ── BACK SLATS (3 vertical planks at the REAR of the seat) ──────────────────
back_z_bot = SH + 0.06
back_z_top = BH
back_h = back_z_top - back_z_bot
for i, bx in enumerate((-BW*0.30, 0.0, BW*0.30)):
    box(f"back_slat_{i+1}", SW, SD, back_h, (bx, BD/2 - SD/2, (back_z_bot+back_z_top)/2), mat_wood, bv=0.006)

# ── CAST IRON END FRAMES (2 decorative S-curve legs per side) ────────────────
for side, x in (("left", -BW/2 + 0.10), ("right", BW/2 - 0.10)):
    # Ground foot plate
    box(f"foot_plate_{side}", 0.38, 0.07, 0.025, (x, 0, 0.0125), mat_iron, bv=0.010)
    # Front leg
    cyl(f"front_leg_{side}", 0.028, SH, (x, BD/2 - 0.06, SH/2), mat_iron)
    # Rear leg (angled back)
    cyl(f"rear_leg_{side}",  0.028, SH + 0.08, (x, -BD/2 + 0.05, (SH+0.08)/2), mat_iron)
    # Side rail (horizontal tube connecting legs at mid height)
    cyl(f"side_rail_{side}", 0.018, BD, (x, 0, SH*0.55), mat_iron, axis='X')
    # Back upright (continues from seat to back top)
    cyl(f"back_post_{side}", 0.022, BH - SH, (x, BD/2 - 0.04, SH + (BH-SH)/2), mat_iron)
    # Arm rest curved top bar
    box(f"arm_rest_{side}", 0.04, BD*0.60, 0.04, (x, -BD/2*0.20, SH + 0.06), mat_iron, bv=0.014)
    # Decorative scrollwork block (centre of frame)
    box(f"scroll_{side}", 0.045, 0.22, 0.18, (x, -BD/2*0.30, SH*0.40), mat_iron, bv=0.018)

# ── LOWER STRETCHER (connects the two frames at ground level) ────────────────
cyl("stretcher_bar", 0.022, BW - 0.20, (0, -BD*0.10, 0.08), mat_iron, axis='X')

# ── SEAT SUPPORT RAILS ───────────────────────────────────────────────────────
for y in (BD/2 - 0.06, -BD/2 + 0.08):
    cyl(f"seat_rail_{int(y*100)}", 0.016, BW - 0.20, (0, y, SH - 0.04), mat_iron, axis='X')

# ── BOLTS (4 visible fastener heads) ─────────────────────────────────────────
for x in (-BW/2 + 0.10, BW/2 - 0.10):
    for y in (-BD/2 + 0.08, BD/2 - 0.06):
        cyl(f"bolt_{int(x*10)}_{int(y*100)}", 0.013, 0.022, (x, y, SH - 0.01), mat_bolt, segs=6)

bpy.context.view_layer.update()
''')


def _metal_barrel_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"

def mat(name, col, met=0.0, rgh=0.55):
    m = bpy.data.materials.new(name=name); m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    o = n.new("ShaderNodeOutputMaterial"); p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], o.inputs["Surface"])
    p.inputs["Base Color"].default_value = col; p.inputs["Metallic"].default_value = met
    p.inputs["Roughness"].default_value = rgh; return m

mat_body  = mat("painted_steel",  (0.10, 0.14, 0.22, 1), 0.30, 0.52)
mat_band  = mat("galv_band",      (0.42, 0.42, 0.44, 1), 0.85, 0.30)
mat_lid   = mat("lid_metal",      (0.12, 0.16, 0.24, 1), 0.40, 0.48)
mat_bung  = mat("bung_plug",      (0.08, 0.08, 0.10, 1), 0.80, 0.25)
mat_rust  = mat("rust_accent",    (0.42, 0.18, 0.06, 1), 0.05, 0.90)

def obj(name, bm_, mat_):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces[:])
    ng = [f for f in bm_.faces if len(f.verts) > 4]
    if ng: bmesh.ops.triangulate(bm_, faces=ng)
    bm_.to_mesh(mesh); bm_.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = True
    mesh.update(); o_ = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o_); o_.parent = root
    o_.matrix_parent_inverse = root.matrix_world.inverted()
    o_.data.materials.append(mat_); return o_

def vcyl(name, r1, r2, depth, z_bot, mat_, segs=36, bv=0.0):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=depth, radius1=r1, radius2=r2)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((0, 0, depth/2 + z_bot)))
    if bv: bmesh.ops.bevel(bm_, geom=bm_.edges[:], offset=bv, segments=2, profile=0.6)
    return obj(name, bm_, mat_)

def cyl_ring(name, r_out, thick, h, z_bot, mat_, segs=36):
    """Thin band/ring as flat disc scaled inward."""
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=h,
                          radius1=r_out, radius2=r_out)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((0, 0, h/2 + z_bot)))
    return obj(name, bm_, mat_)

# ── BODY ─────────────────────────────────────────────────────────────────────
R       = 0.285   # body radius
H       = 0.875   # body height
vcyl("barrel_body",  R, R*0.96, H,        0.0,  mat_body, segs=40)
vcyl("barrel_base",  R*1.04, R*1.01, 0.040, 0.0, mat_band, segs=40, bv=0.006)
vcyl("barrel_top_taper", R*0.96, R*0.86, 0.065, H,      mat_body, segs=40)

# ── ROLLING BANDS (3 raised ribs) ────────────────────────────────────────────
for i, bz in enumerate((0.175, 0.445, 0.715)):
    cyl_ring(f"rolling_band_{i+1}", R + 0.012, 0.012, 0.038, bz, mat_band)

# ── LID / CHIME ───────────────────────────────────────────────────────────────
vcyl("lid_disc",     R*0.86, R*0.84, 0.025, H + 0.065,  mat_lid, segs=40)
vcyl("chime_rim",    R*1.04, R*1.02, 0.030, H + 0.020,  mat_band, segs=40, bv=0.006)
vcyl("neck_flange",  R*0.90, R*0.88, 0.020, H + 0.055,  mat_band, segs=40)

# ── BUNG PLUGS (2 × filling ports on lid) ────────────────────────────────────
for bx in (R*0.38, -R*0.38):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=8, depth=0.028,
                          radius1=0.035, radius2=0.030)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((bx, 0, H + 0.082)))
    obj(f"bung_plug_{int(bx*100)}", bm_, mat_bung)

# ── SIDE SEAM / WELD LINE ─────────────────────────────────────────────────────
bm_ = bmesh.new()
bmesh.ops.create_cube(bm_, size=1.0)
bmesh.ops.scale(bm_, vec=mathutils.Vector((0.010, 0.010, H + 0.04)),
                verts=bm_.verts[:], space=mathutils.Matrix.Identity(4))
bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((R, 0, H/2)))
obj("weld_seam", bm_, mat_band)

# ── RUST PATCHES (2 small surface details) ────────────────────────────────────
for ang, rz in ((30, 0.22), (200, 0.50)):
    a = math.radians(ang)
    bm_ = bmesh.new()
    bmesh.ops.create_cube(bm_, size=1.0)
    bmesh.ops.scale(bm_, vec=mathutils.Vector((0.055, 0.012, 0.075)),
                    verts=bm_.verts[:], space=mathutils.Matrix.Identity(4))
    rot = mathutils.Matrix.Rotation(a, 4, 'Z')
    bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)), matrix=rot, verts=bm_.verts[:])
    bmesh.ops.translate(bm_, verts=bm_.verts[:],
                        vec=mathutils.Vector((R*math.cos(a), R*math.sin(a), rz)))
    obj(f"rust_patch_{ang}", bm_, mat_rust)

bpy.context.view_layer.update()
''')


def _treasure_chest_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"

def mat(name, col, met=0.0, rgh=0.75):
    m = bpy.data.materials.new(name=name); m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    o = n.new("ShaderNodeOutputMaterial"); p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], o.inputs["Surface"])
    p.inputs["Base Color"].default_value = col
    p.inputs["Metallic"].default_value = met; p.inputs["Roughness"].default_value = rgh; return m

mat_wood  = mat("dark_oak",      (0.28, 0.16, 0.07, 1), 0.0,  0.82)
mat_iron  = mat("blackened_iron",(0.06, 0.06, 0.07, 1), 0.75, 0.42)
mat_gold  = mat("brass_lock",    (0.72, 0.55, 0.18, 1), 0.90, 0.22)
mat_hinge = mat("iron_hinge",    (0.10, 0.10, 0.11, 1), 0.82, 0.38)
mat_nail  = mat("iron_nail",     (0.18, 0.18, 0.20, 1), 0.85, 0.30)

def obj(name, bm_, mat_):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces[:])
    ng = [f for f in bm_.faces if len(f.verts) > 4]
    if ng: bmesh.ops.triangulate(bm_, faces=ng)
    bm_.to_mesh(mesh); bm_.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = False
    o_ = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o_); o_.parent = root
    o_.matrix_parent_inverse = root.matrix_world.inverted()
    o_.data.materials.append(mat_); return o_

def box(name, sx, sy, sz, loc, mat_, bv=0.010):
    bm_ = bmesh.new()
    bmesh.ops.create_cube(bm_, size=1.0)
    bmesh.ops.scale(bm_, vec=mathutils.Vector((sx,sy,sz)),
                    verts=bm_.verts[:], space=mathutils.Matrix.Identity(4))
    if bv: bmesh.ops.bevel(bm_, geom=bm_.edges[:], offset=bv, segments=2, profile=0.6)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector(loc))
    return obj(name, bm_, mat_)

def cyl(name, r, h, loc, mat_, segs=12, axis='Z'):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=h, radius1=r, radius2=r)
    if axis == 'X':
        bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)),
                         matrix=mathutils.Matrix.Rotation(math.pi/2,4,'Y'), verts=bm_.verts[:])
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector(loc))
    return obj(name, bm_, mat_)

# ── DIMENSIONS ────────────────────────────────────────────────────────────────
CW, CD, CH = 1.10, 0.65, 0.52   # chest width, depth, height
LH = CH * 0.38                   # lid height (arched portion)
BH = CH - LH                     # base height

# ── BASE BODY ────────────────────────────────────────────────────────────────
box("chest_base_body", CW, CD, BH, (0, 0, BH/2), mat_wood, bv=0.016)

# ── ARCHED LID (approximated as a slightly raised box with curved suggestion) ─
box("lid_main", CW, CD, LH, (0, 0, BH + LH/2), mat_wood, bv=0.025)
# Lid ridge strip (suggests arch peak)
box("lid_ridge", CW, 0.06, LH*0.25, (0, 0, BH + LH*0.82), mat_iron, bv=0.005)

# ── HORIZONTAL IRON BANDS (3 wrapping the chest) ─────────────────────────────
for i, bz in enumerate((BH*0.22, BH*0.55, BH*0.88)):
    box(f"hband_front_{i+1}",   CW, 0.020, 0.055, (0, -CD/2, bz), mat_iron, bv=0.004)
    box(f"hband_back_{i+1}",    CW, 0.020, 0.055, (0,  CD/2, bz), mat_iron, bv=0.004)
    box(f"hband_left_{i+1}",  0.020, CD, 0.055, (-CW/2, 0, bz), mat_iron, bv=0.004)
    box(f"hband_right_{i+1}", 0.020, CD, 0.055, ( CW/2, 0, bz), mat_iron, bv=0.004)
# Lid band
box("lid_front_band",   CW, 0.018, 0.040, (0, -CD/2, BH + LH*0.50), mat_iron, bv=0.004)

# ── CORNER IRON PIECES (8 L-shaped corner guards) ───────────────────────────
for xi, x in ((-1, -CW/2), (1, CW/2)):
    for yi, y in ((-1, -CD/2), (1, CD/2)):
        for zi, z in ((0, 0.065), (1, BH - 0.065)):
            box(f"corner_{xi}{yi}{zi}", 0.040, 0.040, 0.12,
                (x + xi*0.020, y + yi*0.020, z), mat_iron, bv=0.008)

# ── HINGES (2 strap hinges on back) ──────────────────────────────────────────
for hx in (-CW*0.28, CW*0.28):
    # Hinge barrel
    cyl(f"hinge_barrel_{int(hx*10)}", 0.018, 0.10, (hx, -CD/2 - 0.005, BH), mat_hinge, axis='X')
    # Lower strap (on base)
    box(f"hinge_strap_lo_{int(hx*10)}", 0.035, 0.15, 0.012, (hx, -CD/2 + 0.075, BH*0.72), mat_hinge, bv=0.004)
    # Upper strap (on lid)
    box(f"hinge_strap_hi_{int(hx*10)}", 0.035, 0.15, 0.012, (hx, -CD/2 + 0.075, BH + LH*0.30), mat_hinge, bv=0.004)

# ── HASP / LOCK (front centre) ───────────────────────────────────────────────
box("hasp_plate",  0.12, 0.020, 0.10, (0, -CD/2, BH*0.60),       mat_iron, bv=0.006)
box("lock_body",   0.07, 0.04,  0.08, (0, -CD/2 - 0.020, BH*0.44), mat_gold, bv=0.010)
cyl("lock_shackle", 0.012, 0.06,     (0, -CD/2 - 0.025, BH*0.51), mat_gold, segs=8)
box("keyhole_plate", 0.040, 0.018, 0.055, (0, -CD/2 - 0.040, BH*0.44), mat_gold, bv=0.004)

# ── NAILS / RIVET DOTS ───────────────────────────────────────────────────────
for xi in (-CW*0.38, CW*0.38):
    for zi in (BH*0.20, BH*0.60):
        cyl(f"rivet_{int(xi*10)}_{int(zi*10)}", 0.012, 0.016, (xi, -CD/2, zi), mat_nail, segs=6)

# ── FEET (4 iron ball-feet) ───────────────────────────────────────────────────
for fx in (-CW/2 + 0.08, CW/2 - 0.08):
    for fy in (-CD/2 + 0.06, CD/2 - 0.06):
        bm_ = bmesh.new()
        bmesh.ops.create_uvsphere(bm_, u_segments=8, v_segments=6, radius=0.036)
        bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((fx, fy, 0.025)))
        obj(f"foot_{int(fx*10)}_{int(fy*10)}", bm_, mat_iron)

bpy.context.view_layer.update()
''')


def _fire_extinguisher_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"

def mat(name, col, met=0.0, rgh=0.45):
    m = bpy.data.materials.new(name=name); m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    o = n.new("ShaderNodeOutputMaterial"); p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], o.inputs["Surface"])
    p.inputs["Base Color"].default_value = col
    p.inputs["Metallic"].default_value = met; p.inputs["Roughness"].default_value = rgh; return m

mat_red    = mat("fire_red",      (0.75, 0.04, 0.04, 1), 0.10, 0.42)
mat_chrome = mat("chrome_valve",  (0.80, 0.80, 0.82, 1), 0.95, 0.15)
mat_black  = mat("black_hose",    (0.04, 0.04, 0.04, 1), 0.05, 0.88)
mat_label  = mat("white_label",   (0.90, 0.90, 0.90, 1), 0.00, 0.65)
mat_pin    = mat("safety_pin",    (0.85, 0.75, 0.10, 1), 0.80, 0.25)

def obj(name, bm_, mat_):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces[:])
    ng = [f for f in bm_.faces if len(f.verts) > 4]
    if ng: bmesh.ops.triangulate(bm_, faces=ng)
    bm_.to_mesh(mesh); bm_.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = True
    mesh.update(); o_ = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o_); o_.parent = root
    o_.matrix_parent_inverse = root.matrix_world.inverted()
    o_.data.materials.append(mat_); return o_

def vcyl(name, r1, r2, depth, z_bot, mat_, segs=32, bv=0.0):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=depth, radius1=r1, radius2=r2)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((0, 0, depth/2 + z_bot)))
    if bv: bmesh.ops.bevel(bm_, geom=bm_.edges[:], offset=bv, segments=2, profile=0.6)
    return obj(name, bm_, mat_)

def box(name, sx, sy, sz, loc, mat_, bv=0.008):
    bm_ = bmesh.new()
    bmesh.ops.create_cube(bm_, size=1.0)
    bmesh.ops.scale(bm_, vec=mathutils.Vector((sx,sy,sz)),
                    verts=bm_.verts[:], space=mathutils.Matrix.Identity(4))
    if bv: bmesh.ops.bevel(bm_, geom=bm_.edges[:], offset=bv, segments=2, profile=0.6)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector(loc))
    return obj(name, bm_, mat_)

# ── BODY ─────────────────────────────────────────────────────────────────────
R = 0.115
vcyl("body_main",    R, R, 0.46, 0.07, mat_red, segs=36)
vcyl("body_base",    R*1.06, R, 0.07, 0.0,  mat_red, segs=36, bv=0.008)
vcyl("body_shoulder",R, R*0.72, 0.10, 0.53, mat_red, segs=36)
vcyl("neck_tube",    R*0.42, R*0.38, 0.08, 0.63, mat_chrome, segs=24)
vcyl("neck_collar",  R*0.52, R*0.50, 0.020, 0.63, mat_chrome, segs=24)

# ── VALVE HEAD ────────────────────────────────────────────────────────────────
vcyl("valve_body",   R*0.38, R*0.42, 0.055, 0.71, mat_chrome, segs=20)
vcyl("valve_cap",    R*0.36, R*0.28, 0.040, 0.765, mat_chrome, segs=20)
# Trigger handle
box("trigger_handle", 0.010, 0.095, 0.040, (0, R*0.55, 0.745), mat_chrome, bv=0.005)
box("trigger_guard",  0.010, 0.075, 0.030, (0, R*0.30, 0.720), mat_chrome, bv=0.004)
# Pressure gauge
vcyl("gauge_body",  0.028, 0.028, 0.030, 0.700, mat_chrome, segs=16)
vcyl("gauge_face",  0.022, 0.022, 0.008, 0.730, mat_label,  segs=16)

# ── SAFETY PIN ────────────────────────────────────────────────────────────────
bm_ = bmesh.new()
bmesh.ops.create_cone(bm_, cap_ends=True, segments=8, depth=0.11, radius1=0.006, radius2=0.006)
bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)),
                 matrix=mathutils.Matrix.Rotation(math.pi/2,4,'X'), verts=bm_.verts[:])
bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((0, R*0.28, 0.750)))
obj("safety_pin", bm_, mat_pin)
box("pin_ring", 0.028, 0.028, 0.008, (0, R*0.30, 0.765), mat_pin, bv=0.008)

# ── HOSE ─────────────────────────────────────────────────────────────────────
HOSE_SEGS = 10
hose_r = 0.022
for i in range(HOSE_SEGS):
    t0 = i / HOSE_SEGS; t1 = (i+1) / HOSE_SEGS
    # hose hangs from valve side, arcs down to nozzle
    ang0 = math.radians(t0 * 150 - 30)
    ang1 = math.radians(t1 * 150 - 30)
    cx, cz = R * 0.5, 0.68
    x0 = cx + 0.22*math.sin(ang0); z0 = cz - 0.22*math.cos(ang0)
    x1 = cx + 0.22*math.sin(ang1); z1 = cz - 0.22*math.cos(ang1)
    seg_len = math.sqrt((x1-x0)**2+(z1-z0)**2)
    mx=(x0+x1)/2; mz=(z0+z1)/2
    ang_s = math.atan2(z1-z0, x1-x0)
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=8, depth=seg_len*1.1,
                          radius1=hose_r, radius2=hose_r)
    rot = mathutils.Matrix.Rotation(-ang_s+math.pi/2,4,'Y')
    bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)), matrix=rot, verts=bm_.verts[:])
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((mx, 0, mz)))
    obj(f"hose_seg_{i+1}", bm_, mat_black)

# Nozzle
bm_ = bmesh.new()
bmesh.ops.create_cone(bm_, cap_ends=True, segments=16, depth=0.095,
                      radius1=0.038, radius2=0.024)
bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)),
                 matrix=mathutils.Matrix.Rotation(math.pi*0.30,4,'Y'), verts=bm_.verts[:])
bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((R*0.8, 0, 0.28)))
obj("hose_nozzle", bm_, mat_black)

# ── LABEL BAND ────────────────────────────────────────────────────────────────
vcyl("label_band", R+0.002, R+0.002, 0.22, 0.22, mat_label, segs=36)

# ── MOUNTING BRACKET ─────────────────────────────────────────────────────────
box("wall_bracket_back",  0.030, 0.085, 0.30, (-R - 0.015, 0, 0.38), mat_chrome, bv=0.006)
box("wall_bracket_strap", 0.24,  0.020, 0.035,(0, 0, 0.50), mat_chrome, bv=0.005)
box("bracket_bolt_top",   0.016, 0.016, 0.030,(-R - 0.020, 0, 0.54), mat_chrome, bv=0.004)
box("bracket_bolt_bot",   0.016, 0.016, 0.030,(-R - 0.020, 0, 0.28), mat_chrome, bv=0.004)

bpy.context.view_layer.update()
''')


def _jerrycan_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"

def mat(name, col, met=0.55, rgh=0.45):
    m = bpy.data.materials.new(name=name); m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    o = n.new("ShaderNodeOutputMaterial"); p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], o.inputs["Surface"])
    p.inputs["Base Color"].default_value = col
    p.inputs["Metallic"].default_value = met; p.inputs["Roughness"].default_value = rgh; return m

mat_body   = mat("olive_steel",    (0.18, 0.22, 0.12, 1), 0.55, 0.48)
mat_handle = mat("dark_steel",     (0.08, 0.08, 0.09, 1), 0.80, 0.38)
mat_spout  = mat("spout_metal",    (0.22, 0.22, 0.24, 1), 0.85, 0.30)
mat_cap    = mat("red_cap",        (0.62, 0.06, 0.04, 1), 0.05, 0.55)
mat_weld   = mat("weld_seam",      (0.14, 0.14, 0.15, 1), 0.65, 0.52)

def obj(name, bm_, mat_):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces[:])
    ng = [f for f in bm_.faces if len(f.verts) > 4]
    if ng: bmesh.ops.triangulate(bm_, faces=ng)
    bm_.to_mesh(mesh); bm_.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = False
    o_ = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o_); o_.parent = root
    o_.matrix_parent_inverse = root.matrix_world.inverted()
    o_.data.materials.append(mat_); return o_

def box(name, sx, sy, sz, loc, mat_, bv=0.015):
    bm_ = bmesh.new()
    bmesh.ops.create_cube(bm_, size=1.0)
    bmesh.ops.scale(bm_, vec=mathutils.Vector((sx,sy,sz)),
                    verts=bm_.verts[:], space=mathutils.Matrix.Identity(4))
    if bv: bmesh.ops.bevel(bm_, geom=bm_.edges[:], offset=bv, segments=2, profile=0.6)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector(loc))
    return obj(name, bm_, mat_)

def vcyl(name, r1, r2, h, z_bot, mat_, segs=20, bv=0.0):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=h, radius1=r1, radius2=r2)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((0, 0, h/2+z_bot)))
    if bv: bmesh.ops.bevel(bm_, geom=bm_.edges[:], offset=bv, segments=2, profile=0.6)
    return obj(name, bm_, mat_)

def hcyl(name, r, h, loc, mat_, segs=16):
    """Horizontal cylinder along X."""
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=h, radius1=r, radius2=r)
    bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)),
                     matrix=mathutils.Matrix.Rotation(math.pi/2,4,'Y'), verts=bm_.verts[:])
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector(loc))
    return obj(name, bm_, mat_)

# ── DIMENSIONS ────────────────────────────────────────────────────────────────
CW, CD, CH = 0.245, 0.145, 0.390   # jerrycan body W, D, H (standard NATO ~20L)

# ── MAIN BODY ─────────────────────────────────────────────────────────────────
box("body_main",   CW, CD, CH*0.78, (0, 0, CH*0.39), mat_body, bv=0.018)
# Shoulder taper
box("body_shoulder", CW*0.80, CD*0.80, CH*0.14, (0, 0, CH*0.85), mat_body, bv=0.020)
# Neck
vcyl("neck_tube",  CW*0.22, CW*0.18, CH*0.12, CH*0.88, mat_body, segs=16)

# ── WELDED SIDE SEAMS ─────────────────────────────────────────────────────────
for x in (-CW/2, CW/2):
    box(f"side_seam_{'l' if x<0 else 'r'}", 0.012, 0.012, CH*0.82,
        (x, 0, CH*0.41), mat_weld, bv=0.003)

# ── HORIZONTAL REINFORCEMENT RIBS (3 indented bands) ─────────────────────────
for i, z_frac in enumerate((0.25, 0.48, 0.70)):
    box(f"rib_{i+1}", CW+0.004, CD+0.004, 0.016,
        (0, 0, CH*z_frac), mat_weld, bv=0.004)

# ── SPINE RIDGE (vertical strengthening rib on front face) ───────────────────
box("spine_ridge", 0.018, 0.012, CH*0.75, (0, -CD/2, CH*0.38), mat_weld, bv=0.004)

# ── BASE FOOT RING ────────────────────────────────────────────────────────────
box("foot_ring", CW+0.008, CD+0.008, 0.020, (0, 0, 0.010), mat_handle, bv=0.006)

# ── POUR SPOUT CAP ────────────────────────────────────────────────────────────
vcyl("spout_collar", CW*0.22+0.010, CW*0.20, 0.032, CH*0.97, mat_spout, segs=16)
vcyl("spout_cap",    CW*0.20, CW*0.22, 0.040, CH+0.002, mat_cap, segs=16, bv=0.006)
# Cap lug (wing-nut grip)
for ang in (0, 180):
    a = math.radians(ang)
    box(f"cap_lug_{ang}", 0.055, 0.014, 0.020,
        (math.cos(a)*CW*0.16, math.sin(a)*CW*0.16, CH+0.042), mat_spout, bv=0.005)

# ── FOLDING CARRY HANDLE ──────────────────────────────────────────────────────
HANDLE_W = CW*0.80
HANDLE_H = 0.08
PIVOT_Z  = CH*0.85
PIVOT_Y  = -CD*0.10

# Pivot posts (two small cylinders)
for hx in (-HANDLE_W/2, HANDLE_W/2):
    hcyl(f"handle_pivot_{'l' if hx<0 else 'r'}", 0.012, 0.030,
         (hx, PIVOT_Y, PIVOT_Z), mat_handle, segs=8)

# Handle bar (top horizontal)
hcyl("handle_bar",  0.014, HANDLE_W*0.90, (0, PIVOT_Y, PIVOT_Z + HANDLE_H), mat_handle, segs=10)

# Handle side arms (connecting pivot to bar)
for hx in (-HANDLE_W/2 + 0.02, HANDLE_W/2 - 0.02):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=8, depth=HANDLE_H,
                          radius1=0.012, radius2=0.012)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((hx, PIVOT_Y, PIVOT_Z + HANDLE_H/2)))
    obj(f"handle_arm_{'l' if hx<0 else 'r'}", bm_, mat_handle)

# ── GRIP TEXTURE BARS (3 raised ridges on handle) ────────────────────────────
for i in range(3):
    ox = -HANDLE_W*0.25 + i * HANDLE_W*0.25
    box(f"grip_ridge_{i+1}", 0.008, 0.028, 0.014,
        (ox, PIVOT_Y - 0.014, PIVOT_Z + HANDLE_H), mat_handle, bv=0.002)

bpy.context.view_layer.update()
''')


def _traffic_cone_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _o in list(bpy.data.objects): bpy.data.objects.remove(_o, do_unlink=True)
for _m in list(bpy.data.meshes): bpy.data.meshes.remove(_m)
for _t in list(bpy.data.materials): bpy.data.materials.remove(_t)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"

def mat(name, col, met=0.0, rgh=0.55):
    m = bpy.data.materials.new(name=name); m.use_nodes = True
    n = m.node_tree.nodes; n.clear()
    o = n.new("ShaderNodeOutputMaterial"); p = n.new("ShaderNodeBsdfPrincipled")
    m.node_tree.links.new(p.outputs["BSDF"], o.inputs["Surface"])
    p.inputs["Base Color"].default_value = col
    p.inputs["Metallic"].default_value = met; p.inputs["Roughness"].default_value = rgh; return m

mat_orange  = mat("safety_orange",  (0.90, 0.32, 0.02, 1), 0.0, 0.60)
mat_white   = mat("reflective_band",(0.88, 0.88, 0.88, 1), 0.0, 0.40)
mat_base    = mat("black_base",     (0.04, 0.04, 0.04, 1), 0.0, 0.75)

def obj(name, bm_, mat_):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm_, faces=bm_.faces[:])
    ng = [f for f in bm_.faces if len(f.verts) > 4]
    if ng: bmesh.ops.triangulate(bm_, faces=ng)
    bm_.to_mesh(mesh); bm_.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = True
    mesh.update(); o_ = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o_); o_.parent = root
    o_.matrix_parent_inverse = root.matrix_world.inverted()
    o_.data.materials.append(mat_); return o_

def vcyl(name, r1, r2, depth, z_bot, mat_, segs=32):
    bm_ = bmesh.new()
    bmesh.ops.create_cone(bm_, cap_ends=True, segments=segs, depth=depth, radius1=r1, radius2=r2)
    bmesh.ops.translate(bm_, verts=bm_.verts[:], vec=mathutils.Vector((0, 0, depth/2 + z_bot)))
    return obj(name, bm_, mat_)

# ── WEIGHTED BASE ─────────────────────────────────────────────────────────────
vcyl("base_main",     0.280, 0.260, 0.055, 0.000, mat_base, segs=40)
vcyl("base_lip",      0.295, 0.285, 0.012, 0.055, mat_base, segs=40)
vcyl("base_shoulder", 0.210, 0.175, 0.040, 0.067, mat_orange, segs=36)

# ── CONE BODY ─────────────────────────────────────────────────────────────────
vcyl("cone_lower",    0.175, 0.130, 0.18,  0.107, mat_orange, segs=32)
vcyl("cone_mid",      0.130, 0.072, 0.18,  0.287, mat_orange, segs=28)
vcyl("cone_upper",    0.072, 0.022, 0.16,  0.467, mat_orange, segs=24)
vcyl("cone_tip",      0.022, 0.010, 0.048, 0.627, mat_orange, segs=16)

# ── REFLECTIVE BANDS (2 white rings) ─────────────────────────────────────────
vcyl("band_lower",    0.148, 0.142, 0.055, 0.175, mat_white, segs=32)
vcyl("band_upper",    0.096, 0.090, 0.050, 0.348, mat_white, segs=28)

# ── STACKING SLOT (groove near top) ──────────────────────────────────────────
vcyl("stack_slot",    0.038, 0.032, 0.018, 0.610, mat_base, segs=16)

# ── BASE VENTILATION SLOTS (4 recesses on base perimeter) ────────────────────
for i in range(4):
    ang = math.radians(i * 90 + 45)
    bm_ = bmesh.new()
    bmesh.ops.create_cube(bm_, size=1.0)
    bmesh.ops.scale(bm_, vec=mathutils.Vector((0.055, 0.025, 0.038)),
                    verts=bm_.verts[:], space=mathutils.Matrix.Identity(4))
    rot = mathutils.Matrix.Rotation(ang, 4, 'Z')
    bmesh.ops.rotate(bm_, cent=mathutils.Vector((0,0,0)), matrix=rot, verts=bm_.verts[:])
    bmesh.ops.translate(bm_, verts=bm_.verts[:],
                        vec=mathutils.Vector((math.cos(ang)*0.235, math.sin(ang)*0.235, 0.028)))
    obj(f"base_slot_{i+1}", bm_, mat_base)

bpy.context.view_layer.update()
''')


def _wooden_crate_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"
root.location = mathutils.Vector((0.0, 0.0, 0.0))

def make_mat(name, color, metallic=0.0, roughness=0.75):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    pbr = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(pbr.outputs["BSDF"], out.inputs["Surface"])
    pbr.inputs["Base Color"].default_value = color
    pbr.inputs["Metallic"].default_value = metallic
    pbr.inputs["Roughness"].default_value = roughness
    return mat

mat_wood       = make_mat("rough_pine_wood",      (0.62, 0.44, 0.25, 1), 0.0, 0.82)
mat_wood_dark  = make_mat("weathered_wood_post",  (0.40, 0.27, 0.14, 1), 0.0, 0.88)
mat_metal      = make_mat("galvanised_bracket",   (0.48, 0.48, 0.50, 1), 0.85, 0.38)
mat_rope       = make_mat("natural_hemp_rope",    (0.70, 0.56, 0.30, 1), 0.0, 0.90)
mat_nail       = make_mat("iron_nail",            (0.25, 0.25, 0.28, 1), 0.80, 0.42)

def obj_from_bm(name, bm, mat):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    _ng = [f for f in bm.faces if len(f.verts) > 4]
    if _ng: bmesh.ops.triangulate(bm, faces=_ng)
    bm.to_mesh(mesh); bm.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = False   # flat shading for wood planks
    mesh.update()
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.parent = root
    o.matrix_parent_inverse = root.matrix_world.inverted()
    o.data.materials.append(mat)
    return o

def plank(name, length, width, thickness, loc, rot_z, mat):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector((length, width, thickness)),
                    verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    bmesh.ops.bevel(bm, geom=bm.edges[:], offset=0.005, segments=1, profile=0.5)
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector(loc))
    if rot_z:
        rot = mathutils.Matrix.Rotation(math.radians(rot_z), 4, 'Z')
        bmesh.ops.rotate(bm, cent=mathutils.Vector(loc), matrix=rot, verts=bm.verts[:])
    return obj_from_bm(name, bm, mat)

def bracket(name, w, h, depth, loc, rot_z, mat):
    """L-bracket or corner bracket: two thin rectangular plates at 90°."""
    bm = bmesh.new()
    # Vertical plate
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector((depth, w, h)),
                    verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    # Horizontal plate (flap)
    bm2 = bmesh.new()
    bmesh.ops.create_cube(bm2, size=1.0)
    bmesh.ops.scale(bm2, vec=mathutils.Vector((w, depth, h*0.15)),
                    verts=bm2.verts[:], space=mathutils.Matrix.Identity(4))
    bmesh.ops.translate(bm2, verts=bm2.verts[:],
                        vec=mathutils.Vector((w/2 - depth/2, 0, 0)))
    for v in bm2.verts:
        bm.verts.new(v.co)
    bm.verts.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector(loc))
    if rot_z:
        rot = mathutils.Matrix.Rotation(math.radians(rot_z), 4, 'Z')
        bmesh.ops.rotate(bm, cent=mathutils.Vector(loc), matrix=rot, verts=bm.verts[:])
    bm2.free()
    return obj_from_bm(name, bm, mat)

def nail(name, loc):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=6, depth=0.03,
                          radius1=0.008, radius2=0.006)
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector(loc))
    return obj_from_bm(name, bm, mat_nail)

def rope_loop(name, cx, cy, z, r_outer, r_cross, mat, segs=20):
    bm = bmesh.new()
    # Torus approximation: ring of small cylinders
    for i in range(segs):
        ang = math.radians(i * 360.0 / segs)
        x = cx + r_outer * math.cos(ang)
        y = cy + r_outer * math.sin(ang)
        seg_ang = math.radians((i+1) * 360.0 / segs)
        x2 = cx + r_outer * math.cos(seg_ang)
        y2 = cy + r_outer * math.sin(seg_ang)
        mid_x = (x + x2) / 2
        mid_y = (y + y2) / 2
        seg_len = math.sqrt((x2-x)**2 + (y2-y)**2)
        direction = math.atan2(y2-y, x2-x)
        bm2 = bmesh.new()
        bmesh.ops.create_cone(bm2, cap_ends=True, segments=6,
                              depth=seg_len*1.05, radius1=r_cross, radius2=r_cross)
        rot_z = mathutils.Matrix.Rotation(direction, 4, 'Z')
        rot_x = mathutils.Matrix.Rotation(math.pi/2, 4, 'X')
        bmesh.ops.rotate(bm2, cent=mathutils.Vector((0,0,0)), matrix=rot_z * rot_x, verts=bm2.verts[:])
        bmesh.ops.translate(bm2, verts=bm2.verts[:], vec=mathutils.Vector((mid_x, mid_y, z)))
        for v in bm2.verts:
            bm.verts.new(v.co)
        bm2.free()
    bm.verts.ensure_lookup_table()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return obj_from_bm(name, bm, mat)

# ── CRATE DIMENSIONS ─────────────────────────────────────────────────────────
CW  = 1.05   # Width (X)
CD  = 0.75   # Depth (Y)
CH  = 0.70   # Height (Z)
PT  = 0.030  # Plank thickness
PW  = 0.115  # Plank width
PG  = 0.010  # Gap between planks

# ── CORNER POSTS (4 vertical posts) ─────────────────────────────────────────
POST_W = 0.055
for xi, x in enumerate((-CW/2 + POST_W/2, CW/2 - POST_W/2)):
    for yi, y in enumerate((-CD/2 + POST_W/2, CD/2 - POST_W/2)):
        plank(f"corner_post_x{xi}y{yi}", POST_W, POST_W, CH, (x, y, CH/2), 0, mat_wood_dark)

# ── TOP & BOTTOM RAILS ────────────────────────────────────────────────────────
RAIL_W = 0.05; RAIL_T = 0.04
inner_w = CW - 2*POST_W
inner_d = CD - 2*POST_W
# Bottom rails X-direction
for y in (-CD/2 + POST_W/2, CD/2 - POST_W/2):
    plank(f"rail_bottom_x_y{int(y*100)}", inner_w, RAIL_W, RAIL_T, (0, y, RAIL_T/2), 0, mat_wood_dark)
# Bottom rails Y-direction
for x in (-CW/2 + POST_W/2, CW/2 - POST_W/2):
    plank(f"rail_bottom_y_x{int(x*100)}", RAIL_W, inner_d, RAIL_T, (x, 0, RAIL_T/2), 0, mat_wood_dark)
# Top rails
for y in (-CD/2 + POST_W/2, CD/2 - POST_W/2):
    plank(f"rail_top_x_y{int(y*100)}", inner_w, RAIL_W, RAIL_T, (0, y, CH - RAIL_T/2), 0, mat_wood_dark)
for x in (-CW/2 + POST_W/2, CW/2 - POST_W/2):
    plank(f"rail_top_y_x{int(x*100)}", RAIL_W, inner_d, RAIL_T, (x, 0, CH - RAIL_T/2), 0, mat_wood_dark)

# ── SIDE PLANKS (4 walls, 3 planks each side at mid-heights) ────────────────
PLANK_ZS = [0.13, 0.35, 0.58]
# Front & Back walls (planks along X)
for zi, pz in enumerate(PLANK_ZS):
    for y_sign, side in ((-1, "front"), (1, "back")):
        y = y_sign * CD/2
        plank(f"wall_{side}_plank_{zi+1}", inner_w, PT, PW,
              (0, y, pz), 0, mat_wood)

# Left & Right walls (planks along Y)
for zi, pz in enumerate(PLANK_ZS):
    for x_sign, side in ((-1, "left"), (1, "right")):
        x = x_sign * CW/2
        plank(f"wall_{side}_plank_{zi+1}", PT, inner_d, PW,
              (x, 0, pz), 0, mat_wood)

# Bottom floor planks (3 planks along Y)
FLOOR_ZS = [0.015]
for yi in range(3):
    y = -CD/2 + POST_W + CD/3*(yi+0.5) - POST_W/3
    plank(f"floor_plank_{yi+1}", inner_w, CD/3 - 0.01, PT, (0, y, PT/2), 0, mat_wood)

# ── LID PLANKS (4 horizontal planks on top) ────────────────────────────────
for xi in range(4):
    x = -CW/2 + POST_W + (inner_w/4)*(xi+0.5)
    plank(f"lid_plank_{xi+1}", inner_w/4 - PG, inner_d, PT, (x, 0, CH + PT/2), 0, mat_wood)

# ── METAL CORNER BRACKETS ────────────────────────────────────────────────────
BW = 0.06; BH = 0.09; BD = 0.008
corners = [
    (-CW/2, -CD/2, 0.08,   0,  "fl_lo"),
    ( CW/2, -CD/2, 0.08,  90,  "fr_lo"),
    (-CW/2,  CD/2, 0.08, -90,  "bl_lo"),
    ( CW/2,  CD/2, 0.08, 180,  "br_lo"),
    (-CW/2, -CD/2, CH-0.10,  0,  "fl_hi"),
    ( CW/2, -CD/2, CH-0.10, 90,  "fr_hi"),
    (-CW/2,  CD/2, CH-0.10,-90,  "bl_hi"),
    ( CW/2,  CD/2, CH-0.10,180,  "br_hi"),
]
for cx, cy, cz, rz, name in corners:
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector((BH, BD, BW)),
                    verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector((cx, cy, cz)))
    if rz:
        rot = mathutils.Matrix.Rotation(math.radians(rz), 4, 'Z')
        bmesh.ops.rotate(bm, cent=mathutils.Vector((cx, cy, cz)), matrix=rot, verts=bm.verts[:])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    obj_from_bm(f"corner_bracket_{name}", bm, mat_metal)

# ── NAILS / FASTENERS ────────────────────────────────────────────────────────
for zi, pz in enumerate(PLANK_ZS):
    for x_nail in (-CW/2 + POST_W*0.6, CW/2 - POST_W*0.6):
        nail(f"nail_side_l{zi}", (x_nail, -CD/2, pz))
        nail(f"nail_side_r{zi}", (x_nail,  CD/2, pz))

# ── ROPE HANDLES ─────────────────────────────────────────────────────────────
rope_loop("handle_front", 0, -CD/2, CH*0.5, 0.065, 0.012, mat_rope)
rope_loop("handle_back",  0,  CD/2, CH*0.5, 0.065, 0.012, mat_rope)
rope_loop("handle_left",  -CW/2, 0, CH*0.5, 0.065, 0.012, mat_rope)
rope_loop("handle_right",  CW/2, 0, CH*0.5, 0.065, 0.012, mat_rope)

bpy.context.view_layer.update()
''')


def _street_lamp_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"
root.location = mathutils.Vector((0.0, 0.0, 0.0))

def make_mat(name, color, metallic=0.85, roughness=0.45, emission=None):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    pbr = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(pbr.outputs["BSDF"], out.inputs["Surface"])
    pbr.inputs["Base Color"].default_value = color
    pbr.inputs["Metallic"].default_value = metallic
    pbr.inputs["Roughness"].default_value = roughness
    if emission:
        pbr.inputs["Emission Color"].default_value = emission[0]
        pbr.inputs["Emission Strength"].default_value = emission[1]
    return mat

mat_pole   = make_mat("black_metal_pole",  (0.02, 0.02, 0.025, 1), 0.88, 0.40)
mat_base   = make_mat("cast_iron_base",    (0.04, 0.04, 0.05, 1),  0.70, 0.55)
mat_collar = make_mat("decorative_collar", (0.05, 0.05, 0.06, 1),  0.85, 0.38)
mat_globe  = make_mat("globe_glass",       (0.85, 0.92, 1.0, 0.25), 0.0, 0.05)
mat_glow   = make_mat("warm_glow",        (1.0, 0.85, 0.55, 1),   0.0, 0.05,
                      emission=((1.0, 0.82, 0.50, 1.0), 4.0))
mat_arm    = make_mat("arm_metal",        (0.025, 0.025, 0.03, 1), 0.85, 0.42)

def obj_from_bm(name, bm, mat):
    mesh = bpy.data.meshes.new(name)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    _ng = [f for f in bm.faces if len(f.verts) > 4]
    if _ng: bmesh.ops.triangulate(bm, faces=_ng)
    bm.to_mesh(mesh); bm.free(); mesh.update()
    if not mesh.uv_layers: mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons: p.use_smooth = True
    mesh.update()
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.parent = root
    o.matrix_parent_inverse = root.matrix_world.inverted()
    o.data.materials.append(mat)
    return o

def vcyl(name, r1, r2, depth, z_bottom, mat, segs=32, bevel=0.0):
    """Vertical cylinder with bottom at z_bottom."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs,
                          depth=depth, radius1=r1, radius2=r2)
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector((0, 0, depth/2 + z_bottom)))
    if bevel > 0:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel, segments=2, profile=0.6)
    return obj_from_bm(name, bm, mat)

def ring(name, r_outer, r_inner, height, z_bottom, mat, segs=32):
    """Annular ring (hollow cylinder)."""
    bm = bmesh.new()
    outer = bmesh.new()
    bmesh.ops.create_cone(outer, cap_ends=True, segments=segs, depth=height,
                          radius1=r_outer, radius2=r_outer)
    inner = bmesh.new()
    bmesh.ops.create_cone(inner, cap_ends=True, segments=segs, depth=height*1.1,
                          radius1=r_inner, radius2=r_inner)
    # Approximate with outer cylinder scaled — good enough for collars
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, depth=height,
                          radius1=r_outer, radius2=r_outer*0.9)
    outer.free(); inner.free()
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector((0, 0, height/2 + z_bottom)))
    return obj_from_bm(name, bm, mat)

# ── HEAVY CAST BASE ─────────────────────────────────────────────────────────
vcyl("base_disc_lower",   0.42, 0.38, 0.06, 0.00, mat_base, segs=40, bevel=0.012)
vcyl("base_disc_upper",   0.32, 0.28, 0.05, 0.06, mat_base, segs=40)
vcyl("base_taper",        0.28, 0.14, 0.12, 0.11, mat_base, segs=32)

# Decorative ribs on base
for i in range(8):
    ang = math.radians(i * 45)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector((0.04, 0.28, 0.10)),
                    verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    rot = mathutils.Matrix.Rotation(ang, 4, 'Z')
    bmesh.ops.rotate(bm, cent=mathutils.Vector((0,0,0)), matrix=rot, verts=bm.verts[:])
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector((0, 0, 0.06)))
    o = obj_from_bm(f"base_rib_{i+1}", bm, mat_base)

# ── MAIN POLE ─────────────────────────────────────────────────────────────
POLE_H = 2.90
vcyl("pole_lower",  0.085, 0.075, POLE_H * 0.55, 0.23, mat_pole, segs=24)
vcyl("pole_upper",  0.075, 0.055, POLE_H * 0.45, 0.23 + POLE_H * 0.55, mat_pole, segs=24)

# Decorative collars on pole
COLLAR_HEIGHTS = [0.38, 0.65, 1.10]
for i, ch in enumerate(COLLAR_HEIGHTS):
    vcyl(f"pole_collar_{i+1}", 0.105, 0.10, 0.055, ch, mat_collar, segs=28)

# ── CURVED LAMP ARM ─────────────────────────────────────────────────────────
# Build arm from tapered cylinder segments forming a curve
ARM_TOP_Z = 0.23 + POLE_H * 0.92   # where arm meets the pole top
ARM_SEGS = 8
arm_r = 0.035
for i in range(ARM_SEGS):
    t0 = i / ARM_SEGS
    t1 = (i+1) / ARM_SEGS
    # parametric curve: start vertical at pole top, arc outward
    ang0 = math.radians(t0 * 80)   # sweep 80° from vertical to near-horizontal
    ang1 = math.radians(t1 * 80)
    # arc centered at (0, -0.50, ARM_TOP_Z)
    cx, cz = 0.0, ARM_TOP_Z + 0.50
    x0 = -0.50 * math.sin(ang0)
    z0 = cz - 0.50 * math.cos(ang0)
    x1 = -0.50 * math.sin(ang1)
    z1 = cz - 0.50 * math.cos(ang1)
    seg_len = math.sqrt((x1-x0)**2 + (z1-z0)**2)
    mid_x = (x0+x1)/2; mid_z = (z0+z1)/2
    seg_ang = math.atan2(z1-z0, x1-x0)  # angle from horizontal
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=16,
                          depth=seg_len, radius1=arm_r, radius2=arm_r*0.92)
    # rotate so cylinder axis aligns with segment direction
    rot_y = mathutils.Matrix.Rotation(-seg_ang + math.pi/2, 4, 'Y')
    bmesh.ops.rotate(bm, cent=mathutils.Vector((0,0,0)), matrix=rot_y, verts=bm.verts[:])
    bmesh.ops.translate(bm, verts=bm.verts[:], vec=mathutils.Vector((mid_x, 0, mid_z)))
    o = obj_from_bm(f"lamp_arm_seg_{i+1}", bm, mat_arm)

# ── LAMP HEAD ───────────────────────────────────────────────────────────────
# Position at end of arm
ARM_END_X = -0.50 * math.sin(math.radians(80))
ARM_END_Z = ARM_TOP_Z + 0.50 - 0.50 * math.cos(math.radians(80))

vcyl("lamp_head_neck",    0.055, 0.065, 0.12, ARM_END_Z, mat_collar, segs=24)
vcyl("lamp_head_housing", 0.13, 0.08, 0.25, ARM_END_Z + 0.12, mat_base, segs=32)
vcyl("lamp_head_rim",     0.145, 0.14, 0.03, ARM_END_Z + 0.12, mat_collar, segs=32)

# Globe glass (translucent sphere)
bm_globe = bmesh.new()
bmesh.ops.create_uvsphere(bm_globe, u_segments=24, v_segments=12, radius=0.115)
bmesh.ops.translate(bm_globe, verts=bm_globe.verts[:],
                    vec=mathutils.Vector((ARM_END_X, 0, ARM_END_Z + 0.22)))
o_globe = obj_from_bm("lamp_globe_glass", bm_globe, mat_globe)

# Inner glow sphere
bm_glow = bmesh.new()
bmesh.ops.create_uvsphere(bm_glow, u_segments=12, v_segments=8, radius=0.070)
bmesh.ops.translate(bm_glow, verts=bm_glow.verts[:],
                    vec=mathutils.Vector((ARM_END_X, 0, ARM_END_Z + 0.22)))
o_glow = obj_from_bm("lamp_inner_bulb", bm_glow, mat_glow)

# Finial on top
vcyl("lamp_finial_shaft", 0.018, 0.012, 0.12, ARM_END_Z + 0.355, mat_collar, segs=12)
vcyl("lamp_finial_ball",  0.028, 0.022, 0.04, ARM_END_Z + 0.475, mat_collar, segs=16)

bpy.context.view_layer.update()
''')


def _fire_hydrant_fallback_script() -> str:
    return sanitize_generated_code(r'''
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
random.seed(42)

root = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root)
root.empty_display_type = "PLAIN_AXES"
root.location = mathutils.Vector((0.0, 0.0, 0.0))

def make_mat(name, color, metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    pbr = nodes.new("ShaderNodeBsdfPrincipled")
    mat.node_tree.links.new(pbr.outputs["BSDF"], out.inputs["Surface"])
    pbr.inputs["Base Color"].default_value = color
    pbr.inputs["Metallic"].default_value = metallic
    pbr.inputs["Roughness"].default_value = roughness
    return mat

mat_red   = make_mat("red_painted_metal",   (0.72, 0.06, 0.04, 1), 0.05, 0.42)
mat_dark  = make_mat("dark_metal",          (0.05, 0.05, 0.06, 1), 0.80, 0.35)
mat_bolt  = make_mat("bolt_metal",          (0.22, 0.22, 0.24, 1), 0.90, 0.28)
mat_rim   = make_mat("chrome_rim",          (0.75, 0.75, 0.78, 1), 0.95, 0.18)

def obj_from_bm(name, bm, mat):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons:
        p.use_smooth = True
    mesh.update()
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.parent = root
    o.matrix_parent_inverse = root.matrix_world.inverted()
    o.data.materials.append(mat)
    return o

def cyl(name, r1, r2, depth, loc, mat, segs=32, rot=None, bevel_amt=0.0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs,
                          depth=depth, radius1=r1, radius2=r2)
    if bevel_amt > 0:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=bevel_amt, segments=2, profile=0.6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    _ng = [f for f in bm.faces if len(f.verts) > 4]
    if _ng: bmesh.ops.triangulate(bm, faces=_ng)
    o = obj_from_bm(name, bm, mat)
    o.location = mathutils.Vector(loc)
    if rot:
        o.rotation_euler = mathutils.Euler(rot, 'XYZ')
    return o

def pentagon_cap(name, radius, height, loc, mat):
    # Build pentagon prism from pydata — 5 verts bottom + 5 verts top
    a = [math.radians(90 + 72*i) for i in range(5)]
    bot = [(radius*math.cos(ai), radius*math.sin(ai), 0.0) for ai in a]
    top = [(radius*math.cos(ai), radius*math.sin(ai), height) for ai in a]
    verts = bot + top
    # 5 side quads + top/bottom pentagons
    faces = []
    for i in range(5):
        j = (i+1) % 5
        faces.append([i, j, j+5, i+5])
    faces.append(list(range(4, -1, -1)))      # bottom (reversed for outward normal)
    faces.append(list(range(5, 10)))           # top
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")
    for p in mesh.polygons:
        p.use_smooth = False
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.parent = root
    o.matrix_parent_inverse = root.matrix_world.inverted()
    o.location = mathutils.Vector(loc)
    o.data.materials.append(mat)
    return o

def bolt_ring(name_prefix, count, ring_r, z, bolt_r, bolt_h, mat):
    objs = []
    for i in range(count):
        ang = math.radians(i * 360.0 / count)
        x = ring_r * math.cos(ang)
        y = ring_r * math.sin(ang)
        o = cyl(f"{name_prefix}_{i+1}", bolt_r, bolt_r, bolt_h, (x, y, z), mat, segs=8)
        objs.append(o)
    return objs

# ── BASE FLANGE ─────────────────────────────────────────────────────────────
cyl("base_flange_disc",    0.28, 0.28, 0.055, (0, 0, 0.0275),   mat_dark, segs=32, bevel_amt=0.008)
cyl("base_flange_skirt",   0.26, 0.24, 0.04,  (0, 0, 0.055),    mat_dark, segs=32)
bolt_ring("base_bolt", 6, 0.235, 0.04, 0.014, 0.038, mat_bolt)

# ── LOWER BODY ──────────────────────────────────────────────────────────────
cyl("body_lower",    0.19, 0.17, 0.30, (0, 0, 0.245), mat_red, segs=32, bevel_amt=0.006)
cyl("body_mid_ring", 0.205, 0.205, 0.03, (0, 0, 0.40),  mat_dark, segs=32)

# ── UPPER BONNET ────────────────────────────────────────────────────────────
cyl("body_upper",   0.17, 0.155, 0.26, (0, 0, 0.565), mat_red, segs=32, bevel_amt=0.005)
cyl("bonnet_ring",  0.178, 0.178, 0.025, (0, 0, 0.70), mat_dark, segs=32)
cyl("bonnet_taper", 0.155, 0.105, 0.12, (0, 0, 0.78), mat_red, segs=24)

# ── PENTAGONAL TOP CAP ───────────────────────────────────────────────────────
pentagon_cap("top_cap_pentagon", 0.11, 0.07, (0, 0, 0.90), mat_dark)
cyl("top_cap_dome",  0.09, 0.04, 0.09, (0, 0, 0.97), mat_red, segs=24)
cyl("top_nut",       0.025, 0.025, 0.05, (0, 0, 1.06), mat_bolt, segs=6)   # hex nut
cyl("top_nipple",    0.015, 0.010, 0.04, (0, 0, 1.11), mat_dark, segs=12)

# ── SIDE VALVES (left and right, correctly connected) ────────────────────────
# Valve body extends from body surface outward; center at body_r + half_len from axis
VALVE_R       = 0.075
VALVE_LEN     = 0.22
VALVE_CENTER  = 0.17 + VALVE_LEN / 2   # starts at body radius, extends outward
VALVE_Z       = 0.41

for side, x_sign in (("left", -1), ("right", 1)):
    # Main valve cylinder (horizontal, along X)
    v_bm = bmesh.new()
    bmesh.ops.create_cone(v_bm, cap_ends=True, segments=24,
                          depth=VALVE_LEN, radius1=VALVE_R, radius2=VALVE_R*0.88)
    bmesh.ops.recalc_face_normals(v_bm, faces=v_bm.faces[:])
    _ng = [f for f in v_bm.faces if len(f.verts) > 4]
    if _ng: bmesh.ops.triangulate(v_bm, faces=_ng)
    valve_obj = obj_from_bm(f"side_valve_{side}", v_bm, mat_red)
    valve_obj.location = mathutils.Vector((x_sign * VALVE_CENTER, 0, VALVE_Z))
    valve_obj.rotation_euler = mathutils.Euler((0, math.radians(90), 0), 'XYZ')

    # Nozzle collar (where cap sits)
    collar_x = x_sign * (0.17 + VALVE_LEN + 0.015)
    cyl(f"nozzle_collar_{side}", 0.07, 0.065, 0.04,
        (collar_x, 0, VALVE_Z), mat_dark, segs=20,
        rot=(0, math.radians(90), 0))

    # Nozzle cap (hexagonal end cap)
    cap_x = x_sign * (0.17 + VALVE_LEN + 0.04)
    cyl(f"nozzle_cap_{side}", 0.065, 0.065, 0.055,
        (cap_x, 0, VALVE_Z), mat_bolt, segs=6,
        rot=(0, math.radians(90), 0))

    # Chain anchor stub
    stub_x = x_sign * (0.17 + VALVE_LEN + 0.06)
    cyl(f"chain_stub_{side}", 0.008, 0.008, 0.03,
        (stub_x, 0, VALVE_Z - 0.035), mat_dark, segs=8)

# ── FRONT STEAMER PORT ────────────────────────────────────────────────────────
STEAMER_Z = 0.42
cyl("front_steamer_body",   0.065, 0.055, 0.20, (0, 0.17 + 0.10, STEAMER_Z), mat_red, segs=20,
    rot=(math.radians(90), 0, 0))
cyl("front_steamer_collar", 0.07, 0.065, 0.035, (0, 0.17 + 0.205, STEAMER_Z), mat_dark, segs=20,
    rot=(math.radians(90), 0, 0))
cyl("front_steamer_cap",    0.065, 0.065, 0.04, (0, 0.17 + 0.24, STEAMER_Z), mat_bolt, segs=6,
    rot=(math.radians(90), 0, 0))

# ── BONNET BOLTS ─────────────────────────────────────────────────────────────
bolt_ring("bonnet_bolt", 5, 0.16, 0.715, 0.012, 0.03, mat_bolt)

bpy.context.view_layer.update()
''')


_TEXTURE_UPGRADE_TEMPLATE = '''

# ── ARGUS texture upgrade (appended) ─────────────────────────────────────────
# The handcrafted fallback scripts hardcode flat make_mat() colours, so an asset
# that fell back to one shipped as untextured plastic even when real PBR maps had
# already been downloaded for it. Attach them here, matching planner material
# names to the script's own material names by shared word tokens. Fully guarded:
# any failure leaves the original flat material exactly as it was.
_ARGUS_TEX_SETS = {tex_sets_json}

def _argus_apply_downloaded_textures():
    import bpy
    _stop = {{"the", "a", "of", "and", "mat", "material", "color", "colour"}}

    def _tokens(name):
        out = set()
        for chunk in str(name).lower().replace("-", "_").replace(" ", "_").split("_"):
            chunk = chunk.strip()
            if len(chunk) > 2 and chunk not in _stop:
                out.add(chunk)
        return out

    for mat in list(bpy.data.materials):
        if not mat or not mat.use_nodes or mat.name.startswith("ARGUS_GROUND"):
            continue
        nodes = mat.node_tree.nodes
        if any(n.type == "TEX_IMAGE" for n in nodes):
            continue  # already textured — never clobber a good material
        bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue

        mat_tokens = _tokens(mat.name)
        best, best_score = None, 0
        for purpose, maps in _ARGUS_TEX_SETS.items():
            score = len(mat_tokens & _tokens(purpose))
            if score > best_score:
                best, best_score = maps, score
        if not best:
            continue

        links = mat.node_tree.links
        try:
            if best.get("diff"):
                img = bpy.data.images.load(best["diff"], check_existing=True)
                tex = nodes.new("ShaderNodeTexImage")
                tex.image = img
                # Multiply by the script's own tint so painted colours survive
                # (a red hydrant stays red rather than turning raw-wood grey).
                tint = tuple(bsdf.inputs["Base Color"].default_value)
                mix = nodes.new("ShaderNodeMixRGB")
                mix.blend_type = "MULTIPLY"
                mix.inputs[0].default_value = 1.0
                mix.inputs[1].default_value = tint
                links.new(tex.outputs["Color"], mix.inputs[2])
                links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
            if best.get("rough"):
                img = bpy.data.images.load(best["rough"], check_existing=True)
                img.colorspace_settings.name = "Non-Color"
                tex = nodes.new("ShaderNodeTexImage")
                tex.image = img
                links.new(tex.outputs["Color"], bsdf.inputs["Roughness"])
            if best.get("nor_gl"):
                img = bpy.data.images.load(best["nor_gl"], check_existing=True)
                img.colorspace_settings.name = "Non-Color"
                tex = nodes.new("ShaderNodeTexImage")
                tex.image = img
                nmap = nodes.new("ShaderNodeNormalMap")
                links.new(tex.outputs["Color"], nmap.inputs["Color"])
                links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
            print("[ARGUS tex upgrade] " + mat.name)
        except Exception as _tex_exc:
            print("[ARGUS tex upgrade skip] " + mat.name + ": " + str(_tex_exc))

try:
    _argus_apply_downloaded_textures()
except Exception as _tex_outer:
    print("[ARGUS tex upgrade failed] " + str(_tex_outer))
'''


def _texture_upgrade_epilogue(part_data: Optional[dict]) -> str:
    """Blender code that attaches already-downloaded PBR maps to a handcrafted
    fallback's flat materials. Empty string when nothing was downloaded."""
    tex_sets = (part_data or {}).get("poly_haven_texture_paths") or {}
    usable = {
        str(name): {k: str(v) for k, v in (maps or {}).items()
                    if k in ("diff", "rough", "nor_gl") and v}
        for name, maps in tex_sets.items()
    }
    usable = {k: v for k, v in usable.items() if v}
    if not usable:
        return ""
    return _TEXTURE_UPGRADE_TEMPLATE.format(
        tex_sets_json=json.dumps(usable, ensure_ascii=True)
    )


def _procedural_blueprint_fallback(part_data: Optional[dict], user_prompt: str = "") -> str:
    """Handcrafted keyword-matched fallback, upgraded with real PBR textures when
    the pipeline already downloaded some for this asset."""
    script = _procedural_blueprint_fallback_raw(part_data, user_prompt)
    if not script:
        return script
    return script + _texture_upgrade_epilogue(part_data)


def _procedural_blueprint_fallback_raw(part_data: Optional[dict], user_prompt: str = "") -> str:
    blueprint = str((part_data or {}).get("structure_blueprint", "")).lower()
    category = str((part_data or {}).get("category", "")).lower()
    style = str((part_data or {}).get("style", "")).lower()
    haystack = " ".join([blueprint, category, style, user_prompt.lower()])
    _hs = f" {haystack} "

    def _word(term):
        return f" {term} " in _hs

    def _any_word(*terms):
        return any(_word(t) for t in terms)

    def _any_sub(*terms):
        """Substring match — safe for unambiguous multi-word terms only."""
        return any(t in haystack for t in terms)

    if _any_sub("gaming_chair", "gaming chair", "racing chair", "ergonomic chair"):
        return _semantic_game_asset_fallback_script("gaming_chair")
    if _any_sub("sports_car", "vehicle_four_wheel", "go_kart", "rc_car") or \
       _any_word("car", "sedan", "truck", "bus"):
        return _semantic_game_asset_fallback_script("car")
    if _any_sub("bicycle", "bike", "motorcycle", "motorbike", "scooter", "vehicle_two_wheel"):
        return _semantic_game_asset_fallback_script("bike")
    if any(term in haystack for term in ("backpack", "school bag", "rucksack", "luggage")):
        return _semantic_game_asset_fallback_script("backpack")
    if any(term in haystack for term in ("shoe", "sneaker", "trainer", "boot")):
        return _semantic_game_asset_fallback_script("shoe")
    if any(term in haystack for term in ("fire_hydrant", "fire hydrant", "hydrant")):
        return _fire_hydrant_fallback_script()
    if any(term in haystack for term in ("street_lamp", "street lamp", "lamp post", "lamppost", "streetlight")):
        return _street_lamp_fallback_script()
    if any(term in haystack for term in ("park_bench", "park bench", "bench")):
        return _park_bench_fallback_script()
    if any(term in haystack for term in ("fire_extinguisher", "fire extinguisher", "extinguisher")):
        return _fire_extinguisher_fallback_script()
    if any(term in haystack for term in ("traffic_cone", "traffic cone", "road cone")):
        return _traffic_cone_fallback_script()
    if any(term in haystack for term in ("treasure_chest", "treasure chest", "chest")):
        return _treasure_chest_fallback_script()
    if any(term in haystack for term in ("jerry_can", "jerrycan", "jerry can", "fuel_can", "fuel can",
                                          "gas can", "petrol can", "nato can")):
        return _jerrycan_fallback_script()
    if any(term in haystack for term in ("metal_barrel", "barrel", "drum")):
        return _metal_barrel_fallback_script()
    _scifi_crate_terms = (
        "sci_fi_cargo_crate", "sci_fi_crate", "sci-fi crate", "sci fi crate",
        "ammo crate", "ammunition crate", "supply crate", "heavy crate",
        "sci-fi prop", "sci fi prop",
    )
    _has_scifi = any(t in haystack for t in ("sci-fi", "sci_fi", "scifi", "sci fi"))
    _has_crate = any(t in haystack for t in ("crate", "container", "box"))
    if any(t in haystack for t in _scifi_crate_terms) or (_has_scifi and _has_crate):
        return _sci_fi_crate_fallback_script()
    if any(term in haystack for term in ("wooden_crate", "wooden crate", "wood crate", "plank crate")):
        return _wooden_crate_fallback_script()
    if _has_crate:
        return _semantic_game_asset_fallback_script("crate")
    if "castle" in blueprint or "castle" in category or "fortress" in category:
        return _castle_fallback_script()
    if "banyan_tree" in blueprint or "banyan tree" in category:
        return _banyan_tree_fallback_script()
    if "banana_tree" in blueprint or "banana tree" in category or "banana plant" in category:
        return _banana_tree_fallback_script()
    return ""


