
from __future__ import annotations
import re as _re

_MD_FENCE = _re.compile(r"```[a-zA-Z]*\s*")

def sanitize_generated_code(code: str) -> str:
    code = _MD_FENCE.sub("", code).replace("```", "").strip()
    return code


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
