
from __future__ import annotations

from core.prompt_modules import select_rule_modules
from core.components import component_prompt_block


_REQUIRED_PART_DATA_KEYS = {"category", "parts", "style", "material"}

_POLY_TARGETS: dict[str, tuple[int, int, str]] = {
    "low":    (500,    1_500,  "mobile / LOD2"),
    "medium": (1_500,  5_000,  "game-ready / LOD0"),
    "high":   (5_000, 20_000,  "cinematic / hero prop"),
}

MAX_REPAIR_ATTEMPTS = 5


_REPAIR_TAXONOMY = """\nREPAIR TAXONOMY — match error to canonical fix:

AttributeError "inputs": node.inputs["Metallic"/"Roughness"/"Base Color"].default_value
AttributeError "use_auto_smooth"/"auto_smooth_angle": delete those lines; use
  shade_smooth_by_angle(angle=math.radians(30)) inside try/except instead.
AttributeError "objects.link"/"NoneType": bpy.context.collection.objects.link(obj)
AttributeError "clear" on bpy_collection: use list(col) + remove() loop.
TypeError "bpy_prop_collection.__contains__": use `if mat.name not in obj.data.materials:`
TypeError "from_pydata": pass int indices — [v.index for v in f.verts]
TypeError "Vector": mathutils.Vector((x, y, z))
RuntimeError "poll() failed": switch mode first, or use bmesh instead.
RuntimeError "is not iterable" bpy.data.objects: iterate list(bpy.data.objects).
KeyError "ARGUS_ROOT": create empty named exactly "ARGUS_ROOT".
KeyError socket: use only Base Color/Metallic/Roughness/IOR Level/Emission Color/Emission Strength.
FileNotFoundError export: os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
FileNotFoundError images.load: wrap in try/except; do NOT change the path.
MemoryError/infinite geo: reduce segment counts; add poly-budget guard.
ZeroDivisionError: assert scale > 0.001 before bmesh ops.
AssertionError poly budget: reduce segments; bmesh.ops.dissolve_limit(__POLY_MAX__).
Flat wheels/rings: rotate 90° around X or Y before bm.to_mesh().
Removed Blender 5.0 params: export_selected→use_selection; apply_modifiers removed; mesh_smooth_type removed.

"""

def build_object_prompt(
    user_prompt: str,
    part_data: dict,
    mcp_mode: bool = True,
    repair_context: dict | None = None,
) -> str:
    """
    Build the full code-generation system prompt for the coding LLM.

    Parameters
    ----------
    user_prompt     : raw natural-language asset request
    part_data       : structured metadata from semantic decomposition stage
                      (see schema comment above)
    mcp_mode        : True  → code runs inside a live Blender session via MCP
                      False → code runs in Blender 5.0 batch mode
    repair_context  : dict produced by the traceback parser on prior failure
                      (see schema comment above)
    """

    missing = _REQUIRED_PART_DATA_KEYS - part_data.keys()
    if missing:
        raise ValueError(
            f"part_data is missing required keys: {missing}. "
            "Check the semantic decomposition output."
        )

    category      : str        = part_data["category"]
    parts         : list[str]  = part_data["parts"]
    style         : str        = part_data["style"]
    material      : str        = part_data["material"]
    poly_budget   : str        = part_data.get("poly_budget",   "medium")
    export_format : str        = part_data.get("export_format", "glb").lower()
    asset_scale   : float      = float(part_data.get("asset_scale", 1.0))
    output_dir    : str        = part_data.get("output_dir",    "/tmp")
    seed          : int | None = part_data.get("seed",          None)
    planner_reasoning: str = str(part_data.get("reasoning", "") or "")
    structure_summary: str = part_data.get(
        "structure_summary",
        "A stable hard-surface object with connected parts.",
    )
    structure_blueprint: str = part_data.get("structure_blueprint", "")
    blueprint_count = part_data.get("blueprint_count", "")
    memory_success_count = part_data.get("memory_success_count", 0)
    memory_last_prompt: str = part_data.get("memory_last_prompt", "")
    support_strategy: str = part_data.get(
        "support_strategy",
        "Keep the asset grounded with a stable support footprint.",
    )
    poly_haven_texture_paths: dict = part_data.get("poly_haven_texture_paths") or {}
    geometry_mode: str  = part_data.get("geometry_mode", "hard_surface")
    organic_parts: list = part_data.get("organic_parts") or []
    organic_terms = (
        "tree",
        "plant",
        "leaf",
        "leaves",
        "fruit",
        "banana",
        "palm",
        "trunk",
        "branch",
        "bark",
        "root",
        "foliage",
        "flower",
    )
    organic_text = " ".join(
        [
            user_prompt,
            category,
            style,
            material,
            structure_blueprint,
            " ".join(str(part) for part in parts),
        ]
    ).lower()
    is_botanical_asset = any(term in organic_text for term in organic_terms)
    wants_high_detail = any(
        term in user_prompt.lower()
        for term in ("high res", "high-res", "high resolution", "high detail", "hero", "cinematic")
    )
    if is_botanical_asset and wants_high_detail:
        poly_budget = "high"

    def _list_field(key: str, fallback: list[str]) -> list[str]:
        value = part_data.get(key, fallback)
        if not isinstance(value, list) or not value:
            return fallback
        return [str(item) for item in value]

    primary_forms = _list_field("primary_forms", ["main body volume"])
    assembly_order = _list_field("assembly_order", ["build main body", "attach details"])
    attachment_points = _list_field(
        "attachment_points",
        ["visible structural joints between parts"],
    )
    orientation_notes = _list_field(
        "orientation_notes",
        ["Z is up; align circular mechanical parts to their real axis"],
    )
    texture_plan = _list_field(
        "texture_plan",
        [f"{material} base material with subtle roughness variation"],
    )
    shape_refinement_plan = _list_field(
        "shape_refinement_plan",
        [
            "beveled hard-surface edges",
            "rounded or tapered forms where the object is not naturally box-shaped",
            "surface details that break up large plain faces",
        ],
    )
    critical_constraints = _list_field(
        "critical_constraints",
        ["no floating parts", "all major parts visibly connected"],
    )

    if poly_budget not in _POLY_TARGETS:
        poly_budget = "medium"

    poly_min, poly_max, poly_use = _POLY_TARGETS[poly_budget]
    poly_range = f"{poly_min:,}–{poly_max:,}"

    part_text = "\n".join(f"- {p}" for p in parts)
    primary_form_text = "\n".join(f"- {p}" for p in primary_forms)
    assembly_text = "\n".join(f"- {p}" for p in assembly_order)
    attachment_text = "\n".join(f"- {p}" for p in attachment_points)
    orientation_text = "\n".join(f"- {p}" for p in orientation_notes)
    texture_text = "\n".join(f"- {p}" for p in texture_plan)
    shape_refinement_text = "\n".join(f"- {p}" for p in shape_refinement_plan)
    constraint_text = "\n".join(f"- {p}" for p in critical_constraints)

    category_rule_modules = select_rule_modules(
        user_prompt, category, style, structure_blueprint, " ".join(str(p) for p in parts)
    )

    repair_taxonomy_block = (
        _REPAIR_TAXONOMY.replace("__POLY_MAX__", str(poly_max))
        if repair_context else ""
    )

    component_block = component_prompt_block()

    if planner_reasoning:
        planner_reasoning_block = (
            "PLANNER SPATIAL REASONING\n"
            "(Use these exact positions and sizes when placing parts — "
            "the planner worked these out from real-world dimensions)\n\n"
            + planner_reasoning.strip()
            + "\n"
        )
    else:
        planner_reasoning_block = ""

    scene_graph = part_data.get("scene_graph") or []
    if isinstance(scene_graph, list) and scene_graph:
        _sg_lines = [
            "SCENE GRAPH — PRECISION LAYOUT (derive all positions from these numbers)",
            "",
            "GEOMETRIC PRECISION RULES:",
            "1. Define every key dimension as a NAMED CONSTANT at the top of your script",
            "   e.g.  HULL_LEN = 9.0; HULL_BEAM = 3.1; HULL_DEPTH = 1.4; DECK_Z = 1.4",
            "2. Compute ALL part positions/sizes from those constants — never hardcode",
            "   magic numbers that don't relate to each other.",
            "3. Every joint/attachment must be EXACT: if deck sits on hull at Z=HULL_DEPTH,",
            "   the cabin base must be at Z=HULL_DEPTH, not Z=1.42 or Z=1.38.",
            "4. Symmetric parts (wheels L/R, panels, fins) MUST be mirrored, not eyeballed.",
            "5. After placing all parts, the combined bounding box must match the target",
            "   asset_scale = {asset_scale} m (longest dimension).",
            "",
            "Part layout (centre pos x,y,z in metres; Z=0 is ground level):",
            "",
        ]
        for _node in scene_graph[:40]:
            if not isinstance(_node, dict):
                continue
            _nm = str(_node.get("name", "part"))
            _pos = _node.get("pos") or _node.get("position") or [0, 0, 0]
            _sz = _node.get("size") or _node.get("dims") or [0.2, 0.2, 0.2]
            try:
                _pos = [round(float(v), 3) for v in list(_pos)[:3]]
                _sz = [round(float(v), 3) for v in list(_sz)[:3]]
            except (TypeError, ValueError):
                continue
            _attach = _node.get("attach_to") or _node.get("parent") or ""
            _suffix = f"  → attaches to: {_attach}" if _attach else ""
            _sg_lines.append(
                f"  {_nm}: centre=({_pos[0]}, {_pos[1]}, {_pos[2]})  "
                f"size=({_sz[0]}w × {_sz[1]}d × {_sz[2]}h){_suffix}"
            )
        scene_graph_block = "\n".join(_sg_lines) + "\n"
    else:
        scene_graph_block = (
            f"GEOMETRIC PRECISION RULES:\n"
            f"1. Define key dimensions as named constants: e.g. LENGTH = {asset_scale:.1f}; BEAM = LENGTH/3.0\n"
            f"2. Derive ALL positions from those constants — no isolated magic numbers.\n"
            f"3. Every joint must be exact: if part A ends at Z=H, part B must start at Z=H.\n"
            f"4. Symmetric parts (wheels, fins, struts) must be mirrored or computed as ±offset.\n"
            f"5. Final bounding box longest dimension must ≈ {asset_scale} m.\n"
        )

    if poly_haven_texture_paths:
        _ph_lines = ["POLY HAVEN PBR TEXTURES (pre-downloaded, CC0)\n"]
        _ph_lines.append(
            "These real PBR textures have been downloaded for this asset. "
            "Apply them via ShaderNodeTexImage in the material node tree "
            "INSTEAD of using plain Base Color values for those materials.\n"
        )
        for mat_name, maps in poly_haven_texture_paths.items():
            _ph_lines.append(f"  Material purpose: {mat_name}")
            for map_key, path in maps.items():
                _ph_lines.append(f"    {map_key}: r\"{path}\"")
        _ph_lines.append("""
Texture setup — wrap every load in try/except (missing file must NOT crash):
  try: img = bpy.data.images.load(r"<path>"); img.colorspace_settings.name="sRGB"
  except Exception: img = None
  if img:
      tex = nodes.new("ShaderNodeTexImage"); tex.image = img
      # Tint multiply keeps painted color (red hydrant stays red, not grey):
      tint = principled.inputs["Base Color"].default_value[:]
      mix = nodes.new("ShaderNodeMixRGB"); mix.blend_type="MULTIPLY"
      mix.inputs[0].default_value=1.0; mix.inputs[1].default_value=tint
      links.new(tex.outputs["Color"], mix.inputs[2])
      links.new(mix.outputs["Color"], principled.inputs["Base Color"])
Roughness (Non-Color): links.new(tex_rough.outputs["Color"], principled.inputs["Roughness"])
Normal map: ShaderNodeTexImage (Non-Color) → ShaderNodeNormalMap → principled["Normal"]
Use ONLY the exact paths listed above. Do NOT invent paths.
""")
        poly_haven_block = "\n".join(_ph_lines)
    else:
        poly_haven_block = ""

    use_metaballs = geometry_mode in ("organic", "hybrid")
    if use_metaballs:
        _organic_list = "\n".join(f"  - {p}" for p in organic_parts) if organic_parts else "  - (all parts)"
        metaball_block = f"""
ORGANIC GEOMETRY — NATIVE BLENDER METABALLS
============================================

geometry_mode : {geometry_mode}
Organic parts that MUST use metaballs:
{_organic_list}

Use Blender metaballs for every organic part listed above.
Use bmesh primitives (create_cone, create_cube, etc.) for all hard-surface parts.

METABALL ELEMENT TYPES
  BALL      → sphere — use for fruit, joints, knobs, root bulges
  CAPSULE   → elongated sphere — use for branches, roots, stems, tubes
              (set elem.size_x for the capsule half-length along local X)
  ELLIPSOID → tri-axis stretch — use for trunk sections, body masses, leaf crowns
              (set elem.size_x, size_y, size_z independently)

RESOLUTION GUIDE
  mb_data.resolution = 0.05   # coarse — use for large background masses
  mb_data.resolution = 0.03   # standard — good balance of detail and speed
  mb_data.resolution = 0.02   # fine — use only for hero/cinematic assets

STIFFNESS (controls blend sharpness between elements)
  stiffness = 1.0   # heavy blending — elements melt into each other
  stiffness = 2.0   # moderate — clear shapes with smooth transitions
  stiffness = 3.0+  # sharp — distinct elements, tight blend zone only at contact

METABALL BUILD PATTERN (one per organic cluster)

  # --- CREATE METABALL DATA ---
  mb_data = bpy.data.metaballs.new("trunk_meta")
  mb_data.resolution        = 0.03
  mb_data.render_resolution = 0.03
  mb_data.threshold         = 0.6

  mb_obj = bpy.data.objects.new("trunk_meta", mb_data)
  bpy.context.collection.objects.link(mb_obj)

  # --- ADD ELEMENTS ---
  e = mb_data.elements.new(type='CAPSULE')
  e.co        = mathutils.Vector((0, 0, 0.8))   # center in local space
  e.radius    = 0.25
  e.size_x    = 0.7                              # half-length for CAPSULE
  e.stiffness = 2.0

  e2 = mb_data.elements.new(type='BALL')
  e2.co        = mathutils.Vector((0.3, 0, 1.4))
  e2.radius    = 0.18
  e2.stiffness = 2.0

  # Elements in the SAME metaball data object always blend together.
  # For separate clusters that must NOT blend, create a new metaball object
  # with a different name (e.g. "leaf_cluster_meta" vs "trunk_meta").

  # --- FORCE EVALUATION ---
  bpy.context.view_layer.update()

  # --- CONVERT TO REAL MESH (no bpy.ops needed) ---
  depsgraph = bpy.context.evaluated_depsgraph_get()
  eval_mb   = mb_obj.evaluated_get(depsgraph)
  mesh      = bpy.data.meshes.new_from_object(eval_mb)
  mesh.name = "trunk"

  final_obj = bpy.data.objects.new("trunk", mesh)
  bpy.context.collection.objects.link(final_obj)
  final_obj.parent = root_obj

  # Remove temporary metaball objects
  bpy.data.objects.remove(mb_obj, do_unlink=True)
  bpy.data.metaballs.remove(mb_data)

  # --- POST-PROCESS (required after every metaball conversion) ---
  bm = bmesh.new()
  bm.from_mesh(mesh)
  bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(5),
                            verts=bm.verts[:], edges=bm.edges[:])
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
  _ngons = [f for f in bm.faces if len(f.verts) > 4]
  if _ngons: bmesh.ops.triangulate(bm, faces=_ngons)  # n-gons only, never quads
  bm.to_mesh(mesh); bm.free(); mesh.update()

  if not mesh.uv_layers:
      mesh.uv_layers.new(name="ARGUS_UV")

  final_obj.data.materials.append(mat_bark)   # assign your material

RULES FOR ORGANIC PARTS
- Every organic part MUST use the metaball pattern above — no cone stacks or cubes.
- Group blending elements (trunk + roots + stubs) in ONE metaball object.
- Separate metaball objects for leaf clusters and fruit so they don't merge with trunk.
- Post-process every converted mesh (dissolve_limit → recalc_normals → n-gons only).
- Resolution 0.04–0.06 for medium budget, 0.025–0.035 for high.
- Hard-surface hardware stays as bmesh objects; parent to ARGUS_ROOT.
"""
    else:
        metaball_block = ""

    if is_botanical_asset:
        object_type_rules = """
OBJECT TYPE RULES

Generate INANIMATE BOTANICAL / ORGANIC VEGETATION assets when requested.

Allowed organic targets:
- trees, palms, banana plants, shrubs, flowers, leaves, fruit, roots, vines,
  bark, stems, branches, seed pods, and stylized low/high-poly vegetation.

Still DO NOT generate:
- humans / humanoids / creatures / animal anatomy
- faces / eyes / mouths
- characters / NPCs
- armatures / rigging / shape keys
- fluid / smoke / cloth simulations
- particles / hair systems
- terrain or landscape meshes
- animation keyframes
"""
        geometry_rules = f"""
GEOMETRY RULES

USE:
- Procedural botanical geometry made from bpy + bmesh mesh objects.
- Modular construction: one mesh per logical cluster such as trunk sections,
  leaf fans, fruit bunches, roots, stems, flowers, and bark plates.
- Real-world scale: longest dimension ≈ {asset_scale} m.
- Game-ready high-detail silhouette within the triangle budget.

SURFACE CONSTRUCTION RULE

Build surface-only geometry. Every part is a shell — the interior does not
exist and does not matter. Do NOT add Solidify modifiers, interior face loops,
thickness walls, or closed manifold volumes. Model only what the camera sees.

BOTANICAL QUALITY RULES
- Tapered curved trunk (overlapping frustums rotated a few degrees, not a pole).
- Root flare or basal leaves grounding the plant at Z=0.
- Leaf blades: tapered mesh cards with center vein, not flat rectangles. Layered
  radial clusters with varied yaw/pitch/scale — not one flat ring.
- 4+ materials: bark, leaf dark, leaf light/vein, fruit.
- Banana tree: fibrous pseudostem, 12–20 large leaves, hanging banana bunch.
- Banyan tree: multi-column trunk, buttress/prop roots, aerial roots, wide canopy
  (wider than tall). Never a single pole + sphere canopy.
- No floating foliage; all leaves/fruit visibly attached to stems or stalks.
"""
    else:
        object_type_rules = """
{object_type_rules}
"""
        geometry_rules = f"""
GEOMETRY RULES

- Procedural hard-surface geometry, one mesh per logical part.
- Real-world scale: longest dimension ≈ {asset_scale} m.
- Surface-only geometry (shells). No interior faces or Solidify unless a part
  is literally solid (e.g. rubber tire cross-section).

SHAPE QUALITY RULES
- No unmodified cubes unless the object is truly box-shaped. Bevel, taper, inset.
- Part-appropriate primitives: cushions/pads → scaled uvsphere ellipsoids;
  wheels/bolts/pipes → create_cone(radius1=r, radius2=r), ≥16 segments;
  tapered parts (feet, nozzles) → frustums with differing radius1/radius2.
- Every meaningful visible part is a separate named mesh under ARGUS_ROOT.
- Clean names (no ARGUS_ prefix): car_body_shell, rubber_black, glass_dark.
- Add seam lines, panel breaks, raised rims, bolts, insets to break up flat faces.
- Prefer fewer well-shaped parts over many blocky filler parts.
"""

    seed_line = (
        f"random.seed({seed})   # ARGUS reproducibility contract"
        if seed is not None
        else "# random.seed() not set — non-deterministic run (set part_data['seed'] for reproducibility)"
    )

    repair_block = ""
    if repair_context:
        ec  = repair_context.get("error_class",   "UnknownError")
        em  = repair_context.get("error_msg",     "")
        fl  = repair_context.get("failed_line",   "")
        att = repair_context.get("attempt_number", 1)

        repair_block = f"""
REPAIR CONTEXT  (attempt {att} of {MAX_REPAIR_ATTEMPTS})

Previous execution FAILED.

Error class  : {ec}
Error message: {em}
Failed line  : {fl}

Consult REPAIR TAXONOMY below for the canonical fix.
Do NOT repeat the pattern that caused the failure.
{"WARNING: approaching max attempts — simplify geometry if stuck." if att >= MAX_REPAIR_ATTEMPTS - 1 else ""}

"""

    if mcp_mode:
        context_block = """
MCP EXECUTION CONTEXT

Code runs inside a LIVE Blender 5.0 session via the
Blender MCP server (execute_blender_code tool).

Implications:
- bpy.context IS valid; active window exists
- A live scene already exists — clear it with the
  SCENE INIT TEMPLATE below before building geometry
- bpy.ops calls are allowed only for Blender 5.0 operators listed in this prompt
- MCP captures stdout — use print() for checkpoints:
    print("ARGUS: building base mesh")
    print("ARGUS: applying materials")
    print("ARGUS: exporting")
- End script with the ARGUS_OK sentinel (see below)

MCP TOOL INTERACTION ORDER (pipeline contract):
  1. execute_blender_code  → generate geometry
  2. get_object_info       → validate mesh stats
  3. get_viewport_screenshot (optional) → visual QA
  4. execute_blender_code  → run export block

DO NOT make socket, HTTP, or subprocess calls
inside generated scripts. MCP handles all I/O.
"""
    else:
        context_block = """
HEADLESS EXECUTION CONTEXT

Code runs in Blender batch mode:
  blender --background --python script.py

Implications:
- bpy.context window may be None
- There may be no active window, screen, or VIEW_3D area
- Avoid bpy.ops calls that require UI context
- Prefer bmesh and direct data API operations
- Never access bpy.context.window_manager.windows[0] in headless mode
- Never call bpy.ops without first proving the required context exists
"""

    if mcp_mode:
        uv_block = """\
UV LAYER:
  Every mesh MUST have exactly one UV layer named "ARGUS_UV":
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")

  Smart UV project (MCP / live session — safe):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66))
    bpy.ops.object.mode_set(mode="OBJECT")"""
    else:
        uv_block = """\
UV LAYER:
  Every mesh MUST have exactly one UV layer named "ARGUS_UV":
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")

  Headless mode has no reliable VIEW_3D context. Do NOT call
  bpy.ops.uv.smart_project() or access window_manager.windows[0].
  If UV coordinates are needed, assign them through mesh.uv_layers
  data directly, or leave the ARGUS_UV layer empty."""

    export_path_expr = (
        f'os.path.join(r"{output_dir}", '
        f'f"ARGUS_{{ARGUS_EXPORT_OBJ.name}}.{export_format}")'
    )

    if export_format == "fbx":
        export_op_block = """\
  bpy.ops.export_scene.fbx(
      filepath=ARGUS_EXPORT_PATH,
      use_selection=True,
      bake_anim=False,
      use_mesh_edges=False,
      axis_forward="-Z",
      axis_up="Y",
  )"""
    else:
        export_op_block = """\
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
  )"""

    return f"""
Generate Blender 5.0 compatible Python code using bpy + bmesh.
{repair_block}
CRITICAL OUTPUT RULES

- Output ONLY executable Python code
- NO markdown fences
- NO explanations, comments to the reader, or prose
- The script must run without modification

REQUIRED IMPORTS  (first 6 lines, exact order)

import bpy
import bmesh
import math
import mathutils
import random
import os

FORBIDDEN IMPORTS

DO NOT USE:
- from core / import core
- from utils / import utils
- from ml / import ml
- numpy        (unavailable in many Blender builds)
- scipy / PIL / cv2 / torch
- external addons
- socket / http / requests / urllib / subprocess

OBJECT SPECIFICATION

User Prompt  : {user_prompt}
Category     : {category}
Blueprint    : {structure_blueprint or "none"}{f" ({blueprint_count} available)" if blueprint_count else ""}
Memory       : {memory_success_count} strict success run(s){f" | last: {memory_last_prompt}" if memory_last_prompt else ""}
Style        : {style}
Material     : {material}
Parts        :
{part_text}
Asset scale  : {asset_scale} metres (real-world longest dimension)
Poly budget  : {poly_range} triangles  ({poly_use})
Export format: {export_format.upper()}
Output dir   : {output_dir}

STRUCTURAL PLAN FROM PLANNER

Structure summary:
{structure_summary}

Primary forms:
{primary_form_text}

Assembly order:
{assembly_text}

Attachment points:
{attachment_text}

Orientation notes:
{orientation_text}

Support strategy:
{support_strategy}

Texture/material plan:
{texture_text}

{poly_haven_block}
Shape refinement plan:
{shape_refinement_text}

Critical constraints:
{constraint_text}

{scene_graph_block}
{planner_reasoning_block}
QUALITY FLOOR (NON-NEGOTIABLE)

Asset must be recognizable as the requested object in the first viewport render.
- 8+ named mesh objects for simple props; 14+ for industrial/furniture/vehicles.
- 3+ distinct materials when the brief mentions multiple colors, rubber, glass,
  emissive, wood, painted metal, or raw metal.
- All blueprint signature parts present as named objects, not just comments.
- Small repeated details: bolt rows, seam strips, panel grooves, ribs, vents.
- No floating parts — every detail must touch or slightly overlap its surface.
- Distinctive colors (red, blue, black rubber, wood) → separate named material.
- Layered build: base volume → structural parts → hardware → surface detail.

OBJECT TYPE RULES

Generate ONLY INANIMATE HARD-SURFACE OBJECTS.

DO NOT generate:
- humans / humanoids / creatures / anatomy
- faces / eyes / mouths
- characters / NPCs
- armatures / rigging / shape keys
- fluid / smoke / cloth simulations
- particles / hair systems
- terrain or landscape meshes
- animation keyframes

BLENDER VERSION CONTRACT

Target  : Blender 4.1+ (validated on Blender 4.x and 5.x)

Line 7 MUST be this exact single-line assert:
  assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 4.1+ , got {{bpy.app.version}}"

BLENDER 4.1+/5.0 API CONTRACT

Use only stable Blender 4.1+ Python APIs and bmesh operators that also exist in
Blender 5.0. Do not generate code for Blender 2.x / 3.x-only or 6.x-only APIs.

Banned non-5.0 / deprecated / removed patterns:
  bpy.context.scene.objects.link()       → collection.objects.link()
  mesh.update(calc_edges=True)           → mesh.update()
  bpy.ops.mesh.uv_texture_add()          → mesh.uv_layers.new()
  ob.data.show_double_sided              → removed before Blender 5.0
  material.use_nodes = False then direct → always use nodes
  bmesh.ops.extrude_edge_region          → unavailable in Blender 5.0
  bpy.ops.wm.collada_export()            → unsupported
  bpy.ops.object.parent_set()            → use direct parent assignment
  bpy.context.window_manager.windows[0]  → unavailable in headless mode
  bmesh.ops.create_cylinder              → not a valid BMesh operator; use create_cone with radius1 == radius2

Allowed Blender 5.0 modules/imports are exactly:
  bpy, bmesh, math, mathutils, random, os

Allowed bmesh operators:
  create_cube, create_cone, create_circle, create_grid, create_uvsphere,
  recalc_face_normals, triangulate, delete, extrude_face_region,
  inset_region, bevel, translate, scale, rotate, subdivide_edges

Allowed bpy.ops operators:
  bpy.ops.object.select_all
  bpy.ops.object.transform_apply
  bpy.ops.object.mode_set            # MCP/live only when context exists
  bpy.ops.mesh.select_all            # MCP/live edit mode only
  bpy.ops.uv.smart_project           # MCP/live edit mode only
  bpy.ops.export_scene.gltf
  bpy.ops.export_scene.fbx

{context_block}

SCENE INITIALIZATION TEMPLATE

EVERY script MUST begin with this exact block
immediately after imports and the version assert.
Do NOT skip or modify it.

# ── ARGUS scene init ──────────────────────────
for _obj  in list(bpy.data.objects):    # list() copy — safe to mutate
    bpy.data.objects.remove(_obj,  do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat  in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
# ─────────────────────────────────────────────

REPRODUCIBILITY CONTRACT

The NEXT line after scene init MUST be:
  {seed_line}

If seed is set, the asset MUST be geometrically
identical across runs with the same seed value.
Use random only for non-structural variation
(e.g. panel line jitter, rivet placement).
Never use random for core structural dimensions.

ACTIVE OBJECT & CONTEXT MANAGEMENT

After creating any object you MUST:

1. Link to the active collection:
     bpy.context.collection.objects.link(obj)

2. Set as active and selected:
     bpy.context.view_layer.objects.active = obj
     obj.select_set(True)

3. Call update after bulk operations:
     bpy.context.view_layer.update()

MODE SWITCHING rules:
- Enter edit mode:  bpy.ops.object.mode_set(mode="EDIT")
- Exit edit mode:   bpy.ops.object.mode_set(mode="OBJECT")
- NEVER call bpy.ops.mesh.* while in OBJECT mode
- NEVER call bpy.ops.object.* while in EDIT mode
- PREFERRED: use bmesh directly — no mode switching needed:
    bm = bmesh.new()
    # build geometry
    bm.to_mesh(mesh)
    bm.free()

PRINCIPLED BSDF RULES

DO NOT use direct attribute access (AttributeError in Blender 5.0):
  node.metallic / node.roughness / node.specular / node.base_color

ALWAYS use input socket dict:
  node.inputs["Metallic"].default_value
  node.inputs["Roughness"].default_value
  node.inputs["Base Color"].default_value
  node.inputs["IOR Level"].default_value   # Blender 5.0 replacement for Specular

If uncertain about a socket name, print all available:
  [i.name for i in principled.inputs]

Canonical material setup:
  mat        = bpy.data.materials.new(name="brushed_metal")
  mat.use_nodes = True
  nodes      = mat.node_tree.nodes
  nodes.clear()
  output     = nodes.new("ShaderNodeOutputMaterial")
  principled = nodes.new("ShaderNodeBsdfPrincipled")
  mat.node_tree.links.new(
      principled.outputs["BSDF"], output.inputs["Surface"]
  )
  principled.inputs["Base Color"].default_value = (0.6, 0.6, 0.65, 1.0)
  principled.inputs["Metallic"].default_value   = 0.9
  principled.inputs["Roughness"].default_value  = 0.25

MATERIAL PRESETS

Use clean material names without an ARGUS_ prefix unless a specific internal
pipeline marker requires it. Good material names: brushed_metal, painted_metal,
rubber_black, glass_dark, fabric_canvas, leather_dark, plastic_black,
paint_red, emissive_cyan.
Every mesh MUST have at least one material:
  obj.data.materials.append(mat)

Brushed metal    : Base Color (0.55,0.55,0.58,1) | Metallic 0.95 | Rough 0.35
Painted metal    : Base Color (0.15,0.18,0.22,1) | Metallic 0.0  | Rough 0.45
Plastic          : Base Color (0.08,0.08,0.10,1) | Metallic 0.0  | Rough 0.60
Rubber           : Base Color (0.03,0.03,0.03,1) | Metallic 0.0  | Rough 0.90
Glass-like       : Base Color (0.55,0.85,1.0,0.45) | Metallic 0.0 | Rough 0.05
                   Avoid transmission sockets. Socket names vary by Blender build.
                   Use alpha blending only if needed, otherwise keep it opaque.
Sci-fi emissive  : Metallic 0.8 | Rough 0.2
                   + inputs["Emission Color"].default_value    = (0.1,0.8,1.0,1.0)
                   + inputs["Emission Strength"].default_value = 3.0
Tree bark        : Base Color (0.28,0.16,0.08,1) | Metallic 0.0 | Rough 0.82
Banana stem      : Base Color (0.25,0.42,0.16,1) | Metallic 0.0 | Rough 0.75
Leaf dark green  : Base Color (0.03,0.24,0.06,1) | Metallic 0.0 | Rough 0.68
Leaf light green : Base Color (0.18,0.55,0.12,1) | Metallic 0.0 | Rough 0.62
Ripe banana      : Base Color (0.95,0.74,0.10,1) | Metallic 0.0 | Rough 0.55

TOPOLOGY CONTRACT

Triangle budget: {poly_range} triangles  ({poly_use})

POLYGON RULE — quads required, n-gons forbidden, unnecessary triangles forbidden:
  # !! NEVER call bmesh.ops.triangulate(bm, faces=bm.faces[:]) !!
  # Only remove actual n-gons (5+ vertex faces) — keep quads intact:
  _ngons = [f for f in bm.faces if len(f.verts) > 4]
  if _ngons:
      bmesh.ops.triangulate(bm, faces=_ngons)
  bm.to_mesh(mesh); bm.free(); mesh.update()
  # Merge stray tris back to quads (context-safe — no mode switch needed):
  _bm2 = bmesh.new(); _bm2.from_mesh(mesh)
  bmesh.ops.join_triangles(_bm2, faces=_bm2.faces[:],
                           angle_face_threshold=0.698, angle_shape_threshold=0.698)
  _bm2.to_mesh(mesh); _bm2.free(); mesh.update()
  # Result: almost entirely quads, zero n-gons.

SMOOTH SHADING — apply per mesh after bm.to_mesh() / mesh.update():
  # Smooth everything; shade_smooth_by_angle handles the hard-edge transitions.
  for poly in mesh.polygons:
      poly.use_smooth = True
  mesh.update()
  # Blender 4.1+ removed use_auto_smooth — use shade_smooth_by_angle instead:
  bpy.context.view_layer.objects.active = obj
  obj.select_set(True)
  try:
      bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
  except Exception:
      pass  # older Blender builds: smooth shading already applied above

NORMAL DIRECTION — enforce after every bmesh build:
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
  # Inverted normals → black faces in-engine.

MANIFOLD RULE — validate after booleans:
  bad = [e for e in bm.edges if not e.is_manifold]
  if bad:
      bmesh.ops.delete(bm, geom=bad, context="EDGES")

EDGE CREASE — mark hard corners after recalc_face_normals:
  for edge in bm.edges:
      if edge.calc_face_angle(0.0) > math.radians(30):
          edge.smooth = False

{uv_block}

POLY BUDGET — soft decimate, never assert/crash (add after mesh.update()):
  # Count equivalent triangles: quads = 2 tris, tris = 1 tri.
  # DO NOT raise AssertionError on overage — gracefully dissolve coplanar
  # detail instead so a slightly heavy mesh still exports successfully.
  _tri_count = sum(len(p.vertices) - 2 for p in mesh.polygons)
  if _tri_count > {poly_max}:
      _bm2 = bmesh.new()
      _bm2.from_mesh(mesh)
      try:
          bmesh.ops.dissolve_limit(_bm2, angle_limit=math.radians(3),
                                   verts=_bm2.verts[:], edges=_bm2.edges[:])
      except Exception:
          pass
      _bm2.to_mesh(mesh)
      _bm2.free()
      mesh.update()
      _tri_count = sum(len(p.vertices) - 2 for p in mesh.polygons)
  # quads are preferred — an all-quad mesh uses half the face count of a
  # triangulated mesh for the same silhouette. A small overage is acceptable.

{geometry_rules}

{metaball_block}
{component_block}
STRUCTURAL STABILITY RULES

- Load-bearing base, feet, or frame; lowest verts at Z=0.0.
- Connect all parts with visible joints (bolts, collars, welds, hinges).
- No floating panels, handles, wheels, or decorative parts.
- Tall objects: widen base or add braces. Rings/wheels: use axle/hub/yoke.
- One shared coordinate frame — no subassemblies at far-apart offsets.
- Position details from already-built parts (legs under seat corners, etc.).
- Joints overlap/contact: brackets penetrate surfaces, axles pass through hubs.
- All parts fit in one compact bounding box; no part isolated by more than its size.

{category_rule_modules}
ORIENTATION AND AXIS RULES

Blender convention for ARGUS assets:
- Z is vertical/up.
- X/Y are horizontal ground-plane axes.
- Circular caps made by bmesh.ops.create_cone() are created in the XY plane
  with depth along Z by default.

Before finalizing any circular/mechanical part, decide its real-world axis:
- Tires, road wheels, pulleys, side wheels, gears, handwheels, valve wheels,
  steering wheels, and vertical rings MUST stand upright, with their circular
  face in a vertical plane. Their axle/spindle must run horizontally along X
  or Y, not vertically along Z.
- Cylindrical posts, pipes, poles, tanks, bolts, stems, and vertical valve
  bodies may keep their cylinder axis along Z.
- Horizontal pipes or axles must be rotated 90 degrees from the default
  create_cone orientation.

For a default Z-axis cylinder/cone that must become a wheel/tire standing
upright, rotate all bmesh verts by 90 degrees about X or Y before bm.to_mesh():
  rot_mat = mathutils.Matrix.Rotation(math.radians(90), 4, "X")
  bmesh.ops.rotate(bm, cent=mathutils.Vector((0, 0, 0)),
                   matrix=rot_mat, verts=bm.verts[:])

Sanity check:
- A car tire lying flat on the ground is WRONG.
- A valve handwheel lying like a table top is WRONG.
- A wheel should read as a vertical disk/torus with a visible horizontal axle.

PREFERRED bmesh primitives (context-safe):
  bmesh.ops.create_cube(bm, size=1.0)
  bmesh.ops.create_cone(bm, cap_ends=True, segments=32,
                         depth=1.0, radius1=0.5, radius2=0.0)
  bmesh.ops.create_circle(bm, cap_tris=False, segments=32,
                            radius=0.5)
  bmesh.ops.create_grid(bm, x_segments=4, y_segments=4,
                          size=1.0)
  bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16,
                              radius=0.5)

SEGMENT COUNT GUIDE — controls curve smoothness:
  Tiny bolts, rivets, small details : segments=8  (low-poly, barely visible)
  Standard pipes, poles, handles    : segments=16  (acceptable for medium budget)
  Visible cylinders, wheels, barrels: segments=32  (smooth silhouette)
  Hero props, large prominent curves : segments=48  (cinematic quality)
  DO NOT use segments=8 for any part that is large or prominently visible.
  More segments = rounder curves = better silhouette = quad loops not triangles.

ADVANCED GEOMETRY TECHNIQUES — use these for curved/compound shapes.
STRONGLY PREFERRED over stacking flat boxes/cylinders.

1. SUBSURF CAGE — vehicle bodies, hulls, cushions: low-poly cage (8–30 faces)
   + mod.levels=2 SubdivisionSurface applied. Shape cage verts to define curve.
2. SIMPLE_DEFORM — TAPER/BEND/TWIST/STRETCH: curved hulls, bent pipes, tapered masts.
   Set deform_method, deform_axis, factor/angle, then modifier_apply.
3. BRIDGE LOOPS — lofted hulls/fuselages: build cross-section profiles at different
   Y positions, then bmesh.ops.bridge_loops(bm, edges=bm.edges[:], use_cyclic=False).
   Mirror for symmetry: bmesh.ops.mirror(bm, geom=geom,
   matrix=mathutils.Matrix.Scale(-1,4,(1,0,0)), merge_dist=0.01, axis="X").
4. MIRROR MODIFIER — ALL symmetric objects: mod.use_axis[0]=True,
   mod.use_bisect_axis[0]=True, mod.merge_threshold=0.001. Apply before export.
5. ARRAY MODIFIER — repeated elements (ribs, bolts, steps):
   mod.fit_type="FIXED_COUNT", mod.count=N, mod.use_relative_offset=True.
6. SCREW MODIFIER — lathe/revolution (bottles, barrels, bowls): draw 2D profile
   as edges in XZ plane, mod.axis="Z", mod.steps=36, mod.screw_offset=0.
7. SOLIDIFY — hollow shells (hull panels, body panels): mod.thickness=0.04,
   mod.offset=-1.0 (inward). Model zero-thickness surface first.
8. from_pydata — parametric shapes: build verts/faces lists with math, call
   mesh.from_pydata(verts, [], faces); mesh.update().

MODIFIER APPLY ORDER: MIRROR → ARRAY → SUBSURF → SOLIDIFY → SIMPLE_DEFORM
Always apply before export. SubSurf level 2 adds ~4× faces — stay within budget.

AVOID:
- bpy.ops.mesh.primitive_* (needs edit-mode context)
- Subdivision Surface modifier with levels > 2
- Multiresolution modifier
- Any modifier that multiplies face count beyond budget

BMESH RULES

Standard per-part workflow:
  mesh = bpy.data.meshes.new("part_name")
  obj  = bpy.data.objects.new("part_name", mesh)
  bpy.context.collection.objects.link(obj)
  bm = bmesh.new()
  # --- build geometry ---
  bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
  _ngons = [f for f in bm.faces if len(f.verts) > 4]
  if _ngons: bmesh.ops.triangulate(bm, faces=_ngons)
  bm.to_mesh(mesh); bm.free(); mesh.update()
  # apply smooth-shading block from TOPOLOGY CONTRACT

Preferred ops: extrude_face_region, inset_region, bevel, translate, scale,
  rotate, subdivide_edges, join_triangles, bridge_loops, mirror.
DO NOT USE: extrude_edge_region (removed), create_cylinder (use create_cone).
Conversion: bm.to_mesh(mesh); bm.free(); mesh.update()
  from_pydata: pass int indices — [[v.index for v in f.verts] for f in bm.faces]

INTER-PART PARENTING & NAMING

Naming convention:
  Root empty: "ARGUS_ROOT"                     (exactly this string)
  Objects   : clean semantic names without ARGUS_ prefix
              e.g. car_body_shell, car_tire_front_left, chair_lumbar_pillow
  Meshes    : same clean semantic name as object
  Materials : clean names without ARGUS_ prefix
              e.g. rubber_black, glass_dark, fabric_canvas, brushed_metal

Parenting (context-safe — do NOT use bpy.ops.object.parent_set()):
  part_obj.parent = root_obj
  part_obj.matrix_parent_inverse = root_obj.matrix_world.inverted()

Root empty creation:
  root_obj = bpy.data.objects.new("ARGUS_ROOT", None)
  bpy.context.collection.objects.link(root_obj)
  root_obj.empty_display_type = "PLAIN_AXES"
  root_obj.location = mathutils.Vector((0.0, 0.0, 0.0))

Scene origin:
  - ROOT empty at (0, 0, 0)
  - Lowest vertex of any mesh at Z = 0.0
  - All parts positioned relative to ROOT

EXPORT CODE TEMPLATE

The script MUST end with this exact block verbatim.

  # ── ARGUS export ─────────────────────────────────
  import os
  ARGUS_EXPORT_OBJ  = bpy.data.objects["ARGUS_ROOT"]
  ARGUS_EXPORT_PATH = {export_path_expr}

  os.makedirs(os.path.dirname(ARGUS_EXPORT_PATH) or ".", exist_ok=True)

  bpy.ops.object.select_all(action="DESELECT")
  for _o in bpy.data.objects:
      if _o.type == "MESH":
          _o.select_set(True)
  bpy.ops.object.transform_apply(
      location=True, rotation=True, scale=True
  )

{export_op_block}

  print(f"ARGUS_OK:{{ARGUS_EXPORT_OBJ.name}}")
  print(f"ARGUS_PATH:{{ARGUS_EXPORT_PATH}}")
  # ─────────────────────────────────────────────────

DO NOT use bpy.ops.wm.collada_export() — unsupported.

OUTPUT CONTRACT (pipeline-critical)

  ARGUS_EXPORT_OBJ = bpy.data.objects["ARGUS_ROOT"]

This variable name is parsed by the pipeline.
The sentinel  print(f"ARGUS_OK:{{ARGUS_EXPORT_OBJ.name}}")
must appear as the last print() in the script.

{repair_taxonomy_block}QUALITY TARGETS

- ROOT empty at (0,0,0); all geometry grounded (lowest vertex Z=0); stable footprint
- Major parts connected into one coherent assembly; no floating/disconnected geometry
- No coincident/overlapping faces; no zero-length edges; transforms applied (template handles)
- Single-run execution; silhouette readable at 50 m; reproducible given same seed
- Poly budget: {poly_range} triangles; all normals outward-facing
- All faces tris or quads (zero N-gons); one UV layer ("ARGUS_UV") + 1 material per mesh
- Export-ready GLB/FBX, no manual cleanup needed

ALLOWED MODIFIERS (use freely — apply before export)
  SubdivisionSurface, SimpleDeform, Mirror, Array, Screw, Solidify,
  Bevel (modifier), Displace, Lattice, Skin, Curve, Shrinkwrap

AVOID

- random.*() for structural dimensions; organic/sculpt workflows
- animation / drivers / shape keys; N-gons (5+ vert faces); interior duplicate faces
- bpy.ops.* without correct mode/context; hardcoded absolute paths (output_dir injected)
- imports beyond the six required modules; Collada export
- bmesh.ops.create_cylinder (does not exist — use create_cone with equal radii)
- bmesh.ops.create_torus   (does not exist — use two create_cone + bridge_loops, or a tube + deform)
- bmesh.ops.create_capsule (does not exist — build from two hemispheres + cylinder)
- bmesh.ops.create_plane   (does not exist — use create_grid with x_segments=1, y_segments=1)
- MultiresolutionModifier (too memory-heavy for batch/headless)

OUTPUT FORMAT

The FIRST six lines MUST be exactly:

import bpy
import bmesh
import math
import mathutils
import random
import os

Line 7 MUST be exactly:
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 4.1+ , got {{bpy.app.version}}"

Then: scene init block (list() copy pattern).
Then: {seed_line}

Then: geometry construction for each part.
Then: inter-part parenting.
Then: export block verbatim.
Last print: ARGUS_OK sentinel.

Output ONLY Python code. Nothing else.
"""
