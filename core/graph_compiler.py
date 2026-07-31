
from __future__ import annotations

import math
import os
import random
import re

from core.components import COMPONENT_LIBRARY_SOURCE
from core.spec import expand_parts

COMPILER_VERSION = "1.0"

_TEX_MAP_KEYS = ("diff", "rough", "nor_gl")


def _fmt(value) -> str:
    """Compact, deterministic literal for floats/ints in emitted code."""
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e9:
            return f"{int(value)}.0"
        return repr(round(value, 5))
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in value) + ("," if len(value) == 1 else "") + ")"
    return repr(value)


def _ident(name: str, used: set[str]) -> str:
    """Sanitised, unique Python identifier for a material variable."""
    base = "".join(c if c.isalnum() else "_" for c in name.upper()).strip("_") or "MAT"
    if base[0].isdigit():
        base = "M_" + base
    ident = "MAT_" + base
    n = 2
    while ident in used:
        ident = f"MAT_{base}_{n}"
        n += 1
    used.add(ident)
    return ident


_MATERIAL_BUILDER = r'''
def _argus_pbr(name, color, metallic, roughness, tex_maps=None):
    """Principled-BSDF material; optional Poly Haven PBR maps (guarded —
    a missing texture file can never crash the build)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if not tex_maps:
        return mat
    x = -560
    if tex_maps.get("diff"):
        try:
            img = bpy.data.images.load(tex_maps["diff"])
            img.colorspace_settings.name = "sRGB"
            node = nodes.new("ShaderNodeTexImage")
            node.image = img
            node.location = (x, 300)
            links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
        except Exception as exc:
            print("[ARGUS tex skip diff] " + str(exc))
    if tex_maps.get("rough"):
        try:
            img = bpy.data.images.load(tex_maps["rough"])
            img.colorspace_settings.name = "Non-Color"
            node = nodes.new("ShaderNodeTexImage")
            node.image = img
            node.location = (x, 0)
            links.new(node.outputs["Color"], bsdf.inputs["Roughness"])
        except Exception as exc:
            print("[ARGUS tex skip rough] " + str(exc))
    if tex_maps.get("nor_gl"):
        try:
            img = bpy.data.images.load(tex_maps["nor_gl"])
            img.colorspace_settings.name = "Non-Color"
            node = nodes.new("ShaderNodeTexImage")
            node.image = img
            node.location = (x, -300)
            nrm = nodes.new("ShaderNodeNormalMap")
            nrm.location = (x + 280, -300)
            links.new(node.outputs["Color"], nrm.inputs["Color"])
            links.new(nrm.outputs["Normal"], bsdf.inputs["Normal"])
        except Exception as exc:
            print("[ARGUS tex skip nor] " + str(exc))
    return mat


def _argus_smart_mat(name, color, metallic, roughness, style="generic", tex_maps=None):
    """Procedural 'smart material': noise colour/roughness variation, cavity
    grime (AO) and pointiness edge wear — Substance-style richness with zero
    texture files. Tagged for the export wrapper's Cycles bake. Poly Haven
    maps take precedence; any node failure falls back to the flat material."""
    if tex_maps:
        return _argus_pbr(name, color, metallic, roughness, tex_maps)
    try:
        if style == "ceramic":
            roughness = min(roughness, 0.2)
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        mat["argus_bake"] = True
        mat["argus_metallic"] = float(metallic)
        mat["argus_color"] = [float(color[0]), float(color[1]), float(color[2])]
        mat["argus_rough"] = float(roughness)
        nt = mat.node_tree
        nd = nt.nodes
        ln = nt.links
        bsdf = nd.get("Principled BSDF") or nd.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Metallic"].default_value = metallic

        if style == "ceramic":
            # Glazed-ceramic clearcoat: thin glassy layer over the body colour
            # gives the sharp specular highlight typical of glazed pottery.
            # Stashed as custom props too so the post-bake material rebuild
            # (core/blender.py) can carry the coat over to the final export.
            mat["argus_coat"] = 0.8
            mat["argus_coat_rough"] = 0.08
            try:
                bsdf.inputs["Coat Weight"].default_value = 0.8
                bsdf.inputs["Coat Roughness"].default_value = 0.08
            except KeyError:
                pass

        base = (color[0], color[1], color[2], 1.0)
        lite = tuple(min(1.0, c * 1.12) for c in color) + (1.0,)
        dark = tuple(c * 0.82 for c in color) + (1.0,)

        # Colour breakup: large soft noise between dark/light variants.
        var = nd.new("ShaderNodeTexNoise")
        var.inputs["Scale"].default_value = 7.0 if style != "wood" else 3.0
        var.inputs["Detail"].default_value = 4.0
        mix_var = nd.new("ShaderNodeMixRGB")
        mix_var.inputs["Color1"].default_value = dark
        mix_var.inputs["Color2"].default_value = lite
        ln.new(var.outputs["Fac"], mix_var.inputs["Fac"])
        col_out = mix_var.outputs["Color"]

        if style == "wood":
            grain = nd.new("ShaderNodeTexWave")
            grain.inputs["Scale"].default_value = 2.2
            grain.inputs["Distortion"].default_value = 6.0
            grain.inputs["Detail"].default_value = 2.0
            ramp = nd.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.35
            ramp.color_ramp.elements[0].color = (0.55, 0.5, 0.45, 1.0)
            ramp.color_ramp.elements[1].position = 0.75
            ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
            ln.new(grain.outputs["Fac"], ramp.inputs["Fac"])
            gmix = nd.new("ShaderNodeMixRGB")
            gmix.blend_type = "MULTIPLY"
            gmix.inputs["Fac"].default_value = 0.55
            ln.new(col_out, gmix.inputs["Color1"])
            ln.new(ramp.outputs["Color"], gmix.inputs["Color2"])
            col_out = gmix.outputs["Color"]

        # Cavity grime: AO inverted → darken crevices.
        ao = nd.new("ShaderNodeAmbientOcclusion")
        ao.inputs["Distance"].default_value = 0.12
        ao_ramp = nd.new("ShaderNodeValToRGB")
        ao_ramp.color_ramp.elements[0].position = 0.35
        ao_ramp.color_ramp.elements[1].position = 0.9
        ln.new(ao.outputs["Color"], ao_ramp.inputs["Fac"])
        grime = nd.new("ShaderNodeMixRGB")
        grime.blend_type = "MULTIPLY"
        grime.inputs["Fac"].default_value = 0.45
        ln.new(col_out, grime.inputs["Color1"])
        ln.new(ao_ramp.outputs["Color"], grime.inputs["Color2"])
        col_out = grime.outputs["Color"]

        rough_out = None
        if style in ("paint", "metal", "wood"):
            # Edge wear: pointiness picks out convex edges (Cycles/bake only).
            geo = nd.new("ShaderNodeNewGeometry")
            wear_ramp = nd.new("ShaderNodeValToRGB")
            wear_ramp.color_ramp.elements[0].position = 0.55
            wear_ramp.color_ramp.elements[1].position = 0.68
            ln.new(geo.outputs["Pointiness"], wear_ramp.inputs["Fac"])
            wear_noise = nd.new("ShaderNodeTexNoise")
            wear_noise.inputs["Scale"].default_value = 30.0
            wear_mask = nd.new("ShaderNodeMath")
            wear_mask.operation = "MULTIPLY"
            ln.new(wear_ramp.outputs["Color"], wear_mask.inputs[0])
            ln.new(wear_noise.outputs["Fac"], wear_mask.inputs[1])
            worn = {"paint": (0.35, 0.34, 0.33, 1.0),
                    "metal": (0.75, 0.74, 0.72, 1.0),
                    "wood": (0.72, 0.62, 0.48, 1.0)}[style]
            wmix = nd.new("ShaderNodeMixRGB")
            wmix.inputs["Color2"].default_value = worn
            ln.new(col_out, wmix.inputs["Color1"])
            ln.new(wear_mask.outputs["Value"], wmix.inputs["Fac"])
            col_out = wmix.outputs["Color"]
            wr = nd.new("ShaderNodeMath")
            wr.operation = "MULTIPLY_ADD"
            ln.new(wear_mask.outputs["Value"], wr.inputs[0])
            wr.inputs[1].default_value = -0.25 if style != "wood" else -0.1
            wr.inputs[2].default_value = roughness
            rough_out = wr.outputs["Value"]

        if rough_out is None:
            rn = nd.new("ShaderNodeTexNoise")
            rn.inputs["Scale"].default_value = 14.0
            rr = nd.new("ShaderNodeMath")
            rr.operation = "MULTIPLY_ADD"
            ln.new(rn.outputs["Fac"], rr.inputs[0])
            rr.inputs[1].default_value = 0.22
            rr.inputs[2].default_value = max(0.0, roughness - 0.11)
            rough_out = rr.outputs["Value"]

        if style == "fabric":
            weave = nd.new("ShaderNodeTexNoise")
            weave.inputs["Scale"].default_value = 180.0
            bump = nd.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.25
            ln.new(weave.outputs["Fac"], bump.inputs["Height"])
            ln.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        if style == "glass":
            try:
                bsdf.inputs["Transmission Weight"].default_value = 0.7
            except KeyError:
                pass

        ln.new(col_out, bsdf.inputs["Base Color"])
        ln.new(rough_out, bsdf.inputs["Roughness"])
        return mat
    except Exception as exc:
        print("[ARGUS smart_mat fallback] " + str(exc))
        return _argus_pbr(name, color, metallic, roughness, None)
'''


_STYLE_WORDS = {
    "ceramic": ("ceramic", "porcelain", "glaze", "glazed", "stoneware", "china",
                "earthenware", "pottery"),
    "wood": ("wood", "oak", "timber", "plank", "bark", "birch", "pine", "walnut"),
    "fabric": ("fabric", "cloth", "cushion", "leather", "canvas", "upholstery", "linen"),
    "rubber": ("rubber", "tire", "tyre", "hose"),
    "glass": ("glass", "lens", "bulb", "window"),
    "plastic": ("plastic", "abs", "vinyl", "polymer"),
    "paint": ("paint", "enamel", "coated", "body", "shell"),
    "metal": ("metal", "steel", "iron", "chrome", "brass", "copper", "aluminum",
              "aluminium", "silver", "gold", "tin", "zinc"),
}


def _mat_style(name: str, purpose: str, metallic: float) -> str:
    """Pick a smart-material template from the material's name/texture purpose,
    falling back to metallic value (painted metal vs bare metal vs generic)."""
    hay = f"{name} {purpose}".lower()
    for style, words in _STYLE_WORDS.items():
        if any(w in hay for w in words):
            return style
    if metallic >= 0.6:
        return "metal"
    if metallic >= 0.15:
        return "paint"
    return "generic"


def _emit_materials(spec: dict, texture_paths: dict | None) -> tuple[list[str], dict]:
    """Emit material constants. Returns (lines, material_name → identifier)."""
    texture_paths = texture_paths or {}
    lines: list[str] = []
    idents: dict[str, str] = {}
    used: set[str] = set()
    for name, mat in spec["materials"].items():
        ident = _ident(name, used)
        idents[name] = ident
        maps_arg = "None"
        purpose = mat.get("texture", "")
        maps = texture_paths.get(purpose) if purpose else None
        if maps:
            usable = {k: str(maps[k]) for k in _TEX_MAP_KEYS if maps.get(k)}
            if usable:
                pairs = ", ".join(f"{k!r}: {v!r}" for k, v in usable.items())
                maps_arg = "{" + pairs + "}"
        c = mat["color"]
        style = _mat_style(name, purpose, float(mat["metallic"]))
        lines.append(
            f"{ident} = _argus_smart_mat({('ARGUS_' + name)!r}, "
            f"({_fmt(c[0])}, {_fmt(c[1])}, {_fmt(c[2])}), "
            f"{_fmt(mat['metallic'])}, {_fmt(mat['roughness'])}, "
            f"style={style!r}, tex_maps={maps_arg})"
        )
    return lines, idents


_VESSEL_TOKENS = frozenset((
    "mug", "cup", "teacup", "bowl", "vase", "bucket", "pot", "teapot", "jar",
    "jug", "pitcher", "glass", "tankard", "stein", "basin", "planter",
    "beaker", "goblet", "chalice", "crucible", "urn", "vessel", "bin",
))
# Parts that carry a vessel word but are solid pieces of one, not the cavity.
_SOLID_TOKENS = frozenset((
    "rim", "base", "foot", "lid", "cap", "handle", "knob", "saucer", "stand",
    "ring", "stem", "neck", "spout", "leg", "post", "coaster", "tray",
))


def _hollow_vessel_profile(profile: list, part_id: str) -> list:
    """argus_lathe fan-caps both end rings, so a vessel body spun from an
    outer-wall-only profile comes out sealed shut — a mug reads as a closed
    canister. When a lathe part is recognisably an open container, extend the
    profile back down the inside: rim annulus → inner wall → interior floor.
    The added segment keeps the mesh a closed manifold solid."""
    tokens = set(re.split(r"[^a-z]+", part_id.lower())) - {""}
    if not (tokens & _VESSEL_TOKENS) or (tokens & _SOLID_TOKENS):
        return profile
    prof = [(float(r), float(z)) for r, z in profile]
    if len(prof) < 2:
        return profile
    r_top, z_top = prof[-1]
    z_bot = min(z for _, z in prof)
    height = z_top - z_bot
    # Must end open at the top and be tall enough to plausibly hold anything.
    if r_top < 0.008 or height < max(0.015, 0.4 * r_top):
        return profile
    # Profile already turns back downward → planner modelled the interior.
    if any(b[1] < a[1] - 1e-9 for a, b in zip(prof, prof[1:])):
        return profile
    wall = min(max(0.06 * r_top, 0.0015), 0.01)
    floor_z = z_bot + max(1.5 * wall, 0.003)
    inner_r = max(r_top - wall, 0.5 * r_top)
    return prof + [
        (inner_r, z_top),
        (max(inner_r - 0.1 * wall, 0.25 * r_top), floor_z),
        (0.0, floor_z),
    ]


def _auto_snap_islands(parts: list[dict]) -> list[dict]:
    """Connectivity repair, shape-agnostic: the planner's geometry is kept
    exactly as designed, but disconnected islands are translated until the
    WHOLE assembly is one connected body. Each pass merges the single closest
    pair of clusters (moving the non-grounded / smaller side), so N islands
    collapse to one in at most N-1 passes — a 13-part scatter fully connects
    instead of stalling after a few. The grounded cluster never moves, so the
    asset stays on the floor. Novelty lives in the shapes; physics only
    insists that everything touches."""
    from core.spec_scorer import _overlaps, _part_aabb

    n = len(parts)
    if n < 2:
        return parts

    def _vol(boxes, idxs):
        return sum(
            (boxes[i][1][0] - boxes[i][0][0])
            * (boxes[i][1][1] - boxes[i][0][1])
            * (boxes[i][1][2] - boxes[i][0][2]) for i in idxs)

    def _clusters(boxes):
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if _overlaps(boxes[i], boxes[j], tol=0.015):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        cl: dict[int, list[int]] = {}
        for i in range(n):
            cl.setdefault(find(i), []).append(i)
        return cl

    for _ in range(n + 2):  # guaranteed to reach one cluster well within this
        boxes = [_part_aabb(p) for p in parts]
        cl = _clusters(boxes)
        if len(cl) <= 1:
            break

        # Grounded (lowest, tie-break biggest) cluster is the fixed anchor.
        anchor = min(cl, key=lambda k: (min(boxes[i][0][2] for i in cl[k]),
                                        -_vol(boxes, cl[k])))
        keys = list(cl.keys())
        best = None  # (d2, move_key, push_delta)
        for ai in range(len(keys)):
            for bi in range(ai + 1, len(keys)):
                ka, kb = keys[ai], keys[bi]
                for i in cl[ka]:
                    a = boxes[i]
                    for j in cl[kb]:
                        b = boxes[j]
                        delta = [0.0, 0.0, 0.0]
                        d2 = 0.0
                        for ax in range(3):
                            if a[0][ax] > b[1][ax]:
                                g = b[1][ax] - a[0][ax]
                            elif a[1][ax] < b[0][ax]:
                                g = b[0][ax] - a[1][ax]
                            else:
                                g = 0.0
                            delta[ax] = g
                            d2 += g * g
                        if best is not None and d2 >= best[0]:
                            continue
                        # delta moves cluster ka to touch kb. Pick the side to
                        # move: never the anchor; otherwise the smaller volume.
                        if ka == anchor:
                            mv_key, mv = kb, [-x for x in delta]
                        elif kb == anchor:
                            mv_key, mv = ka, delta
                        elif _vol(boxes, cl[ka]) <= _vol(boxes, cl[kb]):
                            mv_key, mv = ka, delta
                        else:
                            mv_key, mv = kb, [-x for x in delta]
                        best = (d2, mv_key, mv)
        if best is None:
            break
        _, mv_key, mv = best
        # overshoot contact by 6 mm so surfaces genuinely intersect (well
        # within the validator's >=3 cm connection threshold either way).
        push = [d + (0.006 if d > 1e-9 else (-0.006 if d < -1e-9 else 0.0))
                for d in mv]
        if all(abs(x) < 1e-9 for x in push):
            break
        for i in cl[mv_key]:
            pos = list(parts[i].get("pos") or (0.0, 0.0, 0.0))
            parts[i]["pos"] = [pos[0] + push[0], pos[1] + push[1], pos[2] + push[2]]
    return parts


def _sanitize_lathe_profile(profile: list) -> list:
    """Drop planner-emitted degeneracies that spin into non-manifold shells:
    exact consecutive duplicates, and runs of 3+ points at the same height —
    the middle points fold the surface back on itself in-plane, and the
    coincident rings weld into edges shared by more than two faces."""
    prof = [(float(r), float(z)) for r, z in profile]
    dedup: list = []
    for pt in prof:
        if dedup and abs(pt[0] - dedup[-1][0]) < 1e-9 and abs(pt[1] - dedup[-1][1]) < 1e-9:
            continue
        dedup.append(pt)
    cleaned: list = []
    i = 0
    while i < len(dedup):
        j = i
        while j + 1 < len(dedup) and abs(dedup[j + 1][1] - dedup[i][1]) < 1e-9:
            j += 1
        cleaned.append(dedup[i])
        if j > i and (abs(dedup[j][0] - dedup[i][0]) > 1e-9):
            cleaned.append(dedup[j])
        i = j + 1
    return cleaned if len(cleaned) >= 2 else profile


_RIM_TOKENS = frozenset(("rim", "lip", "collar"))


def _annular_rim_profile(profile: list, part_id: str) -> list:
    """A rim/lip/collar spun from an open profile gets fan-capped into a
    solid disc that seals the vessel mouth it sits on. Close the profile
    into a loop instead — outer face as the planner gave it, inner face
    offset inward — so argus_lathe spins a true annulus with a hole."""
    tokens = set(re.split(r"[^a-z]+", part_id.lower())) - {""}
    if not (tokens & _RIM_TOKENS):
        return profile
    prof = [(float(r), float(z)) for r, z in profile]
    if len(prof) < 2:
        return profile
    r_min = min(r for r, _ in prof)
    # Touches the axis (solid by design) or already a closed loop → leave it.
    if r_min < 1e-6:
        return profile
    if abs(prof[0][0] - prof[-1][0]) < 1e-9 and abs(prof[0][1] - prof[-1][1]) < 1e-9:
        return profile
    wall = min(max(0.15 * r_min, 0.0015), 0.008)
    inner = [(max(r - wall, 0.3 * r_min), z) for r, z in reversed(prof)]
    return prof + inner + [prof[0]]


def _emit_part(part: dict, mat_idents: dict) -> list[str]:
    """Emit the builder call for one expanded part."""
    prim = part["primitive"]
    p = part["params"]
    mat = mat_idents[part["material"]]
    rot = part.get("rot")
    loc = (0.0, 0.0, 0.0) if rot else tuple(part["pos"])
    name = part["id"]

    if prim == "box":
        extra = f", bevel={_fmt(p['bevel'])}" if "bevel" in p else ""
        call = (f"argus_box({name!r}, {_fmt(loc)}, {_fmt(tuple(p['size']))}, "
                f"{mat}, ROOT{extra})")
        if p.get("smooth"):
            call = (f"argus_smooth({call}, levels={p['smooth']}, "
                    f"crease={_fmt(float(p.get('crease', 0.0)))})")
    elif prim == "cylinder":
        extra = f", r_top={_fmt(p['r_top'])}" if "r_top" in p else ""
        call = (f"argus_cylinder({name!r}, {_fmt(loc)}, {_fmt(p['radius'])}, "
                f"{_fmt(p['depth'])}, {mat}, ROOT, axis={p['axis']!r}, "
                f"segments={p['segments']}{extra})")
    elif prim == "sphere":
        call = (f"argus_sphere({name!r}, {_fmt(loc)}, {_fmt(tuple(p['size']))}, "
                f"{mat}, ROOT)")
    elif prim == "wheel":
        call = (f"argus_wheel({name!r}, {_fmt(loc)}, {_fmt(p['radius'])}, "
                f"{_fmt(p['width'])}, {mat}, ROOT, axis={p['axis']!r}, "
                f"segments={p['segments']})")
    elif prim == "bolt":
        call = (f"argus_bolt({name!r}, {_fmt(loc)}, {_fmt(p['radius'])}, "
                f"{_fmt(p['height'])}, {mat}, ROOT, axis={p['axis']!r})")
    elif prim == "panel":
        call = (f"argus_panel({name!r}, {_fmt(loc)}, {_fmt(tuple(p['size']))}, "
                f"{mat}, ROOT, inset={_fmt(p['inset'])})")
    elif prim == "lathe":
        pts = _sanitize_lathe_profile(p["profile"])
        pts = _annular_rim_profile(_hollow_vessel_profile(pts, name), name)
        profile = "[" + ", ".join(_fmt(tuple(pt)) for pt in pts) + "]"
        call = (f"argus_lathe({name!r}, {_fmt(loc)}, {profile}, {mat}, ROOT, "
                f"segments={p['segments']})")
    elif prim == "tube":
        path = "[" + ", ".join(_fmt(tuple(pt)) for pt in p["path"]) + "]"
        call = (f"argus_tube({name!r}, {_fmt(loc)}, {path}, {_fmt(p['radius'])}, "
                f"{mat}, ROOT, segments={p['segments']})")
    elif prim == "ring":
        call = (f"argus_ring({name!r}, {_fmt(loc)}, {_fmt(p['radius'])}, "
                f"{_fmt(p['thickness'])}, {mat}, ROOT, axis={p['axis']!r}, "
                f"segments={p['segments']})")
    elif prim == "leaf_card":
        call = (f"argus_leaf_card({name!r}, {_fmt(tuple(part['pos']))}, "
                f"{_fmt(p['length'])}, {_fmt(p['width'])}, {mat}, ROOT, "
                f"curve={_fmt(p['curve'])}, yaw={_fmt(p['yaw'])}, "
                f"pitch={_fmt(p['pitch'])})")
        return [f"{call}"]
    else:  # pragma: no cover — validate_spec guarantees membership
        raise ValueError(f"unknown primitive '{prim}'")

    if rot:
        rad = tuple(round(math.radians(a), 6) for a in rot)
        return [
            f"_obj = {call}",
            f"_obj.rotation_euler = {_fmt(rad)}",
            f"_obj.location = {_fmt(tuple(part['pos']))}",
        ]
    return [call]


def _auto_greebles(parts: list[dict]) -> list[dict]:
    """Add connective detail where vertical cylinders meet their parent: a
    snug collar ring at the contact line, like real posts/legs/poles have.
    Conservative by design — wrong greebles look worse than none.
    ARGUS_GREEBLES=0 disables."""
    if os.environ.get("ARGUS_GREEBLES", "1") == "0":
        return parts
    by_id = {p["id"]: p for p in parts}
    children: dict[str, list[dict]] = {}
    for q in parts:
        children.setdefault(q.get("attach_to") or "", []).append(q)
    out = list(parts)
    added = 0
    for p in parts:
        if added >= 6:
            break
        if p["primitive"] != "cylinder" or p.get("rot"):
            continue
        prm = p["params"]
        if prm.get("axis", "Z") != "Z" or "r_top" in prm:
            continue
        r = prm.get("radius", 0.0)
        if not 0.015 <= r <= 0.3:
            continue
        # joint partner: the parent, or a slab child attached to this cylinder
        # (a tabletop on a leg, a seat frame on chair legs, ...)
        partner = by_id.get(p.get("attach_to") or "")
        if partner is None or partner["primitive"] not in ("box", "panel", "lathe"):
            partner = next((c for c in children.get(p["id"], [])
                            if c["primitive"] in ("box", "panel")), None)
        if partner is None:
            continue
        half = prm["depth"] / 2.0
        bottom, top = p["pos"][2] - half, p["pos"][2] + half
        pz = partner["pos"][2]
        p_half = partner.get("params", {}).get(
            "size", [0, 0, prm["depth"]])[2] / 2.0 + 0.05
        # collar at whichever end sits at/inside the partner
        if abs(bottom - pz) < p_half:
            collar_z = bottom + max(0.012, r * 0.25)
        elif abs(top - pz) < p_half:
            collar_z = top - max(0.012, r * 0.25)
        else:
            continue
        out.append({
            "id": f"auto_collar_{p['id']}",
            "primitive": "ring",
            "pos": [p["pos"][0], p["pos"][1], collar_z],
            "params": {"radius": r * 1.02,
                       "thickness": max(0.006, r * 0.16),
                       "axis": "Z",
                       "segments": prm.get("segments", 32)},
            "material": p["material"],
            "attach_to": p["id"],
        })
        added += 1
    return out


def _imperfection_jitter(asset_name: str, parts: list[dict]) -> list[dict]:
    """Deterministic micro-jitter (seeded by asset+part id) so nothing is
    mathematically perfect: ±1.2° yaw, ±0.5° tilt, ±1.5% size. Skips wheels,
    axles and auto-greebles. ARGUS_JITTER=0 disables."""
    if os.environ.get("ARGUS_JITTER", "1") == "0":
        return parts
    for p in parts:
        pid = p["id"]
        if (p["primitive"] == "wheel" or pid.startswith("auto_")
                or "axle" in pid.lower()):
            continue
        rng = random.Random(f"{asset_name}:{pid}")
        rot = list(p.get("rot") or [0.0, 0.0, 0.0])
        rot[0] += rng.uniform(-0.5, 0.5)
        rot[1] += rng.uniform(-0.5, 0.5)
        rot[2] += rng.uniform(-1.2, 1.2)
        p["rot"] = rot
        size = p.get("params", {}).get("size")
        if size:
            f = 1.0 + rng.uniform(-0.015, 0.015)
            p["params"]["size"] = [s * f for s in size]
    return parts


def compile_spec(spec: dict, texture_paths: dict | None = None) -> str:
    """Compile a validated build spec into a standalone Blender script.

    texture_paths: part_data["poly_haven_texture_paths"] — maps a material's
    "texture" purpose key to {diff/rough/nor_gl: absolute file path}.
    """
    parts = expand_parts(spec)
    parts = _auto_greebles(parts)
    parts = _imperfection_jitter(spec["name"], parts)
    # Snap LAST so greebles and jitter can't re-separate parts after the
    # connectivity guarantee — the final positions are what gets built.
    parts = _auto_snap_islands(parts)
    mat_lines, mat_idents = _emit_materials(spec, texture_paths)

    root_name = "".join(
        w.capitalize() for w in spec["name"].replace("-", "_").split("_") if w
    ) or "Asset"

    header = [
        f"# ARGUS compiled build — {spec['name']}",
        f"# scene-graph compiler v{COMPILER_VERSION}; "
        f"{len(spec['parts'])} spec parts → {len(parts)} objects",
        "import bpy",
        "import bmesh",
        "import math",
        "import mathutils",
        "",
        'assert bpy.app.version >= (4, 1, 0), "ARGUS requires Blender 4.1+"',
        "",
        "# Clean scene (guarded; the export wrapper clears again)",
        "for _o in list(bpy.data.objects):",
        "    try:",
        "        bpy.data.objects.remove(_o, do_unlink=True)",
        "    except Exception:",
        "        pass",
    ]

    body = [
        COMPONENT_LIBRARY_SOURCE.rstrip(),
        "",
        _MATERIAL_BUILDER.strip(),
        "",
        "# ── Materials ──────────────────────────────────────────────────────",
        *mat_lines,
        "",
        "# ── Assembly root ──────────────────────────────────────────────────",
        f"ROOT = bpy.data.objects.new({(root_name + '_Root')!r}, None)",
        "bpy.context.collection.objects.link(ROOT)",
        "",
        "# ── Parts ──────────────────────────────────────────────────────────",
    ]
    for part in parts:
        body.extend(_emit_part(part, mat_idents))

    # SDF-style smooth union: parts flagged "fuse": true melt into their
    # attach_to parent (voxel remesh + relax) — clay-like blended joints.
    # Chains (tab→curve→body) collapse into ONE group, since the first join
    # consumes the intermediate object names.
    part_ids = {p["id"] for p in parts}
    fuse_parent: dict[str, str] = {pid: pid for pid in part_ids}

    def _fuse_root(pid: str) -> str:
        while fuse_parent[pid] != pid:
            fuse_parent[pid] = fuse_parent[fuse_parent[pid]]
            pid = fuse_parent[pid]
        return pid

    fused_any = False
    for p in parts:
        target = p.get("attach_to") or ""
        if p.get("fuse") and target in part_ids and target != p["id"]:
            ra, rb = _fuse_root(p["id"]), _fuse_root(target)
            if ra != rb:
                fuse_parent[ra] = rb
                fused_any = True
    if fused_any:
        groups: dict[str, list[str]] = {}
        for pid in part_ids:
            root = _fuse_root(pid)
            if root != pid:
                groups.setdefault(root, []).append(pid)
        body.append("")
        body.append("# ── Smooth-union fused joints ──────────────────────────────────────")
        for root, members in groups.items():
            body.append(f"argus_fuse({[root] + sorted(members)!r})")

    body.append("")
    body.append(f'print("[ARGUS COMPILED] {spec["name"]}: ' + str(len(parts)) + ' parts built")')

    return "\n".join(header + body) + "\n"
