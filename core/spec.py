
from __future__ import annotations

import math
from dataclasses import dataclass, field


MIN_PARTS = 3
MAX_EXPANDED_PARTS = 80
MAX_ARRAY_COUNT = 24
MAX_SEGMENTS = 64
MIN_SEGMENTS = 3
CONTACT_TOL = 1.5e-3
SNAP_OVERLAP = 0.02
MAX_DIM = 50.0

VALID_PRIMITIVES = {
    "box", "cylinder", "sphere", "wheel", "bolt", "panel",
    "leaf_card", "lathe", "tube", "ring",
}
VALID_AXES = {"X", "Y", "Z"}
VALID_MIRRORS = {"X", "Y"}

_SNAP_EXEMPT = {"leaf_card"}


@dataclass
class SpecReport:
    ok: bool
    spec: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _num(value, default=None, lo=None, hi=None):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def _vec3(value, default=(0.0, 0.0, 0.0)):
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return list(default)
    out = []
    for i in range(3):
        v = _num(value[i])
        if v is None:
            return list(default)
        out.append(v)
    return out


def _dim(value, default=0.1):
    """A positive geometric dimension, clamped to sane bounds."""
    v = _num(value, default=default)
    v = abs(v) if v is not None else default
    if v < 1e-4:
        v = default
    return min(v, MAX_DIM)


def _color3(value):
    rgb = _vec3(value, default=(0.5, 0.5, 0.5))
    if any(c > 1.001 for c in rgb):
        rgb = [c / 255.0 for c in rgb]
    return [min(1.0, max(0.0, c)) for c in rgb]


def _segments(value, default=32):
    v = _num(value, default=default)
    return int(min(MAX_SEGMENTS, max(MIN_SEGMENTS, round(v if v is not None else default))))


def _clean_box(node):
    size = node.get("size") or node.get("dims")
    if size is None and node.get("radius") is not None:
        r = _dim(node.get("radius"))
        size = [r * 2, r * 2, r * 2]
    sx, sy, sz = [_dim(v) for v in _vec3(size, default=(0.2, 0.2, 0.2))]
    params = {"size": [sx, sy, sz]}
    smooth = node.get("smooth")
    if smooth:
        lv = 2 if smooth is True else _num(smooth, default=2)
        params["smooth"] = int(min(3, max(1, round(lv if lv is not None else 2))))
        params["crease"] = _num(node.get("crease"), default=0.0, lo=0.0, hi=1.0)
    bevel = _num(node.get("bevel"), default=None, lo=0.0)
    if bevel is None and smooth:
        bevel = 0.0  # raw cage — the subdivision does the rounding
    if bevel is not None:
        params["bevel"] = min(bevel, min(sx, sy, sz) / 2.0)
    return params, None


def _clean_cylinder(node):
    radius = _dim(node.get("radius"), default=0.1)
    depth = _dim(node.get("depth") or node.get("height") or node.get("length"), default=0.2)
    params = {"radius": radius, "depth": depth,
              "axis": _axis(node), "segments": _segments(node.get("segments"))}
    r_top = _num(node.get("r_top", node.get("radius_top")))
    if r_top is not None:
        params["r_top"] = max(0.0, min(abs(r_top), MAX_DIM))
    return params, None


def _clean_sphere(node):
    size = node.get("size") or node.get("dims")
    if size is None:
        r = _dim(node.get("radius"), default=0.1)
        size = [r * 2, r * 2, r * 2]
    sx, sy, sz = [_dim(v) for v in _vec3(size, default=(0.2, 0.2, 0.2))]
    return {"size": [sx, sy, sz]}, None


def _clean_wheel(node):
    radius = _dim(node.get("radius"), default=0.3)
    width = _dim(node.get("width") or node.get("depth"), default=0.15)
    axis = _axis(node, default="X")
    if axis == "Z":
        axis = "X"
    return {"radius": radius, "width": width, "axis": axis,
            "segments": _segments(node.get("segments"))}, None


def _clean_bolt(node):
    radius = _dim(node.get("radius"), default=0.012)
    height = _dim(node.get("height") or node.get("depth"), default=0.02)
    return {"radius": radius, "height": height, "axis": _axis(node)}, None


def _clean_panel(node):
    size = node.get("size") or node.get("dims")
    sx, sy, sz = [_dim(v) for v in _vec3(size, default=(0.3, 0.3, 0.03))]
    inset = _num(node.get("inset"), default=0.05, lo=0.005, hi=0.5)
    return {"size": [sx, sy, sz], "inset": inset}, None


def _clean_leaf_card(node):
    length = node.get("length")
    size = node.get("size")
    if length is None and isinstance(size, (list, tuple)) and len(size) >= 2:
        length = size[1]
    length = _dim(length, default=0.3)
    width = _dim(node.get("width"), default=0.1)
    yaw = _num(node.get("yaw"), default=0.0)
    pitch = _num(node.get("pitch"), default=0.0)
    curve = _num(node.get("curve"), default=0.12, lo=0.0, hi=1.0)
    return {"length": length, "width": width, "yaw": yaw, "pitch": pitch,
            "curve": curve}, None


def _clean_lathe(node):
    raw = node.get("profile")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None, "lathe needs a profile of at least 2 [radius, z] points"
    profile = []
    for pt in raw[:32]:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        r = _num(pt[0])
        z = _num(pt[1])
        if r is None or z is None:
            continue
        profile.append([min(abs(r), MAX_DIM), max(-MAX_DIM, min(z, MAX_DIM))])
    if len(profile) < 2:
        return None, "lathe profile had no usable points"
    profile.sort(key=lambda p: p[1])
    if max(p[0] for p in profile) < 1e-4:
        return None, "lathe profile has zero radius everywhere"
    return {"profile": profile, "segments": _segments(node.get("segments"))}, None


def _clean_tube(node):
    raw = node.get("path")
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None, "tube needs a path of at least 2 [x, y, z] points"
    path = []
    for pt in raw[:32]:
        v = _vec3(pt, default=None) if isinstance(pt, (list, tuple)) else None
        if v is None:
            continue
        path.append([max(-MAX_DIM, min(c, MAX_DIM)) for c in v])
    deduped = [path[0]] if path else []
    for p in path[1:]:
        if sum((a - b) ** 2 for a, b in zip(p, deduped[-1])) > 1e-10:
            deduped.append(p)
    if len(deduped) < 2:
        return None, "tube path had no usable points"
    radius = _dim(node.get("radius"), default=0.02)
    return {"path": deduped, "radius": radius,
            "segments": _segments(node.get("segments"), default=12)}, None


def _clean_ring(node):
    radius = _dim(node.get("radius"), default=0.15)
    thickness = _dim(node.get("thickness") or node.get("minor_radius"), default=0.02)
    if thickness >= radius:
        thickness = radius * 0.4
    return {"radius": radius, "thickness": thickness, "axis": _axis(node),
            "segments": _segments(node.get("segments"))}, None


def _axis(node, default="Z"):
    axis = str(node.get("axis", default)).strip().upper()
    return axis if axis in VALID_AXES else default


_CLEANERS = {
    "box": _clean_box,
    "cylinder": _clean_cylinder,
    "sphere": _clean_sphere,
    "wheel": _clean_wheel,
    "bolt": _clean_bolt,
    "panel": _clean_panel,
    "leaf_card": _clean_leaf_card,
    "lathe": _clean_lathe,
    "tube": _clean_tube,
    "ring": _clean_ring,
}

_PRIMITIVE_ALIASES = {
    "cube": "box", "block": "box", "plank": "box",
    "cone": "cylinder", "rod": "cylinder", "pipe": "cylinder", "post": "cylinder",
    "tyre": "wheel", "tire": "wheel",
    "ellipsoid": "sphere", "ball": "sphere",
    "torus": "ring", "band": "ring",
    "hose": "tube", "cable": "tube", "wire": "tube",
    "leaf": "leaf_card", "blade": "leaf_card",
    "plate": "panel",
    "revolve": "lathe", "lathe_profile": "lathe",
}


def _local_bbox(part) -> tuple[list, list]:
    """Axis-aligned local bbox as ([minx,miny,minz], [maxx,maxy,maxz])."""
    prim = part["primitive"]
    p = part["params"]
    if prim in ("box", "sphere", "panel"):
        h = [s * 0.5 for s in p["size"]]
        return [-h[0], -h[1], -h[2]], [h[0], h[1], h[2]]
    if prim in ("cylinder", "bolt", "wheel"):
        r = p["radius"]
        length = p.get("depth") or p.get("height") or p.get("width")
        axis = p.get("axis", "Z")
        half = {"X": [length * 0.5, r, r],
                "Y": [r, length * 0.5, r],
                "Z": [r, r, length * 0.5]}[axis]
        return [-half[0], -half[1], -half[2]], half
    if prim == "ring":
        outer = p["radius"] + p["thickness"]
        t = p["thickness"]
        axis = p.get("axis", "Z")
        half = {"X": [t, outer, outer],
                "Y": [outer, t, outer],
                "Z": [outer, outer, t]}[axis]
        return [-half[0], -half[1], -half[2]], half
    if prim == "lathe":
        r = max(pt[0] for pt in p["profile"])
        zs = [pt[1] for pt in p["profile"]]
        return [-r, -r, min(zs)], [r, r, max(zs)]
    if prim == "tube":
        r = p["radius"]
        xs = [pt[0] for pt in p["path"]]
        ys = [pt[1] for pt in p["path"]]
        zs = [pt[2] for pt in p["path"]]
        return ([min(xs) - r, min(ys) - r, min(zs) - r],
                [max(xs) + r, max(ys) + r, max(zs) + r])
    if prim == "leaf_card":
        w, length = p["width"], p["length"]
        return [-w * 0.5, 0.0, 0.0], [w * 0.5, length, p["curve"] * 0.4 * length]
    return [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]


def _rotate_point(pt, rot_deg):
    """Rotate a point by XYZ euler angles given in degrees."""
    rx, ry, rz = [math.radians(a) for a in rot_deg]
    x, y, z = pt
    cy_, sy_ = math.cos(rx), math.sin(rx)
    y, z = y * cy_ - z * sy_, y * sy_ + z * cy_
    cy_, sy_ = math.cos(ry), math.sin(ry)
    x, z = x * cy_ + z * sy_, -x * sy_ + z * cy_
    cy_, sy_ = math.cos(rz), math.sin(rz)
    x, y = x * cy_ - y * sy_, x * sy_ + y * cy_
    return [x, y, z]


def world_bbox(part) -> tuple[list, list]:
    """World-space AABB for a cleaned part (pos + optional rot applied)."""
    lo, hi = _local_bbox(part)
    rot = part.get("rot") or [0.0, 0.0, 0.0]
    pos = part.get("pos") or [0.0, 0.0, 0.0]
    if any(abs(a) > 1e-6 for a in rot):
        corners = [[x, y, z] for x in (lo[0], hi[0])
                   for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        rotated = [_rotate_point(c, rot) for c in corners]
        lo = [min(c[i] for c in rotated) for i in range(3)]
        hi = [max(c[i] for c in rotated) for i in range(3)]
    return ([lo[i] + pos[i] for i in range(3)],
            [hi[i] + pos[i] for i in range(3)])


def _clean_materials(raw, warnings: list[str]) -> dict:
    materials: dict[str, dict] = {}
    if isinstance(raw, dict):
        for name, mat in raw.items():
            key = str(name).strip()
            if not key:
                continue
            if not isinstance(mat, dict):
                mat = {}
            entry = {
                "color": _color3(mat.get("color") or mat.get("base_color")),
                "metallic": _num(mat.get("metallic"), default=0.0, lo=0.0, hi=1.0),
                "roughness": _num(mat.get("roughness"), default=0.6, lo=0.0, hi=1.0),
            }
            texture = str(mat.get("texture") or "").strip()
            if texture:
                entry["texture"] = texture
            materials[key] = entry
    if not materials:
        warnings.append("no materials palette — using a neutral default")
        materials["default"] = {"color": [0.55, 0.55, 0.55],
                                "metallic": 0.0, "roughness": 0.6}
    return materials


def _clean_array(raw, warnings: list[str], part_id: str):
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("type", "")).strip().lower()
    count = _num(raw.get("count"), default=0)
    count = int(count) if count else 0
    if count < 2:
        return None
    if count > MAX_ARRAY_COUNT:
        warnings.append(f"part '{part_id}': array count {count} capped to {MAX_ARRAY_COUNT}")
        count = MAX_ARRAY_COUNT
    if kind == "ring":
        ring_radius = _dim(raw.get("ring_radius") or raw.get("radius"), default=0.15)
        out = {"type": "ring", "count": count, "ring_radius": ring_radius,
               "rotate": bool(raw.get("rotate", True))}
        center = raw.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            cx = _num(center[0], default=0.0)
            cy = _num(center[1], default=0.0)
            out["center"] = [cx, cy]
        z = _num(raw.get("z"))
        if z is not None:
            out["z"] = z
        return out
    if kind == "linear":
        step = _vec3(raw.get("step"), default=(0.0, 0.0, 0.0))
        if all(abs(s) < 1e-6 for s in step):
            warnings.append(f"part '{part_id}': linear array with zero step dropped")
            return None
        return {"type": "linear", "count": count, "step": step}
    warnings.append(f"part '{part_id}': unknown array type '{kind}' dropped")
    return None


def validate_spec(raw: dict) -> SpecReport:
    """Validate/coerce a raw planner spec. Permissive: unusable parts are
    dropped with warnings; errors only when the whole spec is unusable."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw, dict):
        return SpecReport(ok=False, errors=["spec is not an object"])

    raw_parts = raw.get("parts")
    if raw_parts is None:
        raw_parts = raw.get("scene_graph")
    if not isinstance(raw_parts, list) or not raw_parts:
        return SpecReport(ok=False, errors=["spec has no parts list"])

    materials = _clean_materials(raw.get("materials"), warnings)
    default_material = next(iter(materials))

    parts: list[dict] = []
    seen_ids: set[str] = set()

    for idx, node in enumerate(raw_parts):
        if not isinstance(node, dict):
            warnings.append(f"part #{idx}: not an object — dropped")
            continue

        pid = str(node.get("id") or node.get("name") or "").strip()
        if not pid:
            warnings.append(f"part #{idx}: missing id — dropped")
            continue
        pid_base = pid
        n = 2
        while pid in seen_ids:
            pid = f"{pid_base}_{n}"
            n += 1
        if pid != pid_base:
            warnings.append(f"part '{pid_base}': duplicate id renamed to '{pid}'")

        prim = str(node.get("primitive") or node.get("type") or "").strip().lower()
        prim = _PRIMITIVE_ALIASES.get(prim, prim)
        if prim not in VALID_PRIMITIVES:
            if node.get("profile"):
                prim = "lathe"
            elif node.get("path"):
                prim = "tube"
            elif node.get("radius") is not None:
                prim = "cylinder"
            elif node.get("size") or node.get("dims"):
                prim = "box"
            else:
                warnings.append(f"part '{pid}': unknown primitive — dropped")
                continue
            warnings.append(f"part '{pid}': primitive guessed as '{prim}'")

        params, warn = _CLEANERS[prim](node)
        if params is None:
            warnings.append(f"part '{pid}': {warn} — dropped")
            continue
        if warn:
            warnings.append(f"part '{pid}': {warn}")

        material = str(node.get("material") or "").strip()
        if material not in materials:
            if material:
                warnings.append(f"part '{pid}': unknown material '{material}' → '{default_material}'")
            material = default_material

        part = {
            "id": pid,
            "primitive": prim,
            "pos": _vec3(node.get("pos") or node.get("position")),
            "params": params,
            "material": material,
            "attach_to": str(node.get("attach_to") or node.get("parent") or "").strip(),
        }

        rot = node.get("rot") or node.get("rotation")
        if rot is not None:
            rot = _vec3(rot)
            if any(abs(a) > 1e-6 for a in rot):
                part["rot"] = [max(-360.0, min(a, 360.0)) for a in rot]

        mirror = str(node.get("mirror") or "").strip().upper()
        if mirror:
            if mirror in VALID_MIRRORS:
                part["mirror"] = mirror
            else:
                warnings.append(f"part '{pid}': mirror '{mirror}' ignored (use X or Y)")

        if node.get("fuse") in (True, "true", "True", 1):
            part["fuse"] = True

        array = _clean_array(node.get("array"), warnings, pid)
        if array:
            part["array"] = array

        seen_ids.add(pid)
        parts.append(part)

    if len(parts) < MIN_PARTS:
        errors.append(
            f"only {len(parts)} usable part(s) after cleaning (need >= {MIN_PARTS})"
        )
        return SpecReport(ok=False, errors=errors, warnings=warnings)

    ids = {p["id"] for p in parts}
    by_id = {p["id"]: p for p in parts}
    for p in parts:
        if p["attach_to"] and p["attach_to"] not in ids:
            warnings.append(
                f"part '{p['id']}': attach_to '{p['attach_to']}' not found → re-rooted"
            )
            p["attach_to"] = ""
        if p["attach_to"] == p["id"]:
            warnings.append(f"part '{p['id']}': attached to itself → re-rooted")
            p["attach_to"] = ""

    for p in parts:
        seen = {p["id"]}
        cur = p
        while cur["attach_to"]:
            nxt = by_id[cur["attach_to"]]
            if nxt["id"] in seen:
                warnings.append(
                    f"attach cycle broken at '{cur['id']}' (was → '{nxt['id']}')"
                )
                cur["attach_to"] = ""
                break
            seen.add(nxt["id"])
            cur = nxt

    spec = {
        "name": str(raw.get("name") or "argus_asset").strip() or "argus_asset",
        "materials": materials,
        "parts": parts,
    }

    expanded_count = sum(_expansion_factor(p) for p in parts)
    if expanded_count > MAX_EXPANDED_PARTS:
        errors.append(
            f"spec expands to {expanded_count} parts (max {MAX_EXPANDED_PARTS})"
        )
        return SpecReport(ok=False, errors=errors, warnings=warnings)

    snap_warnings = snap_spec(spec)
    warnings.extend(snap_warnings)

    return SpecReport(ok=True, spec=spec, errors=errors, warnings=warnings)


def _expansion_factor(part) -> int:
    n = part.get("array", {}).get("count", 1) if part.get("array") else 1
    if part.get("mirror"):
        n *= 2
    return n


def _children_map(parts):
    children: dict[str, list] = {}
    for p in parts:
        children.setdefault(p["attach_to"], []).append(p)
    return children


def _subtree_ids(part_id, children) -> list[str]:
    out = []
    stack = [part_id]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, []):
            out.append(child["id"])
            stack.append(child["id"])
    return out


def snap_spec(spec: dict) -> list[str]:
    """Translate attached parts so their AABB touches/overlaps the parent's.
    Children move with their subtree. Returns warnings for large corrections."""
    warnings: list[str] = []
    parts = spec["parts"]
    by_id = {p["id"]: p for p in parts}
    children = _children_map(parts)

    order: list[dict] = []
    queue = list(children.get("", []))
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        queue.extend(children.get(cur["id"], []))

    for part in order:
        parent_id = part["attach_to"]
        if not parent_id:
            continue
        if part["primitive"] in _SNAP_EXEMPT or part.get("array"):
            continue
        parent = by_id[parent_id]
        lo_a, hi_a = world_bbox(part)
        lo_b, hi_b = world_bbox(parent)

        delta = [0.0, 0.0, 0.0]
        for ax in range(3):
            gap_pos = lo_a[ax] - hi_b[ax]
            gap_neg = lo_b[ax] - hi_a[ax]
            if gap_pos > CONTACT_TOL:
                overlap = min(SNAP_OVERLAP,
                              (hi_a[ax] - lo_a[ax]) * 0.25,
                              (hi_b[ax] - lo_b[ax]) * 0.25)
                delta[ax] = -(gap_pos + overlap)
            elif gap_neg > CONTACT_TOL:
                overlap = min(SNAP_OVERLAP,
                              (hi_a[ax] - lo_a[ax]) * 0.25,
                              (hi_b[ax] - lo_b[ax]) * 0.25)
                delta[ax] = gap_neg + overlap

        if any(abs(d) > 1e-9 for d in delta):
            moved = math.sqrt(sum(d * d for d in delta))
            if moved > 0.25:
                warnings.append(
                    f"part '{part['id']}': snapped {moved:.2f} m to touch "
                    f"'{parent_id}' — planner positions were far off"
                )
            ids_to_move = [part["id"]] + _subtree_ids(part["id"], children)
            for pid in ids_to_move:
                node = by_id[pid]
                node["pos"] = [node["pos"][i] + delta[i] for i in range(3)]

    min_z = min(world_bbox(p)[0][2] for p in parts)
    if abs(min_z) > 1e-6:
        for p in parts:
            p["pos"][2] -= min_z

    return warnings


def expand_parts(spec: dict) -> list[dict]:
    """Expand mirror/array declarations into concrete parts. Output parts have
    no 'mirror'/'array' keys and unique ids; attach_to is preserved."""
    out: list[dict] = []
    for part in spec["parts"]:
        instances = _expand_array(part)
        mirrored: list[dict] = []
        for inst in instances:
            mirrored.append(inst)
            if part.get("mirror"):
                mirrored.append(_mirror_part(inst, part["mirror"]))
        out.extend(mirrored)
    return out


def _copy_part(part) -> dict:
    import copy
    clone = copy.deepcopy(part)
    clone.pop("mirror", None)
    clone.pop("array", None)
    return clone


def _expand_array(part) -> list[dict]:
    array = part.get("array")
    if not array:
        return [_copy_part(part)]
    out = []
    count = array["count"]
    if array["type"] == "ring":
        cx, cy = array.get("center", [0.0, 0.0])
        rr = array["ring_radius"]
        z = array.get("z", part["pos"][2])
        for i in range(count):
            ang = 2.0 * math.pi * i / count
            clone = _copy_part(part)
            clone["id"] = f"{part['id']}_{i + 1}"
            clone["pos"] = [cx + rr * math.cos(ang), cy + rr * math.sin(ang), z]
            if array.get("rotate", True):
                rot = clone.get("rot", [0.0, 0.0, 0.0])
                clone["rot"] = [rot[0], rot[1], rot[2] + math.degrees(ang)]
            out.append(clone)
    else:
        step = array["step"]
        for i in range(count):
            clone = _copy_part(part)
            clone["id"] = f"{part['id']}_{i + 1}"
            clone["pos"] = [part["pos"][j] + step[j] * i for j in range(3)]
            out.append(clone)
    return out


def _mirror_part(part, axis: str) -> dict:
    clone = _copy_part(part)
    clone["id"] = f"{part['id']}_m{axis.lower()}"
    idx = 0 if axis == "X" else 1
    clone["pos"] = list(part["pos"])
    clone["pos"][idx] = -clone["pos"][idx]
    if clone.get("rot"):
        rx, ry, rz = clone["rot"]
        if axis == "X":
            clone["rot"] = [rx, -ry, -rz]
        else:
            clone["rot"] = [-rx, ry, -rz]
    params = clone["params"]
    if "path" in params:
        params["path"] = [
            [(-p[0] if axis == "X" else p[0]),
             (-p[1] if axis == "Y" else p[1]), p[2]]
            for p in params["path"]
        ]
    if clone["primitive"] == "leaf_card":
        params["yaw"] = -params.get("yaw", 0.0)
    return clone


_EDITABLE_FIELDS = {
    "pos", "rot", "size", "radius", "depth", "height", "width", "length",
    "r_top", "axis", "segments", "profile", "path", "thickness", "inset",
    "material", "attach_to", "mirror", "array", "yaw", "pitch", "curve",
    "bevel", "smooth", "crease",
}


def _flatten_part(part: dict) -> dict:
    """Cleaned part → raw-style node (params hoisted to the top level) so
    validate_spec can re-clean it. Raw nodes pass through unchanged."""
    if "params" not in part:
        return part
    flat = {"id": part["id"], "primitive": part["primitive"],
            "pos": part["pos"], "material": part["material"],
            "attach_to": part["attach_to"]}
    for key in ("rot", "mirror", "array"):
        if part.get(key):
            flat[key] = part[key]
    flat.update(part["params"])
    return flat


def apply_spec_edits(spec: dict, edits: list) -> SpecReport:
    """Apply a vision-model edit list to a validated spec, then re-validate.

    Edit forms:
      {"id": "part_id", "set": {field: value, ...}}     — update fields
      {"id": "part_id", "delete": true}                 — remove a part
      {"add": {<raw part node>}}                        — append a new part
      {"material": "name", "set": {color/metallic/roughness/texture}}
    """
    import copy
    if not isinstance(edits, list) or not edits:
        return SpecReport(ok=False, errors=["empty edit list"])

    draft = copy.deepcopy(spec)
    parts = draft["parts"]
    by_id = {p["id"]: p for p in parts}
    applied = 0
    warnings: list[str] = []

    for edit in edits:
        if not isinstance(edit, dict):
            continue

        if edit.get("add"):
            node = edit["add"]
            if isinstance(node, dict):
                parts.append(node)
                applied += 1
            continue

        mat_name = str(edit.get("material") or "").strip()
        if mat_name and isinstance(edit.get("set"), dict):
            mat = draft["materials"].get(mat_name)
            if mat is None:
                warnings.append(f"edit: unknown material '{mat_name}' skipped")
                continue
            changes = edit["set"]
            if "color" in changes:
                mat["color"] = _color3(changes["color"])
            for key in ("metallic", "roughness"):
                if key in changes:
                    v = _num(changes[key], default=None, lo=0.0, hi=1.0)
                    if v is not None:
                        mat[key] = v
            if "texture" in changes:
                tex = str(changes["texture"] or "").strip()
                if tex:
                    mat["texture"] = tex
                else:
                    mat.pop("texture", None)
            applied += 1
            continue

        pid = str(edit.get("id") or "").strip()
        if not pid:
            continue
        part = by_id.get(pid)
        if part is None:
            warnings.append(f"edit: unknown part '{pid}' skipped")
            continue

        if edit.get("delete"):
            parts.remove(part)
            by_id.pop(pid, None)
            applied += 1
            continue

        changes = edit.get("set")
        if not isinstance(changes, dict):
            continue
        flat = _flatten_part(part)
        for key, value in changes.items():
            if key in _EDITABLE_FIELDS:
                flat[key] = value
            else:
                warnings.append(f"edit '{pid}': field '{key}' not editable — skipped")
        idx = parts.index(part)
        parts[idx] = flat
        by_id[pid] = flat
        applied += 1

    if applied == 0:
        return SpecReport(ok=False, errors=["no edits could be applied"],
                          warnings=warnings)

    draft["parts"] = [_flatten_part(p) for p in draft["parts"]]
    report = validate_spec(draft)
    report.warnings = warnings + report.warnings
    return report


def spec_summary(spec: dict) -> str:
    """One-line description for pipeline logs."""
    parts = spec.get("parts", [])
    prims: dict[str, int] = {}
    for p in parts:
        prims[p["primitive"]] = prims.get(p["primitive"], 0) + 1
    hist = ", ".join(f"{k}×{v}" for k, v in sorted(prims.items()))
    expanded = sum(_expansion_factor(p) for p in parts)
    return (f"{len(parts)} parts ({expanded} expanded), "
            f"{len(spec.get('materials', {}))} materials [{hist}]")


def legacy_scene_graph(spec: dict) -> list[dict]:
    """Derive the old-style scene_graph hint list (name/pos/size/attach_to)
    from a build spec, for the LLM-codegen fallback path."""
    out = []
    for p in spec.get("parts", []):
        lo, hi = world_bbox(p)
        out.append({
            "name": p["id"],
            "pos": [round((lo[i] + hi[i]) * 0.5, 4) for i in range(3)],
            "size": [round(hi[i] - lo[i], 4) for i in range(3)],
            "attach_to": p["attach_to"],
        })
    return out
