"""Render-free quality scoring for validated build specs.

Scores a spec on mechanical criteria (grounding, connectivity, symmetry,
proportions, variety) in milliseconds — no Blender, no LLM. Used to pick
the best of N planner candidates before anything is compiled or rendered.
"""
from __future__ import annotations

import math

from core.spec import expand_parts


def _part_aabb(part: dict) -> tuple[list[float], list[float]]:
    """Approximate world AABB (min, max) for one expanded part. Rotation is
    handled by inflating to the bounding sphere — coarse but fine for scoring."""
    p = part.get("params", {})
    pos = list(part.get("pos", [0.0, 0.0, 0.0]))
    prim = part.get("primitive", "box")
    half = [0.1, 0.1, 0.1]
    centre = pos

    if prim in ("box", "panel", "sphere"):
        s = p.get("size", [0.2, 0.2, 0.2])
        half = [s[0] / 2, s[1] / 2, s[2] / 2]
    elif prim == "cylinder":
        r = max(p.get("radius", 0.1), p.get("r_top", 0.0))
        d = p.get("depth", 0.2) / 2
        axis = p.get("axis", "Z")
        half = {"X": [d, r, r], "Y": [r, d, r], "Z": [r, r, d]}[axis]
    elif prim == "wheel":
        r = p.get("radius", 0.3)
        w = p.get("width", 0.15) / 2
        half = [w, r, r] if p.get("axis", "X") == "X" else [r, w, r]
    elif prim == "bolt":
        r = p.get("radius", 0.012)
        h = p.get("height", 0.02) / 2
        axis = p.get("axis", "Z")
        half = {"X": [h, r, r], "Y": [r, h, r], "Z": [r, r, h]}[axis]
    elif prim == "ring":
        r = p.get("radius", 0.1) + p.get("thickness", 0.02)
        t = p.get("thickness", 0.02)
        axis = p.get("axis", "Z")
        half = {"X": [t, r, r], "Y": [r, t, r], "Z": [r, r, t]}[axis]
    elif prim == "lathe":
        prof = p.get("profile") or [[0.1, 0.0], [0.1, 0.2]]
        maxr = max(pt[0] for pt in prof)
        zs = [pt[1] for pt in prof]
        zmin, zmax = min(zs), max(zs)
        centre = [pos[0], pos[1], pos[2] + (zmin + zmax) / 2]
        half = [maxr, maxr, max((zmax - zmin) / 2, 1e-3)]
    elif prim == "tube":
        path = p.get("path") or [[0, 0, 0], [0, 0, 0.2]]
        r = p.get("radius", 0.02)
        lo = [min(pt[i] for pt in path) - r + pos[i] for i in range(3)]
        hi = [max(pt[i] for pt in path) + r + pos[i] for i in range(3)]
        return lo, hi
    elif prim == "leaf_card":
        length = p.get("length", 0.3)
        w = p.get("width", 0.1)
        centre = [pos[0], pos[1] + length / 2, pos[2]]
        half = [w, length / 2, 0.05]

    if part.get("rot"):
        rad = math.sqrt(sum(h * h for h in half))
        half = [rad, rad, rad]
    lo = [centre[i] - half[i] for i in range(3)]
    hi = [centre[i] + half[i] for i in range(3)]
    return lo, hi


def _overlaps(a, b, tol=0.03) -> bool:
    return all(a[0][i] - tol <= b[1][i] and b[0][i] - tol <= a[1][i] for i in range(3))


def score_spec(spec: dict, prompt: str = "") -> tuple[float, list[str]]:
    """Score a validated spec 0-100. Returns (score, reasons)."""
    score = 50.0
    reasons: list[str] = []
    parts = spec.get("parts") or []
    if not parts:
        return 0.0, ["empty spec"]

    try:
        expanded = expand_parts(spec)
    except Exception:
        expanded = parts
    boxes = [_part_aabb(pt) for pt in expanded]
    n = len(expanded)

    if 5 <= n <= 30:
        score += 10
    elif n < 5:
        score += 2
        reasons.append(f"only {n} parts")
    elif n > 40:
        score -= 8
        reasons.append(f"{n} parts is excessive")

    glo = [min(b[0][i] for b in boxes) for i in range(3)]
    ghi = [max(b[1][i] for b in boxes) for i in range(3)]
    dims = [max(ghi[i] - glo[i], 1e-4) for i in range(3)]

    grounded = sum(1 for b in boxes if b[0][2] < 0.02)
    if grounded:
        score += 5
    else:
        score -= 10
        reasons.append("nothing touches the ground")

    by_id = {pt["id"]: _part_aabb(pt) for pt in parts}
    attached = [pt for pt in parts if pt.get("attach_to")]
    if attached:
        connected = sum(
            1 for pt in attached
            if pt["attach_to"] in by_id
            and _overlaps(_part_aabb(pt), by_id[pt["attach_to"]])
        )
        frac = connected / len(attached)
        score += frac * 10
        if frac < 1.0:
            floaters = [pt["id"] for pt in attached
                        if pt["attach_to"] in by_id
                        and not _overlaps(_part_aabb(pt), by_id[pt["attach_to"]])]
            reasons.append("floating: " + ", ".join(floaters[:4]))

    vol_total = 0.0
    cx_weighted = 0.0
    for b in boxes:
        v = max((b[1][0] - b[0][0]) * (b[1][1] - b[0][1]) * (b[1][2] - b[0][2]), 1e-9)
        vol_total += v
        cx_weighted += v * (b[0][0] + b[1][0]) / 2
    cx = cx_weighted / vol_total
    if abs(cx) < 0.05 * dims[0]:
        score += 8
    else:
        reasons.append("asymmetric about X")
    if any(pt.get("mirror") or pt.get("array") for pt in parts):
        score += 4

    extreme = 0
    for b in boxes:
        d = sorted(max(b[1][i] - b[0][i], 1e-4) for i in range(3))
        if d[2] / d[0] > 60:
            extreme += 1
    if extreme:
        score -= min(extreme * 3, 9)
        reasons.append(f"{extreme} sliver part(s)")
    if max(dims) / min(dims) > 15:
        score -= 6
        reasons.append("overall bbox is extremely elongated")

    if 0.1 <= max(dims) <= 12.0:
        score += 5
    else:
        score -= 5
        reasons.append(f"overall size {max(dims):.2f}m implausible")

    mats = {pt.get("material") for pt in parts}
    if len(mats) >= 2:
        score += 4
    else:
        reasons.append("single material")

    prims = {pt.get("primitive") for pt in parts}
    if len(prims) >= 3:
        score += 4
    elif prims <= {"box", "panel"}:
        score -= 3
        reasons.append("all-box construction")

    hidden = 0
    for i, b in enumerate(boxes):
        for j, other in enumerate(boxes):
            if i == j:
                continue
            if all(other[0][k] <= b[0][k] and b[1][k] <= other[1][k] for k in range(3)):
                hidden += 1
                break
    if hidden:
        score -= min(hidden * 2, 8)
        reasons.append(f"{hidden} fully-hidden part(s)")

    # Deeply-buried feature parts: a round "feature" (wheel, dome, stack,
    # nozzle...) that sits >55% inside a much larger part reads as embedded,
    # not attached — this is the locomotive failure (1.8 m drive wheels whose
    # top half is swallowed by the boiler). Boxes/panels/greebles are exempt
    # because flush surfaces and collars are meant to sit against the body.
    _ROUND = {"wheel", "sphere", "cylinder", "tube", "lathe", "ring"}

    def _vol_box(bx):
        return max(
            (bx[1][0] - bx[0][0]) * (bx[1][1] - bx[0][1]) * (bx[1][2] - bx[0][2]),
            1e-9)

    def _embed_frac(a, bx):
        inter = 1.0
        for k in range(3):
            inter *= max(0.0, min(a[1][k], bx[1][k]) - max(a[0][k], bx[0][k]))
        return inter / _vol_box(a)

    embedded = 0
    for i, pt in enumerate(expanded):
        if (pt.get("primitive") not in _ROUND
                or str(pt.get("id", "")).startswith("auto_")):
            continue
        vi = _vol_box(boxes[i])
        for j, other in enumerate(boxes):
            if i == j or _vol_box(other) < 3.0 * vi:
                continue
            if _embed_frac(boxes[i], other) > 0.55:
                embedded += 1
                break
    if embedded:
        score -= min(embedded * 6, 24)
        reasons.append(
            f"{embedded} feature part(s) buried in a larger part "
            "(e.g. wheels sunk into the body)")

    # Cleaner-default fingerprints: a cylinder of exactly r=0.1 ∧ d=0.2 (etc.)
    # almost always means the planner omitted dimensions and spec-cleaning
    # back-filled them — the built proportions will be nonsense.
    defaulted = 0
    for pt in expanded:
        p = pt.get("params", {})
        prim = pt.get("primitive")
        if prim == "cylinder" and p.get("radius") == 0.1 and p.get("depth") == 0.2:
            defaulted += 1
        elif (prim in ("box", "panel", "sphere")
              and list(p.get("size") or ()) == [0.2, 0.2, 0.2]):
            defaulted += 1
        elif prim == "wheel" and p.get("radius") == 0.3 and p.get("width") == 0.15:
            defaulted += 1
    if defaulted:
        score -= min(defaulted * 6, 18)
        reasons.append(
            f"{defaulted} default-sized part(s) — planner omitted dimensions")

    # Whole-assembly connectivity: union-find over AABB contact. attach_to
    # checks above only cover declared pairs — a spec can pass those and
    # still fall into separate islands (observed: a "crate" in 6 clusters).
    parent = list(range(len(boxes)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlaps(boxes[i], boxes[j], tol=0.01):
                ri, rj = _find(i), _find(j)
                if ri != rj:
                    parent[ri] = rj
    clusters = len({_find(i) for i in range(len(boxes))})
    if clusters > 1:
        score -= min((clusters - 1) * 10, 20)
        reasons.append(f"parts form {clusters} disconnected islands")

    # Identical primitive+params at the same position is a z-fighting copy,
    # not a deliberate array (arrays expand to distinct positions).
    seen: set = set()
    coincident = 0
    for pt in expanded:
        key = (pt.get("primitive"),
               repr(sorted((pt.get("params") or {}).items())),
               tuple(round(float(c), 4) for c in (pt.get("pos") or (0, 0, 0))))
        if key in seen:
            coincident += 1
        else:
            seen.add(key)
    if coincident:
        score -= min(coincident * 8, 16)
        reasons.append(f"{coincident} coincident duplicate part(s)")

    return max(0.0, min(100.0, score)), reasons
