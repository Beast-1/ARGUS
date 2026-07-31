
from __future__ import annotations

COMPONENT_LIBRARY_SOURCE = r'''
# ── ARGUS parametric component library ────────────────────────────────────
# Reusable, well-shaped builders. Prefer these over ad-hoc primitives so every
# part is bevelled, has outward normals, a UV layer, and a clean material.

def _argus_finalize(mesh, bm, *, bevel_edges=True, bevel_off=0.012, bevel_seg=2):
    """Shared bmesh finishing: optional bevel, normals, n-gon cleanup, UV."""
    if bevel_edges and bevel_off > 0.0 and bm.edges:
        try:
            bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel_off,
                            segments=bevel_seg, profile=0.7, affect="EDGES")
        except Exception:
            pass
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    _ng = [f for f in bm.faces if len(f.verts) > 4]
    if _ng:
        # n-gons here are almost all circular caps (cylinder/bolt/wheel/cone).
        # Ear-clipping a 32-gon cap fans slivers from one corner; poke adds a
        # centre vertex and fans evenly (umbrella) → ~equal triangles, clean
        # shading, no centre pinch.
        try:
            bmesh.ops.poke(bm, faces=_ng)
        except Exception:
            try:
                bmesh.ops.triangulate(bm, faces=_ng,
                                      quad_method="BEAUTY", ngon_method="BEAUTY")
            except TypeError:
                bmesh.ops.triangulate(bm, faces=_ng)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    for _p in mesh.polygons:
        _p.use_smooth = True
    mesh.update()
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="ARGUS_UV")


def _argus_link(name, mesh, mat, root):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    if root is not None:
        obj.parent = root
        obj.matrix_parent_inverse = root.matrix_world.inverted()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(33))
    except Exception:
        pass
    return obj


def _argus_uv_layer(bm):
    return bm.loops.layers.uv.get("ARGUS_UV") or bm.loops.layers.uv.new("ARGUS_UV")


def _argus_uv_box(bm, faces=None):
    """Box (triplanar) projection: each face is mapped on the two axes
    perpendicular to its dominant normal axis. 1 world metre = 1 UV unit,
    so texel density is uniform across parts of any size."""
    uv = _argus_uv_layer(bm)
    bm.normal_update()
    for f in (faces if faces is not None else bm.faces):
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        ui, vi = [i for i in range(3) if i != ax]
        for loop in f.loops:
            co = loop.vert.co
            loop[uv].uv = (co[ui], co[vi])


def _argus_uv_cyl(bm, radius, cap_limit=0.7):
    """Cylindrical wrap for Z-aligned geometry: side faces get u = arc length
    (seam-corrected), v = z; near-horizontal faces (caps) get planar XY.
    Call BEFORE any axis rotation/translation. World-scale texel density."""
    uv = _argus_uv_layer(bm)
    bm.normal_update()
    two_pi = 2.0 * math.pi
    circ = two_pi * max(radius, 1e-6)
    for f in bm.faces:
        if abs(f.normal.z) > cap_limit:
            for loop in f.loops:
                co = loop.vert.co
                loop[uv].uv = (co.x, co.y)
            continue
        us = []
        for loop in f.loops:
            co = loop.vert.co
            us.append(math.atan2(co.y, co.x) / two_pi)
        u0 = us[0]
        us = [u + 1.0 if (u - u0) < -0.5 else (u - 1.0 if (u - u0) > 0.5 else u)
              for u in us]
        for loop, u in zip(f.loops, us):
            loop[uv].uv = (u * circ, loop.vert.co.z)


def _argus_fan_cap(bm, ring_verts):
    """Triangulate a round end-cap as an umbrella fan from its centre vertex.
    A circular cap left as one n-gon gets ear-clipped into slivers radiating
    from a single corner (ugly shading + a pinch); a centre fan gives even,
    well-shaped triangles. Winding is fixed later by recalc_face_normals."""
    c = mathutils.Vector((0.0, 0.0, 0.0))
    for v in ring_verts:
        c = c + v.co
    c = c / len(ring_verts)
    cv = bm.verts.new(c)
    n = len(ring_verts)
    for i in range(n):
        bm.faces.new((cv, ring_verts[i], ring_verts[(i + 1) % n]))


def argus_box(name, loc, size, mat, root, bevel=0.012):
    """Rounded box. size=(sx,sy,sz) full dimensions; loc=center."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector((size[0], size[1], size[2])),
                    verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    _argus_uv_box(bm)
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    _argus_finalize(mesh, bm, bevel_off=bevel)
    return _argus_link(name, mesh, mat, root)


def argus_cylinder(name, loc, radius, depth, mat, root, *, axis="Z",
                   segments=32, r_top=None, bevel=0.0):
    """Capped cylinder / tapered cone. axis in {'X','Y','Z'}. r_top=None -> straight."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    r2 = radius if r_top is None else r_top
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                          radius1=radius, radius2=r2, depth=depth)
    # Length loop-cuts: a plain cone has verts only at its two end rings, which
    # both shades flatly and hides mid-body attachments from the connectivity
    # check. Add horizontal loops (~1 per 10 cm, capped) so the side carries
    # real geometry where other parts join it.
    _cuts = min(int(abs(depth) / 0.1), 10)
    if _cuts > 0:
        _side = [e for e in bm.edges
                 if abs(e.verts[0].co.z - e.verts[1].co.z) > abs(depth) * 0.5]
        if _side:
            bmesh.ops.subdivide_edges(bm, edges=_side, cuts=_cuts,
                                      use_grid_fill=True)
    _argus_uv_cyl(bm, max(radius, r2))
    if axis == "X":
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, "Y")
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=rot, verts=bm.verts[:])
    elif axis == "Y":
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, "X")
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=rot, verts=bm.verts[:])
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    _argus_finalize(mesh, bm, bevel_edges=bevel > 0.0, bevel_off=bevel)
    return _argus_link(name, mesh, mat, root)


def argus_sphere(name, loc, scale, mat, root, *, u=24, v=14):
    """Ellipsoid via scaled UV sphere — cushions, knobs, fruit, soft pads."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=0.5)
    bmesh.ops.scale(bm, vec=mathutils.Vector((scale[0], scale[1], scale[2])),
                    verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    # Cylindrical wrap; only the pole fans fall back to planar (cap_limit 0.95)
    _argus_uv_cyl(bm, 0.25 * (scale[0] + scale[1]), cap_limit=0.95)
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    _argus_finalize(mesh, bm, bevel_edges=False)
    return _argus_link(name, mesh, mat, root)


def argus_wheel(name, center, radius, width, mat, root, *, axis="X", segments=32):
    """Upright wheel/tyre: circular face in a VERTICAL plane, axle horizontal.
    Use axis='X' for left/right axle, 'Y' for front/back axle."""
    return argus_cylinder(name, center, radius, width, mat, root,
                          axis=axis, segments=segments, bevel=0.01)


def argus_bolt(name, loc, radius, height, mat, root, *, axis="Z", segments=6):
    """Hex bolt head — low-poly, for greebling/detail rows."""
    return argus_cylinder(name, loc, radius, height, mat, root,
                          axis=axis, segments=segments, bevel=0.0)


def argus_panel(name, loc, size, mat, root, *, inset=0.05, depth=0.012):
    """Flat detail panel with a recessed inset face — breaks up large surfaces."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=mathutils.Vector((size[0], size[1], max(size[2], 0.02))),
                    verts=bm.verts[:], space=mathutils.Matrix.Identity(4))
    top = max(bm.faces, key=lambda f: f.calc_center_median().z)
    try:
        res = bmesh.ops.inset_region(bm, faces=[top], thickness=inset, depth=-depth)
    except Exception:
        res = None
    _argus_uv_box(bm)
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    _argus_finalize(mesh, bm, bevel_off=0.006)
    return _argus_link(name, mesh, mat, root)


def argus_lathe(name, loc, profile, mat, root, *, segments=32):
    """Surface of revolution around local +Z — hydrants, barrels, bottles,
    lamp posts, vases. profile = [(radius, z), ...] bottom to top; a radius
    of 0 closes the shape at that point. A profile whose last point repeats
    its first spins into a ring/annulus (no end caps) — for rims, lips,
    collars, mouldings. loc = where profile z=0 sits."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    prof = [(max(0.0, float(r)), float(z)) for r, z in profile]
    closed = (len(prof) > 2
              and abs(prof[0][0] - prof[-1][0]) < 1e-6
              and abs(prof[0][1] - prof[-1][1]) < 1e-6)
    rings = []
    for r, z in prof:
        if r <= 1e-6:
            rings.append([bm.verts.new((0.0, 0.0, z))])
            continue
        ring = []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
        rings.append(ring)
    for ring_a, ring_b in zip(rings, rings[1:]):
        if len(ring_a) == 1 and len(ring_b) == 1:
            continue
        if len(ring_a) == 1:
            apex = ring_a[0]
            for i in range(segments):
                bm.faces.new((apex, ring_b[i], ring_b[(i + 1) % segments]))
        elif len(ring_b) == 1:
            apex = ring_b[0]
            for i in range(segments):
                bm.faces.new((ring_a[(i + 1) % segments], ring_a[i], apex))
        else:
            for i in range(segments):
                j = (i + 1) % segments
                bm.faces.new((ring_a[i], ring_a[j], ring_b[j], ring_b[i]))
    if not closed:
        if len(rings[0]) > 1:
            _argus_fan_cap(bm, rings[0])
        if len(rings[-1]) > 1:
            _argus_fan_cap(bm, rings[-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-6)
    _avg_r = sum(p[0] for p in prof) / max(len(prof), 1)
    _argus_uv_cyl(bm, max(_avg_r, 1e-3))
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    _argus_finalize(mesh, bm, bevel_edges=False)
    return _argus_link(name, mesh, mat, root)


def argus_tube(name, loc, path, radius, mat, root, *, segments=12):
    """Polyline swept with a circular cross-section — hoses, pipes, cables,
    curved handles. path = [(x,y,z), ...] relative to loc. Frames are built
    per-point from averaged directions (fine for gentle prop-scale curves)."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    pts = [mathutils.Vector(p) for p in path]
    if len(pts) >= 3:
        # Planner paths are sparse waypoints; sweeping them directly gives
        # sharp elbows (an angular "C" for a mug handle). Resample through a
        # Catmull-Rom spline so multi-point tubes bend smoothly while still
        # passing through every waypoint.
        ctrl = [pts[0]] + pts + [pts[-1]]
        smooth = [pts[0]]
        for i in range(1, len(ctrl) - 2):
            p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2]
            for s in range(1, 7):
                t = s / 6.0
                smooth.append(
                    ((p1 * 2.0)
                     + (p2 - p0) * t
                     + (p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3) * (t * t)
                     + (p1 * 3.0 - p2 * 3.0 + p0 - p3) * (t * t * t)) * 0.5)
        pts = smooth
    n = len(pts)
    rings = []
    vert_meta = {}
    cum = 0.0
    for idx, p in enumerate(pts):
        if idx > 0:
            cum += (pts[idx] - pts[idx - 1]).length
        if idx == 0:
            d = pts[1] - pts[0]
        elif idx == n - 1:
            d = pts[-1] - pts[-2]
        else:
            d = pts[idx + 1] - pts[idx - 1]
        if d.length < 1e-9:
            d = mathutils.Vector((0.0, 0.0, 1.0))
        d.normalize()
        up = mathutils.Vector((0.0, 0.0, 1.0)) if abs(d.z) < 0.95 \
            else mathutils.Vector((1.0, 0.0, 0.0))
        x = d.cross(up)
        x.normalize()
        y = d.cross(x)
        y.normalize()
        ring = []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            v = bm.verts.new(
                p + x * (radius * math.cos(a)) + y * (radius * math.sin(a)))
            vert_meta[v] = (cum, i)
            ring.append(v)
        rings.append(ring)
    for ring_a, ring_b in zip(rings, rings[1:]):
        for i in range(segments):
            j = (i + 1) % segments
            bm.faces.new((ring_a[i], ring_a[j], ring_b[j], ring_b[i]))
    _argus_fan_cap(bm, rings[0])
    _argus_fan_cap(bm, rings[-1])
    # UVs: u = arc length around the ring (seam-corrected), v = distance along
    # the path — so a texture flows naturally down a hose. Caps (which contain
    # the fan-centre verts) fall back to box projection.
    uv = _argus_uv_layer(bm)
    bm.normal_update()
    _step = (2.0 * math.pi * radius) / segments
    for f in bm.faces:
        metas = [vert_meta.get(l.vert) for l in f.loops]
        if all(m is not None for m in metas):
            ss = [m[1] for m in metas]
            wrap = (0 in ss) and (segments - 1 in ss)
            for loop, m in zip(f.loops, metas):
                s = m[1] + (segments if (wrap and m[1] == 0) else 0)
                loop[uv].uv = (s * _step, m[0])
        else:
            _argus_uv_box(bm, faces=[f])
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    _argus_finalize(mesh, bm, bevel_edges=False)
    return _argus_link(name, mesh, mat, root)


def argus_ring(name, loc, radius, thickness, mat, root, *, axis="Z",
               segments=32, ring_segments=12):
    """Torus — bands, rims, hoops, round handles. radius = major (centreline),
    thickness = minor (cross-section) radius. axis = hole direction."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    rings = []
    vert_meta = {}
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        ca, sa = math.cos(a), math.sin(a)
        ring = []
        for k in range(ring_segments):
            b = 2.0 * math.pi * k / ring_segments
            rr = radius + thickness * math.cos(b)
            v = bm.verts.new((rr * ca, rr * sa, thickness * math.sin(b)))
            vert_meta[v] = (i, k)
            ring.append(v)
        rings.append(ring)
    for i in range(segments):
        ring_a = rings[i]
        ring_b = rings[(i + 1) % segments]
        for k in range(ring_segments):
            kk = (k + 1) % ring_segments
            bm.faces.new((ring_a[k], ring_a[kk], ring_b[kk], ring_b[k]))
    # UVs: u = major arc length, v = minor arc length; both axes seam-wrap.
    uv = _argus_uv_layer(bm)
    _mu = (2.0 * math.pi * radius) / segments
    _mv = (2.0 * math.pi * thickness) / ring_segments
    for f in bm.faces:
        metas = [vert_meta[l.vert] for l in f.loops]
        is_ = [m[0] for m in metas]
        ks_ = [m[1] for m in metas]
        wrap_i = (0 in is_) and (segments - 1 in is_)
        wrap_k = (0 in ks_) and (ring_segments - 1 in ks_)
        for loop, (i_, k_) in zip(f.loops, metas):
            iu = i_ + (segments if (wrap_i and i_ == 0) else 0)
            kv = k_ + (ring_segments if (wrap_k and k_ == 0) else 0)
            loop[uv].uv = (iu * _mu, kv * _mv)
    if axis == "X":
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, "Y")
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=rot, verts=bm.verts[:])
    elif axis == "Y":
        rot = mathutils.Matrix.Rotation(math.radians(90), 4, "X")
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=rot, verts=bm.verts[:])
    bmesh.ops.translate(bm, vec=mathutils.Vector(loc), verts=bm.verts[:])
    _argus_finalize(mesh, bm, bevel_edges=False)
    return _argus_link(name, mesh, mat, root)


def argus_leaf_card(name, loc, length, width, mat, root, *, curve=0.12, yaw=0.0, pitch=0.0):
    """Tapered double-sided leaf blade with a centre ridge — for foliage."""
    mesh = bpy.data.meshes.new(name)
    L, W = length, width
    verts = [
        (0.0, 0.0, 0.0), (0.0, L * 0.5, curve * 0.4 * L), (0.0, L, 0.0),
        (-W * 0.5, L * 0.4, 0.0), (W * 0.5, L * 0.4, 0.0),
        (-W * 0.3, L * 0.75, 0.0), (W * 0.3, L * 0.75, 0.0),
    ]
    faces = [(0, 3, 1), (0, 1, 4), (3, 5, 1), (1, 5, 2), (1, 2, 6), (4, 1, 6)]
    mesh.from_pydata([list(v) for v in verts], [], [list(f) for f in faces])
    mesh.update()
    for _p in mesh.polygons:
        _p.use_smooth = True
    _uvl = mesh.uv_layers.get("ARGUS_UV") or mesh.uv_layers.new(name="ARGUS_UV")
    for _lp in mesh.loops:
        _co = mesh.vertices[_lp.vertex_index].co
        _uvl.data[_lp.index].uv = (_co.x + W * 0.5, _co.y)
    obj = _argus_link(name, mesh, mat, root)
    obj.rotation_euler = (math.radians(pitch), 0.0, math.radians(yaw))
    obj.location = mathutils.Vector(loc)
    return obj


def argus_smooth(obj, levels=2, crease=0.0):
    """Subdivision-surface smoothing: coarse cage → soft rounded form.
    crease 0 = fully soft (cushions/pillows); near 1 = edges stay sharp.
    Bottom-face edges are always pinned sharp (crease=1) so that face stays
    perfectly flat at its original height — the part keeps a flush base on
    whatever it rests on, and only the top/sides round into a cushion shape.
    The export wrapper applies the modifier before its finishing passes."""
    try:
        me = obj.data
        if me is not None and len(me.edges):
            attr = me.attributes.get("crease_edge")
            if attr is None:
                attr = me.attributes.new("crease_edge", 'FLOAT', 'EDGE')
            val = min(1.0, max(0.0, float(crease)))
            min_z = min((v.co.z for v in me.vertices), default=0.0)
            values = []
            for e in me.edges:
                v0, v1 = e.vertices
                on_bottom = (me.vertices[v0].co.z <= min_z + 1e-6
                              and me.vertices[v1].co.z <= min_z + 1e-6)
                values.append(1.0 if on_bottom else val)
            attr.data.foreach_set("value", values)
        mod = obj.modifiers.new("ARGUS_SUBSURF", 'SUBSURF')
        mod.levels = int(levels)
        mod.render_levels = int(levels)
    except Exception as exc:
        print("[ARGUS smooth skip] " + str(exc))
    return obj


def argus_fuse(names, voxel=0.0, smooth_iters=10):
    """SDF-style smooth union: join the named objects and voxel-remesh them
    into ONE watertight surface, then relax it so the junctions blend with a
    natural fillet — a handle fuses into a mug like pulled clay instead of
    butt-jointing with a seam. Shape-agnostic: whatever geometry the parts
    have, only the join is changed. First name keeps the object identity,
    materials collapse to the first part's material."""
    objs = [bpy.data.objects[n] for n in names
            if n in bpy.data.objects and bpy.data.objects[n].type == "MESH"]
    if len(objs) < 2:
        return objs[0] if objs else None
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        # subsurf cages must become real before remeshing
        for o in objs:
            for m in list(o.modifiers):
                try:
                    bpy.context.view_layer.objects.active = o
                    bpy.ops.object.modifier_apply(modifier=m.name)
                except Exception:
                    o.modifiers.remove(m)
        # Voxel size must resolve the THINNEST member or it dissolves into a
        # blob — a 5 mm handle needs ~1 mm voxels even on a 10 cm body.
        if voxel <= 0.0:
            feature = min(
                min((d for d in o.dimensions if d > 1e-5), default=0.02)
                for o in objs)
            voxel = max(feature / 6.0, 0.0008)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
        voxel = max(voxel, max(obj.dimensions) / 320.0)  # poly-count ceiling
        obj.data.remesh_voxel_size = voxel
        obj.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
        # gentle relax: fillets the junctions without eroding thin parts
        mod = obj.modifiers.new("ARGUS_BLEND", "SMOOTH")
        mod.factor = 0.5
        mod.iterations = min(int(smooth_iters), 5)
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.ops.object.shade_smooth()
        # remesh discards UVs — re-project so textures still map
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=1.15)
        bpy.ops.object.mode_set(mode="OBJECT")
        return obj
    except Exception as exc:
        print("[ARGUS fuse skip] " + str(exc))
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass
        return objs[0]
# ── end ARGUS parametric component library ────────────────────────────────
'''


def component_prompt_block() -> str:
    """Short prompt section offering the helpers to the code-gen model."""
    return (
        "PARAMETRIC COMPONENT HELPERS (recommended)\n\n"
        "You MAY define and reuse these canonical helper functions instead of "
        "re-deriving primitive maths for every part. Each returns a linked, "
        "parented, smooth-shaded, UV-mapped mesh object:\n"
        "  argus_box(name, loc, (sx,sy,sz), mat, root, bevel=0.012)\n"
        "  argus_cylinder(name, loc, radius, depth, mat, root, axis='Z', segments=32, r_top=None, bevel=0.0)\n"
        "  argus_sphere(name, loc, (sx,sy,sz), mat, root)            # cushions/knobs/fruit\n"
        "  argus_wheel(name, center, radius, width, mat, root, axis='X')  # UPRIGHT wheel, horizontal axle\n"
        "  argus_bolt(name, loc, radius, height, mat, root, axis='Z')     # hex detail\n"
        "  argus_panel(name, loc, (sx,sy,sz), mat, root, inset=0.05)      # recessed surface panel\n"
        "  argus_lathe(name, loc, [(r,z),...], mat, root, segments=32)    # revolve profile: hydrants/barrels/bottles/posts\n"
        "  argus_tube(name, loc, [(x,y,z),...], radius, mat, root)        # swept hose/pipe/cable/handle\n"
        "  argus_ring(name, loc, radius, thickness, mat, root, axis='Z')  # torus band/rim/hoop\n"
        "  argus_leaf_card(name, loc, length, width, mat, root, yaw=0, pitch=0)  # foliage blade\n\n"
        "If you use any helper you MUST paste its definition (and the shared "
        "_argus_finalize/_argus_link helpers) into the script — the runtime has "
        "no local imports. Prefer argus_wheel for any tyre/wheel/handwheel so it "
        "is automatically upright with a horizontal axle.\n"
    )
