"""
GPU-accelerated mesh renderer — ModernGL offscreen context.

Prefers the NVIDIA discrete GPU on Optimus laptops via:
  1. Loading nvapi64.dll (tells Windows/NVIDIA this process wants the dGPU)
  2. Creating a pyglet window with OpenGL 4.6 core profile (forces discrete GPU
     selection on most Optimus systems)

Visual style exactly matches the CPU painter:
  - SOLID  : flat Lambert shading with the same 3-point light rig
  - WIRE   : barycentric-coordinates wireframe, cyan-on-dark (matches Blender)

Usage:
    renderer = GLRenderer.create()          # returns None if GPU init fails
    if renderer:
        print(renderer.gpu_name)            # "NVIDIA GeForce RTX 2050 ..."
        renderer.upload_mesh(V, F, C, N)    # once per model
        raw = renderer.render_frame(yaw, pitch, zoom, w, h, mode, pan_x, pan_y)
        # raw is RGBA bytes, OpenGL bottom-up order — flip vertically before display
"""
from __future__ import annotations

import math
import sys
from typing import List, Optional

import numpy as np

try:
    import moderngl
    _MODERNGL_OK = True
except ImportError:
    _MODERNGL_OK = False


_PYGLET_WINDOWS: List = []


_SOLID_VERT = """
#version 330 core
in vec3 in_vert;
in vec3 in_norm;
in vec3 in_color;
uniform mat4 u_mvp;
uniform mat3 u_rot;
out vec3 v_norm;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(in_vert, 1.0);
    v_norm  = normalize(u_rot * in_norm);
    v_color = in_color;
}
"""

_SOLID_FRAG = """
#version 330 core
in  vec3 v_norm;
in  vec3 v_color;
uniform float u_ambient;
uniform bool  u_use_flat;
uniform vec3  u_flat_color;
out vec4 f_color;
void main() {
    vec3 key  = normalize(vec3( 0.45, -0.70,  0.55));
    vec3 fill = normalize(vec3(-0.60,  0.30,  0.20));
    vec3 rim  = normalize(vec3( 0.10,  0.80, -0.20));
    float d = max(dot(v_norm, key),  0.0) * 0.65
            + max(dot(v_norm, fill), 0.0) * 0.25
            + max(dot(v_norm, rim),  0.0) * 0.15
            + u_ambient;
    vec3 base = u_use_flat ? u_flat_color : v_color;
    f_color = vec4(clamp(base * d, 0.0, 1.0), 1.0);
}
"""


_FLOOR_VERT = """
#version 330 core
in vec3 in_vert;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_vert, 1.0);
}
"""

_FLOOR_FRAG = """
#version 330 core
uniform vec3 u_color;
out vec4 f_color;
void main() {
    f_color = vec4(u_color, 1.0);
}
"""


_WIRE_VERT = """
#version 330 core
in vec3 in_vert;
in vec3 in_bary;
in vec3 in_evis;
uniform mat4 u_mvp;
out vec3 v_bary;
out vec3 v_evis;
void main() {
    gl_Position = u_mvp * vec4(in_vert, 1.0);
    v_bary = in_bary;
    v_evis = in_evis;
}
"""

_WIRE_FRAG = """
#version 330 core
in  vec3 v_bary;
in  vec3 v_evis;
out vec4 f_color;
void main() {
    // Replace hidden edges (evis~0) with 1.0 so they never trigger the draw
    float bx = v_evis.x > 0.5 ? v_bary.x : 1.0;
    float by = v_evis.y > 0.5 ? v_bary.y : 1.0;
    float bz = v_evis.z > 0.5 ? v_bary.z : 1.0;
    float edge = min(bx, min(by, bz));

    // fwidth() can return ~0 in offscreen FBO contexts.
    // Use a fixed threshold (0.03 = 3% of bary range) as the base so
    // edges are always visible regardless of triangle screen size.
    // The fwidth term adds anti-aliasing on top when it works correctly.
    float fw = fwidth(edge);
    float w  = (fw > 0.0001) ? max(fw * 2.0, 0.025) : 0.035;
    float a  = smoothstep(0.0, w, edge);

    // Bright cyan wire on very dark background — matches Blender wireframe
    vec3 wire = vec3(0.10, 0.88, 1.00);
    vec3 bg   = vec3(0.012, 0.016, 0.024);
    f_color   = vec4(mix(wire, bg, a), 1.0);
}
"""


def _create_context() -> "moderngl.Context":
    """
    Create a ModernGL context, strongly preferring the NVIDIA discrete GPU.

    On Windows Optimus (integrated Intel + discrete NVIDIA) the default
    OpenGL context goes to Intel unless the process signals it wants the dGPU.
    Two complementary hints achieve reliable NVIDIA selection:

      1.  Loading nvapi64.dll before context creation tells the NVIDIA driver
          that this process is GPU-aware.
      2.  Requesting an OpenGL 4.6 core-profile context via pyglet triggers
          the Windows GPU arbiter to assign the discrete GPU.
    """
    if sys.platform == "win32":
        import ctypes
        for _lib in ("nvapi64.dll", "nvapi.dll"):
            try:
                ctypes.cdll.LoadLibrary(_lib)
                break
            except OSError:
                pass

        try:
            import pyglet
            from pyglet.gl import Config
            for _major, _minor in ((4, 6), (4, 5), (4, 1), (3, 3)):
                try:
                    _cfg = Config(
                        major_version=_major,
                        minor_version=_minor,
                        depth_size=24,
                        forward_compatible=True,
                    )
                    _win = pyglet.window.Window(1, 1, visible=False, config=_cfg)
                    _PYGLET_WINDOWS.append(_win)
                    return moderngl.create_context()
                except Exception:
                    continue
        except Exception:
            pass

    return moderngl.create_standalone_context(require=330)


def _build_mvp(
    yaw: float,
    pitch: float,
    zoom: float,
    w: int,
    h: int,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mvp_col16, rot3_col9) in column-major float32 ready for GLSL.

    The projection exactly matches the CPU painter:
        screen_x = (w/2 + pan_x) + rx   * scale
        screen_y = (h/2 + pan_y) - rz2  * scale
    where  scale = min(w, h) * 1.55 * zoom
    and    (rx, ry2, rz2) = R_pitch(pitch) @ R_yaw(yaw) @ vertex.

    The FBO is OpenGL bottom-up, so the caller must flip the image vertically
    before display — which exactly inverts the y-axis and matches the CPU path.
    """
    scale = min(w, h) * 1.55 * zoom
    hw = w / (2.0 * scale)
    hh = h / (2.0 * scale)

    yc = math.cos(yaw);  ys = math.sin(yaw)
    pc = math.cos(pitch); ps = math.sin(pitch)

    M = np.array([
        [ yc,       -ys,       0.0,  0.0],
        [ ys * pc,   yc * pc, -ps,   0.0],
        [ ys * ps,   yc * ps,  pc,   0.0],
        [ 0.0,       0.0,      0.0,  1.0],
    ], dtype=np.float32)

    T = np.array([
        [1.0/hw, 0.0,     0.0,      0.0],
        [0.0,    0.0,     1.0/hh,   0.0],
        [0.0,    0.1,     0.0,      0.0],
        [0.0,    0.0,     0.0,      1.0],
    ], dtype=np.float32)

    MVP = T @ M

    MVP[0, 3] += 2.0 * pan_x / w
    MVP[1, 3] -= 2.0 * pan_y / h

    rot3 = M[:3, :3]

    return np.ascontiguousarray(MVP.T), np.ascontiguousarray(rot3.T)


class GLRenderer:
    """
    Off-screen GPU renderer using ModernGL.

    All rasterisation and lighting run on the GPU; only the final RGBA pixels
    are transferred back to the CPU for display in the tkinter canvas.
    """

    def __init__(self) -> None:
        if not _MODERNGL_OK:
            raise RuntimeError("moderngl not installed — run: pip install moderngl")

        import moderngl as mgl
        self._mgl = mgl
        self.ctx: moderngl.Context = _create_context()

        self._solid_prog: moderngl.Program = self.ctx.program(
            vertex_shader=_SOLID_VERT,
            fragment_shader=_SOLID_FRAG,
        )
        self._wire_prog: moderngl.Program = self.ctx.program(
            vertex_shader=_WIRE_VERT,
            fragment_shader=_WIRE_FRAG,
        )
        self._solid_vao: Optional[moderngl.VertexArray] = None
        self._wire_vao:  Optional[moderngl.VertexArray] = None
        self._n_verts:   int = 0
        self._fbo:       Optional[moderngl.Framebuffer] = None
        self._fbo_size:  tuple[int, int] = (0, 0)

        self._floor_prog: moderngl.Program = self.ctx.program(
            vertex_shader=_FLOOR_VERT,
            fragment_shader=_FLOOR_FRAG,
        )
        self._floor_vao:   Optional[moderngl.VertexArray] = None
        self._fline_vao:   Optional[moderngl.VertexArray] = None
        self._fline_count: int = 0
        self._xaxis_vao:   Optional[moderngl.VertexArray] = None
        self._yaxis_vao:   Optional[moderngl.VertexArray] = None

        info = self.ctx.info
        self._gpu_renderer: str = info.get("GL_RENDERER", "Unknown GPU")
        self._gpu_vendor:   str = info.get("GL_VENDOR",   "Unknown")


    @classmethod
    def create(cls) -> Optional["GLRenderer"]:
        """Return a GLRenderer, or None if GPU context creation fails."""
        try:
            return cls()
        except Exception:
            return None


    @property
    def gpu_name(self) -> str:
        return self._gpu_renderer

    @property
    def gpu_vendor(self) -> str:
        return self._gpu_vendor


    def _ensure_fbo(self, width: int, height: int) -> None:
        if self._fbo is not None and self._fbo_size == (width, height):
            return
        if self._fbo is not None:
            self._fbo.release()
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)],
            depth_attachment=self.ctx.depth_renderbuffer((width, height)),
        )
        self._fbo_size = (width, height)


    def upload_floor(self, floor_z: float, half_size: float = 1.1) -> None:
        """
        Upload a rotating floor plane + grid at the given world Z.
        Call once after upload_mesh with the model's minimum Z coordinate.
        """
        if self._floor_vao is not None:
            self._floor_vao.release()
        if self._fline_vao is not None:
            self._fline_vao.release()

        S = half_size
        fz = float(floor_z)

        floor_pts = np.array([
            [-S, -S, fz],  [ S, -S, fz],  [ S,  S, fz],
            [-S, -S, fz],  [ S,  S, fz],  [-S,  S, fz],
        ], dtype=np.float32)
        fvbo = self.ctx.buffer(floor_pts.tobytes())
        self._floor_vao = self.ctx.vertex_array(
            self._floor_prog, [(fvbo, "3f", "in_vert")]
        )

        STEP = 0.2
        n_steps = int(S / STEP) + 1
        segs: list = []
        for i in range(-n_steps, n_steps + 1):
            t = i * STEP
            if abs(t) > S + 1e-4:
                continue
            if abs(t) > 1e-4:
                segs += [[t, -S, fz], [t, S, fz]]
                segs += [[-S, t, fz], [S, t, fz]]
        line_arr = np.array(segs, dtype=np.float32)
        lvbo = self.ctx.buffer(line_arr.tobytes())
        self._fline_vao = self.ctx.vertex_array(
            self._floor_prog, [(lvbo, "3f", "in_vert")]
        )
        self._fline_count = len(line_arr)

        if self._xaxis_vao is not None:
            self._xaxis_vao.release()
        if self._yaxis_vao is not None:
            self._yaxis_vao.release()
        x_arr = np.array([[-S, 0, fz], [S, 0, fz]], dtype=np.float32)
        y_arr = np.array([[0, -S, fz], [0, S, fz]], dtype=np.float32)
        xvbo = self.ctx.buffer(x_arr.tobytes())
        yvbo = self.ctx.buffer(y_arr.tobytes())
        self._xaxis_vao = self.ctx.vertex_array(self._floor_prog, [(xvbo, "3f", "in_vert")])
        self._yaxis_vao = self.ctx.vertex_array(self._floor_prog, [(yvbo, "3f", "in_vert")])

    def upload_mesh(
        self,
        vertices:  np.ndarray,
        faces:     np.ndarray,
        colors:    np.ndarray,
        normals:   np.ndarray,
        edge_vis:  Optional[np.ndarray] = None,
    ) -> None:
        """
        Upload mesh geometry to GPU VBOs.

        The mesh is "exploded" (no shared vertices) so every triangle carries
        its own flat-shaded normal, color, barycentric coordinates, and
        per-edge visibility flags.

        edge_vis[f, k] = 1  → draw the edge where bary[k]≈0 (real feature edge)
        edge_vis[f, k] = 0  → hide it (flat triangulation diagonal, dihedral≈0)
        Pass None to show every edge (wireframe shows all triangulation).
        """
        if self._solid_vao is not None:
            self._solid_vao.release()
            self._wire_vao.release()

        F = len(faces)
        self._n_verts = F * 3

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        if edge_vis is None:
            edge_vis = np.ones((F, 3), dtype=np.float32)

        solid = np.empty((F * 3, 9), dtype=np.float32)
        solid[0::3, 0:3] = v0;       solid[1::3, 0:3] = v1;       solid[2::3, 0:3] = v2
        solid[0::3, 3:6] = normals;  solid[1::3, 3:6] = normals;  solid[2::3, 3:6] = normals
        solid[0::3, 6:9] = colors;   solid[1::3, 6:9] = colors;   solid[2::3, 6:9] = colors

        BARY = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        wire = np.empty((F * 3, 9), dtype=np.float32)
        wire[0::3, 0:3] = v0;  wire[1::3, 0:3] = v1;  wire[2::3, 0:3] = v2
        wire[0::3, 3:6] = BARY[0];  wire[1::3, 3:6] = BARY[1];  wire[2::3, 3:6] = BARY[2]
        wire[0::3, 6:9] = edge_vis
        wire[1::3, 6:9] = edge_vis
        wire[2::3, 6:9] = edge_vis

        svbo = self.ctx.buffer(solid.tobytes())
        self._solid_vao = self.ctx.vertex_array(
            self._solid_prog,
            [(svbo, "3f 3f 3f", "in_vert", "in_norm", "in_color")],
        )
        wvbo = self.ctx.buffer(wire.tobytes())
        self._wire_vao = self.ctx.vertex_array(
            self._wire_prog,
            [(wvbo, "3f 3f 3f", "in_vert", "in_bary", "in_evis")],
        )

    def render_frame(
        self,
        yaw:          float,
        pitch:        float,
        zoom:         float,
        width:        int,
        height:       int,
        mode:         str   = "solid",
        pan_x:        float = 0.0,
        pan_y:        float = 0.0,
        ambient:      float = 0.20,
        bg_brightness: float = 1.0,
    ) -> Optional[bytes]:
        """
        Render the current mesh and return raw RGBA bytes (bottom-up, OpenGL order).

        The caller must flip the image vertically before displaying — PIL:
            img = Image.fromarray(np.frombuffer(raw, uint8).reshape(h,w,4)[::-1])

        Returns None if no mesh has been uploaded yet.
        """
        if self._solid_vao is None:
            return None

        self._ensure_fbo(width, height)
        self._fbo.use()
        self.ctx.viewport = (0, 0, width, height)

        mvp, rot3 = _build_mvp(yaw, pitch, zoom, width, height, pan_x, pan_y)

        mvp_bytes = mvp.tobytes()

        bb = max(0.0, min(1.0, bg_brightness))
        amb = max(0.0, min(1.0, ambient))

        self.ctx.clear(0.012 * bb, 0.016 * bb, 0.024 * bb, 1.0)
        self.ctx.enable(self._mgl.DEPTH_TEST)

        self.ctx.depth_func = '<'
        if self._fline_vao is not None:
            self._floor_prog["u_mvp"].write(mvp_bytes)
            self._floor_prog["u_color"].value = (0.040 * bb, 0.068 * bb, 0.095 * bb)
            self._fline_vao.render(self._mgl.LINES, vertices=self._fline_count)
        if self._xaxis_vao is not None:
            self._floor_prog["u_color"].value = (0.42 * bb, 0.08 * bb, 0.08 * bb)
            self._xaxis_vao.render(self._mgl.LINES)
        if self._yaxis_vao is not None:
            self._floor_prog["u_color"].value = (0.08 * bb, 0.28 * bb, 0.08 * bb)
            self._yaxis_vao.render(self._mgl.LINES)

        self._solid_prog["u_mvp"].write(mvp_bytes)
        self._solid_prog["u_rot"].write(rot3.tobytes())
        self._solid_prog["u_ambient"].value = 0.0
        self._solid_prog["u_use_flat"].value = True
        self._solid_prog["u_flat_color"].value = (0.012 * bb, 0.016 * bb, 0.024 * bb)
        self._solid_vao.render(self._mgl.TRIANGLES)
        self._solid_prog["u_use_flat"].value = False

        self.ctx.depth_func = '<='
        self._wire_prog["u_mvp"].write(mvp_bytes)
        self._wire_vao.render(self._mgl.TRIANGLES)
        self.ctx.depth_func = '<'

        return self._fbo.read(components=4)

    def release(self) -> None:
        """Free all GPU resources."""
        if self._solid_vao:
            self._solid_vao.release()
        if self._wire_vao:
            self._wire_vao.release()
        if self._floor_vao:
            self._floor_vao.release()
        if self._fline_vao:
            self._fline_vao.release()
        if self._xaxis_vao:
            self._xaxis_vao.release()
        if self._yaxis_vao:
            self._yaxis_vao.release()
        if self._fbo:
            self._fbo.release()
        self._solid_prog.release()
        self._wire_prog.release()
        self._floor_prog.release()
        self.ctx.release()
