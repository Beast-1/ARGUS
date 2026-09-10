
from __future__ import annotations

import json
import logging
import os
import secrets
import socket

from dataclasses import dataclass, asdict, field
from enum import Enum

from core.script_safety import check_script_safety

logger = logging.getLogger("ARGUS.MCP")

MCP_HOST = os.getenv("ARGUS_MCP_HOST", os.getenv("BLENDER_HOST", "127.0.0.1"))
MCP_PORT = int(os.getenv("ARGUS_MCP_PORT", "9876"))
# BLENDER_MCP_TIMEOUT is honoured as a legacy alias, but it must not carry its
# own default here: core/blender.py:30 already defaults that same variable to
# 120 for the headless topology pass, so declaring 180 for it here meant one
# unset variable silently meant two different things depending on which path
# ran. Exactly one literal default (180, this path's own) now.
MCP_TIMEOUT = int(os.getenv("ARGUS_MCP_TIMEOUT") or os.getenv("BLENDER_MCP_TIMEOUT") or "180")

# The live server does `exec()` on whatever it receives over this socket, so it must
# never accept a payload without proof the sender is this same ARGUS process (or one
# that was told the token out of band). setdefault() both generates a fresh token
# once per process AND writes it into os.environ, so `ensure_mcp_server()`'s
# `env=dict(os.environ)` Popen call below hands it to the launched Blender process
# automatically — no separate plumbing needed. Set ARGUS_MCP_TOKEN yourself to pin a
# token (e.g. to talk to an already-running third-party BlenderMCP-compatible addon
# that was configured with the same value).
MCP_TOKEN = os.environ.setdefault("ARGUS_MCP_TOKEN", secrets.token_hex(32))


class McpRepairClass(str, Enum):

    CLEAN = "clean"

    NGON_ERROR = "ngon_error"
    OPEN_EDGE_ERROR = "open_edge_error"
    MANIFOLD_ERROR = "manifold_error"

    EMPTY_SCENE = "empty_scene"

    UNKNOWN = "unknown"


@dataclass
class McpValidationResult:

    success: bool

    repair_class: McpRepairClass

    mesh_count: int = 0
    object_count: int = 0

    triangle_count: int = 0

    tri_faces: int = 0
    quad_faces: int = 0
    ngon_faces: int = 0

    # Same field names the headless report and the structure-memory gate use —
    # previously these were missing here, so the gate silently read zeros.
    ngon_count: int = 0
    non_manifold_faces: int = 0
    isolated_verts: int = 0

    open_edges: int = 0
    manifold_errors: int = 0

    severity: str = "clean"
    disconnected_components: int = 0
    floating_components: int = 0

    grounded: bool = True

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    raw_response: str = ""

    @property
    def topology_clean(self):

        return (
            self.ngon_count == 0
            and self.open_edges == 0
            and self.manifold_errors == 0
            and self.isolated_verts == 0
            and self.disconnected_components <= 1
        )

    def as_dict(self):

        d = asdict(self)

        d["repair_class"] = self.repair_class.value

        return d


class McpClient:

    CONNECT_TIMEOUT = 5  # connect fast or fail fast; reads use MCP_TIMEOUT

    def send(self, payload):

        raw = json.dumps(payload)

        last_exc: Exception | None = None
        for _attempt in (1, 2):
            try:
                with socket.create_connection(
                    (MCP_HOST, MCP_PORT),
                    timeout=self.CONNECT_TIMEOUT,
                ) as sock:
                    sock.sendall(raw.encode("utf-8"))
                    response = self._receive_json(sock)
                break
            except (ConnectionError, socket.timeout, OSError) as exc:
                last_exc = exc
                continue
            except Exception as exc:  # noqa: BLE001
                return {"success": False, "error": str(exc)}
        else:
            return {"success": False, "error": str(last_exc)}

        try:

            return json.loads(
                response.decode("utf-8")
            )

        except Exception:

            return {
                "success": False,
                "error": "invalid_json",
            }

    def _receive_json(self, sock) -> bytes:
        chunks = []
        sock.settimeout(MCP_TIMEOUT)

        while True:
            chunk = sock.recv(8192)
            if not chunk:
                if chunks:
                    return b"".join(chunks)
                raise ConnectionError("connection closed before response")

            chunks.append(chunk)
            data = b"".join(chunks)

            try:
                json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            return data


def _normalize_addon_response(response: dict) -> dict:
    if not isinstance(response, dict):
        return {
            "success": False,
            "error": "invalid_response",
        }

    if response.get("success") is not None:
        return response

    status = response.get("status")

    if status == "success":
        result = response.get("result", {})
        stdout = ""

        if isinstance(result, dict):
            stdout = str(result.get("result", ""))
        elif result is not None:
            stdout = str(result)

        return {
            "success": True,
            "stdout": stdout,
            "stderr": "",
            "raw_response": response,
        }

    return {
        "success": False,
        "stdout": "",
        "stderr": response.get("message", ""),
        "error": response.get("message", "mcp_command_failed"),
        "raw_response": response,
    }


def execute_blender_code(code: str):
    """Every caller today already runs generated code through
    core.llm.validate_generated_script (which calls check_script_safety) before
    it gets here — this is defense-in-depth for the live-exec path specifically,
    not the primary gate, so a future caller that skips that upstream validation
    doesn't silently get an unchecked exec() on a persistent GUI process."""
    safe, reason = check_script_safety(code)
    if not safe:
        return {"success": False, "error": f"blocked by script safety check: {reason}"}

    client = McpClient()

    response = client.send({
        "type": "execute_code",
        "token": MCP_TOKEN,
        "params": {
            "code": code,
        },
    })

    return _normalize_addon_response(response)


def ping_mcp_server() -> tuple[bool, str]:
    try:
        with socket.create_connection((MCP_HOST, MCP_PORT), timeout=2):
            return True, f"{MCP_HOST}:{MCP_PORT}"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


# In-Blender TCP server, protocol-compatible with execute_blender_code.
# A socket thread accepts connections and queues requests; a bpy.app.timers
# callback executes them on the MAIN thread (Blender's API is not thread-safe)
# and replies. Launched via `blender --python <this script>` in GUI mode.
_LIVE_SERVER_SCRIPT = r'''
import bpy
import contextlib
import io
import json
import os
import queue
import socket
import threading
import traceback

HOST = os.environ.get("ARGUS_MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("ARGUS_MCP_PORT", "9876"))
TOKEN = os.environ.get("ARGUS_MCP_TOKEN", "")
_q = queue.Queue()


def _client_thread(conn):
    try:
        conn.settimeout(300)
        chunks = []
        payload = None
        while True:
            c = conn.recv(8192)
            if not c:
                break
            chunks.append(c)
            try:
                payload = json.loads(b"".join(chunks).decode("utf-8"))
                break
            except json.JSONDecodeError:
                continue
        if payload is None:        # bare connect = ping; just close
            conn.close()
            return
        _q.put((payload, conn))
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


def _accept_loop(srv):
    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_client_thread, args=(conn,),
                             daemon=True).start()
        except OSError:
            break


def _handle(payload):
    if not TOKEN or payload.get("token") != TOKEN:
        return {"success": False, "error": "unauthorized"}
    if payload.get("type") != "execute_code":
        return {"success": False, "error": "unknown_command"}
    code = payload.get("params", {}).get("code", "")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<argus_mcp>", "exec"), {"__name__": "__main__"})
        return {"success": True, "stdout": buf.getvalue(), "stderr": ""}
    except Exception:
        tb = traceback.format_exc()
        return {"success": False, "stdout": buf.getvalue(),
                "stderr": tb, "error": tb.strip().splitlines()[-1]}


def _process():
    try:
        while True:
            payload, conn = _q.get_nowait()
            resp = _handle(payload)
            try:
                conn.sendall(json.dumps(resp).encode("utf-8"))
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except queue.Empty:
        pass
    return 0.2


_srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
_srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
_srv.bind((HOST, PORT))
_srv.listen(5)
threading.Thread(target=_accept_loop, args=(_srv,), daemon=True).start()
bpy.app.timers.register(_process, persistent=True)
print("[ARGUS MCP] live server listening on %s:%d" % (HOST, PORT))
'''

_launched_proc = None


def ensure_mcp_server(auto_launch: bool = True, timeout: float = 90.0):
    """Ping the live MCP server; if it's down, launch a GUI Blender that runs
    our embedded server and wait for the port to come up.

    Returns (ok, detail). ARGUS_MCP_AUTOLAUNCH=0 disables launching."""
    global _launched_proc
    import subprocess
    import tempfile
    import time
    from pathlib import Path

    ok, detail = ping_mcp_server()
    if ok:
        return True, detail
    if not auto_launch or os.getenv("ARGUS_MCP_AUTOLAUNCH", "1") == "0":
        return False, detail

    blender = os.getenv("BLENDER_PATH", "blender")
    script = Path(tempfile.gettempdir()) / "argus_mcp_server.py"
    script.write_text(_LIVE_SERVER_SCRIPT, encoding="utf-8")
    logger.info("[MCP] launching Blender GUI with live server (%s)", blender)
    try:
        env = dict(os.environ)
        env["ARGUS_MCP_HOST"] = MCP_HOST
        env["ARGUS_MCP_PORT"] = str(MCP_PORT)
        _launched_proc = subprocess.Popen(
            [blender, "--python", str(script)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        return False, f"could not launch Blender: {exc}"

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _launched_proc.poll() is not None:
            return False, f"Blender exited early (code {_launched_proc.returncode})"
        ok, detail = ping_mcp_server()
        if ok:
            return True, f"{detail} (auto-launched)"
        time.sleep(1.0)
    return False, f"server did not come up within {timeout:.0f}s"


_TOPOLOGY_SCRIPT = r'''
import bpy
import bmesh
import json

report = {
    "success": True,
    "mesh_count": 0,
    "object_count": len(bpy.data.objects),

    "triangle_count": 0,

    "tri_faces": 0,
    "quad_faces": 0,
    "ngon_count": 0,

    "open_edges": 0,
    "manifold_errors": 0,
    "non_manifold_faces": 0,
    "isolated_verts": 0,
    "floating_components": 0,

    "grounded": True,

    "warnings": [],
    "errors": [],
}

meshes = [o for o in bpy.data.objects if o.type == "MESH"]

report["mesh_count"] = len(meshes)

if not meshes:
    report["success"] = False
    report["errors"].append("no_meshes")

scene_min_z = None
bounds = []  # (name, x0, x1, y0, y1, z0, z1) world-space AABB per object

for obj in meshes:

    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(obj.data)

    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    for f in bm.faces:

        verts = len(f.verts)

        if verts == 3:
            report["tri_faces"] += 1

        elif verts == 4:
            report["quad_faces"] += 1

        else:
            report["ngon_count"] += 1

        if not f.is_valid:
            report["non_manifold_faces"] += 1

        report["triangle_count"] += max(1, verts - 2)

    for e in bm.edges:
        nf = len(e.link_faces)
        if nf != 2:
            report["manifold_errors"] += 1
            if nf == 0:
                report["open_edges"] += 1

    for v in bm.verts:
        if not v.link_edges:
            report["isolated_verts"] += 1

    # World space — local v.co.z lies once objects carry transforms (jitter,
    # planner rot, parenting all do).
    coords = [mw @ v.co for v in bm.verts]
    if coords:
        xs = [c.x for c in coords]
        ys = [c.y for c in coords]
        zs = [c.z for c in coords]
        bounds.append((obj.name, min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
        obj_min_z = min(zs)
        if scene_min_z is None or obj_min_z < scene_min_z:
            scene_min_z = obj_min_z

    bm.free()

# An object only counts as "floating" if it's well above the ground AND
# nothing else's top surface sits close underneath it with an overlapping
# XY footprint — this catches genuinely detached debris without flagging
# every tabletop/seat/shelf that's correctly resting on legs/supports.
SUPPORT_GAP = 0.02
for i, (name, x0, x1, y0, y1, z0, z1) in enumerate(bounds):
    if z0 <= 0.05:
        continue
    supported = False
    for j, (oname, ox0, ox1, oy0, oy1, oz0, oz1) in enumerate(bounds):
        if i == j:
            continue
        if ox1 < x0 or ox0 > x1 or oy1 < y0 or oy0 > y1:
            continue
        if abs(oz1 - z0) <= SUPPORT_GAP:
            supported = True
            break
    if not supported:
        report["floating_components"] += 1

if scene_min_z is not None and scene_min_z < -0.01:
    report["grounded"] = False
    report["warnings"].append("geometry_below_ground_plane")
if report["floating_components"]:
    report["warnings"].append(
        str(report["floating_components"]) + "_floating_mesh_objects")

if report["isolated_verts"] > 0:
    report["severity"] = "critical"
elif report["manifold_errors"] > 0:
    report["severity"] = "high"
elif report["ngon_count"] > 0:
    report["severity"] = "medium"
else:
    report["severity"] = "clean"
report["success"] = report["success"] and report["isolated_verts"] == 0

print("ARGUS_MCP_REPORT:" + json.dumps(report))
'''


def run_argus_mcp_validation():

    result = execute_blender_code(
        _TOPOLOGY_SCRIPT
    )

    if not result.get("success", False):

        return McpValidationResult(
            success=False,
            repair_class=McpRepairClass.UNKNOWN,
            errors=[result.get("error")],
        )

    stdout = result.get("stdout", "")

    sentinel = "ARGUS_MCP_REPORT:"

    idx = stdout.find(sentinel)

    if idx == -1:

        return McpValidationResult(
            success=False,
            repair_class=McpRepairClass.UNKNOWN,
            errors=["missing_report"],
        )

    try:
        data = json.loads(stdout[idx + len(sentinel):].splitlines()[0])
    except (ValueError, IndexError):
        return McpValidationResult(
            success=False,
            repair_class=McpRepairClass.UNKNOWN,
            errors=["unparseable_report"],
        )

    ngons = data.get("ngon_count", data.get("ngon_faces", 0))

    repair_class = McpRepairClass.CLEAN

    if data.get("mesh_count", 0) == 0:
        repair_class = McpRepairClass.EMPTY_SCENE

    elif ngons > 0:
        repair_class = McpRepairClass.NGON_ERROR

    elif data.get("open_edges", 0) > 0:
        repair_class = McpRepairClass.OPEN_EDGE_ERROR

    elif data.get("manifold_errors", 0) > 0:
        repair_class = McpRepairClass.MANIFOLD_ERROR

    return McpValidationResult(
        success=data.get("success", False),
        repair_class=repair_class,

        mesh_count=data.get("mesh_count", 0),
        object_count=data.get("object_count", 0),

        triangle_count=data.get("triangle_count", 0),

        tri_faces=data.get("tri_faces", 0),
        quad_faces=data.get("quad_faces", 0),
        ngon_faces=ngons,
        ngon_count=ngons,

        non_manifold_faces=data.get("non_manifold_faces", 0),
        isolated_verts=data.get("isolated_verts", 0),

        open_edges=data.get("open_edges", 0),
        manifold_errors=data.get("manifold_errors", 0),

        severity=data.get("severity", "clean"),
        floating_components=data.get("floating_components", 0),

        grounded=data.get("grounded", True),

        warnings=data.get("warnings", []),
        errors=data.get("errors", []),

        raw_response=stdout,
    )
