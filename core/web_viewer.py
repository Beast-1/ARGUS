"""Embedded 3D asset viewer.

Tkinter can't display a textured, animated glTF, so the real preview opens in
the browser via Google's <model-viewer> web component — it shows the baked
materials AND auto-plays the rig animation (spinning wheels). The asset is
served over a tiny local HTTP server so model-viewer's fetch() works (a
file:// page is blocked by WebGL/CORS in Chrome/Edge).

Public API: open_3d_viewer(glb_path) -> bool
"""
from __future__ import annotations

import http.server
import logging
import socket
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path

logger = logging.getLogger("ARGUS.viewer")

_SERVER: dict = {}  # {"root": str, "port": int, "httpd": server}

_VIEWER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARGUS — {title}</title>
<script type="module"
  src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.5.0/dist/model-viewer.min.js"></script>
<style>
  :root {{ --bg:#0b0b0e; --panel:#141417; --line:#232329; --violet:#8b5cf6;
           --violet2:#a78bfa; --text:#fafafa; --muted:#a1a1aa; }}
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; height:100%; background:var(--bg); color:var(--text);
    font-family:'Segoe UI',system-ui,sans-serif; }}
  header {{ display:flex; align-items:center; gap:12px; padding:14px 22px;
    border-bottom:1px solid var(--line); background:#09090b; }}
  .logo {{ display:flex; gap:3px; align-items:flex-end; }}
  .logo i {{ width:3px; border-radius:2px; display:block; }}
  .logo i:nth-child(1){{height:10px;background:#7c3aed;}}
  .logo i:nth-child(2){{height:16px;background:var(--violet);}}
  .logo i:nth-child(3){{height:12px;background:var(--violet2);}}
  h1 {{ font-size:15px; font-weight:700; margin:0; letter-spacing:.5px; }}
  .name {{ color:var(--muted); font-weight:400; }}
  model-viewer {{ width:100%; height:calc(100vh - 53px);
    background:radial-gradient(circle at 50% 38%, #1a1a22 0%, #0b0b0e 72%); }}
  .bar {{ position:absolute; bottom:18px; left:50%; transform:translateX(-50%);
    display:flex; gap:8px; background:rgba(20,20,23,.85); border:1px solid var(--line);
    border-radius:10px; padding:7px 10px; backdrop-filter:blur(6px); }}
  .bar button {{ background:var(--panel); color:var(--text); border:1px solid var(--line);
    border-radius:7px; padding:7px 13px; font-size:12px; font-weight:600; cursor:pointer; }}
  .bar button:hover {{ border-color:var(--violet); color:var(--violet2); }}
  .pill {{ position:absolute; top:68px; right:20px; background:rgba(20,20,23,.85);
    border:1px solid var(--line); border-radius:8px; padding:8px 12px; font-size:12px;
    color:var(--muted); }}
  .pill b {{ color:var(--violet2); }}
</style>
</head>
<body>
<header>
  <span class="logo"><i></i><i></i><i></i></span>
  <h1>ARGUS 3D <span class="name">/ {title}</span></h1>
</header>
<model-viewer id="mv" src="{glb}" alt="{title}"
  camera-controls touch-action="pan-y" interaction-prompt="none"
  autoplay shadow-intensity="1" shadow-softness="0.9" exposure="1.15"
  environment-image="neutral" camera-orbit="-35deg 72deg auto"
  min-camera-orbit="auto auto auto" max-camera-orbit="auto auto auto">
</model-viewer>
<div class="pill" id="pill">loading…</div>
<div class="bar">
  <button onclick="mvSpin()">▶ / ⏸ animation</button>
  <button onclick="mvRotate()">orbit on/off</button>
  <button onclick="document.getElementById('mv').resetTurntableRotation?.()">reset view</button>
</div>
<script>
  const mv = document.getElementById('mv');
  const pill = document.getElementById('pill');
  let spinning = true, rotating = false;
  function mvSpin() {{ spinning = !spinning; spinning ? mv.play() : mv.pause(); }}
  function mvRotate() {{ rotating = !rotating; mv.toggleAttribute('auto-rotate', rotating); }}
  mv.addEventListener('load', () => {{
    const anims = mv.availableAnimations || [];
    pill.innerHTML = anims.length
      ? 'rig: <b>' + anims.length + ' animation</b> · drag to orbit'
      : 'static model · drag to orbit';
    if (anims.length) mv.play();
  }});
  mv.addEventListener('error', () => {{ pill.textContent = 'failed to load model'; }});
</script>
</body>
</html>"""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ensure_server(root: Path) -> int | None:
    """Start (once) a daemon HTTP server rooted at `root`; reuse if unchanged."""
    root_s = str(root)
    if _SERVER.get("root") == root_s and _SERVER.get("port"):
        return _SERVER["port"]
    # different root → stop the old one
    old = _SERVER.get("httpd")
    if old is not None:
        try:
            old.shutdown()
        except Exception:
            pass
    try:
        port = _free_port()
        handler = partial(http.server.SimpleHTTPRequestHandler, directory=root_s)
        httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
        httpd.daemon_threads = True
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        _SERVER.update(root=root_s, port=port, httpd=httpd)
        return port
    except Exception as exc:  # noqa: BLE001
        logger.warning("[VIEWER] server start failed: %s", exc)
        return None


def open_3d_viewer(glb_path: "str | Path") -> bool:
    """Open the textured, animated asset in the browser via <model-viewer>.
    Returns True on success."""
    glb = Path(glb_path).resolve()
    if not glb.exists():
        logger.warning("[VIEWER] GLB not found: %s", glb)
        return False

    # Serve from the asset's parent so the GLB is same-origin with viewer.html.
    root = glb.parent
    viewer = root / "_argus_viewer.html"
    try:
        viewer.write_text(
            _VIEWER_HTML.format(title=glb.stem.replace("_", " "), glb=glb.name),
            encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[VIEWER] could not write viewer html: %s", exc)
        return False

    port = _ensure_server(root)
    if port is None:
        # last resort: open the file directly (works for static, no animation
        # in some browsers due to CORS, but better than nothing)
        webbrowser.open(viewer.as_uri())
        return True
    webbrowser.open(f"http://127.0.0.1:{port}/{viewer.name}")
    return True
