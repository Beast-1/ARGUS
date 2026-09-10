"""FastAPI backend for the Tauri + React desktop UI.

Run from the repo root (main.py resolves OUTPUT_ROOT/LOG_DIR relative to CWD):
    .venv\\Scripts\\python.exe -m uvicorn service.api:app --host 127.0.0.1 --port 8420
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import queue
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from service import projects
from service.auth import require_token
from service.run_manager import run_manager
from service.schemas import (
    GenerateRequest,
    MemoryApprovalRequest,
    ProjectDetail,
    ProjectListResponse,
)

_AUTH = [Depends(require_token)]

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Print the API token once, on the operator's own console.

    The desktop flow never needed this — run_app.bat puts the token in an env
    var both processes inherit, so the Tauri shell hands it to the frontend
    invisibly. A browser has no shared environment, so whoever opens the hosted
    UI has to type the token in, and until this existed there was no way to find
    out what it was.

    stdout rather than the logger on purpose: this is a one-time startup notice
    to a terminal, not something that should accumulate in logs/argus.log. A
    startup hook rather than module scope so importing the app in tests doesn't
    print it.
    """
    from service.auth import API_TOKEN
    print(f"\n  ARGUS API token: {API_TOKEN}")
    print("  (needed once by the browser UI; the desktop app picks it up automatically)\n")
    yield


app = FastAPI(title="ARGUS API", lifespan=_lifespan)

# Desktop origins are fixed; the hosted frontend's is not, so it's configured.
# Deliberately an explicit allowlist rather than "*": every /api/* route already
# requires a bearer token, but a wildcard would let any page a browser happens to
# load read responses from a backend reachable at that origin, and a tunnelled
# backend is reachable from anywhere.
_CORS_ORIGINS = [
    "http://localhost:1420",       # vite dev
    "tauri://localhost",           # desktop build
    "http://tauri.localhost",      # desktop build (windows webview2)
]
_extra_origins = os.getenv("ARGUS_ALLOWED_ORIGINS", "")
if _extra_origins:
    _CORS_ORIGINS += [o.strip().rstrip("/") for o in _extra_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# .glb has no standard mimetypes entry; be explicit so <model-viewer>'s fetch()
# gets the right Content-Type instead of a generic octet-stream fallback.
mimetypes.add_type("model/gltf-binary", ".glb")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/projects", response_model=ProjectListResponse, dependencies=_AUTH)
def get_projects() -> ProjectListResponse:
    items = projects.list_projects()
    return ProjectListResponse(projects=items, count=len(items))


@app.get("/api/projects/{name}", response_model=ProjectDetail, dependencies=_AUTH)
def get_project(name: str) -> ProjectDetail:
    detail = projects.get_project_detail(name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return detail


@app.get("/api/projects/{name}/files/{filename}", dependencies=_AUTH)
def get_project_file(name: str, filename: str) -> FileResponse:
    path = projects.resolve_project_file(name, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.post("/api/projects/{name}/open-in-blender", dependencies=_AUTH)
def open_project_in_blender(name: str) -> dict:
    ok, message = projects.open_in_blender(name)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in message else 500, detail=message)
    return {"ok": True, "message": message}


@app.post("/api/projects/{name}/open-folder", dependencies=_AUTH)
def open_project_folder(name: str) -> dict:
    ok, message = projects.open_folder(name)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in message else 500, detail=message)
    return {"ok": True, "message": message}


@app.delete("/api/projects/{name}", dependencies=_AUTH)
def delete_project(name: str) -> dict:
    ok, message = projects.delete_project(name)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in message else 500, detail=message)
    return {"ok": True, "message": message}


@app.post("/api/generate", status_code=202, dependencies=_AUTH)
def start_generation(req: GenerateRequest) -> dict:
    started = run_manager.start(
        prompt=req.prompt,
        poly_budget=req.poly_budget,
        mcp_mode=req.mcp_mode,
        use_concept_pipeline=req.use_concept_pipeline,
    )
    if not started:
        raise HTTPException(status_code=409, detail="A generation is already running.")
    return {"status": "started"}


@app.post("/api/generate/memory-approval", dependencies=_AUTH)
def resolve_memory_approval(req: MemoryApprovalRequest) -> dict:
    if not run_manager.resolve_memory_approval(req.approved):
        raise HTTPException(status_code=409, detail="Nothing is awaiting approval.")
    return {"ok": True}


@app.post("/api/generate/cancel", dependencies=_AUTH)
def cancel_generation() -> dict:
    if not run_manager.request_cancel():
        raise HTTPException(status_code=409, detail="Nothing is running.")
    return {"ok": True}


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


@app.get("/api/generate/stream", dependencies=_AUTH)
async def generation_stream(request: Request) -> StreamingResponse:
    async def event_source():
        q, replay = run_manager.subscribe()
        try:
            for event in replay:
                yield _sse(event["type"], event["payload"])
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = q.get_nowait()
                except queue.Empty:
                    # Comment frame doubles as a keepalive between runs.
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.25)
                    continue
                yield _sse(event["type"], event["payload"])
        finally:
            run_manager.unsubscribe(q)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
