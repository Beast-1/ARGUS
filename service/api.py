"""FastAPI backend for the Tauri + React desktop UI.

Run from the repo root (main.py resolves OUTPUT_ROOT/LOG_DIR relative to CWD):
    .venv\\Scripts\\python.exe -m uvicorn service.api:app --host 127.0.0.1 --port 8420
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import queue

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from service import projects
from service.run_manager import run_manager
from service.schemas import (
    GenerateRequest,
    GenerateStatus,
    MemoryApprovalRequest,
    ProjectDetail,
    ProjectListResponse,
)

app = FastAPI(title="ARGUS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost", "http://tauri.localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# .glb has no standard mimetypes entry; be explicit so <model-viewer>'s fetch()
# gets the right Content-Type instead of a generic octet-stream fallback.
mimetypes.add_type("model/gltf-binary", ".glb")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/projects", response_model=ProjectListResponse)
def get_projects() -> ProjectListResponse:
    items = projects.list_projects()
    return ProjectListResponse(projects=items, count=len(items))


@app.get("/api/projects/{name}", response_model=ProjectDetail)
def get_project(name: str) -> ProjectDetail:
    detail = projects.get_project_detail(name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return detail


@app.get("/api/projects/{name}/files/{filename}")
def get_project_file(name: str, filename: str) -> FileResponse:
    path = projects.resolve_project_file(name, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.post("/api/projects/{name}/open-in-blender")
def open_project_in_blender(name: str) -> dict:
    ok, message = projects.open_in_blender(name)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in message else 500, detail=message)
    return {"ok": True, "message": message}


@app.post("/api/projects/{name}/open-folder")
def open_project_folder(name: str) -> dict:
    ok, message = projects.open_folder(name)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in message else 500, detail=message)
    return {"ok": True, "message": message}


@app.delete("/api/projects/{name}")
def delete_project(name: str) -> dict:
    ok, message = projects.delete_project(name)
    if not ok:
        raise HTTPException(status_code=404 if "not found" in message else 500, detail=message)
    return {"ok": True, "message": message}


@app.post("/api/generate", status_code=202)
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


@app.get("/api/generate/status", response_model=GenerateStatus)
def generation_status() -> GenerateStatus:
    return GenerateStatus(**run_manager.snapshot())


@app.post("/api/generate/memory-approval")
def resolve_memory_approval(req: MemoryApprovalRequest) -> dict:
    if not run_manager.resolve_memory_approval(req.approved):
        raise HTTPException(status_code=409, detail="Nothing is awaiting approval.")
    return {"ok": True}


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


@app.get("/api/generate/stream")
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
