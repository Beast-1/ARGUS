"""Shared-secret auth for the desktop API.

The backend binds to 127.0.0.1 by convention, but CORS is a browser-only defense —
any local process (or another user on a shared machine) could otherwise call
/api/generate, DELETE a project, or trigger open-in-blender/open-folder with zero
auth. A random per-process token, generated once and handed to the Tauri frontend
via an IPC command (not an HTTP round trip, so there's no bootstrapping problem),
closes that gap without needing real user accounts for a single-user local app.
"""
from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Request

# setdefault() both generates a fresh token once per process AND writes it into
# os.environ, so run_app.bat's uvicorn process and the sibling `npm run tauri dev`
# process (which inherits the same shell environment) can share it without any
# extra plumbing — the Tauri side reads it straight from its own process env.
API_TOKEN = os.environ.setdefault("ARGUS_API_TOKEN", secrets.token_urlsafe(32))


def require_token(request: Request) -> None:
    """Dependency for every /api/* route except /api/health.

    Two transports, because several browser APIs cannot attach a header at all:
    EventSource (/api/generate/stream), <img src>, <a download>, and three.js's
    GLTFLoader. Those pass the token as a `?token=` query param; everything else
    uses `Authorization: Bearer <token>`.

    The query-param form is the weaker of the two — it lands in server access
    logs, in browser history for <a download> navigations, and in the logs of any
    tunnel the backend is published through. It is accepted because the
    alternative (blob URLs everywhere) buys little for a single-user app holding
    a per-process token that dies with the process. If ARGUS ever grows real
    multi-user auth, this is the line to revisit.
    """
    auth = request.headers.get("authorization", "")
    header_token = auth[7:] if auth.lower().startswith("bearer ") else ""
    query_token = request.query_params.get("token", "")
    supplied = header_token or query_token or ""
    if not secrets.compare_digest(supplied, API_TOKEN):
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")
