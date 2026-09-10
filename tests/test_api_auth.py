"""The backend previously had zero auth on any route — any local process could
POST /api/generate, DELETE a project, or trigger open-in-blender. This proves the
new shared-secret token is actually enforced by the running app, not just present
in source. Hermetic: FastAPI's TestClient, no real network, no Blender.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Request
from fastapi.testclient import TestClient  # noqa: E402

import service.api as api_mod  # noqa: E402
from service.auth import API_TOKEN, require_token  # noqa: E402

client = TestClient(api_mod.app)


def test_health_needs_no_token():
    res = client.get("/api/health")
    assert res.status_code == 200


def test_protected_route_rejects_missing_token():
    res = client.get("/api/projects")
    assert res.status_code == 401


def test_protected_route_rejects_wrong_token():
    res = client.get("/api/projects", headers={"Authorization": "Bearer wrong-token"})
    assert res.status_code == 401


def test_protected_route_accepts_correct_bearer_token():
    res = client.get("/api/projects", headers={"Authorization": f"Bearer {API_TOKEN}"})
    assert res.status_code == 200


def test_generate_endpoint_requires_auth():
    res = client.post("/api/generate", json={"prompt": "a crate"})
    assert res.status_code == 401


def test_delete_project_requires_auth():
    res = client.delete("/api/projects/nonexistent-asset")
    assert res.status_code == 401


def test_stream_endpoint_rejects_bad_query_param_token():
    """The auth dependency runs before the route body, so a bad token never
    reaches generation_stream()'s (deliberately never-ending, keepalive-looping)
    generator — this returns a plain 401, not an opened stream."""
    res = client.get("/api/generate/stream?token=nope")
    assert res.status_code == 401


def test_require_token_accepts_query_param_form():
    """EventSource can't set headers, so /api/generate/stream also accepts
    ?token= — checked directly against the dependency rather than by opening the
    real (intentionally infinite) SSE stream in a test."""
    request = Request(scope={
        "type": "http",
        "headers": [],
        "query_string": f"token={API_TOKEN}".encode(),
    })
    require_token(request)  # must not raise


def test_require_token_rejects_query_param_mismatch():
    from fastapi import HTTPException
    import pytest

    request = Request(scope={
        "type": "http",
        "headers": [],
        "query_string": b"token=nope",
    })
    with pytest.raises(HTTPException) as exc_info:
        require_token(request)
    assert exc_info.value.status_code == 401


def test_generate_status_endpoint_was_removed():
    """Confirmed orphaned (zero frontend callers) during the remediation pass —
    SSE replay already covers reconnect, so it was deleted rather than fixed."""
    res = client.get("/api/generate/status")
    assert res.status_code == 404
