"""The live MCP server does `exec()` on whatever it receives over a local TCP
socket — it must never accept a request without a matching shared-secret token.
Hermetic: no socket connection, no Blender, no network. What actually happens
inside the Blender-side `_handle()` at runtime is exercised end-to-end by the
manual MCP-mode smoke check in the plan, not here — this locks in the pieces
that are testable without a live Blender process.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.mcp_server as mcp  # noqa: E402


def test_token_is_generated_and_nontrivial():
    assert mcp.MCP_TOKEN
    assert len(mcp.MCP_TOKEN) >= 32


def test_execute_blender_code_attaches_token(monkeypatch):
    captured = {}

    def fake_send(self, payload):
        captured.update(payload)
        return {"success": True, "stdout": "", "stderr": ""}

    monkeypatch.setattr(mcp.McpClient, "send", fake_send)
    mcp.execute_blender_code("print('hi')")
    assert captured.get("token") == mcp.MCP_TOKEN


def test_live_server_script_rejects_missing_or_wrong_token():
    """The embedded server source (shipped into a launched Blender process) must
    actually contain the token check, not just this module's client side."""
    src = mcp._LIVE_SERVER_SCRIPT
    assert 'payload.get("token")' in src
    assert "unauthorized" in src
    # The check must happen before dispatching to execute_code, not after.
    token_check_pos = src.index('payload.get("token")')
    execute_dispatch_pos = src.index('payload.get("type") != "execute_code"')
    assert token_check_pos < execute_dispatch_pos


def test_live_server_script_reads_token_from_env():
    src = mcp._LIVE_SERVER_SCRIPT
    assert 'TOKEN = os.environ.get("ARGUS_MCP_TOKEN"' in src


def test_ensure_mcp_server_passes_token_env_to_launched_process(monkeypatch, tmp_path):
    """The launched Blender subprocess must inherit ARGUS_MCP_TOKEN so its embedded
    server can actually validate against the same value the client sends."""
    import os

    captured_env = {}

    def fake_popen(cmd, env=None, **kw):
        captured_env.update(env or {})

        class _FakeProc:
            def poll(self):
                return None

        return _FakeProc()

    # False before the launch attempt (so ensure_mcp_server takes the launch
    # branch), then True (so the post-launch poll loop returns immediately
    # instead of sleeping a full second waiting for a port that will never open).
    ping_results = iter([(False, "not up"), (True, "up")])
    monkeypatch.setattr(mcp, "ping_mcp_server", lambda: next(ping_results))
    monkeypatch.setenv("ARGUS_MCP_AUTOLAUNCH", "1")
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    mcp.ensure_mcp_server(auto_launch=True, timeout=5.0)
    assert captured_env.get("ARGUS_MCP_TOKEN") == os.environ["ARGUS_MCP_TOKEN"]
