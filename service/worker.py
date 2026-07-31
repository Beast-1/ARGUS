from __future__ import annotations

import json
import logging
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ARGUS.worker")

# Tier -> env overrides applied for the duration of a single job. "free" reuses
# ARGUS_FAST's lighter loop budgets (see main.py) to keep cost/time predictable
# on the MVP's single self-hosted worker; "pro" runs the full pipeline. Both
# get ARGUS_MAX_SECONDS as a hard backstop so one prompt can never block every
# other user's queued job indefinitely.
TIER_ENV: dict[str, dict[str, str]] = {
    "free": {"ARGUS_FAST": "1", "ARGUS_MAX_SECONDS": "300"},
    "pro": {"ARGUS_FAST": "0", "ARGUS_MAX_SECONDS": "1800"},
}

# Different users can submit prompts that sanitize to the same asset name.
# ARGUS_KEEP_HISTORY=1 stops main.make_run_id() from wiping one user's
# out/final/<name> folder when another user's job reuses that name (the
# default behavior assumes a single desktop user working on one asset at a
# time, which doesn't hold for a shared multi-tenant worker).
_BASE_ENV = {
    "ARGUS_KEEP_HISTORY": "1",
    "ARGUS_AUTO_APPROVE_MEMORY": "0",
}

_FINAL_ASSET_RE = re.compile(r"^Final Asset\s*:\s*(.+)$")
_PROJECT_RE = re.compile(r"^Project\s*:\s*(.+)$")
_REJECTED_PREFIX = "[ARGUS] Request rejected:"


class _JobLog:
    """Captures run_pipeline() stdout/stderr the same way desktop_app.py's
    _QueueWriter does (desktop_app.py:33-44), but into an in-memory buffer
    instead of a Tk queue, and picks the same stable stdout markers
    desktop_app.py already relies on ([ARGUS] Request rejected:, and here the
    [ARGUS COMPLETE] block's "Final Asset :" / "Project     :" lines) instead
    of threading extra return values through run_pipeline()."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.final_asset: str | None = None
        self.run_id: str | None = None
        self.rejection_reason: str | None = None

    def write(self, text: str) -> int:
        if not text:
            return 0
        self.lines.append(text)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            m = _FINAL_ASSET_RE.match(line)
            if m:
                self.final_asset = m.group(1).strip()
                continue
            m = _PROJECT_RE.match(line)
            if m:
                self.run_id = m.group(1).strip()
                continue
            if line.startswith(_REJECTED_PREFIX):
                self.rejection_reason = line[len(_REJECTED_PREFIX):].strip()
        return len(text)

    def flush(self) -> None:
        return

    @property
    def text(self) -> str:
        return "".join(self.lines)


def generate_asset_job(prompt: str, tier: str = "free") -> dict:
    """RQ task entry point: run one ARGUS generation end-to-end and upload the
    result to object storage.

    Mirrors desktop_app.py's own pattern (desktop_app.py:2952-2982) of
    importing run_pipeline and calling it fresh per generation inside a
    stdout/stderr redirect — that pattern is already proven safe to repeat
    across many in-process calls, so this worker does the same rather than
    shelling out to a subprocess per job.
    """
    env_overrides = {**_BASE_ENV, **TIER_ENV.get(tier, TIER_ENV["free"])}
    previous = {k: os.environ.get(k) for k in env_overrides}
    os.environ.update(env_overrides)

    try:
        from main import run_pipeline  # deferred import: must see env overrides above

        log = _JobLog()
        with redirect_stdout(log), redirect_stderr(log):
            success = run_pipeline(
                prompt=prompt,
                mcp_mode=False,
                memory_approval_callback=lambda ctx: False,
            )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if log.rejection_reason:
        return {"status": "rejected", "reason": log.rejection_reason, "log": log.text}

    if not success or not log.final_asset or not log.run_id:
        return {"status": "failed", "log": log.text}

    final_dir = Path(log.final_asset).parent
    visual_score = None
    manifest_path = final_dir / "manifest.json"
    if manifest_path.exists():
        try:
            visual_score = json.loads(manifest_path.read_text(encoding="utf-8")).get("visual_score")
        except Exception:  # noqa: BLE001
            pass

    from core.storage import is_configured, upload_job_artifacts

    urls: dict[str, str] = {}
    if is_configured():
        urls = upload_job_artifacts(final_dir, log.run_id)
    else:
        logger.warning("[WORKER] core.storage not configured — artifacts stay on local disk only")

    return {
        "status": "success",
        "run_id": log.run_id,
        "visual_score": visual_score,
        "urls": urls,
        "local_dir": str(final_dir),
        "log": log.text,
    }


def _run_rq_worker() -> None:
    """Entry point for `python -m service.worker` — starts an RQ worker
    listening on the argus-jobs queue. Requires redis + rq (worker-only
    dependencies, see requirements.txt) and REDIS_URL in the environment."""
    from redis import Redis
    from rq import Queue, Worker

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    conn = Redis.from_url(redis_url)
    queue = Queue("argus-jobs", connection=conn)
    logger.info("[WORKER] listening on 'argus-jobs' at %s", redis_url)
    Worker([queue], connection=conn).work()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_rq_worker()
