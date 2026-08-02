"""Single-flight run lifecycle: spawns run_pipeline() on a background thread, fans its
events out to SSE subscribers, and brokers the memory-approval round trip.

Execution model matches desktop_app.py's (in-process background thread, not a subprocess):
`memory_approval_callback` blocks its calling thread on a `threading.Event` until a human
answers, which ports over unchanged here — across a subprocess boundary it would need real
IPC for no benefit at single-user scale.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Optional

from service.pipeline_events import PipelineEventWriter


def _manifest_score(glb_path: Optional[str]) -> Optional[int]:
    """Authoritative final score, read from the manifest main.py writes just before
    `[ARGUS COMPLETE]` (main.py:1457-1472) — same approach service/worker.py uses.

    Deliberately not scraped: the last "N/10" printed in a run is often not the final
    visual score. A reverted texture-paint pass prints
    "Texture paint : reverted — repaint 6/10 < un-painted 7/10", so scraping reports the
    rejected candidate's 6 while the asset that actually shipped scored 7.
    """
    if not glb_path:
        return None
    try:
        manifest = Path(glb_path).with_name("manifest.json")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        score = data.get("visual_score")
        return int(score) if score is not None else None
    except Exception:  # noqa: BLE001 — a missing manifest just means "no score"
        return None

# Terminal + transitional run states surfaced to the UI.
IDLE = "idle"
RUNNING = "running"
AWAITING_APPROVAL = "awaiting_approval"
COMPLETE = "complete"
FAILED = "failed"
REJECTED = "rejected"
ERROR = "error"
CANCELLED = "cancelled"


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._subscribers_guard = threading.Lock()
        self._history: list[dict] = []
        self._history_guard = threading.Lock()

        self.status: str = IDLE
        self.run_id: Optional[str] = None
        self.asset_name: Optional[str] = None
        self.prompt: Optional[str] = None

        self._approval_event: Optional[threading.Event] = None
        self._approval_decision: bool = False
        self._approval_context: Optional[dict] = None

    # -- event plumbing -------------------------------------------------
    def emit(self, event_type: str, payload: dict) -> None:
        event = {"type": event_type, "payload": payload, "ts": time.time()}
        with self._history_guard:
            self._history.append(event)
        with self._subscribers_guard:
            for q in list(self._subscribers):
                q.put(event)

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        """Returns a live queue plus the current run's events so far, so a client that
        connects mid-run (or reconnects) sees the whole run without Last-Event-ID logic."""
        q: queue.Queue = queue.Queue()
        with self._history_guard:
            replay = list(self._history)
        with self._subscribers_guard:
            self._subscribers.append(q)
        return q, replay

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subscribers_guard:
            if q in self._subscribers:
                self._subscribers.remove(q)

    # -- run lifecycle ---------------------------------------------------
    def is_busy(self) -> bool:
        return self.status in (RUNNING, AWAITING_APPROVAL)

    def start(
        self,
        prompt: str,
        poly_budget: Optional[str] = None,
        mcp_mode: bool = False,
        use_concept_pipeline: bool = True,
    ) -> bool:
        """Returns False if a generation is already in flight."""
        if not self._lock.acquire(blocking=False):
            return False

        # The cancel flag is process-wide (main._CANCEL_EVENT) and outlives a run
        # that finished normally — clear it before this one starts, or a prior
        # cancelled run would make this new run appear cancelled from the outset.
        from main import clear_cancel
        clear_cancel()

        with self._history_guard:
            self._history = []
        self.status = RUNNING
        self.run_id = None
        self.asset_name = None
        self.prompt = prompt

        thread = threading.Thread(
            target=self._run,
            args=(prompt, poly_budget, mcp_mode, use_concept_pipeline),
            daemon=True,
        )
        thread.start()
        return True

    def _run(
        self,
        prompt: str,
        poly_budget: Optional[str],
        mcp_mode: bool,
        use_concept_pipeline: bool,
    ) -> None:
        writer = PipelineEventWriter(self.emit)
        self.emit("run_started", {
            "prompt": prompt,
            "poly_budget": poly_budget,
            "mcp_mode": mcp_mode,
            "use_concept_pipeline": use_concept_pipeline,
            "started_at": time.time(),
        })

        try:
            # Deferred import: main.py reads ARGUS_* env vars at module import time
            # (main.py:89-100), matching service/worker.py:96's documented pattern.
            from main import run_pipeline

            with redirect_stdout(writer), redirect_stderr(writer):
                success = run_pipeline(
                    prompt=prompt,
                    poly_budget=poly_budget,
                    mcp_mode=mcp_mode,
                    use_concept_pipeline=use_concept_pipeline,
                    memory_approval_callback=self._memory_approval_callback,
                )
            writer.flush()

            self.run_id = writer.run_id
            self.asset_name = writer.asset_name

            # A cancelled run still goes through run_pipeline's normal completion
            # path (main.py's loops stop cooperatively at the next checkpoint,
            # then the pipeline finishes the remaining stages as usual) — so
            # `success` alone can't tell "finished" from "stopped early because
            # you asked it to". Check the cancel flag to report the state you
            # actually asked for, while still surfacing whatever asset exists.
            from main import is_cancelled
            cancelled = is_cancelled()

            # Terminal state comes from the real return value, cross-checked against the
            # completion markers — the same belt-and-braces service/worker.py:115 uses.
            if cancelled and success and writer.final_asset:
                self.status = CANCELLED
                self.emit("cancelled", {
                    "run_id": writer.run_id,
                    "asset_name": writer.asset_name,
                    "glb_path": writer.final_asset,
                    "preview_path": writer.preview_path,
                    "visual_score": _manifest_score(writer.final_asset),
                })
            elif cancelled:
                self.status = CANCELLED
                self.emit("cancelled", {"reason": "Cancelled before an asset was produced"})
            elif success is None:
                self.status = REJECTED
                self.emit("rejected", {"reason": writer.rejection_reason or "Request rejected"})
            elif success and writer.final_asset:
                self.status = COMPLETE
                self.emit("complete", {
                    "run_id": writer.run_id,
                    "asset_name": writer.asset_name,
                    "glb_path": writer.final_asset,
                    "preview_path": writer.preview_path,
                    "visual_score": _manifest_score(writer.final_asset),
                })
            else:
                self.status = FAILED
                self.emit("failed", {"reason": "Pipeline reported failure"})
        except Exception as exc:  # noqa: BLE001 — surface it, never kill the service
            self.status = ERROR
            self.emit("error", {"message": f"{type(exc).__name__}: {exc}"})
        finally:
            self._approval_event = None
            self._approval_context = None
            self._lock.release()

    # -- memory approval --------------------------------------------------
    def _memory_approval_callback(self, context: dict) -> bool:
        """Runs on the worker thread. Emits a pending event, then blocks until the
        frontend POSTs a decision — same blocking shape as desktop_app.py:3017-3125."""
        event = threading.Event()
        self._approval_event = event
        self._approval_decision = False
        self._approval_context = context
        self.status = AWAITING_APPROVAL

        self.emit("memory_approval_pending", {
            "prompt": context.get("prompt"),
            "asset_name": context.get("asset_name"),
            "run_id": context.get("run_id"),
            "blueprint": context.get("blueprint"),
            "preview_path": context.get("preview_path"),
            "asset_path": context.get("asset_path"),
        })

        event.wait()
        self.status = RUNNING
        return self._approval_decision

    def request_cancel(self) -> bool:
        """Returns False if there's nothing running to cancel."""
        if not self.is_busy():
            return False
        from main import request_cancel as _request_cancel
        _request_cancel()
        self.emit("cancel_requested", {})
        # If the pipeline thread is currently blocked waiting on a memory-approval
        # decision, it can't reach the next _past_deadline() checkpoint until that
        # wait ends — resolve it (as "don't save") so cancellation actually
        # unblocks the thread instead of hanging until someone answers the dialog.
        if self.status == AWAITING_APPROVAL:
            self.resolve_memory_approval(False)
        return True

    def resolve_memory_approval(self, approved: bool) -> bool:
        """Returns False if nothing is currently awaiting a decision."""
        event = self._approval_event
        if event is None or event.is_set():
            return False
        self._approval_decision = approved
        self.emit("memory_approval_resolved", {"approved": approved})
        event.set()
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "run_id": self.run_id,
            "asset_name": self.asset_name,
            "prompt": self.prompt,
        }


run_manager = RunManager()
