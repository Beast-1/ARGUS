"""Single-flight run lifecycle: spawns run_pipeline() on a background thread, fans its
events out to SSE subscribers, and brokers the memory-approval round trip.

Execution model matches desktop_app.py's (in-process background thread, not a subprocess):
`memory_approval_callback` blocks its calling thread on a `threading.Event` until a human
answers, which ports over unchanged here — across a subprocess boundary it would need real
IPC for no benefit at single-user scale.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # import only for annotations — main/core are imported lazily here
    from core.provider_keys import ProviderKeys

from service.pipeline_events import PipelineEventWriter
from service.projects import OUTPUT_ROOT

logger = logging.getLogger("ARGUS.run_manager")

# Bounds how much a single run's event history can grow — matches the 500-line
# display cap the frontend already applies to log_line events (useEventStream.ts),
# but here it protects the *source of truth*: without this, _history and every
# subscriber's queue grow for the life of a long/chatty run with no cap at all.
_MAX_HISTORY = 2000

_STATE_FILE = OUTPUT_ROOT / ".run_state.json"


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
        # Guards status/run_id/asset_name/prompt as one unit, so a concurrent
        # snapshot() (e.g. from a status poll) can't observe a torn combination —
        # e.g. a new run_id paired with the previous run's stale status.
        self._state_guard = threading.Lock()

        self.status: str = IDLE
        self.run_id: Optional[str] = None
        self.asset_name: Optional[str] = None
        self.prompt: Optional[str] = None

        self._approval_event: Optional[threading.Event] = None
        # The current run's cancel scope (core/cancel.py). Created on the request
        # thread in start(), adopted by the worker thread in _run(), signalled
        # from whichever thread calls request_cancel().
        self._cancel_scope: Optional[threading.Event] = None
        # The current run's own provider credentials, or None to use the
        # operator's .env keys. Never persisted — snapshot() must not learn
        # about this field, or the keys would reach out/.run_state.json.
        self._provider_keys: Optional["ProviderKeys"] = None
        self._approval_decision: bool = False
        self._approval_context: Optional[dict] = None

        self._warn_if_state_file_shows_interrupted_run()

    def _warn_if_state_file_shows_interrupted_run(self) -> None:
        """A backend restart mid-run has nothing else to go on — the in-memory
        RunManager is gone, and no run-history database exists. This doesn't
        attempt recovery (out of scope), it just makes sure the interruption isn't
        silent: whoever restarts the service can see what was in flight."""
        try:
            data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if data.get("status") in (RUNNING, AWAITING_APPROVAL):
            logger.warning(
                "[RUN_MANAGER] Previous run appears to have been interrupted by a "
                "backend restart: run_id=%s asset_name=%s prompt=%r status=%s",
                data.get("run_id"), data.get("asset_name"), data.get("prompt"),
                data.get("status"),
            )

    def _persist_state(self) -> None:
        """Best-effort — losing this trace is preferable to a run failing because
        disk I/O for a diagnostic file didn't succeed."""
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.snapshot()), encoding="utf-8")
            tmp.replace(_STATE_FILE)
        except OSError:
            pass

    # -- event plumbing -------------------------------------------------
    def emit(self, event_type: str, payload: dict) -> None:
        event = {"type": event_type, "payload": payload, "ts": time.time()}
        with self._history_guard:
            self._history.append(event)
            if len(self._history) > _MAX_HISTORY:
                # Keep the newest events — a reconnecting client cares about "what's
                # happening now", not the start of a run it already saw once.
                del self._history[: len(self._history) - _MAX_HISTORY]
        with self._subscribers_guard:
            for q in list(self._subscribers):
                try:
                    # Bounded + non-blocking: a subscriber whose consumer has
                    # stopped draining (e.g. a dropped connection request.is_
                    # disconnected() hasn't caught yet) must never be able to
                    # block emit() itself, which runs on the pipeline worker
                    # thread — dropping the event for that one stalled
                    # subscriber is the right trade, not stalling the run.
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self) -> tuple[queue.Queue, list[dict]]:
        """Returns a live queue plus the current run's events so far, so a client that
        connects mid-run (or reconnects) sees the whole run without Last-Event-ID logic."""
        q: queue.Queue = queue.Queue(maxsize=_MAX_HISTORY)
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
        provider_keys: Optional["ProviderKeys"] = None,
    ) -> bool:
        """Returns False if a generation is already in flight."""
        if not self._lock.acquire(blocking=False):
            return False

        # Everything from here until thread.start() must release the lock on any
        # failure — it's only otherwise released in _run()'s finally, which never
        # runs if we never reach the point of starting that thread. Before this
        # fix, an exception here (e.g. the `from main import clear_cancel` below)
        # left the lock held forever, permanently stuck reporting "a generation is
        # already running" until the process was restarted.
        try:
            # Each run gets its own cancel scope (core/cancel.py). The scope is
            # opened here rather than on the worker thread so request_cancel(),
            # which arrives on a FastAPI request thread, has something to signal
            # the moment the run exists — there is no window where a cancel would
            # land on a scope that hasn't been created yet. The worker adopts it
            # as its thread-current scope in _run().
            from core.cancel import new_scope
            self._cancel_scope = new_scope()

            # A run's own credentials, opened here for the same reason as the
            # cancel scope: created on the request thread, handed to the worker,
            # adopted there. None means "use the operator's .env keys", which is
            # the desktop and CLI path.
            self._provider_keys = provider_keys

            with self._history_guard:
                self._history = []
            with self._state_guard:
                self.status = RUNNING
                self.run_id = None
                self.asset_name = None
                self.prompt = prompt
            self._persist_state()

            thread = threading.Thread(
                target=self._run,
                args=(prompt, poly_budget, mcp_mode, use_concept_pipeline),
                daemon=True,
            )
            thread.start()
        except Exception:
            self._lock.release()
            raise
        return True

    def _run(
        self,
        prompt: str,
        poly_budget: Optional[str],
        mcp_mode: bool,
        use_concept_pipeline: bool,
    ) -> None:
        writer = PipelineEventWriter(self.emit)
        # A ContextVar set on the request thread isn't visible here, so adopt the
        # scope start() created — otherwise this thread would check the
        # process-wide fallback and never see the cancel.
        if self._cancel_scope is not None:
            from core.cancel import adopt_scope
            adopt_scope(self._cancel_scope)

        # Same reasoning for the run's credentials: without adopting them here
        # core/llm.py would fall back to the operator's .env pools and silently
        # spend their quota, which is the exact outcome BYOK exists to prevent.
        # Registering the literal values with the redactor is what keeps them out
        # of logs/argus.log, the SSE stream and the eval artifacts.
        run_secrets: list[str] = []
        if self._provider_keys is not None:
            from core.provider_keys import adopt_keys
            from core.secrets import register_run_secrets
            adopt_keys(self._provider_keys)
            run_secrets = self._provider_keys.secret_values()
            register_run_secrets(run_secrets)
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

            with self._state_guard:
                self.run_id = writer.run_id
                self.asset_name = writer.asset_name

            # A cancelled run still goes through run_pipeline's normal completion
            # path (main.py's loops stop cooperatively at the next checkpoint,
            # then the pipeline finishes the remaining stages as usual) — so
            # `success` alone can't tell "finished" from "stopped early because
            # you asked it to". Check the cancel flag to report the state you
            # actually asked for, while still surfacing whatever asset exists.
            from core.cancel import is_cancelled
            cancelled = is_cancelled(self._cancel_scope)

            # Terminal state comes from the real return value, cross-checked against the
            # completion markers — the same belt-and-braces service/worker.py:115 uses.
            if cancelled and success and writer.final_asset:
                with self._state_guard:
                    self.status = CANCELLED
                self.emit("cancelled", {
                    "run_id": writer.run_id,
                    "asset_name": writer.asset_name,
                    "glb_path": writer.final_asset,
                    "preview_path": writer.preview_path,
                    "visual_score": _manifest_score(writer.final_asset),
                })
            elif cancelled:
                with self._state_guard:
                    self.status = CANCELLED
                self.emit("cancelled", {"reason": "Cancelled before an asset was produced"})
            elif success is None:
                with self._state_guard:
                    self.status = REJECTED
                self.emit("rejected", {"reason": writer.rejection_reason or "Request rejected"})
            elif success and writer.final_asset:
                with self._state_guard:
                    self.status = COMPLETE
                self.emit("complete", {
                    "run_id": writer.run_id,
                    "asset_name": writer.asset_name,
                    "glb_path": writer.final_asset,
                    "preview_path": writer.preview_path,
                    "visual_score": _manifest_score(writer.final_asset),
                })
            else:
                with self._state_guard:
                    self.status = FAILED
                self.emit("failed", {"reason": "Pipeline reported failure"})
        except Exception as exc:  # noqa: BLE001 — surface it, never kill the service
            with self._state_guard:
                self.status = ERROR
            # Exception text routinely carries the failing request's URL and
            # sometimes the provider's error envelope, so redact before this
            # crosses to the browser.
            from core.secrets import redact
            self.emit("error", {"message": redact(f"{type(exc).__name__}: {exc}")})
        finally:
            self._approval_event = None
            self._approval_context = None
            # Drop the run's credentials from this thread's context and from the
            # redactor. Clearing the specific values rather than the whole
            # registry leaves any other run's secrets registered.
            if self._provider_keys is not None:
                from core.provider_keys import clear_keys
                from core.secrets import clear_run_secrets
                clear_keys()
                clear_run_secrets(run_secrets)
                self._provider_keys = None
            self._persist_state()
            self._lock.release()

    # -- memory approval --------------------------------------------------
    def _memory_approval_callback(self, context: dict) -> bool:
        """Runs on the worker thread. Emits a pending event, then blocks until the
        frontend POSTs a decision — same blocking shape as desktop_app.py:3017-3125."""
        event = threading.Event()
        self._approval_event = event
        self._approval_decision = False
        self._approval_context = context
        with self._state_guard:
            self.status = AWAITING_APPROVAL
        self._persist_state()

        self.emit("memory_approval_pending", {
            "prompt": context.get("prompt"),
            "asset_name": context.get("asset_name"),
            "run_id": context.get("run_id"),
            "blueprint": context.get("blueprint"),
            "preview_path": context.get("preview_path"),
            "asset_path": context.get("asset_path"),
        })

        event.wait()
        with self._state_guard:
            self.status = RUNNING
        self._persist_state()
        return self._approval_decision

    def request_cancel(self) -> bool:
        """Returns False if there's nothing running to cancel."""
        if not self.is_busy():
            return False
        # Signal this run's scope explicitly. request_cancel() with no argument
        # would target *this* (HTTP request) thread's scope, which is the
        # process-wide fallback — not the worker's — so the run would never see
        # it. Passing the captured scope is what makes cancel cross the thread
        # boundary.
        from core.cancel import request_cancel as _request_cancel
        _request_cancel(self._cancel_scope)
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
        with self._state_guard:
            return {
                "status": self.status,
                "run_id": self.run_id,
                "asset_name": self.asset_name,
                "prompt": self.prompt,
            }


run_manager = RunManager()
