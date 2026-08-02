"""RunManager's cancel handling.

The tricky part: a cancelled run still goes through run_pipeline's normal
completion path (main.py's loops stop cooperatively at the next checkpoint,
then the remaining stages run as usual), so `success=True` alone can't tell
"finished" from "stopped early because you asked it to" — RunManager must check
main.is_cancelled() to report the state actually asked for.

Fully mocked — no LLM, Blender, or network access.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from service.run_manager import AWAITING_APPROVAL, CANCELLED, COMPLETE, RunManager  # noqa: E402

COMPLETION_LOG = (
    "\n[ARGUS COMPLETE]\n"
    "Final Asset : C:\\out\\final\\thing\\thing.glb\n"
    "Project     : thing\n"
)


def test_normal_completion_reports_complete(monkeypatch):
    def fake_run_pipeline(**kwargs):
        print(COMPLETION_LOG)
        return True

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    main.clear_cancel()

    mgr = RunManager()
    mgr._lock.acquire()  # _run()'s own finally releases it, matching start()'s flow
    mgr._run("a thing", None, False, True)

    assert mgr.status == COMPLETE
    assert mgr.run_id == "thing"


def test_cancelled_run_with_asset_reports_cancelled_not_complete(monkeypatch):
    """The core case: the fake pipeline finishes normally (as a cancelled run
    really does — it stops the loop, then still runs export/validation/etc.),
    but cancellation was requested partway through."""
    def fake_run_pipeline(**kwargs):
        print("[STAGE 55] Visual Feedback Loop")
        main.request_cancel()  # simulates cancel arriving mid-run
        print(COMPLETION_LOG)
        return True

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    main.clear_cancel()

    mgr = RunManager()
    mgr._lock.acquire()  # _run()'s own finally releases it, matching start()'s flow
    mgr._run("a thing", None, False, True)

    assert mgr.status == CANCELLED
    assert mgr.run_id == "thing"  # the partial result is still surfaced

    events = [e["type"] for e in mgr._history]
    assert "cancelled" in events
    assert "complete" not in events
    main.clear_cancel()


def test_cancelled_before_any_asset_reports_cancelled_with_reason(monkeypatch):
    def fake_run_pipeline(**kwargs):
        main.request_cancel()
        return False

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    main.clear_cancel()

    mgr = RunManager()
    mgr._lock.acquire()  # _run()'s own finally releases it, matching start()'s flow
    mgr._run("a thing", None, False, True)

    assert mgr.status == CANCELLED
    cancelled_events = [e for e in mgr._history if e["type"] == "cancelled"]
    assert cancelled_events and "reason" in cancelled_events[0]["payload"]
    main.clear_cancel()


def test_request_cancel_returns_false_when_idle():
    mgr = RunManager()
    assert mgr.request_cancel() is False


def test_cancel_while_awaiting_approval_unblocks_the_run(monkeypatch):
    """The pipeline thread blocks inside the approval callback's Event.wait().
    request_cancel() must resolve that wait (as "don't save") or cancellation
    would hang forever waiting for a decision nobody's making."""
    def fake_run_pipeline(**kwargs):
        approved = kwargs["memory_approval_callback"]({
            "prompt": "x", "asset_name": "thing", "run_id": "thing",
        })
        assert approved is False
        print(COMPLETION_LOG)
        return True

    monkeypatch.setattr(main, "run_pipeline", fake_run_pipeline)
    main.clear_cancel()

    mgr = RunManager()
    assert mgr.start("a thing") is True

    deadline = time.monotonic() + 5
    while mgr.status != AWAITING_APPROVAL and time.monotonic() < deadline:
        time.sleep(0.02)
    assert mgr.status == AWAITING_APPROVAL, "never reached the approval gate"

    assert mgr.request_cancel() is True

    deadline = time.monotonic() + 5
    while mgr.is_busy() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert mgr.status == CANCELLED
    main.clear_cancel()
