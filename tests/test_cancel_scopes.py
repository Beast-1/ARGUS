"""core/cancel.py has to satisfy two requirements that pull against each other:

  1. Concurrent runs (core/batch.py --workers N) must not see each other's
     cancel. A single shared flag meant cancelling one job would kill every
     other job's Blender subprocess mid-write, once cancellation stopped being
     purely cooperative.
  2. The thread requesting a cancel is NOT the thread running the pipeline
     (FastAPI request handler vs. RunManager's worker thread). A plain
     thread-local satisfies (1) and breaks (2) completely — cancel would
     silently never reach the run.

Both directions are tested here, because getting one right while breaking the
other is the obvious failure mode and produces no error, just a Cancel button
that does nothing.

Hermetic: threads and Events only.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cancel import (  # noqa: E402
    adopt_scope,
    current_scope,
    is_cancelled,
    new_scope,
    request_cancel,
)


def test_concurrent_scopes_are_isolated():
    """Requirement 1: cancelling one worker must not cancel its siblings."""
    scopes: dict[int, threading.Event] = {}
    observed: dict[int, bool] = {}
    opened = threading.Barrier(3)
    checked = threading.Barrier(3)

    def worker(i: int):
        scopes[i] = new_scope()
        opened.wait(timeout=5)
        # worker 0 is cancelled by the main thread here
        checked.wait(timeout=5)
        observed[i] = is_cancelled()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    opened.wait(timeout=5)
    request_cancel(scopes[0])          # cancel exactly one job
    checked.wait(timeout=5)
    for t in threads:
        t.join(timeout=5)

    assert observed[0] is True, "the cancelled worker should see its own cancel"
    assert observed[1] is False, (
        "a sibling worker saw another job's cancel — this is the bug that made "
        "--workers > 1 unsafe"
    )


def test_cancel_crosses_the_thread_boundary_when_given_the_scope():
    """Requirement 2: the desktop app's actual shape — scope created on one
    thread, adopted by the worker, signalled from the first."""
    scope = new_scope()                     # 'request' thread creates it
    seen = threading.Event()
    result = {}

    def worker():
        adopt_scope(scope)                  # worker adopts it
        assert not is_cancelled()
        seen.set()
        cancelled_at_checkpoint.wait(timeout=5)
        result["cancelled"] = is_cancelled()

    cancelled_at_checkpoint = threading.Event()
    t = threading.Thread(target=worker)
    t.start()
    assert seen.wait(timeout=5)
    request_cancel(scope)                   # signalled from this thread
    cancelled_at_checkpoint.set()
    t.join(timeout=5)

    assert result["cancelled"] is True, (
        "worker did not observe a cancel signalled from another thread — a "
        "thread-local would fail exactly here, and the Cancel button would "
        "silently do nothing"
    )


def test_threads_without_a_scope_all_share_the_process_fallback():
    """The CLI never opens a scope; it must still be cancellable, and every
    scope-less caller must agree on which Event that is.

    Both observations are taken from fresh threads on purpose: new_scope()
    durably rebinds the *calling* thread's context, so the main thread here has
    a scope left over from the tests above. That is correct behaviour (a worker
    thread keeps its run's scope for the life of the run) but it makes the main
    thread a bad place to sample the fallback from.
    """
    seen: list[threading.Event] = []

    def scopeless_worker():
        seen.append(current_scope())   # no new_scope() call at all

    threads = [threading.Thread(target=scopeless_worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert len(seen) == 2
    assert seen[0] is seen[1], "scope-less threads must share one fallback Event"
    assert not seen[0].is_set()


def test_run_manager_signals_the_worker_scope_not_its_own(monkeypatch):
    """Integration-shaped: RunManager must pass its captured scope to
    request_cancel, not call the no-arg form (which would target the HTTP
    thread's scope and never reach the run)."""
    import service.run_manager as rm

    mgr = rm.RunManager()
    mgr._cancel_scope = threading.Event()
    mgr.status = rm.RUNNING

    assert mgr.request_cancel() is True
    assert mgr._cancel_scope.is_set(), (
        "request_cancel did not signal the run's own scope"
    )
