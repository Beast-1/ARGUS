"""Per-run cancellation scopes.

main.py originally held one process-wide `threading.Event`, justified by a true
observation: RunManager's single-flight lock means the desktop app only ever has
one generation in flight, so "cancel" was unambiguous. core/batch.py breaks that
premise — it runs N pipelines concurrently in one process via a ThreadPoolExecutor
— and its own docstring already warned that `--workers > 1` was unsafe.

That was survivable while cancellation was purely cooperative (a shared flag that
only stopped loops at their next checkpoint). It stopped being survivable once
core/blender.py:_run_cancellable started *terminating the Blender subprocess* on
that flag: with one shared Event, cancelling any run in a concurrent batch would
kill every other run's render mid-write.

The design has to satisfy two things that pull in opposite directions:

  1. Concurrent runs must not see each other's cancel   -> per-run state
  2. The thread that requests a cancel is NOT the thread running the pipeline
     (FastAPI's request handler vs. RunManager's worker thread)  -> a plain
     thread-local would break cancel for the desktop app entirely

So: a ContextVar holds *which* scope the current thread's pipeline belongs to
(giving isolation without threading a parameter through every loop in main.py),
while the Event itself is an ordinary object any thread can signal once it holds
a reference. RunManager captures that reference when the run starts and signals
it from the HTTP thread; batch workers each open their own scope and never see
each other's.

A process-wide fallback scope keeps the CLI and any caller that never opens a
scope working exactly as before.
"""
from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Optional

# The scope belonging to the pipeline running on this thread. None = "no scope
# opened here", which falls back to the process-wide Event below.
_current_scope: ContextVar[Optional[threading.Event]] = ContextVar(
    "argus_cancel_scope", default=None
)

# Fallback for single-run callers (the CLI) that never call new_scope().
_process_scope = threading.Event()


def new_scope() -> threading.Event:
    """Open a fresh cancel scope for one run and make it current on this thread.

    Call this at the start of a run, on the thread that will run the pipeline.
    Returns the Event so the caller can hand it to whoever might cancel it — see
    service/run_manager.py, which stores it so the HTTP thread can signal a run
    executing on a worker thread.
    """
    event = threading.Event()
    _current_scope.set(event)
    return event


def adopt_scope(event: threading.Event) -> None:
    """Make an already-created scope current on *this* thread.

    Needed because a ContextVar set on one thread is invisible to another: the
    scope is created when the run is accepted (on the HTTP thread, so a cancel
    arriving immediately has something to signal) but must then be adopted by
    the worker thread that actually runs the pipeline, or that worker would fall
    back to the process-wide scope and ignore the cancel entirely.
    """
    _current_scope.set(event)


def current_scope() -> threading.Event:
    """The Event this thread's pipeline should be checking."""
    return _current_scope.get() or _process_scope


def request_cancel(scope: Optional[threading.Event] = None) -> None:
    """Signal cancellation. Pass an explicit scope to cancel a run executing on
    a different thread; omit it to cancel this thread's own run."""
    (scope or current_scope()).set()


def clear_cancel(scope: Optional[threading.Event] = None) -> None:
    """Reset a scope. Only meaningful for the process-wide fallback, which
    outlives a run; a per-run scope is discarded with the run."""
    (scope or current_scope()).clear()


def is_cancelled(scope: Optional[threading.Event] = None) -> bool:
    return (scope or current_scope()).is_set()
