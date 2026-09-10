"""core/blender.py's _run_cancellable(): previously every Blender invocation used
subprocess.run() with a hard timeout as its ONLY exit path, so clicking Cancel
mid-render just set a flag nothing checked until the pipeline's next loop-boundary
— the process kept running untouched for up to BLENDER_EXEC_TIMEOUT (5 minutes
default). This locks in that a cancel request now actually terminates the child
process promptly.

Hermetic in the sense the repo cares about (no Blender, no network) — it does spawn
a real short-lived `python` subprocess (via sys.executable) to verify the actual
Popen/communicate/terminate mechanics, since mocking Popen entirely would just
re-assert the implementation rather than prove the polling loop really works.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.blender as blender_mod  # noqa: E402

_SLEEP_30S = [sys.executable, "-c", "import time; time.sleep(30)"]
_SLEEP_02S = [sys.executable, "-c", "import time; time.sleep(0.2); print('done')"]


def test_returns_normally_when_process_finishes_before_cancel_or_timeout():
    with patch("main.is_cancelled", return_value=False):
        result = blender_mod._run_cancellable(_SLEEP_02S, timeout=10)
    assert result.returncode == 0
    assert "done" in result.stdout


def test_terminates_promptly_on_cancel_instead_of_waiting_out_the_timeout():
    """The actual bug: previously nothing would touch the subprocess until a
    5-minute timeout. Here the subprocess is given 30s to run but cancel fires
    after ~0.5s — a correct implementation returns in low single-digit seconds,
    not anywhere close to 30."""
    cancelled = threading.Event()

    def _is_cancelled():
        return cancelled.is_set()

    def _flip_after_delay():
        time.sleep(0.5)
        cancelled.set()

    threading.Thread(target=_flip_after_delay, daemon=True).start()

    t0 = time.monotonic()
    with patch("main.is_cancelled", side_effect=_is_cancelled):
        try:
            blender_mod._run_cancellable(_SLEEP_30S, timeout=30)
            raised = False
        except subprocess.TimeoutExpired:
            raised = True
    elapsed = time.monotonic() - t0

    assert raised, "cancelling should surface as TimeoutExpired, same as a real timeout"
    assert elapsed < 5, (
        f"took {elapsed:.1f}s to react to cancel — should be ~0.5-1s, not "
        "anywhere near the 30s process timeout"
    )


def test_kills_the_process_it_terminated_on_cancel():
    """Not just fast — the child must actually be gone, not orphaned."""
    cancelled = threading.Event()
    proc_holder: dict = {}

    real_popen = subprocess.Popen

    def _tracking_popen(*args, **kwargs):
        p = real_popen(*args, **kwargs)
        proc_holder["proc"] = p
        return p

    def _flip_after_delay():
        time.sleep(0.5)
        cancelled.set()

    threading.Thread(target=_flip_after_delay, daemon=True).start()

    with patch("main.is_cancelled", side_effect=lambda: cancelled.is_set()), \
         patch("core.blender.subprocess.Popen", side_effect=_tracking_popen):
        try:
            blender_mod._run_cancellable(_SLEEP_30S, timeout=30)
        except subprocess.TimeoutExpired:
            pass

    proc = proc_holder["proc"]
    # Give the OS a brief moment to reflect the termination in poll().
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    assert proc.poll() is not None, "child process should have exited, not been left running"


def test_still_raises_timeout_expired_when_never_cancelled():
    with patch("main.is_cancelled", return_value=False):
        t0 = time.monotonic()
        try:
            blender_mod._run_cancellable(_SLEEP_30S, timeout=1)
            raised = False
        except subprocess.TimeoutExpired:
            raised = True
        elapsed = time.monotonic() - t0
    assert raised
    assert elapsed < 5
