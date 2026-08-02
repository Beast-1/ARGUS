"""User-initiated cancellation. Never existed before this — not in the original
Tkinter app either — so this is new capability, not a restored one.

Hooked into the SAME checkpoints ARGUS_MAX_SECONDS already gates
(_past_deadline(), checked in the visual-improvement loop, the outer regen loop,
and best-of-N) rather than threading a new parameter through every loop. One
process-wide flag is correct because exactly one generation runs at a time
(RunManager's single-flight lock).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402


def test_idle_state_is_not_cancelled():
    main.clear_cancel()
    assert main.is_cancelled() is False


def test_request_cancel_sets_the_flag():
    main.clear_cancel()
    main.request_cancel()
    assert main.is_cancelled() is True


def test_clear_cancel_resets_it():
    main.request_cancel()
    main.clear_cancel()
    assert main.is_cancelled() is False


def test_past_deadline_trips_immediately_on_cancel_with_no_time_deadline():
    """The common case: ARGUS_MAX_SECONDS=0 (unbounded, the desktop/CLI default),
    so deadline=None — cancellation must still stop the loop."""
    main.clear_cancel()
    assert main._past_deadline(None) is False
    main.request_cancel()
    assert main._past_deadline(None) is True
    main.clear_cancel()


def test_past_deadline_still_works_without_cancellation():
    """Cancellation must be additive, not a regression of the existing
    ARGUS_MAX_SECONDS behavior other tests / real deployments depend on."""
    main.clear_cancel()
    future = 1e18
    past = 0.0
    assert main._past_deadline(future) is False
    assert main._past_deadline(past) is True
    main.clear_cancel()


def test_stop_reason_reflects_which_one_tripped():
    main.clear_cancel()
    assert main._stop_reason() == "deadline exceeded"
    main.request_cancel()
    assert main._stop_reason() == "cancelled by user"
    main.clear_cancel()
