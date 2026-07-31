"""Regression test for a real incident: a 15-cell sweep hit a mid-run network
outage (every provider failed with NameResolutionError from ~cell 3 onward), and
13 of 15 cells silently recorded visual_score=None with no way to tell "the model
tried and produced nothing" apart from "the network was down and nothing ran".

These lock in the fix: env_failure detection, exclusion from aggregate stats
(rather than being counted as failed attempts), and automatic retry via the
existing resume mechanism.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import summarise  # noqa: E402
from eval.runner import _ENV_FAILURE_SIGNATURES, completed_cells  # noqa: E402

REAL_OUTAGE_LOG_EXCERPT = (
    "[GEMINI_CODE] gemini-2.0-flash failed: HTTPSConnectionPool"
    "(host='generativelanguage.googleapis.com', port=443): Max retries exceeded "
    "with url: /v1beta/models/gemini-2.0-flash:generateContent?key=AIza... "
    "(Caused by NameResolutionError(\"HTTPSConnection(host="
    "'generativelanguage.googleapis.com', port=443): Failed to resolve "
    "'generativelanguage.googleapis.com' ([Errno 11001] getaddrinfo failed)\"))"
)


def test_the_actual_outage_log_is_detected():
    assert any(sig in REAL_OUTAGE_LOG_EXCERPT for sig in _ENV_FAILURE_SIGNATURES)


def test_a_normal_failure_log_is_not_misclassified():
    """A model that genuinely produces a bad script must not be excused as an
    environment failure just because some unrelated word overlaps."""
    normal_failure = (
        "[STAGE 5] Blender Execution and Export\n"
        "Blender status    : failed\n"
        "Error type        : SyntaxError\n"
    )
    assert not any(sig in normal_failure for sig in _ENV_FAILURE_SIGNATURES)


def test_env_failed_cells_are_excluded_from_aggregates():
    records = [
        {"exists": True, "visual_score": 7, "topology_severity": "clean",
         "textured_materials": 1, "triangles": 100, "elapsed_sec": 600},
        {"exists": True, "visual_score": 7, "topology_severity": "clean",
         "textured_materials": 1, "triangles": 100, "elapsed_sec": 600},
        # 13 corrupted cells, exactly as the real incident produced:
        *[{"exists": True, "visual_score": None, "topology_severity": None,
           "textured_materials": 0, "triangles": 44, "elapsed_sec": 5,
           "env_failure": "NameResolutionError"} for _ in range(13)],
    ]
    s = summarise(records)
    # The 2 genuine results must read as a clean 7.0 average, not be diluted by
    # 13 meaningless None scores averaging in as failures.
    assert s["n_attempted"] == 2
    assert s["n_env_failures"] == 13
    assert s["mean_visual_score"] == 7
    assert s["build_rate"] == 1.0


def test_env_failed_cells_are_not_counted_as_completed(tmp_path):
    """This is what makes resuming a corrupted sweep a single command instead of
    a manual diagnosis: re-running the same invocation must retry exactly the
    cells an outage corrupted."""
    results = tmp_path / "runs.jsonl"
    results.write_text(
        '{"condition": "full", "prompt_id": "b01", "visual_score": 7}\n'
        '{"condition": "full", "prompt_id": "b02", "visual_score": null, '
        '"env_failure": "NameResolutionError"}\n',
        encoding="utf-8",
    )
    done = completed_cells(results)
    assert ("full", "b01") in done
    assert ("full", "b02") not in done


def test_malformed_final_line_does_not_crash_resume(tmp_path):
    """A sweep killed mid-write (Ctrl-C, power loss) can leave a truncated last
    line; that must not crash resume, just be re-run like a missing cell."""
    results = tmp_path / "runs.jsonl"
    results.write_text(
        '{"condition": "full", "prompt_id": "b01", "visual_score": 7}\n'
        '{"condition": "full", "prompt_id": "b02", "visual_sc',
        encoding="utf-8",
    )
    done = completed_cells(results)
    assert done == {("full", "b01")}
