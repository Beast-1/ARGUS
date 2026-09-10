"""Regression test for a real incident: the first --workers>1 sweep run pushed a
cell past its 2100s timeout under machine load, and run_cell's TimeoutExpired
handler crashed the ENTIRE sweep (not just that one cell) with
`AttributeError: 'str' object has no attribute 'decode'` — subprocess.run(...,
text=True) already decodes captured output, so TimeoutExpired.stdout is a str
here too, not bytes, and the old code unconditionally called .decode() on it.

Hermetic: mocks subprocess.run, no real Blender/network/2100s wait.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.runner import _find_asset, run_cell  # noqa: E402


def test_timeout_with_str_stdout_does_not_crash(tmp_path):
    """The exact incident: text=True means exc.stdout is already a str."""
    partial_output = "[STAGE 1] Planning\n[STAGE 3] Blender Script Generation\n"
    timeout_exc = subprocess.TimeoutExpired(cmd=["python", "main.py"], timeout=2100)
    timeout_exc.stdout = partial_output  # str, as text=True produces

    with patch("eval.runner.subprocess.run", side_effect=timeout_exc):
        record = run_cell(
            {"id": "b01", "prompt": "a wooden crate", "tier": "simple", "category": "container"},
            "full",
            tmp_path,
            timeout=2100,
            log_dir=tmp_path / "logs",
        )

    assert record["timed_out"] is True
    assert record["returncode"] == -1
    assert record["exists"] is False
    log_file = tmp_path / "logs" / "full__b01.log"
    assert log_file.read_text(encoding="utf-8") == partial_output


def test_process_killed_before_printing_anything_is_flagged_for_retry(tmp_path):
    """Real incident: the sweep's parent process was torn down mid-run (a session
    boundary), and several not-yet-started `main.py` children were killed near-
    instantly by the OS — 0-byte log, abnormal returncode (observed:
    3221226091 / 0xC0000409 on Windows). Before this fix, run_cell had no way to
    tell that apart from a cell that legitimately produced nothing, so
    completed_cells() wrongly treated 72 of these as done and eval.report averaged
    in whatever _find_asset's stale "newest folder" fallback happened to match."""
    class _FakeCompletedProcess:
        returncode = 3221226091
        stdout = ""

    with patch("eval.runner.subprocess.run", return_value=_FakeCompletedProcess()):
        record = run_cell(
            {"id": "b01", "prompt": "a wooden crate", "tier": "simple", "category": "container"},
            "fast",
            tmp_path,
            timeout=2100,
            log_dir=tmp_path / "logs",
        )

    assert record["exists"] is False
    assert record["env_failure"] is not None
    assert "3221226091" in record["env_failure"]


def test_find_asset_does_not_fall_back_to_newest_folder_on_empty_log(tmp_path):
    """out_root/final/ is shared across all 15 prompts in a condition — falling
    back to "newest folder" when the log is empty (nothing to identify which
    prompt this was) silently attributes an unrelated earlier prompt's real
    asset to this cell. Real incident: a 'fast' cell with an empty log (process
    killed before it ran) was matched to a leftover "wheelbarrow" directory from
    a completely different prompt."""
    final = tmp_path / "final"
    (final / "some_earlier_unrelated_asset").mkdir(parents=True)

    assert _find_asset(tmp_path, "") is None
    assert _find_asset(tmp_path, "   \n  ") is None
    # Sanity check the fallback still works when there IS real log content to
    # justify it (the original, legitimate use of "newest folder").
    assert _find_asset(tmp_path, "some real stage output\nBlender status: ok\n") is not None


def test_timeout_with_no_captured_output(tmp_path):
    """A timeout before the subprocess wrote anything at all — exc.stdout is None."""
    timeout_exc = subprocess.TimeoutExpired(cmd=["python", "main.py"], timeout=2100)
    timeout_exc.stdout = None

    with patch("eval.runner.subprocess.run", side_effect=timeout_exc):
        record = run_cell(
            {"id": "b02", "prompt": "a steel oil drum", "tier": "simple", "category": "container"},
            "full",
            tmp_path,
            timeout=2100,
            log_dir=tmp_path / "logs",
        )

    assert record["timed_out"] is True
