"""The one consolidated stdout -> typed-event parser.

`main.py` communicates everything except its bare True/False/None return via `print()`.
Historically two consumers independently regex-scraped overlapping subsets of those lines
(`desktop_app.py`'s `_process_operator_log_line`/`_update_stage_from_log`, and
`service/worker.py`'s `_JobLog`). This module replaces both with a single parser.

Matchers are built from what `main.py` actually prints today (verified against source),
not copied from desktop_app.py — which carries stale matchers for lines the pipeline no
longer emits (e.g. "Critic action").
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from core.secrets import redact

__all__ = ["PipelineEventWriter", "STAGE_LABELS", "score_kind"]

EmitFn = Callable[[str, dict], None]

# "\n[STAGE 55] Visual Feedback Loop"  (main.py:226-228, log_stage)
_STAGE_RE = re.compile(r"^\[STAGE (\d+)\]\s*(.*)$")

# "Asset name        : foo"   (main.py:231-232, log_value — 18-wide label)
# "Final Asset : foo"          (main.py:1489, hardcoded narrower padding)
# Requires whitespace after the colon so Windows paths ("C:\...") never match.
_KV_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 ()/_+-]{0,24}?)\s*:\s+(.+)$")

# any "6/10"-style score  (visual score, candidate score, best-of-N, full regen)
_SCORE_RE = re.compile(r"(-?\d+)\s*/\s*10")

_REJECTED_PREFIX = "[ARGUS] Request rejected:"
_GRAPH_PARTS_PREFIX = "[GRAPH_PARTS]"
_PREVIEW_MARKER = "[PREVIEW IMAGE]"

# Stage number -> short label, ported verbatim from desktop_app.py:2425-2438.
STAGE_LABELS = {
    1: "PLAN",
    2: "NAMING",
    3: "SCRIPT",
    4: "REVIEW",
    5: "EXPORT",
    6: "TOPOLOGY",
    7: "QUALITY",
    8: "REFERENCE",
    54: "CANDIDATES",
    55: "VISUAL",
    56: "REGEN",
    75: "VISUAL QA",
    77: "PAINTER",
}

# Labels worth surfacing as structured metrics rather than only raw log text.
_METRIC_LABELS = {
    "asset name", "project folder", "poly budget", "category", "geometry",
    "blueprint", "memory", "style", "material", "generation", "build spec",
    "blender status", "topology status", "topology source", "severity",
    "n-gons", "open edges", "grounded", "memory status", "reason",
    "repair required", "repair status", "glb", "fbx", "blend", "preview",
    "visual score", "visual target", "visual loop final", "best-of-n final",
    "texture paint", "final asset", "project", "reference", "status",
}


def score_kind(score: Optional[int]) -> Optional[str]:
    """Ported from desktop_app.py:968-974 so the frontend never re-implements
    thresholds. The single canonical definition — service/projects.py imports this
    rather than keeping its own copy, which had drifted into two independently
    maintained but identical implementations."""
    if score is None:
        return None
    if score >= 7:
        return "ok"
    if score >= 4:
        return "run"
    return "fail"


class PipelineEventWriter:
    """File-like stdout/stderr sink that turns `main.py`'s printed lines into typed events.

    Replaces desktop_app.py's `_QueueWriter` + line-parsing methods and
    service/worker.py's `_JobLog` in one place.
    """

    def __init__(self, emit: EmitFn) -> None:
        self._emit = emit
        self._buffer = ""
        self._expect_preview_path = False
        # Captured for the terminal `complete` event, which is emitted by the run
        # manager from real return values rather than scraped text.
        self.final_asset: Optional[str] = None
        self.preview_path: Optional[str] = None
        self.run_id: Optional[str] = None
        self.asset_name: Optional[str] = None
        self.rejection_reason: Optional[str] = None
        self.last_score: Optional[int] = None

    # -- file-like API -------------------------------------------------
    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            line, self._buffer = self._buffer, ""
            self._handle_line(line)

    # -- parsing -------------------------------------------------------
    def _handle_line(self, raw: str) -> None:
        # Redact before anything else touches the line. This is the widest exit
        # the pipeline has: every line of stdout/stderr is forwarded verbatim to
        # the browser below, and on a tunnelled deployment that browser is on the
        # public internet. Logging never reaches this writer (main.py binds its
        # StreamHandler to the real sys.stdout before the redirect), so the
        # logging-side redaction provably does not cover this path.
        line = redact(raw.rstrip())
        # Nothing is silently dropped: every line also reaches the raw console pane.
        self._emit("log_line", {"text": line})

        stripped = line.strip()
        if not stripped:
            return

        # [PREVIEW IMAGE] prints the marker and the path as two separate calls
        # (main.py:1182-1183), so the path arrives on the following line.
        if self._expect_preview_path:
            self._expect_preview_path = False
            if stripped and not stripped.startswith("["):
                self.preview_path = stripped
                self._emit("preview_ready", {"kind": "preview", "path": stripped})
                return

        if stripped.startswith(_PREVIEW_MARKER):
            rest = stripped[len(_PREVIEW_MARKER):].strip()
            if rest:  # inline variant, emitted by core/blender.py during the build
                self.preview_path = rest
                self._emit("preview_ready", {"kind": "preview", "path": rest})
            else:
                self._expect_preview_path = True
            return

        if stripped.startswith(_REJECTED_PREFIX):
            reason = stripped[len(_REJECTED_PREFIX):].strip()
            self.rejection_reason = reason
            self._emit("rejected", {"reason": reason})
            return

        if stripped.startswith(_GRAPH_PARTS_PREFIX):
            payload = stripped[len(_GRAPH_PARTS_PREFIX):]
            try:
                parts = json.loads(payload)
            except (ValueError, TypeError):
                return
            self._emit("graph_parts", {"parts": parts})
            return

        stage_match = _STAGE_RE.match(stripped)
        if stage_match:
            number = int(stage_match.group(1))
            self._emit("stage", {
                "number": number,
                "title": stage_match.group(2).strip(),
                "label": STAGE_LABELS.get(number, f"STAGE {number}"),
            })
            return

        kv_match = _KV_RE.match(stripped)
        if kv_match:
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip()
            self._track_key(key, value)
            if key.lower() in _METRIC_LABELS:
                self._emit("log_value", {"key": key, "value": value})

        # Scores can ride on key/value lines ("Visual score      : 6/10") or on
        # bare progress lines ("Candidate 2/3: score 7/10"), so check separately.
        score_match = _SCORE_RE.search(stripped)
        if score_match:
            value = int(score_match.group(1))
            if 0 <= value <= 10:
                self.last_score = value
                self._emit("score", {
                    "value": value,
                    "kind": score_kind(value),
                    "source": stripped.split(":")[0].strip()[:48],
                })

    def _track_key(self, key: str, value: str) -> None:
        low = key.lower()
        if low == "final asset":
            self.final_asset = value
        elif low == "project":
            self.run_id = value
        elif low == "asset name":
            self.asset_name = value
        elif low == "preview":
            self.preview_path = value
