"""Feeds canned `main.py` stdout through the parser and asserts the emitted events.

Runs without Blender or network access, matching docs/PUBLISHING.md's test bar.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.pipeline_events import PipelineEventWriter, score_kind  # noqa: E402


def collect(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    writer = PipelineEventWriter(lambda t, p: events.append((t, p)))
    writer.write(text)
    writer.flush()
    return events, writer


def types_of(events, wanted):
    return [p for t, p in events if t == wanted]


def test_stage_lines_are_typed_with_labels():
    events, _ = collect("\n[STAGE 55] Visual Feedback Loop\n" + "-" * 52 + "\n")
    stages = types_of(events, "stage")
    assert stages == [{"number": 55, "title": "Visual Feedback Loop", "label": "VISUAL"}]


def test_log_value_lines_become_metrics():
    events, _ = collect("Asset name        : ceramic_coffee_mug\n")
    assert types_of(events, "log_value") == [
        {"key": "Asset name", "value": "ceramic_coffee_mug"}
    ]


def test_windows_paths_are_not_mistaken_for_key_values():
    """A bare path line must not parse as "C: \\Users..." — the colon has no space after."""
    events, _ = collect("C:\\Users\\darpa\\out\\final\\mug\\mug.glb\n")
    assert types_of(events, "log_value") == []


def test_preview_image_marker_and_path_on_separate_lines():
    events, writer = collect("[PREVIEW IMAGE]\nC:\\out\\final\\mug\\mug_preview.png\n")
    previews = types_of(events, "preview_ready")
    assert previews == [{"kind": "preview", "path": "C:\\out\\final\\mug\\mug_preview.png"}]
    assert writer.preview_path == "C:\\out\\final\\mug\\mug_preview.png"


def test_inline_preview_image_marker():
    events, _ = collect("[PREVIEW IMAGE] C:\\out\\mug_preview.png\n")
    assert types_of(events, "preview_ready") == [
        {"kind": "preview", "path": "C:\\out\\mug_preview.png"}
    ]


def test_rejection_is_captured():
    events, writer = collect("\n[ARGUS] Request rejected: Too complex to build reliably.\n")
    assert types_of(events, "rejected") == [{"reason": "Too complex to build reliably."}]
    assert writer.rejection_reason == "Too complex to build reliably."


def test_graph_parts_json_is_parsed():
    events, _ = collect('[GRAPH_PARTS]["body", "handle"]\n')
    assert types_of(events, "graph_parts") == [{"parts": ["body", "handle"]}]


def test_malformed_graph_parts_does_not_raise():
    events, _ = collect("[GRAPH_PARTS]{not json\n")
    assert types_of(events, "graph_parts") == []


def test_scores_are_extracted_with_kind():
    events, writer = collect("Visual score      : 6/10 (multiview)\n")
    scores = types_of(events, "score")
    assert len(scores) == 1
    assert scores[0]["value"] == 6
    assert scores[0]["kind"] == "run"
    assert writer.last_score == 6


def test_completion_markers_are_tracked():
    _, writer = collect(
        "\n[ARGUS COMPLETE]\n"
        "Final Asset : C:\\out\\final\\mug\\mug.glb\n"
        "Preview     : C:\\out\\final\\mug\\mug_preview.png\n"
        "Project     : mug\n"
    )
    assert writer.final_asset == "C:\\out\\final\\mug\\mug.glb"
    assert writer.run_id == "mug"
    assert writer.preview_path == "C:\\out\\final\\mug\\mug_preview.png"


def test_partial_writes_are_buffered_across_calls():
    """print() does not guarantee one write() per line."""
    events: list[tuple[str, dict]] = []
    writer = PipelineEventWriter(lambda t, p: events.append((t, p)))
    writer.write("[STAGE 1] Plan")
    assert types_of(events, "stage") == []  # nothing emitted until the newline lands
    writer.write("ning and Structural Analysis\n")
    assert types_of(events, "stage") == [
        {"number": 1, "title": "Planning and Structural Analysis", "label": "PLAN"}
    ]


def test_every_line_also_reaches_the_raw_log():
    events, _ = collect("[STAGE 1] Planning\nAsset name        : mug\n")
    assert len(types_of(events, "log_line")) == 2


def test_texture_paint_line_carries_two_scores():
    """Regression guard for a real bug: a reverted texture-paint pass prints both the
    rejected and the kept score on one line, so the last scraped "N/10" in a run is NOT
    the final visual score. run_manager reads manifest.json instead — see _manifest_score.
    """
    line = "Texture paint     : reverted — repaint 6/10 < un-painted 7/10 (kept procedural)\n"
    events, writer = collect(line)
    scores = types_of(events, "score")
    # The scraper sees the *rejected* candidate's 6 first...
    assert scores[0]["value"] == 6
    assert writer.last_score == 6
    # ...while the asset that actually shipped scored 7. Hence: don't trust last_score
    # for the terminal `complete` event.
    assert "7/10" in line


def test_score_kind_thresholds_match_desktop_app():
    assert score_kind(10) == "ok"
    assert score_kind(7) == "ok"
    assert score_kind(6) == "run"
    assert score_kind(4) == "run"
    assert score_kind(3) == "fail"
    assert score_kind(0) == "fail"
