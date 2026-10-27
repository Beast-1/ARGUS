"""The visual QA rubric — the instrument every reported quality number is read off.

It had no test coverage at all, which is how four measurement defects survived in
107 lines. These tests pin the legacy rubric in place (it is the frozen baseline
for the correction study), and they assert the defects as documented behaviour
rather than quietly fixing them, so the corrected rubric can be compared against
something stable.

Hermetic — pure functions over dicts; no Blender, network, or vision model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.checklist import (  # noqa: E402
    MAX_JOIN_CHECKS_TOTAL,
    MAX_PART_CHECKS,
    build_checklist,
    summarize_results,
)


def spec_with(n_parts: int, *, parents: bool = True) -> dict:
    parts = []
    for i in range(n_parts):
        part = {"id": f"p{i}", "primitive": "box"}
        if parents and i > 0:
            part["attach_to"] = f"p{i-1}"
        parts.append(part)
    return {"parts": parts}


def ids(checks, prefix: str) -> list[str]:
    return [c["id"] for c in checks if c["id"].startswith(prefix)]


def answer_all(checks, verdict: bool = True) -> dict:
    return {c["id"]: verdict for c in checks}


# --------------------------------------------------------------------------
# Legacy rubric: pinned, including its defects
# --------------------------------------------------------------------------

def test_legacy_truncates_parts_at_the_cap():
    """Defect 2: a 25-part asset is graded on the first 10 parts, in spec order."""
    checks = build_checklist({}, spec_with(25))
    assert len(ids(checks, "part_")) == MAX_PART_CHECKS
    assert "part_p9" in ids(checks, "part_")
    assert "part_p10" not in ids(checks, "part_")


def test_legacy_stops_asking_about_joins_after_seven_parts():
    """Defect 3: the cap counts part+join COMBINED, so connectivity — the property
    the compiler exists to guarantee — goes unchecked on complex assets.

    On a 25-part chain only 6 of the 24 joins are ever asked about: p0 is the
    root, and by p7 the running check count has reached MAX_JOIN_CHECKS_TOTAL,
    after which every remaining part contributes a part check and no join check.
    """
    checks = build_checklist({}, spec_with(25))
    join_ids = ids(checks, "join_")

    assert join_ids == [f"join_p{i}" for i in range(1, 7)]
    assert len(join_ids) == 6, "6 of 24 joins — the rest are never measured"
    assert len(ids(checks, "part_")) == MAX_PART_CHECKS
    # The combined cap only stops JOINS; part checks keep accruing past it.
    assert len(ids(checks, "part_")) + len(join_ids) > MAX_JOIN_CHECKS_TOTAL


def test_unanswered_checks_leave_the_denominator_and_raise_the_score():
    """Defect 1: skipping a check you would have failed IMPROVES the score."""
    checks = build_checklist({}, spec_with(4))
    honest = answer_all(checks, True)
    hard_id = ids(checks, "part_")[0]
    honest[hard_id] = False

    evasive = dict(honest)
    del evasive[hard_id]  # judge declined to answer instead of failing it

    honest_score = summarize_results(checks, honest)["visual_score"]
    evasive_score = summarize_results(checks, evasive)["visual_score"]
    assert evasive_score > honest_score, "the bias this rubric carries"
    assert evasive_score == 10


def test_coverage_exposes_the_unanswered_checks():
    checks = build_checklist({}, spec_with(4))
    answers = answer_all(checks, True)
    dropped = sorted(answers)[:3]
    for d in dropped:
        del answers[d]

    out = summarize_results(checks, answers)
    assert out["checks_asked"] == len(checks)
    assert out["checks_answered"] == len(checks) - 3
    assert out["coverage"] == round((len(checks) - 3) / len(checks), 4)
    assert sorted(out["unanswered"]) == dropped


def test_non_bool_answers_count_as_unanswered():
    checks = build_checklist({}, spec_with(3))
    answers = answer_all(checks, True)
    victim = checks[0]["id"]
    answers[victim] = "yes"  # a stringly-typed judge reply
    out = summarize_results(checks, answers)
    assert victim in out["unanswered"]
    assert out["checks_answered"] == len(checks) - 1


def test_nothing_answered_returns_empty_so_the_caller_can_fall_back():
    checks = build_checklist({}, spec_with(3))
    assert summarize_results(checks, {}) == {}


# --------------------------------------------------------------------------
# Full coverage: the correction
# --------------------------------------------------------------------------

def test_full_coverage_asks_about_every_part_and_every_join():
    checks = build_checklist({}, spec_with(25), full_coverage=True)
    assert len(ids(checks, "part_")) == 25
    assert len(ids(checks, "join_")) == 24  # every part but the root
    assert "part_p24" in ids(checks, "part_")
    assert "join_p24" in ids(checks, "join_")


def test_legacy_subset_is_a_strict_subset_of_full_coverage():
    part_data = {"category": "locomotive", "parts": ["a", "b"],
                 "critical_constraints": ["must sit flat"]}
    spec = spec_with(25)
    legacy = {c["id"] for c in build_checklist(part_data, spec)}
    full = build_checklist(part_data, spec, full_coverage=True)
    full_ids = {c["id"] for c in full}

    assert legacy < full_ids
    assert {c["id"] for c in full if c["legacy"]} == legacy


def test_every_legacy_check_is_flagged_legacy_in_legacy_mode():
    checks = build_checklist({}, spec_with(25))
    assert all(c["legacy"] for c in checks)


def test_full_coverage_does_not_move_the_legacy_score():
    """The frozen-baseline guarantee: visual_score is computed over the legacy
    subset either way, so switching the rubric on changes what is recorded and
    not what the pipeline optimises."""
    spec = spec_with(25)
    legacy_checks = build_checklist({}, spec)
    full_checks = build_checklist({}, spec, full_coverage=True)

    # one shared verdict per question id, so the two rubrics see the same judge
    verdicts = {c["id"]: (i % 3 != 0) for i, c in enumerate(full_checks)}

    legacy_out = summarize_results(legacy_checks, verdicts)
    full_out = summarize_results(full_checks, verdicts)

    assert full_out["visual_score"] == legacy_out["visual_score"]
    assert full_out["score_legacy_subset"] == legacy_out["visual_score"]
    assert full_out["checks_asked"] > legacy_out["checks_asked"]


def test_full_and_legacy_scores_diverge_on_the_parts_legacy_never_sees():
    """Failing only the parts beyond the legacy cap leaves the legacy score
    untouched — which is the defect, stated as a measurement."""
    spec = spec_with(25)
    full_checks = build_checklist({}, spec, full_coverage=True)
    verdicts = {c["id"]: c["legacy"] for c in full_checks}  # non-legacy all fail

    out = summarize_results(full_checks, verdicts)
    assert out["visual_score"] == 10
    assert out["score_full"] < 10


def test_balanced_score_reweights_a_part_heavy_asset():
    """Defect 4: a flat pass fraction lets 25 passing part checks drown three
    failing global checks. Equal category weights do not."""
    spec = spec_with(25)
    checks = build_checklist({}, spec, full_coverage=True)
    verdicts = {c["id"]: c["category"] != "global" for c in checks}

    out = summarize_results(checks, verdicts)
    assert out["score_full"] > 9
    assert out["score_balanced"] < out["score_full"]
    assert out["per_category"]["global"]["passed"] == 0


def test_failed_joins_are_reported_where_missing_parts_cannot_be():
    """missing_parts only ever collects part_ failures, so until now a pure
    connectivity failure was invisible to the repair loop."""
    checks = build_checklist({}, spec_with(5), full_coverage=True)
    verdicts = {c["id"]: c["category"] != "join" for c in checks}

    out = summarize_results(checks, verdicts)
    assert out["missing_parts"] == []
    assert set(out["failed_joins"]) == {f"p{i}" for i in range(1, 5)}


def test_raw_answers_are_returned_so_any_rule_can_be_recomputed_offline():
    checks = build_checklist({}, spec_with(4), full_coverage=True)
    verdicts = {c["id"]: (i % 2 == 0) for i, c in enumerate(checks)}
    verdicts[checks[0]["id"]] = None  # unanswered

    out = summarize_results(checks, verdicts)
    assert checks[0]["id"] not in out["answers"]
    assert all(isinstance(v, bool) for v in out["answers"].values())
    assert len(out["answers"]) == out["checks_answered"]


def test_categories_are_tagged_for_every_check():
    part_data = {"category": "locomotive", "parts": ["a"],
                 "critical_constraints": ["must sit flat"]}
    checks = build_checklist(part_data, spec_with(3), full_coverage=True)
    seen = {c["category"] for c in checks}
    assert seen == {"part", "join", "constraint", "global"}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_qa_history_records_verdicts_and_the_keep_decision(tmp_path):
    from main import _append_qa_history

    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    checks = build_checklist({}, spec_with(3), full_coverage=True)
    vqa = summarize_results(checks, answer_all(checks, True))

    _append_qa_history(script_dir, "visual_initial", 0, vqa, adopted=True)
    _append_qa_history(script_dir, "visual_repair", 1, vqa, adopted=False)

    lines = (tmp_path / "reports" / "qa_history.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["stage"] == "visual_initial" and first["adopted"] is True
    assert second["iteration"] == 1 and second["adopted"] is False
    assert first["answers"] == vqa["answers"]
    assert first["per_category"] == vqa["per_category"]


def test_strict_score_cannot_be_inflated_by_declining_to_answer(tmp_path=None):
    """The audit check that caught a real gap: score_full kept the legacy
    answered-only denominator, so a judge could still raise it from 6.92 to a
    perfect 10 by declining every question it would have failed. score_strict
    divides by the questions ASKED, which makes it invariant under that move."""
    spec = spec_with(12)
    checks = build_checklist({}, spec, full_coverage=True)

    honest = {c["id"]: True for c in checks}
    for c in checks[:8]:
        honest[c["id"]] = False
    evasive = {k: v for k, v in honest.items() if v}  # drop the failures entirely

    h = summarize_results(checks, honest)
    e = summarize_results(checks, evasive)

    assert e["score_full"] > h["score_full"], "score_full is still inflatable, by design"
    assert e["score_strict"] == h["score_strict"], "score_strict must not move"
    assert e["score_strict_balanced"] == h["score_strict_balanced"]
    assert e["coverage"] < h["coverage"], "but the evasion is visible in coverage"


def test_strict_is_a_lower_bound_on_full():
    """strict <= true <= full, so the gap between them bounds the judge's
    non-response bias rather than hiding it inside one number."""
    checks = build_checklist({}, spec_with(9), full_coverage=True)
    verdicts = {c["id"]: (i % 3 != 0) for i, c in enumerate(checks)}
    for c in checks[:4]:
        verdicts.pop(c["id"], None)

    out = summarize_results(checks, verdicts)
    assert out["score_strict"] <= out["score_full"]
    assert out["score_strict_balanced"] <= out["score_balanced"]


def test_full_coverage_keeps_repair_targets_on_the_legacy_subset():
    """Repair decisions must not shift when only the rubric is widened:
    missing_parts still reports legacy-scoped parts, so generate_spec_repair
    sees exactly what it saw before."""
    spec = spec_with(25)
    verdict_map = {}
    legacy = build_checklist({}, spec)
    full = build_checklist({}, spec, full_coverage=True)
    for c in full:
        verdict_map[c["id"]] = False  # everything fails

    legacy_out = summarize_results(legacy, verdict_map)
    full_out = summarize_results(full, verdict_map)
    assert full_out["missing_parts"] == legacy_out["missing_parts"]
    assert full_out["feedback"] == legacy_out["feedback"]
    # ...while the wider rubric still records the connectivity it alone saw
    assert len(full_out["failed_joins"]) > len(legacy_out["failed_joins"])


def test_holistic_mode_vqa_persists_without_the_measurement_keys(tmp_path):
    """The holistic fallback returns a vqa dict with no checklist fields. It must
    still record, with the new fields absent rather than fabricated."""
    from main import _append_qa_history

    holistic = {"visual_score": 6, "visible_parts": [], "missing_parts": ["lid"],
                "feedback": "lid missing", "skipped": False}
    _append_qa_history(tmp_path / "scripts", "visual_initial", 0, holistic, adopted=True)

    rec = json.loads((tmp_path / "reports" / "qa_history.jsonl").read_text(
        encoding="utf-8").strip())
    assert rec["visual_score"] == 6
    assert rec["missing_parts"] == ["lid"]
    assert rec["answers"] is None and rec["coverage"] is None


def test_regeneration_events_are_distinguishable_in_the_history(tmp_path):
    from main import _append_qa_history

    checks = build_checklist({}, spec_with(3), full_coverage=True)
    vqa = summarize_results(checks, answer_all(checks, True))
    _append_qa_history(tmp_path / "scripts", "outer_regen", 2, vqa, adopted=None)

    rec = json.loads((tmp_path / "reports" / "qa_history.jsonl").read_text(
        encoding="utf-8").strip())
    assert rec["stage"] == "outer_regen" and rec["iteration"] == 2
    assert rec["adopted"] is None, "a regeneration's fate is settled later"


def test_new_manifest_fields_are_backward_compatible(tmp_path):
    """eval/metrics.collect reads named keys, so an old manifest keeps working
    and the added keys are ignored rather than tripping anything up."""
    import struct

    from eval.metrics import collect

    def write_glb(path: Path) -> None:
        gltf = json.dumps({"asset": {"version": "2.0"}, "materials": [],
                           "meshes": [], "images": []}).encode()
        gltf += b" " * (-len(gltf) % 4)
        path.write_bytes(
            struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(gltf))
            + struct.pack("<II", len(gltf), 0x4E4F534A) + gltf)

    legacy_manifest = {"asset_name": "x", "prompt": "p", "visual_score": 7,
                       "topology_severity": "clean", "topology_grounded": True}
    extended = dict(legacy_manifest, checks_asked=48, checks_answered=45,
                    coverage=0.9375, score_full=9.11, score_strict=8.54,
                    score_balanced=9.2, score_strict_balanced=8.6,
                    score_legacy_subset=7, legacy_coverage=1.0,
                    per_category={"part": {"asked": 22, "answered": 22, "passed": 20}},
                    failed_joins=["p9"])

    for name, manifest in (("old", legacy_manifest), ("new", extended)):
        d = tmp_path / name
        d.mkdir()
        write_glb(d / "a.glb")
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        m = collect(d)
        assert m.has_manifest is True
        assert m.visual_score == 7
        assert m.topology_severity == "clean"
        assert m.topology_grounded is True
        assert m.notes == [], f"{name} manifest produced notes: {m.notes}"


def test_qa_history_skips_a_skipped_score_and_never_raises(tmp_path):
    from main import _append_qa_history

    _append_qa_history(tmp_path / "scripts", "visual_initial", 0, {"skipped": True})
    _append_qa_history(tmp_path / "scripts", "visual_initial", 0, None)
    assert not (tmp_path / "reports").exists()

    # an unwritable destination must not propagate out of telemetry
    _append_qa_history(tmp_path / "nope" / "\0bad", "visual_initial", 0,
                       {"visual_score": 5})
