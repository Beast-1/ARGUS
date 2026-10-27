"""Spec-derived visual QA checklists.

A holistic "rate this 1-10" is noisy from free-tier vision models; a list of
concrete yes/no questions generated from the build spec is far more reliable,
and every failed check maps directly to a repairable part.

Two rubrics live here. The legacy rubric is the one every number in the current
results was produced under, and it is kept bit-identical so it stays a usable
frozen baseline. `full_coverage` removes the two construction caps that make the
legacy rubric grade different objects on different question sets. Both are
scored from the same answers, so one vision call yields a paired comparison.

Four defects in the legacy rubric are preserved deliberately, so they can be
measured rather than silently corrected:

  1. An unanswered check leaves the denominator instead of failing it, so a
     judge that skips the questions it finds hard scores HIGHER. `coverage`
     records how much of the checklist came back, and `score_strict` divides by
     the questions ASKED rather than answered, which makes it invariant under
     that evasion: declining a check you would have failed no longer helps.
     `score_full` keeps the answered-only denominator and is therefore still
     inflatable, so the pair brackets the truth --
     score_strict <= true quality <= score_full -- and the width of that
     bracket is exactly the judge's non-response bias.
  2. Only the first MAX_PART_CHECKS parts are asked about, in spec order, so a
     25-part asset is graded on an arbitrary subset of itself.
  3. Join checks stop once part+join checks reach MAX_JOIN_CHECKS_TOTAL, so
     connectivity -- the property the compiler exists to guarantee -- goes
     unmeasured on exactly the complex assets where it is hardest to get right.
  4. The score is a flat pass fraction over a question mix that shifts with part
     count, so part-heavy and part-light assets are graded on differently
     weighted rubrics. `score_balanced` weights each category equally instead.

Nothing here changes what the pipeline does: `visual_score` keeps its legacy
value and meaning, and every corrected measurement is an additional key that no
control flow reads.
"""
from __future__ import annotations

MAX_PART_CHECKS = 10
MAX_CONSTRAINT_CHECKS = 5
# Legacy cap on the COMBINED part+join count, not on joins alone -- which is why
# joins stop appearing after roughly seven parts.
MAX_JOIN_CHECKS_TOTAL = MAX_PART_CHECKS + 4

CATEGORIES = ("part", "join", "constraint", "global")

_VEHICLE_TERMS = ("wheel", "vehicle", "car", "truck", "locomotive", "train",
                  "cart", "wagon", "trailer")


def _part_and_join_checks(part_data: dict, spec: dict | None,
                          *, full_coverage: bool) -> list[dict]:
    checks: list[dict] = []
    seen_parts: set[str] = set()

    parts = (spec or {}).get("parts") or []
    for pt in (parts if full_coverage else parts[:MAX_PART_CHECKS]):
        pid = str(pt.get("id", "")).strip()
        if not pid or pid in seen_parts:
            continue
        seen_parts.add(pid)
        prim = pt.get("primitive", "part")
        label = pid.replace("_", " ")
        q = f"Is the {label} ({prim}) clearly visible and correctly shaped?"
        checks.append({"id": f"part_{pid}", "part": pid, "category": "part",
                       "question": q})
        parent = str(pt.get("attach_to") or "").strip()
        if not parent:
            continue
        if not full_coverage and len(checks) >= MAX_JOIN_CHECKS_TOTAL:
            continue
        checks.append({
            "id": f"join_{pid}", "part": pid, "category": "join",
            "question": (f"Does the {label} appear physically connected to the "
                         f"{parent.replace('_', ' ')} (no visible gap)?"),
        })

    if not parts:
        names = part_data.get("parts") or []
        for name in (names if full_coverage else names[:MAX_PART_CHECKS]):
            pid = str(name).strip().lower().replace(" ", "_")
            if not pid or pid in seen_parts:
                continue
            seen_parts.add(pid)
            checks.append({
                "id": f"part_{pid}", "part": pid, "category": "part",
                "question": f"Is the {str(name)} clearly visible?",
            })
    return checks


def _constraint_checks(part_data: dict) -> list[dict]:
    out: list[dict] = []
    raw = (part_data.get("critical_constraints") or [])[:MAX_CONSTRAINT_CHECKS]
    for i, c in enumerate(raw):
        text = str(c).strip()
        if not text or text.lower().startswith("similar structure memory"):
            continue
        out.append({"id": f"constraint_{i}", "part": "", "category": "constraint",
                    "question": f'Is this satisfied: "{text}"?'})
    return out


def _global_checks(part_data: dict) -> list[dict]:
    out = [
        {"id": "grounded", "part": "", "category": "global",
         "question": "Is the object resting on the ground with nothing floating in mid-air?"},
        {"id": "proportions", "part": "", "category": "global",
         "question": "Are the overall proportions plausible for this kind of object?"},
        {"id": "no_embedded", "part": "", "category": "global",
         "question": ("Does every part read as a distinct piece — with NO part "
                      "more than half sunk inside another (e.g. wheels buried in "
                      "the body, parts overlapping into each other)?")},
    ]
    haystack = (str(part_data.get("category", "")) + " "
                + " ".join(map(str, part_data.get("parts", [])))).lower()
    if any(t in haystack for t in _VEHICLE_TERMS):
        out.append({"id": "wheels_under", "part": "", "category": "global",
                    "question": ("Are the wheels sitting UNDER the body (their tops at or "
                                 "below the body's underside), touching the ground, and "
                                 "sized in proportion — not oversized discs cutting into "
                                 "the body?")})
    return out


def build_checklist(part_data: dict, spec: dict | None = None,
                    *, full_coverage: bool = False) -> list[dict]:
    """Return [{id, part, category, question, legacy}, ...] for the vision model.

    full_coverage=False reproduces the legacy rubric exactly, question for
    question. full_coverage=True asks about every part and every join.

    Each check carries `legacy`: whether the legacy rubric would have asked it.
    That membership is computed by running the legacy builder rather than
    restating its rules, so a legacy-comparable score can be recovered from
    full-coverage answers without a second vision call.
    """
    checks = (_part_and_join_checks(part_data, spec, full_coverage=full_coverage)
              + _constraint_checks(part_data)
              + _global_checks(part_data))
    if full_coverage:
        legacy_ids = {c["id"] for c in build_checklist(part_data, spec)}
    else:
        legacy_ids = {c["id"] for c in checks}
    for c in checks:
        c["legacy"] = c["id"] in legacy_ids
    return checks


def _tally(checks: list[dict], answers: dict):
    """Count a check list against an answer map.

    Returns (asked, answered, passed, failed_questions, unanswered_ids). A
    non-bool answer is NOT counted as a failure -- that is defect 1, kept so the
    legacy score stays reproducible; `unanswered` is what makes it visible.
    """
    answered = 0
    passed = 0
    failed_questions: list[str] = []
    unanswered: list[str] = []
    for c in checks:
        verdict = answers.get(c["id"])
        if not isinstance(verdict, bool):
            unanswered.append(c["id"])
            continue
        answered += 1
        if verdict:
            passed += 1
        else:
            failed_questions.append(c["question"])
    return len(checks), answered, passed, failed_questions, unanswered


def summarize_results(checks: list[dict], answers: dict) -> dict:
    """Convert {check_id: bool} answers into the standard vqa dict shape.

    `visual_score` is the LEGACY score: a flat pass fraction over the answered
    legacy-subset checks. It is deliberately unchanged, because the pipeline's
    thresholds and every existing result were computed against it. The corrected
    measurements are added alongside and are read by no control flow, so turning
    on full coverage changes what is recorded, never what the pipeline does.
    """
    all_asked, all_answered, all_passed, all_failed_q, all_unanswered = _tally(checks, answers)
    if all_answered == 0:
        return {}

    legacy_checks = [c for c in checks if c.get("legacy", True)]
    l_asked, l_answered, l_passed, l_failed_q, _ = _tally(legacy_checks, answers)

    # Normally the legacy subset is what scores. If full coverage asked
    # questions the legacy rubric never would and the judge answered only those,
    # score from what came back rather than discard a completed run.
    if l_answered:
        primary, p_answered, p_passed, p_failed_q = (
            legacy_checks, l_answered, l_passed, l_failed_q)
    else:
        primary, p_answered, p_passed, p_failed_q = (
            checks, all_answered, all_passed, all_failed_q)

    score = round(10 * p_passed / p_answered)
    missing_parts = [c["part"] for c in primary
                     if c["id"].startswith("part_") and c["part"]
                     and answers.get(c["id"]) is False]
    feedback = "; ".join(q.rstrip("?") for q in p_failed_q[:5])

    per_category: dict[str, dict] = {}
    for c in checks:
        slot = per_category.setdefault(str(c.get("category") or "global"),
                                       {"asked": 0, "answered": 0, "passed": 0})
        slot["asked"] += 1
        verdict = answers.get(c["id"])
        if isinstance(verdict, bool):
            slot["answered"] += 1
            if verdict:
                slot["passed"] += 1

    rates = [s["passed"] / s["answered"] for s in per_category.values() if s["answered"]]
    # Strict: unanswered counts as failed, so the score cannot be raised by
    # declining a question. Invariant under turning failures into non-answers.
    strict_rates = [s["passed"] / s["asked"] for s in per_category.values() if s["asked"]]

    return {
        # -- unchanged: the keys the pipeline reads ---------------------------
        "visual_score": score,
        "visible_parts": [],
        "missing_parts": missing_parts,
        "feedback": (f"failed {p_answered - p_passed}/{p_answered} checks: {feedback}"
                     if p_failed_q else f"passed all {p_answered} checks"),
        "checklist_total": p_answered,
        "checklist_passed": p_passed,
        "skipped": False,
        # -- added: measurement only, read by no control flow ----------------
        "checks_asked": all_asked,
        "checks_answered": all_answered,
        "checks_passed": all_passed,
        "coverage": round(all_answered / all_asked, 4) if all_asked else None,
        "unanswered": all_unanswered,
        "answers": {c["id"]: answers[c["id"]] for c in checks
                    if isinstance(answers.get(c["id"]), bool)},
        "per_category": per_category,
        "score_full": round(10 * all_passed / all_answered, 2),
        "score_balanced": round(10 * sum(rates) / len(rates), 2) if rates else None,
        "score_strict": round(10 * all_passed / all_asked, 2) if all_asked else None,
        "score_strict_balanced": (round(10 * sum(strict_rates) / len(strict_rates), 2)
                                  if strict_rates else None),
        "score_legacy_subset": round(10 * l_passed / l_answered) if l_answered else None,
        "legacy_coverage": round(l_answered / l_asked, 4) if l_asked else None,
        "failed_joins": [c["part"] for c in checks
                         if c["id"].startswith("join_") and c["part"]
                         and answers.get(c["id"]) is False],
    }
