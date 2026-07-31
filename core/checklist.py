"""Spec-derived visual QA checklists.

A holistic "rate this 1-10" is noisy from free-tier vision models; a list of
concrete yes/no questions generated from the build spec is far more reliable,
and every failed check maps directly to a repairable part.
"""
from __future__ import annotations

MAX_PART_CHECKS = 10
MAX_CONSTRAINT_CHECKS = 5


def build_checklist(part_data: dict, spec: dict | None = None) -> list[dict]:
    """Return [{id, part, question}, ...] for the vision model to answer."""
    checks: list[dict] = []
    seen_parts: set[str] = set()

    parts = (spec or {}).get("parts") or []
    for pt in parts[:MAX_PART_CHECKS]:
        pid = str(pt.get("id", "")).strip()
        if not pid or pid in seen_parts:
            continue
        seen_parts.add(pid)
        prim = pt.get("primitive", "part")
        label = pid.replace("_", " ")
        q = f"Is the {label} ({prim}) clearly visible and correctly shaped?"
        checks.append({"id": f"part_{pid}", "part": pid, "question": q})
        parent = str(pt.get("attach_to") or "").strip()
        if parent and len(checks) < MAX_PART_CHECKS + 4:
            checks.append({
                "id": f"join_{pid}",
                "part": pid,
                "question": (f"Does the {label} appear physically connected to the "
                             f"{parent.replace('_', ' ')} (no visible gap)?"),
            })

    if not parts:
        for name in (part_data.get("parts") or [])[:MAX_PART_CHECKS]:
            pid = str(name).strip().lower().replace(" ", "_")
            if not pid or pid in seen_parts:
                continue
            seen_parts.add(pid)
            checks.append({
                "id": f"part_{pid}", "part": pid,
                "question": f"Is the {str(name)} clearly visible?",
            })

    for i, c in enumerate((part_data.get("critical_constraints") or [])[:MAX_CONSTRAINT_CHECKS]):
        text = str(c).strip()
        if not text or text.lower().startswith("similar structure memory"):
            continue
        checks.append({
            "id": f"constraint_{i}", "part": "",
            "question": f"Is this satisfied: \"{text}\"?",
        })

    checks.append({"id": "grounded", "part": "",
                   "question": "Is the object resting on the ground with nothing floating in mid-air?"})
    checks.append({"id": "proportions", "part": "",
                   "question": "Are the overall proportions plausible for this kind of object?"})
    checks.append({"id": "no_embedded", "part": "",
                   "question": ("Does every part read as a distinct piece — with NO part "
                                "more than half sunk inside another (e.g. wheels buried in "
                                "the body, parts overlapping into each other)?")})
    if any(t in (str(part_data.get("category", "")) + " "
                 + " ".join(map(str, part_data.get("parts", [])))).lower()
           for t in ("wheel", "vehicle", "car", "truck", "locomotive", "train",
                     "cart", "wagon", "trailer")):
        checks.append({"id": "wheels_under", "part": "",
                       "question": ("Are the wheels sitting UNDER the body (their tops at or "
                                    "below the body's underside), touching the ground, and "
                                    "sized in proportion — not oversized discs cutting into "
                                    "the body?")})
    return checks


def summarize_results(checks: list[dict], answers: dict) -> dict:
    """Convert {check_id: bool} answers into the standard vqa dict shape."""
    total = 0
    passed = 0
    failed_questions: list[str] = []
    missing_parts: list[str] = []
    for c in checks:
        verdict = answers.get(c["id"])
        if not isinstance(verdict, bool):
            continue
        total += 1
        if verdict:
            passed += 1
        else:
            failed_questions.append(c["question"])
            if c["id"].startswith("part_") and c["part"]:
                missing_parts.append(c["part"])
    if total == 0:
        return {}
    score = round(10 * passed / total)
    feedback = "; ".join(q.rstrip("?") for q in failed_questions[:5])
    return {
        "visual_score": score,
        "visible_parts": [],
        "missing_parts": missing_parts,
        "feedback": (f"failed {total - passed}/{total} checks: {feedback}"
                     if failed_questions else f"passed all {total} checks"),
        "checklist_total": total,
        "checklist_passed": passed,
        "skipped": False,
    }
