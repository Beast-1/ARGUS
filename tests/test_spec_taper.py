"""Box taper — the missing shape control behind blocky silhouettes.

Cylinders could always taper (r_top), but argus_box could not, so every box-shaped
part came out a perfect rectangular prism. Measured across real runs before this
change: 248 boxes, 0 tapered (no such field); 98 cylinders, only 4 tapered.

Hermetic — spec validation and compiler emission only, no Blender required.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.graph_compiler import compile_spec  # noqa: E402
from core.spec import validate_spec  # noqa: E402

MATS = {"wood": {"color": [0.5, 0.3, 0.1], "metallic": 0.0, "roughness": 0.8}}


def build(*extra_part_fields):
    """Spec with >=3 parts (the validator's minimum); part 'subject' carries the fields."""
    parts = [
        {"id": "base", "primitive": "box", "pos": [0, 0, 0.02],
         "size": [1.0, 0.5, 0.04], "material": "wood"},
        {"id": "other", "primitive": "box", "pos": [-0.3, 0, 0.3],
         "size": [0.2, 0.2, 0.5], "attach_to": "base", "material": "wood"},
        {"id": "subject", "primitive": "box", "pos": [0.3, 0, 0.3],
         "size": [0.2, 0.2, 0.5], "attach_to": "base", "material": "wood",
         **(extra_part_fields[0] if extra_part_fields else {})},
    ]
    return {"asset_name": "t", "parts": parts, "materials": MATS}


def subject_params(**fields):
    report = validate_spec(build(fields))
    assert report.ok, report.errors
    part = next(p for p in report.spec["parts"] if p["id"] == "subject")
    return part["params"], report


def test_taper_is_accepted_and_defaults_to_z():
    params, _ = subject_params(taper=0.6)
    assert params["taper"] == 0.6
    assert params["taper_axis"] == "Z"


def test_taper_axis_can_be_overridden():
    params, _ = subject_params(taper=0.7, taper_axis="x")
    assert params["taper_axis"] == "X"


def test_invalid_taper_axis_falls_back_to_z():
    params, _ = subject_params(taper=0.7, taper_axis="banana")
    assert params["taper_axis"] == "Z"


def test_taper_is_clamped_to_sane_range():
    assert subject_params(taper=9.0)[0]["taper"] == 2.0
    assert subject_params(taper=-3.0)[0]["taper"] == 0.0


def test_taper_of_one_is_dropped_as_a_noop():
    """1.0 means 'no taper' — it should not bloat the spec or the emitted call."""
    params, _ = subject_params(taper=1.0)
    assert "taper" not in params


def test_absent_taper_stays_absent():
    params, _ = subject_params()
    assert "taper" not in params


def box_calls(script: str) -> list[str]:
    """Emitted argus_box(...) call sites only.

    The compiled script embeds COMPONENT_LIBRARY_SOURCE verbatim, so its
    `def argus_box(..., taper=None, ...)` signature would match a naive substring
    search whether or not any part actually tapers.
    """
    return [ln.strip() for ln in script.splitlines()
            if "argus_box(" in ln and not ln.lstrip().startswith("def ")]


def test_compiler_emits_taper_and_axis():
    _, report = subject_params(taper=0.4, taper_axis="Y")
    tapered = [c for c in box_calls(compile_spec(report.spec, None)) if "taper=" in c]
    assert len(tapered) == 1, tapered
    assert "taper=0.4" in tapered[0].replace(" ", "")
    assert "taper_axis='Y'" in tapered[0].replace(" ", "")


def test_compiler_omits_taper_when_unset():
    _, report = subject_params()
    calls = box_calls(compile_spec(report.spec, None))
    assert calls, "expected at least one emitted box"
    assert not any("taper=" in c for c in calls)


def test_taper_survives_a_spec_edit():
    """The visual-repair loop edits specs as JSON; taper must be an editable field
    so 'looks blocky' feedback can actually be acted on."""
    from core.spec import _EDITABLE_FIELDS
    assert "taper" in _EDITABLE_FIELDS
    assert "taper_axis" in _EDITABLE_FIELDS
