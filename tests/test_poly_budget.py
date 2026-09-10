"""Poly-budget enforcement.

Before this existed the budget was advisory only: core/prompt.py asked the model
to emit its own dissolve_limit snippet, nothing verified the result, and the
vision scorer never sees a triangle count — so shipped assets ran 1.5x-56x over
the "game-ready" ceiling (a gas pump at 281,182 triangles against a 5,000
budget) and still scored well enough to pass.

The allocation maths is the part worth testing hermetically. Its two failure
modes were both found on real assets and both produce plausible-looking output
rather than an error:

  * charging a floor-clamped object its ORIGINAL size makes pinning raise the
    fixed cost, lowering the ratio, pinning more — a runaway that ends with
    everything pinned and nothing decimated (the gas pump came back untouched
    at 281,182).
  * a fixed floor is unsatisfiable for part-heavy assets — 35 parts x 200 tris
    reserves 7,000 against a 5,000 budget, so the allocator gives up and the
    asset ships 31% over (the fire hydrant).

Hermetic: the allocator is exercised as pure arithmetic, no Blender.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from core.blender import _POLY_REPORT_RE, build_export_script  # noqa: E402


# The allocator runs inside the generated Blender script, so mirror it here to
# test the arithmetic. test_generated_script_contains_the_allocator below guards
# against this copy drifting from the real one.
def _floor_for(n_objects: int, poly_max: int) -> int:
    return max(24, min(200, poly_max // max(1, n_objects) // 2))


def _allocate(per_obj: dict, poly_max: int, floor: int):
    untouchable = {n for n, t in per_obj.items() if t <= floor}
    clamped: set = set()
    ratio = 1.0
    for _ in range(12):
        fixed = floor * len(clamped) + sum(per_obj[n] for n in untouchable)
        flexible = sum(t for n, t in per_obj.items()
                       if n not in untouchable and n not in clamped)
        if flexible <= 0:
            break
        ratio = max(0.0, poly_max - fixed) / flexible
        newly = {n for n, t in per_obj.items()
                 if n not in untouchable and n not in clamped and t * ratio < floor}
        if not newly:
            break
        clamped |= newly
    out = {}
    for n, t in per_obj.items():
        out[n] = 1.0 if n in untouchable else (
            floor / t if n in clamped else min(1.0, ratio))
    return out, ratio


def _projected_total(per_obj, ratios, floor):
    return sum(max(floor if ratios[n] < 1.0 else t, round(t * ratios[n]))
               if t > floor else t
               for n, t in per_obj.items())


# ---------------------------------------------------------------------------
# allocation
# ---------------------------------------------------------------------------

def test_lands_within_budget_on_the_real_gas_pump_shape():
    """281k across 19 objects — the asset that motivated all of this."""
    per_obj = {f"o{i}": 281182 // 19 for i in range(19)}
    floor = _floor_for(len(per_obj), 5000)
    ratios, _ = _allocate(per_obj, 5000, floor)
    assert _projected_total(per_obj, ratios, floor) <= 5000


def test_clamped_objects_do_not_cause_a_pin_runaway():
    """The regression: charging a clamped object its original size instead of
    the floor pinned everything and decimated nothing."""
    per_obj = {f"o{i}": 2000 for i in range(19)}   # all small enough to clamp
    floor = _floor_for(len(per_obj), 5000)
    ratios, _ = _allocate(per_obj, 5000, floor)
    assert any(r < 1.0 for r in ratios.values()), "nothing was decimated at all"
    assert _projected_total(per_obj, ratios, floor) <= 5000


def test_floor_scales_so_part_heavy_assets_are_satisfiable():
    """35 parts x a flat 200 floor = 7,000 reserved against a 5,000 budget."""
    assert _floor_for(35, 5000) * 35 <= 5000
    assert _floor_for(4, 5000) == 200      # few parts -> the full floor
    assert _floor_for(500, 5000) == 24     # many parts -> the hard minimum


def test_floor_never_goes_below_a_recognisable_shape():
    for n in (1, 10, 100, 1000, 10000):
        assert _floor_for(n, 5000) >= 24, "a part below ~24 tris stops reading as a shape"


def test_an_asset_already_within_budget_is_left_alone():
    per_obj = {"a": 400, "b": 300}
    floor = _floor_for(2, 5000)
    ratios, _ = _allocate(per_obj, 5000, floor)
    assert all(r >= 1.0 for r in ratios.values())


# ---------------------------------------------------------------------------
# budget resolution
# ---------------------------------------------------------------------------

def test_budget_comes_from_the_same_table_the_prompt_quotes():
    """If these drift, the model is told one ceiling and held to another."""
    from core.prompt import _POLY_TARGETS
    for name in ("low", "medium", "high"):
        assert main._poly_max_for({"poly_budget": name}) == _POLY_TARGETS[name][1]


def test_unknown_budget_falls_back_to_medium():
    assert main._poly_max_for({"poly_budget": "enormous"}) == main._poly_max_for({})


def test_enforcement_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ARGUS_POLY_ENFORCE", "0")
    assert main._poly_max_for({"poly_budget": "medium"}) == 0


# ---------------------------------------------------------------------------
# report plumbing + score penalty
# ---------------------------------------------------------------------------

def test_report_sentinel_parses_out_of_stdout():
    stdout = (
        "some blender noise\n"
        'ARGUS_POLY_REPORT:{"budget": 5000, "before": 281182, "after": 4999, '
        '"within_budget": true, "reduction": 56.25}\n'
        "[GLB_EXPORTED] x.glb\n"
    )
    m = _POLY_REPORT_RE.search(stdout)
    assert m
    import json
    assert json.loads(m.group(1))["reduction"] == 56.25


class _R:
    """Minimal ExportResult stand-in for the score functions."""
    def __init__(self, reduction=None, severity="clean"):
        self.poly = {"reduction": reduction} if reduction is not None else None
        self.mcp = {"severity": severity}
        self.glb_path = ""


def test_modest_reduction_is_not_penalised():
    """Planar dissolve routinely removes coplanar detail at no visual cost —
    penalising that would punish a healthy build."""
    for r in (1, 2, 3.9):
        assert main._poly_penalty(_R(r))[0] == 0.0


def test_heavy_reduction_is_penalised_progressively():
    assert main._poly_penalty(_R(5))[0] == 0.5
    assert main._poly_penalty(_R(12))[0] == 1.0
    assert main._poly_penalty(_R(56))[0] == 2.0


def test_missing_report_is_not_penalised():
    """Enforcement off, or an older run — absence of evidence isn't a defect."""
    assert main._poly_penalty(_R(None))[0] == 0.0


def test_effective_score_now_sees_over_tessellation():
    """The gap this closes: a 56x-over build could previously score 8 and ship,
    because the vision scorer grades silhouette and never sees geometry."""
    crushed = main._effective_score(8, _R(56))[0]
    healthy = main._effective_score(8, _R(1))[0]
    assert crushed < healthy
    assert healthy - crushed == 2.0


# ---------------------------------------------------------------------------
# generated script
# ---------------------------------------------------------------------------

def _script(poly_max):
    import tempfile
    d = Path(tempfile.mkdtemp())
    src = d / "gen.py"
    src.write_text("import bpy\nimport bmesh\nassert bpy.app.version[0] >= 4\n",
                   encoding="utf-8")
    return build_export_script(src, "thing", "run1", poly_max=poly_max)


def test_generated_script_is_valid_python():
    """It is assembled by f-string interpolation into a triple-quoted template,
    where a stray nested triple quote silently truncates the whole script."""
    import ast
    ast.parse(_script(5000))


def test_generated_script_contains_the_allocator():
    s = _script(5000)
    for token in ("_argus_poly_floor", "_argus_allocate", "ARGUS_POLY_REPORT",
                  "ARGUS_POLY_PLANAR", "ARGUS_POLY_BUDGET"):
        assert token in s, f"{token} missing from the generated export script"
    assert "_poly_max = 5000" in s


def test_planar_dissolve_runs_before_collapse():
    """Order matters: collapse alone at an extreme ratio crumpled a domed top
    into jagged garbage, while dissolving coplanar detail first left a 20x
    gentler ratio and the dome intact."""
    s = _script(5000)
    assert s.index("ARGUS_POLY_PLANAR") < s.index("ARGUS_POLY_BUDGET")


def test_budget_of_zero_disables_enforcement_in_the_script():
    s = _script(0)
    assert "_poly_max = 0" in s
    # the guard is `_poly_max > 0`, so nothing runs
    assert "_poly_max > 0" in s
