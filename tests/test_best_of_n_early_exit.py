"""run_best_of_n_generation used to always build all N candidates, even when
candidate 1 was already confidently good — burning a full LLM-codegen + Blender
build for zero possible benefit. These tests prove the expensive path is actually
skipped (not just that a flag gets set), by asserting generate_blender_script is
never called once the early-exit threshold is met, and still IS called when it
isn't. Fully mocked — no LLM, Blender, or network access.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from test_candidate_selection import write_glb  # noqa: E402


class FakeResult:
    def __init__(self, glb_path, severity="clean"):
        self.success = True
        self.glb_path = str(glb_path)
        self.mcp = {"severity": severity}


def _run(monkeypatch, tmp_path, scores, textured=2, severity="clean"):
    """scores[0] is candidate 1's score; scores[i] (if reached) is candidate i+2's."""
    codegen_calls = {"n": 0}

    def fake_score_render(glb_path, prompt, part_data, spec=None, reference_image=None):
        # Called once per candidate, in call order.
        idx = min(fake_score_render.calls, len(scores) - 1)
        fake_score_render.calls += 1
        return {"visual_score": scores[idx]}, "multiview", None
    fake_score_render.calls = 0

    def fake_generate_blender_script(*args, **kwargs):
        codegen_calls["n"] += 1
        return "# fake script\n"

    def fake_save_script(script_dir, asset_name, script, suffix=""):
        p = tmp_path / f"cand{suffix}.py"
        p.write_text(script)
        return p

    # **kwargs so adding a parameter to run_blender (poly_max, and whatever
    # comes next) doesn't break every fake in the suite.
    def fake_run_blender(script_path, name, run_id, iter_num=0, **kwargs):
        return FakeResult(write_glb(tmp_path / f"cand_{iter_num}.glb", textured), severity)

    monkeypatch.setattr(main, "_score_render", fake_score_render)
    monkeypatch.setattr(main, "generate_blender_script", fake_generate_blender_script)
    monkeypatch.setattr(main, "save_script", fake_save_script)
    monkeypatch.setattr(main, "run_blender", fake_run_blender)
    monkeypatch.setattr("core.prompt.build_object_prompt", lambda *a, **k: "prompt")

    result = FakeResult(write_glb(tmp_path / "cand_1.glb", textured), severity)
    main.run_best_of_n_generation(
        prompt="a thing",
        part_data={},
        script="# original\n",
        script_path=tmp_path / "cand_1.py",
        result=result,
        asset_name="thing",
        run_id="thing",
        script_dir=tmp_path,
        mcp_mode=False,
        reference_image=None,
        n_candidates=3,
    )
    return codegen_calls["n"]


def test_confidently_good_candidate_skips_remaining_generation(monkeypatch, tmp_path):
    """Candidate 1 scores well above target with clean topology and real materials
    — candidates 2 and 3 must never be generated."""
    calls = _run(monkeypatch, tmp_path, scores=[9], textured=2, severity="clean")
    assert calls == 0


def test_mediocre_candidate_still_generates_more(monkeypatch, tmp_path):
    """Below the early-exit bar — the loop must behave as before and keep trying."""
    calls = _run(monkeypatch, tmp_path, scores=[4, 5, 6], textured=1, severity="clean")
    assert calls == 2  # candidates 2 and 3 both attempted


def test_high_score_with_critical_topology_does_not_trigger_early_exit(monkeypatch, tmp_path):
    """A high raw score hiding critical topology must not look 'confidently good'
    — this is the exact blind spot _effective_score exists to close."""
    calls = _run(monkeypatch, tmp_path, scores=[9, 9], textured=1, severity="critical")
    assert calls > 0


def test_early_exit_can_also_trigger_after_a_later_candidate_improves(monkeypatch, tmp_path):
    """Candidate 1 is mediocre, candidate 2 is confidently good — candidate 3 must
    then be skipped, proving the check re-evaluates each loop iteration, not just
    once at the start."""
    calls = _run(monkeypatch, tmp_path, scores=[3, 9], textured=2, severity="clean")
    assert calls == 1  # candidate 2 generated, candidate 3 skipped
