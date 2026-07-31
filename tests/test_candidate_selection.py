"""Selection guards that keep the improvement loop from shipping a worse asset.

The vision rubric grades silhouette only, so without these guards a flat untextured
build out-scores a correctly textured one, and a "critical"-topology build can win
outright. Both were observed in real runs. Hermetic — synthesises GLB bytes, so no
Blender, network, or out/ directory required.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import _accept_candidate, _defect_penalty, _textured_material_count  # noqa: E402


def write_glb(path: Path, textured_materials: int, plain_materials: int = 1) -> Path:
    """Minimal valid GLB carrying only the glTF JSON chunk."""
    materials = [
        {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        for _ in range(textured_materials)
    ] + [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}
         for _ in range(plain_materials)]
    blob = json.dumps({"asset": {"version": "2.0"}, "materials": materials}).encode()
    blob += b" " * ((4 - len(blob) % 4) % 4)
    glb = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(blob))
    glb += struct.pack("<II", len(blob), 0x4E4F534A) + blob
    path.write_bytes(glb)
    return path


class FakeResult:
    def __init__(self, glb_path, severity=None):
        self.glb_path = str(glb_path)
        self.mcp = {"severity": severity} if severity else None


def make(tmp_path, name, textured=0, severity=None):
    return FakeResult(write_glb(tmp_path / f"{name}.glb", textured), severity)


def test_counts_textured_materials_from_glb(tmp_path):
    assert _textured_material_count(write_glb(tmp_path / "a.glb", 3)) == 3
    assert _textured_material_count(write_glb(tmp_path / "b.glb", 0)) == 0


def test_missing_or_corrupt_glb_counts_as_zero(tmp_path):
    assert _textured_material_count(tmp_path / "nope.glb") == 0
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb")
    assert _textured_material_count(bad) == 0


def test_topology_severity_penalty():
    assert _defect_penalty(FakeResult("x", "clean"))[0] == 0
    assert _defect_penalty(FakeResult("x", "medium"))[0] == 0
    assert _defect_penalty(FakeResult("x", "high"))[0] == 1
    assert _defect_penalty(FakeResult("x", "critical"))[0] == 3
    assert _defect_penalty(FakeResult("x"))[0] == 0  # no report -> no penalty


def test_normal_improvement_is_still_adopted(tmp_path):
    best = make(tmp_path, "best", textured=2, severity="clean")
    cand = make(tmp_path, "cand", textured=2, severity="clean")
    accepted, _ = _accept_candidate(8, 6, cand, best)
    assert accepted


def test_lower_score_is_rejected(tmp_path):
    best = make(tmp_path, "best", textured=2)
    cand = make(tmp_path, "cand", textured=2)
    assert _accept_candidate(4, 6, cand, best)[0] is False


def test_marginal_win_may_not_discard_textures(tmp_path):
    """The observed regression: a flat build out-scored a textured one and shipped."""
    best = make(tmp_path, "textured", textured=2)
    cand = make(tmp_path, "flat", textured=0)
    accepted, why = _accept_candidate(6, 5, cand, best)
    assert accepted is False
    assert "textured materials" in why


def test_decisive_win_may_still_discard_textures(tmp_path):
    """A genuinely much better silhouette should not be blocked by the guard."""
    best = make(tmp_path, "textured", textured=2)
    cand = make(tmp_path, "flat", textured=0)
    assert _accept_candidate(8, 5, cand, best)[0] is True


def test_gaining_textures_is_always_fine(tmp_path):
    best = make(tmp_path, "flat", textured=0)
    cand = make(tmp_path, "textured", textured=2)
    assert _accept_candidate(6, 5, cand, best)[0] is True


def test_critical_topology_cannot_win_on_silhouette(tmp_path):
    """Observed: an asset shipped at visual 8/10 with topology_severity 'critical'."""
    best = make(tmp_path, "clean", textured=1, severity="clean")
    cand = make(tmp_path, "broken", textured=1, severity="critical")
    accepted, why = _accept_candidate(8, 6, cand, best)
    assert accepted is False
    assert "critical" in why


def test_high_severity_costs_one_point(tmp_path):
    best = make(tmp_path, "clean", textured=1, severity="clean")
    cand = make(tmp_path, "high", textured=1, severity="high")
    assert _accept_candidate(7, 6, cand, best)[0] is False   # 7-1 == 6, not better
    assert _accept_candidate(8, 6, cand, best)[0] is True    # 8-1 > 6


def test_equal_score_prefers_more_textured_materials(tmp_path):
    """The point of this change: the loop should actively prefer richer materials
    on a tie, not merely avoid losing them. Equal raw score, equal topology."""
    best = make(tmp_path, "one_texture", textured=1)
    cand = make(tmp_path, "three_textures", textured=3)
    accepted, _ = _accept_candidate(6, 6, cand, best)
    assert accepted is True


def test_texture_bonus_is_capped_not_unbounded():
    from main import _texture_bonus
    assert _texture_bonus(4) == _texture_bonus(40)  # capped at ARGUS_TEXTURE_WORTH
    assert _texture_bonus(0) == 0
