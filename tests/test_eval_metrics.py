"""Metrics must be trustworthy before a sweep spends days of API budget on them.

Hermetic: synthesises GLB bytes and manifests on disk, no Blender or network.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import collect, effective_score, summarise  # noqa: E402


# ---------------------------------------------------------------------------
# effective_score — must mirror main.py's Eq. 1 exactly, since the paper quotes it
# ---------------------------------------------------------------------------

def test_effective_score_matches_the_pipelines_own_formula():
    """s_eff = s_vision - p(severity) + min(0.5 * textured, 2), with
    p(clean)=p(medium)=0, p(high)=1, p(critical)=3 — copied from
    main.py:_SEVERITY_PENALTY / _texture_bonus / _effective_score."""
    # clean topology, no textures: unchanged
    assert effective_score({"visual_score": 7, "topology_severity": "clean",
                            "textured_materials": 0}) == 7
    # medium is explicitly a zero penalty, not a small one
    assert effective_score({"visual_score": 7, "topology_severity": "medium",
                            "textured_materials": 0}) == 7
    # high costs 1, critical costs 3
    assert effective_score({"visual_score": 7, "topology_severity": "high",
                            "textured_materials": 0}) == 6
    assert effective_score({"visual_score": 7, "topology_severity": "critical",
                            "textured_materials": 0}) == 4
    # 0.5 per textured material...
    assert effective_score({"visual_score": 6, "topology_severity": "clean",
                            "textured_materials": 2}) == 7
    # ...capped at 2, so 8 textures is worth the same as 4
    assert effective_score({"visual_score": 6, "topology_severity": "clean",
                            "textured_materials": 8}) == 8
    assert effective_score({"visual_score": 6, "topology_severity": "clean",
                            "textured_materials": 4}) == 8


def test_effective_score_is_none_without_a_vision_score():
    """Nothing built, or scoring itself failed — must not be treated as a 0."""
    assert effective_score({"visual_score": None, "topology_severity": "clean",
                            "textured_materials": 0}) is None
    assert effective_score({}) is None


def test_effective_score_reproduces_the_papers_corrected_figure():
    """The specific error this was added to make impossible to repeat: §V-C
    reported "5 of 15 (33%)... using the effective score", but 33% is the RAW
    score's >=8 rate. A clean-topology, well-textured build gets +2 from the
    texture bonus, which is exactly what moves those cases over the bar."""
    records = [
        # raw 7, clean, 4 textures -> 9.0 effective: clears 8 on the bonus alone
        {"exists": True, "visual_score": 7, "topology_severity": "clean",
         "textured_materials": 4, "triangles": 100, "elapsed_sec": 10},
        # raw 8, but critical topology -> 5.0 effective: does NOT clear 8
        {"exists": True, "visual_score": 8, "topology_severity": "critical",
         "textured_materials": 0, "triangles": 100, "elapsed_sec": 10},
    ]
    s = summarise(records)
    assert s["pct_at_or_above_7"] == 100          # both are raw >=7
    assert s["pct_effective_at_or_above_8"] == 50  # only the textured, clean one
    assert s["mean_effective_score"] == 7.0


def write_glb(path: Path, *, textured=0, plain=1, tris=12, images=0) -> None:
    accessors, meshes = [], []
    for i in range(textured + plain):
        accessors.append({"count": tris * 3})   # indices
        accessors.append({"count": tris * 2})   # positions
        meshes.append({"primitives": [{
            "indices": 2 * i, "attributes": {"POSITION": 2 * i + 1, "material": i},
        }]})
    materials = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
                 for _ in range(textured)]
    materials += [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1]}}
                  for _ in range(plain)]
    blob = json.dumps({
        "asset": {"version": "2.0"}, "accessors": accessors,
        "meshes": meshes, "materials": materials,
        "images": [{"uri": f"i{i}"} for i in range(images)],
    }).encode()
    blob += b" " * ((4 - len(blob) % 4) % 4)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(blob))
        + struct.pack("<II", len(blob), 0x4E4F534A) + blob
    )


def make_asset(tmp_path, name="thing", manifest=None, **glb):
    d = tmp_path / "final" / name
    d.mkdir(parents=True)
    write_glb(d / f"{name}.glb", **glb)
    if manifest is not None:
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def test_disambiguated_folder_name_still_finds_the_glb(tmp_path):
    """Real incident: make_run_id() names the FOLDER 'traffic_cone(2)' on a
    collision, but the exported FILE inside stays 'traffic_cone.glb' — the
    folder and file names diverge. Assuming they matched silently reported
    a cleanly-exported asset as 'no GLB exported'."""
    d = tmp_path / "final" / "traffic_cone(2)"
    d.mkdir(parents=True)
    write_glb(d / "traffic_cone.glb", plain=1, tris=8)
    m = collect(d)
    assert m.exists is True
    assert m.triangles == 8
    assert "no GLB exported" not in m.notes


def test_missing_glb_is_reported_not_crashed(tmp_path):
    d = tmp_path / "final" / "empty"
    d.mkdir(parents=True)
    m = collect(d)
    assert m.exists is False
    assert "no GLB exported" in m.notes


def test_geometry_and_material_counts(tmp_path):
    d = make_asset(tmp_path, textured=2, plain=1, tris=10, images=4)
    m = collect(d)
    assert m.exists and m.meshes == 3
    assert m.triangles == 30          # 3 prims x 10
    assert m.materials == 3
    assert m.textured_materials == 2
    assert m.embedded_images == 4


def test_manifest_fields_are_read(tmp_path):
    d = make_asset(tmp_path, manifest={
        "visual_score": 8, "topology_severity": "clean", "topology_grounded": True})
    m = collect(d)
    assert m.has_manifest and m.visual_score == 8
    assert m.topology_severity == "clean"


def test_legacy_asset_without_manifest_degrades(tmp_path):
    """7 of 13 real assets have no manifest — these must not poison the stats."""
    m = collect(make_asset(tmp_path, name="legacy"))
    assert m.exists is True
    assert m.has_manifest is False
    assert m.visual_score is None


def test_corrupt_manifest_does_not_crash(tmp_path):
    d = make_asset(tmp_path)
    (d / "manifest.json").write_text("{not json", encoding="utf-8")
    m = collect(d)
    assert m.exists and m.visual_score is None
    assert any("manifest" in n for n in m.notes)


def test_shape_vocabulary_counts_taper_usage(tmp_path):
    d = make_asset(tmp_path, name="w")
    run = tmp_path / "runs" / "w" / "scripts"
    run.mkdir(parents=True)
    (run / "w.py").write_text(
        "_argus_smart_mat('m', (1,1,1), 0, 0.5)\n"
        "argus_box('a', (0,0,0), (1,1,1), M, ROOT)\n"
        "argus_box('b', (0,0,0), (1,1,1), M, ROOT, taper=0.6)\n"
        "argus_cylinder('c', (0,0,0), 1, 1, M, ROOT, r_top=0.5)\n"
        "argus_cylinder('d', (0,0,0), 1, 1, M, ROOT)\n",
        encoding="utf-8")
    m = collect(d, tmp_path / "runs" / "w")
    assert (m.boxes, m.boxes_tapered) == (2, 1)
    assert (m.cylinders, m.cylinders_tapered) == (2, 1)


def test_iteration_scripts_are_ignored_for_vocabulary(tmp_path):
    """Only the winning base script counts, not repair/visual/regen variants."""
    d = make_asset(tmp_path, name="w")
    run = tmp_path / "runs" / "w" / "scripts"
    run.mkdir(parents=True)
    (run / "w.py").write_text("_argus_smart_mat()\nargus_box('a',0,0,0,0)\n", encoding="utf-8")
    (run / "w_visual_1.py").write_text("argus_box('x',0,0,0,0, taper=0.5)\n", encoding="utf-8")
    m = collect(d, tmp_path / "runs" / "w")
    assert m.boxes == 1 and m.boxes_tapered == 0


def test_summary_ignores_unbuilt_and_unscored(tmp_path):
    records = [
        {"exists": True, "visual_score": 8, "topology_severity": "clean",
         "textured_materials": 2, "triangles": 100, "elapsed_sec": 10},
        {"exists": True, "visual_score": 6, "topology_severity": "critical",
         "textured_materials": 0, "triangles": 200, "elapsed_sec": 20},
        {"exists": True, "visual_score": None, "topology_severity": None,
         "textured_materials": 1, "triangles": 300, "elapsed_sec": 30},
        {"exists": False},
    ]
    s = summarise(records)
    assert s["n_attempted"] == 4 and s["n_built"] == 3
    assert s["mean_visual_score"] == 7          # unscored row excluded
    assert s["pct_at_or_above_7"] == 50
    assert s["pct_topology_critical"] == 50     # of the 2 with a severity
    assert s["pct_assets_textured"] == 67       # 2 of 3 built


def test_summary_handles_an_empty_sweep():
    s = summarise([])
    assert s["n_attempted"] == 0
    assert s["mean_visual_score"] is None
