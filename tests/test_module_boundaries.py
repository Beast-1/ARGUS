"""Structural invariants that were violated once and would be easy to re-violate.

core/llm.py had grown to 6,145 lines by accumulating ~2,300 lines of hardcoded
Blender geometry — 35% of the file was static mesh code in the module named for
LLM orchestration, even though core/handcrafted_assets.py already existed for
exactly that content. Separately, sanitize_generated_code had been independently
reimplemented three times (llm.py, blender.py, handcrafted_assets.py) with three
different regexes that agreed on the common cases but not on trailing newlines.

These assert the boundary, not a line count — a line-count test would just get
bumped. Hermetic: parses source, imports nothing heavy.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
CORE = REPO / "core"


def _toplevel_defs(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_fallback_scripts_live_in_handcrafted_assets_not_llm():
    """The static geometry belongs with the other static geometry."""
    in_llm = {n for n in _toplevel_defs(CORE / "llm.py") if n.endswith("_fallback_script")}
    assert not in_llm, (
        "hardcoded fallback scripts defined in core/llm.py — these belong in "
        "core/handcrafted_assets.py, which is what keeps llm.py about LLM calls:\n  "
        + "\n  ".join(sorted(in_llm))
    )
    in_handcrafted = {
        n for n in _toplevel_defs(CORE / "handcrafted_assets.py")
        if n.endswith("_fallback_script")
    }
    assert len(in_handcrafted) >= 17, (
        f"expected the full fallback library in handcrafted_assets.py, found "
        f"{len(in_handcrafted)}"
    )


def test_only_one_sanitize_implementation_exists():
    """Three copies with three regexes is how a trailing-newline discrepancy hid
    between the module that produced scripts and the one that concatenated them."""
    definers = [
        path.relative_to(REPO).as_posix()
        for path in CORE.glob("*.py")
        if any(
            n in ("sanitize_generated_code", "sanitize_script")
            for n in _toplevel_defs(path)
        )
    ]
    assert definers == ["core/script_safety.py"], (
        f"sanitize should be defined once, in core/script_safety.py; found in {definers}"
    )


def test_script_safety_is_a_leaf_module():
    """It gates code execution for llm.py, blender.py and mcp_server.py, so it
    must not import any of them — a cycle here would be resolved by someone
    deleting the import, i.e. by deleting the safety check."""
    tree = ast.parse((CORE / "script_safety.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "core" not in imported, f"core/script_safety.py must stay dependency-free, imports: {imported}"
