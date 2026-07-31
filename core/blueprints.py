"""Golden build-spec blueprint library + keyword retrieval.

Hand-verified specs under assets/blueprints/*.json are retrieved by keyword
overlap and injected into the planner prompt as worked examples. Free-tier
models imitate structure far better than they invent it, so two relevant
examples beat any amount of abstract instruction.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

BLUEPRINT_DIR = Path(__file__).resolve().parent.parent / "assets" / "blueprints"

_STOPWORDS = {
    "a", "an", "the", "with", "of", "and", "or", "for", "in", "on", "to",
    "make", "create", "generate", "build", "made", "model", "asset", "object",
    "simple", "small", "big", "large", "two", "three", "four", "some", "its",
    "is", "that", "this", "from", "at", "by", "it", "into", "style", "3d",
}


def _tokens(text: str) -> set[str]:
    raw = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
    return {t for t in raw if t not in _STOPWORDS and len(t) > 2}


@lru_cache(maxsize=1)
def load_blueprints() -> list[dict]:
    out = []
    if not BLUEPRINT_DIR.is_dir():
        return out
    for f in sorted(BLUEPRINT_DIR.glob("*.json")):
        try:
            bp = json.loads(f.read_text(encoding="utf-8"))
            if bp.get("scene_graph") and bp.get("materials_palette"):
                out.append(bp)
        except Exception:
            continue
    return out


def retrieve(prompt: str, k: int = 2) -> list[dict]:
    """Top-k blueprints by keyword overlap with the prompt. Empty when nothing
    genuinely matches — a wrong example is worse than no example."""
    ptoks = _tokens(prompt)
    if not ptoks:
        return []
    scored = []
    for bp in load_blueprints():
        kw = {str(w).lower() for w in (bp.get("keywords") or [])}
        name_toks = _tokens(str(bp.get("name", "")))
        desc_toks = _tokens(str(bp.get("description", "")))
        s = len(ptoks & kw) * 10 + len(ptoks & name_toks) * 10 + len(ptoks & desc_toks) * 2
        if s > 0:
            scored.append((s, bp))
    scored.sort(key=lambda x: -x[0])
    return [bp for _, bp in scored[:k]]


def fewshot_block(prompt: str, k: int = 2) -> str:
    """Planner-prompt section with the most relevant worked examples.
    Falls back to the fire hydrant when nothing matches."""
    picked = retrieve(prompt, k)
    if not picked:
        picked = [bp for bp in load_blueprints() if bp.get("name") == "fire_hydrant"][:1]
    if not picked:
        return ""
    blocks = []
    for bp in picked:
        graph = ",\n   ".join(
            json.dumps(node, separators=(",", ":")) for node in bp["scene_graph"]
        )
        blocks.append(
            f"  WORKED EXAMPLE ({bp.get('description', bp['name'])}) — copy this "
            f"structure and quality level:\n"
            f"  \"materials_palette\": {json.dumps(bp['materials_palette'], separators=(',', ':'))},\n"
            f"  \"scene_graph\": [\n   {graph}\n  ]"
        )
    notes = "\n".join(f"  {n}" for bp in picked for n in (bp.get("notes") or [])[:3])
    return "\n\n".join(blocks) + (("\n" + notes) if notes else "")
