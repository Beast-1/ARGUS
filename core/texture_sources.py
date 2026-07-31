"""Multi-provider material resolution: real PBR textures and measured
material values instead of planner-invented RGB.

Sources (all keyless, license-safe for shipped assets):
  acg/<AssetID>  ambientCG photo PBR texture sets       (CC0,  ~2000 sets)
  ph/<slug>      Poly Haven photo PBR texture sets      (CC0,  ~800 sets)
  pbi/<Name>     physicallybased.info measured values   (MIT,  ~90 materials)

The planner is shown a per-hint shortlist from all three and picks one source
per material (or omits it for the procedural fallback). Photo textures go
through relevance gates so an off-topic pick degrades to measured/procedural
instead of shipping a wrong texture.
"""
from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Optional

import requests

from core.poly_haven import (
    fetch_texture_catalog as _ph_fetch_catalog,
    resolve_texture_paths as _ph_resolve,
)

logger = logging.getLogger("ARGUS.textures")

_ACG_SEARCH = "https://ambientcg.com/api/v2/full_json"
_ACG_GET = "https://ambientcg.com/get?file={asset}_1K-JPG.zip"
_PBI_URL = "https://api.physicallybased.info/materials"

_CACHE_DIR = Path("out") / "texture_cache"
_FETCH_TIMEOUT = 12
_DOWNLOAD_TIMEOUT = 90

_ACG_CACHE: dict[str, list[dict]] = {}
_PBI_CACHE: list[dict] = []

# ambientCG zip map-file suffixes → ARGUS canonical map keys
_ACG_MAP_SUFFIXES = {
    "_Color.": "diff",
    "_Roughness.": "rough",
    "_NormalGL.": "nor_gl",
}

# Planner/material vocabulary → physicallybased.info entry names. PBI names
# real substances ("Porcelain"), not artist words ("ceramic"), so alias them.
_PBI_ALIASES = {
    "ceramic": "Porcelain", "porcelain": "Porcelain", "glaze": "Porcelain",
    "glazed": "Porcelain", "china": "Porcelain", "stoneware": "Porcelain",
    "glass": "Glass", "steel": "Iron", "iron": "Iron", "metal": "Iron",
    "aluminum": "Aluminum", "aluminium": "Aluminum", "chrome": "Chromium",
    "brass": "Brass", "copper": "Copper", "gold": "Gold", "silver": "Silver",
    "lead": "Lead", "nickel": "Nickel", "titanium": "Titanium",
    "plastic": "Plastic (PC)", "acrylic": "Plastic (Acrylic)",
    "pvc": "Plastic (PVC)", "rubber": "Rubber", "paint": "Car Paint",
    "painted": "Car Paint", "enamel": "Car Paint", "marble": "Marble",
    "concrete": "Concrete", "cement": "Concrete", "brick": "Brick",
    "bone": "Bone", "pearl": "Pearl", "ice": "Ice", "wax": "Wax",
    "paper": "Office Paper", "skin": "Skin I", "snow": "Snow",
    "diamond": "Diamond", "charcoal": "Charcoal", "sand": "Sand",
    "salt": "Salt", "soap": "Soap", "wood": "Wood (Oak)", "oak": "Wood (Oak)",
}


def _tokens(text: str) -> set[str]:
    return set(t for t in re.split(r"[^a-z]+", text.lower()) if len(t) > 2)


# ── physicallybased.info ─────────────────────────────────────────────────────

def fetch_pbi_materials() -> list[dict]:
    """Measured material values; cached in-process and on disk (static data)."""
    global _PBI_CACHE
    if _PBI_CACHE:
        return _PBI_CACHE
    cache_file = _CACHE_DIR / "pbi_materials.json"
    if cache_file.exists():
        try:
            _PBI_CACHE = json.loads(cache_file.read_text(encoding="utf-8"))
            return _PBI_CACHE
        except Exception:
            pass
    try:
        res = requests.get(_PBI_URL, timeout=_FETCH_TIMEOUT)
        res.raise_for_status()
        _PBI_CACHE = res.json()
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(_PBI_CACHE), encoding="utf-8")
    except Exception as exc:
        logger.warning("[TEXTURES] physicallybased.info fetch failed: %s", exc)
        _PBI_CACHE = []
    return _PBI_CACHE


def _pbi_entry(name: str) -> Optional[dict]:
    """Find a PBI entry for a material name/word ('ceramic_glaze' → Porcelain)."""
    mats = fetch_pbi_materials()
    if not mats:
        return None
    by_name = {m["name"].lower(): m for m in mats}
    low = name.lower().strip()
    if low in by_name:
        return by_name[low]
    for tok in _tokens(name):
        alias = _PBI_ALIASES.get(tok)
        if alias and alias.lower() in by_name:
            return by_name[alias.lower()]
        if tok in by_name:
            return by_name[tok]
    for m in mats:  # token vs entry tags/name words
        entry_words = _tokens(m["name"]) | set(m.get("tags") or [])
        if _tokens(name) & entry_words:
            return m
    return None


def pbi_values_for(name: str) -> Optional[dict]:
    """Measured {color, metallic, roughness} for a material name, or None."""
    m = _pbi_entry(name)
    if not m:
        return None
    color = m.get("color") or [0.8, 0.8, 0.8]
    return {
        "color": [round(float(c), 4) for c in color[:3]],
        "metallic": float(m.get("metalness", 0.0) or 0.0),
        "roughness": float(m.get("roughness", 0.5) if m.get("roughness") is not None else 0.5),
        "source": f"pbi/{m['name']}",
    }


# ── ambientCG ────────────────────────────────────────────────────────────────

def _acg_search(query: str, limit: int = 8) -> list[dict]:
    """Search ambientCG materials; returns [{asset_id, tags}]."""
    key = f"{query}:{limit}"
    if key in _ACG_CACHE:
        return _ACG_CACHE[key]
    try:
        res = requests.get(
            _ACG_SEARCH,
            params={"type": "Material", "limit": limit, "q": query,
                    "sort": "Popular"},
            timeout=_FETCH_TIMEOUT,
        )
        res.raise_for_status()
        found = res.json().get("foundAssets") or []
        out = [{"asset_id": a["assetId"], "tags": a.get("tags") or []}
               for a in found if a.get("assetId")]
    except Exception as exc:
        logger.warning("[TEXTURES] ambientCG search '%s' failed: %s", query, exc)
        out = []
    _ACG_CACHE[key] = out
    return out


def _acg_relevant(mat_name: str, asset_id: str, tags: list[str] | None = None) -> bool:
    """Does this ambientCG asset plausibly depict the material? Asset ids are
    '<Family><Number><Variant>' (Wood095, Metal055A), so compare the family
    name and tags against the material's words — including category synonyms
    (iron/steel → metal, plank/oak → wood)."""
    family = re.sub(r"\d.*$", "", asset_id).lower()
    words = _tokens(mat_name)
    for t in words:
        if len(t) >= 3 and (t in family or family in t):
            return True
    from core.poly_haven import _KEYWORD_TO_CATEGORY
    for t in words:
        for kw, cats in _KEYWORD_TO_CATEGORY.items():
            if kw in t and family in cats:
                return True
    if tags and words & set(t.lower() for t in tags):
        return True
    return False


def _acg_download(asset_id: str) -> dict[str, str]:
    """Download + extract an ambientCG 1K JPG set into the shared cache.
    Returns {map_key: absolute path}; cached across runs."""
    dest = _CACHE_DIR / "acg" / asset_id
    existing = {
        k: str((dest / f"{asset_id}{suf}jpg").resolve())
        for suf, k in _ACG_MAP_SUFFIXES.items()
        if (dest / f"{asset_id}{suf}jpg").exists()
    }
    if "diff" in existing:
        return existing

    dest.mkdir(parents=True, exist_ok=True)
    url = _ACG_GET.format(asset=asset_id)
    zip_path = dest / f"{asset_id}_1K-JPG.zip"
    try:
        r = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
        result: dict[str, str] = {}
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                for suffix, map_key in _ACG_MAP_SUFFIXES.items():
                    if suffix in member:
                        target = dest / f"{asset_id}{suffix}jpg"
                        target.write_bytes(zf.read(member))
                        result[map_key] = str(target.resolve())
        zip_path.unlink(missing_ok=True)
        if result:
            logger.info("[TEXTURES] ambientCG %s → %d map(s)", asset_id, len(result))
        return result
    except Exception as exc:
        logger.warning("[TEXTURES] ambientCG download failed %s: %s", asset_id, exc)
        zip_path.unlink(missing_ok=True)
        return {}


# ── merged planner catalog ───────────────────────────────────────────────────

def build_material_catalog_block(material_hints: list[str], user_prompt: str = "") -> str:
    """Per-hint shortlists from all providers for the planner prompt. The
    planner picks one source id per material; sparse on purpose — a short
    relevant list beats 60 alphabetical slugs (which once put a concrete
    floor on a glazed mug)."""
    hints = [h for h in material_hints if h] or []
    sections: list[str] = []

    for hint in hints[:4]:
        lines: list[str] = []
        for a in _acg_search(hint, limit=6):
            tag_str = ", ".join(a["tags"][:5])
            lines.append(f"  acg/{a['asset_id']} | {tag_str}")
        try:
            ph_entries = _ph_fetch_catalog([hint])
        except Exception:
            ph_entries = []
        hint_words = _tokens(hint)
        ranked = sorted(
            ph_entries,
            key=lambda e: -len(hint_words & _tokens(e["slug"] + " " + e["name"])),
        )
        for e in ranked[:6]:
            cats = ", ".join(e.get("categories") or [])
            lines.append(f"  ph/{e['slug']} | {e['name']} | [{cats}]")
        if lines:
            sections.append(f"Photo textures for '{hint}':\n" + "\n".join(lines))

    pbi_names = sorted({m["name"] for m in fetch_pbi_materials()})
    pbi_line = ""
    if pbi_names:
        pbi_line = ("Measured plain materials (pbi/<Name>): "
                    + ", ".join(pbi_names))

    if not sections and not pbi_line:
        return ""

    header = (
        "MATERIAL SOURCE CATALOG (all free for any use):\n"
        "For each named material, add an entry to \"poly_haven_textures\" "
        "mapping the material name to ONE source id:\n"
        "  \"acg/<AssetID>\" or \"ph/<slug>\" — photo PBR texture. Pick one "
        "ONLY when the surface genuinely shows that visible pattern at this "
        "object's scale (wood grain, brick, fabric weave...).\n"
        "  \"pbi/<Name>\" — physically measured colour/roughness/metalness. "
        "Prefer this for smooth or plain surfaces (glazed ceramic, painted "
        "metal, glass, bare metal).\n"
        "Omit a material entirely to use the procedural fallback. Never "
        "invent ids that are not listed below.\n"
    )
    return header + "\n\n".join(sections) + ("\n\n" + pbi_line if pbi_line else "")


# ── resolution ───────────────────────────────────────────────────────────────

def resolve_material_assets(
    selections: dict[str, str],
    texture_dir: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict]]:
    """Resolve planner picks into (texture_paths, value_overrides).

    texture_paths:   {material: {diff/rough/nor_gl: abs path}} for photo sets
    value_overrides: {material: {color, metallic, roughness, source}} for
                     measured plain materials
    """
    texture_paths: dict[str, dict[str, str]] = {}
    value_overrides: dict[str, dict] = {}
    ph_selections: dict[str, str] = {}

    for mat_name, source in (selections or {}).items():
        if not source or not isinstance(source, str):
            continue
        source = source.strip()

        if source.lower().startswith("pbi/"):
            vals = pbi_values_for(source[4:]) or pbi_values_for(mat_name)
            if vals:
                value_overrides[mat_name] = vals
            continue

        if source.lower().startswith("acg/"):
            asset_id = source[4:].strip()
            hit = _acg_search(asset_id, limit=3)
            tags = next((a["tags"] for a in hit if a["asset_id"].lower() == asset_id.lower()), [])
            canonical = next((a["asset_id"] for a in hit
                              if a["asset_id"].lower() == asset_id.lower()), asset_id)
            if not _acg_relevant(mat_name, canonical, tags):
                logger.warning(
                    "[TEXTURES] acg/%s looks unrelated to material '%s' — "
                    "skipping (measured/procedural fallback)", canonical, mat_name)
                continue
            maps = _acg_download(canonical)
            if maps:
                texture_paths[mat_name] = maps
            continue

        # "ph/<slug>" or a bare Poly Haven slug (legacy planner output)
        slug = source[3:] if source.lower().startswith("ph/") else source
        ph_selections[mat_name] = slug

    if ph_selections:
        for mat, maps in _ph_resolve(ph_selections, texture_dir).items():
            texture_paths[mat] = {k: str(Path(p).resolve()) for k, p in maps.items()}

    return texture_paths, value_overrides
