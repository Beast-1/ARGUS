from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("ARGUS.poly_haven")

_ASSETS_URL = "https://api.polyhaven.com/assets"
_FILES_URL  = "https://api.polyhaven.com/files/{slug}"

_MAP_KEY_ALIASES: dict[str, list[str]] = {
    "diff":   ["Diffuse", "diffuse", "Albedo", "albedo", "Color", "color", "col"],
    "rough":  ["Rough", "rough", "Roughness", "roughness"],
    "nor_gl": ["nor_gl", "Normal_GL", "normal_gl"],
    "ao":     ["AO", "ao", "Ambient_Occlusion"],
    "arm":    ["arm", "ARM"],
}

_FETCH_TIMEOUT    = int(os.getenv("ARGUS_PH_FETCH_TIMEOUT", "10"))
_DOWNLOAD_TIMEOUT = int(os.getenv("ARGUS_PH_DOWNLOAD_TIMEOUT", "30"))
_DEFAULT_RES      = os.getenv("ARGUS_PH_RESOLUTION", "1k")

_CATALOG_CACHE: dict[str, list[dict]] = {}

_KEYWORD_TO_CATEGORY: dict[str, list[str]] = {
    "metal":    ["metal"],
    "steel":    ["metal"],
    "iron":     ["metal"],
    "rust":     ["metal"],
    "copper":   ["metal"],
    "gold":     ["metal"],
    "wood":     ["wood"],
    "timber":   ["wood"],
    "plank":    ["wood"],
    "concrete": ["concrete"],
    "cement":   ["concrete"],
    "stone":    ["stone"],
    "rock":     ["stone"],
    "brick":    ["brick"],
    "tile":     ["tiles"],
    "fabric":   ["fabric"],
    "cloth":    ["fabric"],
    "leather":  ["fabric"],
    "plastic":  ["plastic"],
    "rubber":   ["plastic"],
    "ground":   ["ground"],
    "soil":     ["ground"],
    "dirt":     ["ground"],
    "grass":    ["ground"],
    "sand":     ["ground"],
    "bark":     ["organic"],
    "organic":  ["organic"],
    "marble":   ["stone"],
    "asphalt":  ["ground"],
    "paint":    ["plaster"],
    "plaster":  ["plaster"],
    "wall":     ["plaster", "brick", "concrete"],
}

_MAP_KEYS = {
    "diff":   "diff",
    "rough":  "rough",
    "nor_gl": "nor_gl",
    "ao":     "ao",
}


def _extract_categories(material_hints: list[str]) -> list[str]:
    categories: list[str] = []
    for hint in material_hints:
        for keyword, cats in _KEYWORD_TO_CATEGORY.items():
            if keyword in hint.lower():
                for c in cats:
                    if c not in categories:
                        categories.append(c)
    return categories or []


def fetch_texture_catalog(material_hints: list[str]) -> list[dict]:
    """
    Fetch a compact texture catalog from Poly Haven filtered to categories
    that match the given material hints. Returns at most 60 entries as
    [{slug, name, categories}] for injection into the planner prompt.
    Results are cached for the lifetime of the process.
    """
    categories = _extract_categories(material_hints)
    cache_key = ",".join(sorted(categories)) if categories else "__uncategorized__"

    if cache_key in _CATALOG_CACHE:
        return _CATALOG_CACHE[cache_key]

    try:
        params: dict = {"t": "textures"}
        if len(categories) == 1:
            params["c"] = categories[0]
        res = requests.get(_ASSETS_URL, params=params, timeout=_FETCH_TIMEOUT)
        res.raise_for_status()
        raw: dict = res.json()
    except Exception as exc:
        logger.warning("[POLY_HAVEN] catalog fetch failed: %s", exc)
        return []

    entries: list[dict] = []
    for slug, meta in raw.items():
        asset_cats = meta.get("categories", [])
        if categories and not any(c in asset_cats for c in categories):
            continue
        entries.append({
            "slug": slug,
            "name": meta.get("name", slug),
            "categories": asset_cats,
        })

    entries.sort(key=lambda e: e["slug"])
    result = entries[:60]
    _CATALOG_CACHE[cache_key] = result
    return result


def _fetch_all_textures() -> list[dict]:
    """Fetch the full (unfiltered) texture catalog — used as last-resort fallback."""
    if "__all__" in _CATALOG_CACHE:
        return _CATALOG_CACHE["__all__"]
    try:
        res = requests.get(_ASSETS_URL, params={"t": "textures"}, timeout=_FETCH_TIMEOUT)
        res.raise_for_status()
        raw: dict = res.json()
        entries = [
            {"slug": slug, "name": meta.get("name", slug), "categories": meta.get("categories", [])}
            for slug, meta in raw.items()
        ]
        entries.sort(key=lambda e: e["slug"])
        _CATALOG_CACHE["__all__"] = entries
        return entries
    except Exception as exc:
        logger.warning("[POLY_HAVEN] full catalog fetch failed: %s", exc)
        return []


def build_catalog_block(material_hints: list[str]) -> str:
    """
    Returns a formatted string block listing available Poly Haven textures
    suitable for embedding in the planner's prompt.
    """
    # Without a recognised category the assets endpoint just returns the
    # first 60 textures alphabetically (aerial/asphalt/brick/concrete…) — an
    # actively misleading list that tempts the planner into off-topic picks
    # (observed: concrete_floor_01 chosen for "glazed ceramic"). Offer no
    # catalog instead; un-textured materials use the procedural builder.
    if not _extract_categories(material_hints):
        return ""
    catalog = fetch_texture_catalog(material_hints)
    if not catalog:
        return ""

    lines = ["POLY HAVEN TEXTURE CATALOG (CC0, free for any use):"]
    lines.append("slug | name | categories")
    for e in catalog:
        cats = ", ".join(e["categories"])
        lines.append(f"  {e['slug']} | {e['name']} | [{cats}]")

    lines.append(
        "\nFor each distinct material in the asset, add a "
        '"poly_haven_textures" key to your JSON output mapping a short '
        "descriptive material name to the best matching slug from this list. "
        "Use {} if nothing matches well.\n"
        'Example: "poly_haven_textures": {"body_metal": "metal_plate_wall_001", '
        '"rubber_tires": "worn_rubber_01"}'
    )
    return "\n".join(lines)


def _fetch_slug_files(slug: str) -> dict:
    """Return the full files-API response dict for a slug, or {} on failure."""
    try:
        r = requests.get(_FILES_URL.format(slug=slug), timeout=_FETCH_TIMEOUT)
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json() or {}
    except Exception:
        return {}


def _slug_exists(slug: str) -> bool:
    """Return True if the given slug is a real, non-empty Poly Haven asset."""
    return bool(_fetch_slug_files(slug))


def _closest_slug(slug: str, categories: list[str]) -> Optional[str]:
    """
    Try to find the closest real slug to a hallucinated one.
    1. Try category-filtered catalog.
    2. Fall back to the full texture catalog if nothing matched.
    Returns a real slug string, or None if the catalog is unreachable.
    """
    # Numeric tokens (the "_001"/"_002" variant suffixes Poly Haven slugs use)
    # carry no semantic meaning and would otherwise "match" almost anything.
    slug_words = set(
        w for w in slug.replace("-", "_").split("_")
        if len(w) > 2 and not w.isdigit()
    )

    def _best_in(catalog: list[dict]) -> tuple[Optional[str], int]:
        best: Optional[str] = None
        best_score = 0
        for entry in catalog:
            entry_words = set(
                w for w in entry["slug"].replace("-", "_").split("_")
                if not w.isdigit()
            )
            score = len(slug_words & entry_words)
            if score > best_score:
                best_score = score
                best = entry["slug"]
        return best, best_score

    catalog = fetch_texture_catalog(categories or ["metal", "wood", "concrete"])
    if catalog:
        best, score = _best_in(catalog)
        if best and score >= 1:
            return best
        logger.debug("[POLY_HAVEN] category search returned %d entries but no word overlap for '%s'",
                     len(catalog), slug)

    logger.debug("[POLY_HAVEN] widening search to full catalog for '%s'", slug)
    full = _fetch_all_textures()
    best, score = _best_in(full)
    # Unconstrained search: a single coincidental word match (e.g. "ceramic"
    # only appearing in a dirty roof-tile texture) is too weak a signal once
    # there's no category to anchor it — require at least 2 shared words.
    if best and score >= 2:
        return best

    logger.debug("[POLY_HAVEN] no word overlap found in %d total textures for '%s'", len(full), slug)
    return None


def _slug_categories(slug: str) -> list[str]:
    """Look up a slug's Poly Haven categories from the cached full catalog."""
    for entry in _fetch_all_textures():
        if entry["slug"] == slug:
            return entry.get("categories", [])
    return []


def _is_relevant(mat_name: str, slug: str) -> bool:
    """
    Sanity gate: does this Poly Haven slug plausibly depict the material?
    A missing texture falls back to the procedural builder, which is far
    better than a wrong one (e.g. a concrete floor on a glazed-ceramic mug).
    """
    def _words(s: str) -> set[str]:
        return set(
            w for w in s.lower().replace("-", "_").split("_")
            if len(w) > 2 and not w.isdigit()
        )

    mat_words, slug_words = _words(mat_name), _words(slug)
    if mat_words & slug_words:
        return True
    # Morphological variants: wood/wooden, brick/bricks, metal/metallic…
    if any(
        mw.startswith(sw) or sw.startswith(mw)
        for mw in mat_words for sw in slug_words
        if min(len(mw), len(sw)) >= 4
    ):
        return True

    implied = set(_extract_categories([mat_name]))
    if implied:
        # The material names a real category (wood, metal…): accept the slug
        # only if Poly Haven files it under that category.
        return bool(implied & set(_slug_categories(slug)))

    # No shared words and the material maps to no known category
    # (ceramic, glass, bone…): Poly Haven has nothing trustworthy for it.
    return False


def download_texture_maps(
    slug: str,
    dest_dir: Path,
    resolution: str = _DEFAULT_RES,
    maps: tuple[str, ...] = ("diff", "rough", "nor_gl"),
) -> dict[str, str]:
    """
    Download PBR texture maps for a Poly Haven slug into dest_dir.
    Uses the files API to get correct download URLs (handles Poly Haven's
    capitalised / varied map-name keys and updated CDN path structure).
    Returns a dict mapping canonical map_key → absolute local file path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, str] = {}

    files_data = _fetch_slug_files(slug)
    if not files_data:
        logger.warning("[POLY_HAVEN] could not fetch file manifest for '%s'", slug)
        return result

    def _pick_res(res_dict: dict) -> Optional[str]:
        if resolution in res_dict:
            return resolution
        for fallback_res in ("1k", "2k", "4k"):
            if fallback_res in res_dict:
                return fallback_res
        return next(iter(res_dict), None)

    for map_key in maps:
        map_block: Optional[dict] = None
        for alias in _MAP_KEY_ALIASES.get(map_key, [map_key]):
            if alias in files_data:
                map_block = files_data[alias]
                break
        if map_block is None:
            logger.debug("[POLY_HAVEN] map '%s' not present in '%s'", map_key, slug)
            continue

        chosen_res = _pick_res(map_block)
        if not chosen_res:
            continue

        fmt_block = map_block[chosen_res]
        fmt = "png" if "png" in fmt_block else ("jpg" if "jpg" in fmt_block else None)
        if fmt is None:
            logger.debug("[POLY_HAVEN] no png/jpg for %s/%s/%s", slug, map_key, chosen_res)
            continue

        url = fmt_block[fmt].get("url")
        if not url:
            continue

        out_path = dest_dir / f"{slug}_{map_key}_{chosen_res}.{fmt}"
        if out_path.exists():
            result[map_key] = str(out_path)
            continue

        try:
            r = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
            r.raise_for_status()
            with open(out_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=65536):
                    fh.write(chunk)
            result[map_key] = str(out_path)
            logger.info("[POLY_HAVEN] downloaded %s/%s@%s → %s", slug, map_key, chosen_res, out_path.name)
        except Exception as exc:
            logger.warning("[POLY_HAVEN] download failed %s/%s: %s", slug, map_key, exc)

    return result


def resolve_texture_paths(
    poly_haven_textures: dict[str, str],
    texture_dir: Path,
    resolution: str = _DEFAULT_RES,
) -> dict[str, dict[str, str]]:
    """
    Given planner-selected {material_name: slug} mapping, download all maps
    and return {material_name: {map_key: local_path}}.
    """
    resolved: dict[str, dict[str, str]] = {}
    seen_slugs: dict[str, dict[str, str]] = {}

    for mat_name, slug in poly_haven_textures.items():
        if not slug or not isinstance(slug, str):
            continue

        if not _is_relevant(mat_name, slug):
            logger.warning(
                "[POLY_HAVEN] slug '%s' looks unrelated to material '%s' — "
                "skipping (procedural fallback)", slug, mat_name,
            )
            continue

        if slug in seen_slugs:
            resolved[mat_name] = seen_slugs[slug]
            continue

        if not _slug_exists(slug):
            logger.warning(
                "[POLY_HAVEN] slug '%s' not found on Poly Haven — trying closest match", slug
            )
            hints = [w for w in slug.replace("-", "_").split("_") if len(w) > 3]
            fallback = _closest_slug(slug, hints)
            if fallback and fallback != slug:
                logger.info("[POLY_HAVEN] substituting '%s' → '%s'", slug, fallback)
                slug = fallback
            else:
                logger.warning("[POLY_HAVEN] no substitute found for '%s', skipping", slug)
                continue

        slug_dir = texture_dir / slug
        paths = download_texture_maps(slug, slug_dir, resolution)
        if paths:
            resolved[mat_name] = paths
            seen_slugs[slug] = paths
        else:
            logger.warning("[POLY_HAVEN] no maps downloaded for slug '%s'", slug)

    return resolved
