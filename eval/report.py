"""Aggregate sweep results into a comparison table.

  python -m eval.report                       # all conditions
  python -m eval.report --by tier             # break down by difficulty tier
  python -m eval.report --format md           # markdown, for pasting into a paper
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.metrics import summarise  # noqa: E402

DEFAULT_RESULTS = Path(__file__).parent / "results" / "runs.jsonl"

COLUMNS = [
    ("n_env_failures", "envFail"),
    ("n_built", "built"),
    ("build_rate", "build%"),
    ("mean_visual_score", "score"),
    ("mean_effective_score", "eff"),
    ("pct_at_or_above_7", ">=7%"),
    ("pct_effective_at_or_above_8", "eff>=8%"),
    ("pct_topology_clean", "clean%"),
    ("pct_topology_critical", "crit%"),
    ("pct_assets_textured", "tex%"),
    ("mean_triangles", "tris"),
    ("mean_elapsed_sec", "sec"),
    ("taper_uptake_boxes", "boxTpr%"),
    ("taper_uptake_cylinders", "cylTpr%"),
]


def load(path: Path, dedupe: bool = True) -> list[dict]:
    """Read runs.jsonl.

    `runs.jsonl` is append-only, so a resumed sweep leaves the original
    env-failure attempt for a cell alongside its later successful retry —
    two rows for one (condition, prompt_id). Env-failure rows have
    exists=False, so score/quality aggregates are already unaffected, but
    n_attempted / build_rate would still double-count the dead attempt.
    Dedupe keeps the LAST row per cell (the most recent attempt is the
    authoritative one) while leaving the file itself untouched, so the full
    attempt history — including env failures — stays on disk for audit.
    Pass dedupe=False to see the raw log, e.g. to count how many env
    failures occurred during collection.
    """
    if not path.exists():
        print(f"no results at {path} — run `python -m eval.runner` first", file=sys.stderr)
        raise SystemExit(1)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    if not dedupe:
        return out
    latest: dict[tuple, dict] = {}
    for rec in out:
        latest[(rec.get("condition"), rec.get("prompt_id"))] = rec
    return list(latest.values())


def fmt(value) -> str:
    if value is None:
        # ASCII: this table is read on a Windows console (cp1252 by default),
        # where an em-dash renders as a replacement glyph and makes a column of
        # "no data" cells look like corruption rather than absence.
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render(rows: list[tuple[str, dict]], style: str) -> str:
    headers = ["condition", *[label for _, label in COLUMNS]]
    body = [[name, *[fmt(stats.get(key)) for key, _ in COLUMNS]] for name, stats in rows]

    if style == "md":
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join("---" for _ in headers) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(lines)

    widths = [max(len(h), *(len(r[i]) for r in body)) if body else len(h)
              for i, h in enumerate(headers)]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append("  ".join("-" * w for w in widths))
    out += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in body]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="summarise an ARGUS ablation sweep")
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--by", choices=["tier", "category"],
                    help="additionally break each condition down by this field")
    ap.add_argument("--format", choices=["text", "md"], default="text")
    args = ap.parse_args()

    records = load(Path(args.results))
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_condition[rec.get("condition", "?")].append(rec)

    # Reference condition first, then alphabetical — reads like a paper table.
    order = sorted(by_condition, key=lambda c: (c != "full", c))
    rows = [(c, summarise(by_condition[c])) for c in order]
    print(render(rows, args.format))
    print(f"\n{len(records)} runs across {len(by_condition)} condition(s)")

    if args.by:
        for cond in order:
            groups: dict[str, list[dict]] = defaultdict(list)
            for rec in by_condition[cond]:
                groups[str(rec.get(args.by))].append(rec)
            print(f"\n[{cond}] by {args.by}")
            sub = [(g, summarise(groups[g])) for g in sorted(groups)]
            print(render(sub, args.format))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
