"""Ablation sweep runner.

Runs (prompt x condition) cells as isolated subprocesses and appends one JSON record
per cell to a results file.

Why a subprocess per cell rather than calling run_pipeline() in-process:
  * main.py and core/blender.py resolve their output roots at MODULE IMPORT time, so
    a per-cell output directory only takes effect in a fresh interpreter;
  * conditions must not leak into one another through module-level config, caches,
    or LLM key-pool state;
  * a crash or hang in one cell cannot take the sweep down.

Resumable by design: a full sweep is (30 prompts x N conditions x ~10 min), i.e. days.
Completed cells are skipped on re-run, so a sweep can be stopped and restarted freely.

Usage
-----
  python -m eval.runner --conditions full,no_compiler --tier simple
  python -m eval.runner --conditions full --limit 3 --dry-run
  python -m eval.runner --list
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics as metrics_mod  # noqa: E402
from eval.conditions import CONDITIONS, env_for, flags_for  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BENCHMARK = Path(__file__).with_name("benchmark.jsonl")
RESULTS_DIR = REPO / "eval" / "results"


def load_benchmark(tier: str | None = None, only: set[str] | None = None) -> list[dict]:
    rows = []
    for line in BENCHMARK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if tier and row.get("tier") != tier:
            continue
        if only and row["id"] not in only:
            continue
        rows.append(row)
    return rows


def completed_cells(results_path: Path) -> set[tuple[str, str]]:
    """Cells that don't need to run again.

    A cell that failed because of the environment (network/DNS down, a provider
    outage — see _ENV_FAILURE_SIGNATURES) is deliberately NOT counted as done: it
    produced no usable measurement, so simply re-running the same command retries
    exactly those cells and nothing else. This is what a 15-cell sweep needed after
    a mid-run network outage silently turned 13 cells into meaningless score=None
    records — resuming should have been one command, not a manual diagnosis.
    """
    if not results_path.exists():
        return set()
    done = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # a partially written final line; it will simply be re-run
        if rec.get("env_failure"):
            continue
        done.add((rec["condition"], rec["prompt_id"]))
    return done


# Substrings that mean "the environment failed the run", not "the model produced a
# bad asset". A sweep is often unattended for hours; a mid-sweep network outage or
# quota exhaustion must not silently masquerade as a quality result — this exact
# failure mode corrupted 13 of 15 cells in the first real sweep (a DNS outage from
# ~20:01 onward made every provider fail, and every cell after it recorded a
# meaningless score=None indistinguishable from "the model tried and produced
# nothing"). Matched against the full log, not just the 4000-char tail, so a long
# retry storm before the final failure doesn't push the signature out of range.
_ENV_FAILURE_SIGNATURES = [
    "NameResolutionError", "getaddrinfo failed", "Failed to resolve",
    "Max retries exceeded", "ConnectionError", "ConnectionResetError",
    "Temporary failure in name resolution",
]


def run_cell(item: dict, condition: str, out_root: Path, timeout: int,
            log_dir: Path | None = None) -> dict:
    """One (prompt, condition) generation, fully isolated."""
    env = dict(os.environ)
    env.update(env_for(condition))
    # Both names: main.py reads ARGUS_OUTPUT_ROOT, core/blender.py historically read
    # ARGUS_OUT_ROOT. Setting both keeps this correct on either version.
    env["ARGUS_OUTPUT_ROOT"] = str(out_root)
    env["ARGUS_OUT_ROOT"] = str(out_root)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [sys.executable, "main.py", "--prompt", item["prompt"], *flags_for(condition)]

    t0 = time.monotonic()
    timed_out = False
    returncode = -1
    full_log = ""
    try:
        proc = subprocess.run(
            cmd, cwd=REPO, env=env, timeout=timeout,
            capture_output=True, text=True, errors="replace",
        )
        returncode = proc.returncode
        full_log = proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        full_log = (exc.stdout or b"").decode("utf-8", "replace") if exc.stdout else ""
    elapsed = round(time.monotonic() - t0, 1)

    # Full stdout is kept on disk, not in the JSON record — a 15-cell sweep would
    # otherwise put megabytes of log text in runs.jsonl. This is what made
    # diagnosing the network outage above require a slow manual repro; it will not
    # next time.
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{condition}__{item['id']}.log"
        log_path.write_text(full_log, encoding="utf-8", errors="replace")

    env_failure = next((s for s in _ENV_FAILURE_SIGNATURES if s in full_log), None)
    log_tail = full_log[-4000:]

    record = {
        "prompt_id": item["id"],
        "prompt": item["prompt"],
        "tier": item.get("tier"),
        "category": item.get("category"),
        "condition": condition,
        "elapsed_sec": elapsed,
        "returncode": returncode,
        "timed_out": timed_out,
        "rejected": "[ARGUS] Request rejected:" in log_tail,
        "env_failure": env_failure,  # e.g. "NameResolutionError" — see above
    }

    asset = _find_asset(out_root, log_tail)
    if asset:
        m = metrics_mod.collect(asset, out_root / "runs" / asset.name)
        record.update(m.as_dict())
    else:
        record.update({"asset": None, "exists": False,
                       "notes": ["no asset directory found"]})
    return record


def _find_asset(out_root: Path, log_tail: str) -> Path | None:
    """Prefer the run id the pipeline printed; fall back to newest folder."""
    for line in reversed(log_tail.splitlines()):
        if line.startswith("Project     :"):
            candidate = out_root / "final" / line.split(":", 1)[1].strip()
            if candidate.is_dir():
                return candidate
    final = out_root / "final"
    if not final.is_dir():
        return None
    dirs = [d for d in final.iterdir() if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def main() -> int:
    ap = argparse.ArgumentParser(description="ARGUS ablation sweep")
    ap.add_argument("--conditions", default="full",
                    help="comma-separated condition names, or 'all'")
    ap.add_argument("--tier", choices=["simple", "moderate", "complex"])
    ap.add_argument("--ids", help="comma-separated prompt ids (e.g. b01,b09)")
    ap.add_argument("--limit", type=int, help="cap prompts per condition")
    ap.add_argument("--timeout", type=int, default=2100, help="per-cell seconds")
    ap.add_argument("--results", default=str(RESULTS_DIR / "runs.jsonl"))
    ap.add_argument("--out-root", default=str(REPO / "eval" / "out"))
    ap.add_argument("--log-dir", default=str(RESULTS_DIR / "logs"),
                    help="full per-cell stdout, one file per (condition, prompt_id)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="list conditions and exit")
    args = ap.parse_args()

    if args.list:
        for name, overlay in CONDITIONS.items():
            print(f"  {name:20s} {overlay or '(reference configuration)'}")
        return 0

    names = sorted(CONDITIONS) if args.conditions == "all" else \
        [c.strip() for c in args.conditions.split(",") if c.strip()]
    for name in names:
        if name not in CONDITIONS:
            print(f"unknown condition: {name}", file=sys.stderr)
            return 2

    only = {i.strip() for i in args.ids.split(",")} if args.ids else None
    items = load_benchmark(args.tier, only)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("no benchmark items selected", file=sys.stderr)
        return 2

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_cells(results_path)

    cells = [(c, it) for c in names for it in items if (c, it["id"]) not in done]
    skipped = len(names) * len(items) - len(cells)
    print(f"conditions={names}  prompts={len(items)}  "
          f"cells={len(cells)} (skipping {skipped} already done)")
    print(f"worst case ~{len(cells) * args.timeout / 3600:.1f}h; results -> {results_path}")

    if args.dry_run:
        for cond, it in cells:
            print(f"  would run [{cond}] {it['id']} {it['prompt'][:52]}")
        return 0

    for n, (cond, it) in enumerate(cells, 1):
        out_root = Path(args.out_root) / cond
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"\n[{n}/{len(cells)}] {cond} :: {it['id']} :: {it['prompt'][:60]}", flush=True)
        record = run_cell(it, cond, out_root, args.timeout, log_dir=Path(args.log_dir))
        # Append immediately so an interrupted sweep loses at most one cell.
        with results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if record.get("env_failure"):
            print(f"    !! ENVIRONMENT FAILURE ({record['env_failure']}) — not a "
                  f"quality result, will retry on next invocation. "
                  f"Full log: {args.log_dir}/{cond}__{it['id']}.log", flush=True)
        else:
            print(f"    -> score={record.get('visual_score')} "
                  f"sev={record.get('topology_severity')} "
                  f"tex={record.get('textured_materials')} "
                  f"tris={record.get('triangles')} {record['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
