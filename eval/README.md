# ARGUS evaluation harness

Reproducible ablation sweeps over a fixed benchmark, for measuring whether a change
to the pipeline actually helps.

## Layout

| file | role |
|---|---|
| `benchmark.jsonl` | 30 prompts, stratified `simple` / `moderate` / `complex`, with the parts a correct result should contain |
| `conditions.py` | experimental conditions as env overlays — an ablation is a dict, not a code branch |
| `metrics.py` | deterministic per-asset metrics read from the exported GLB + manifest |
| `runner.py` | runs (prompt × condition) cells as isolated subprocesses, resumable |
| `report.py` | aggregates results into a comparison table |

## Running

```bash
python -m eval.runner --list                                  # show conditions
python -m eval.runner --conditions full --tier simple --dry-run
python -m eval.runner --conditions full,no_compiler --tier simple
python -m eval.report --by tier
python -m eval.report --format md            # paste into a paper
```

Results append to `eval/results/runs.jsonl`, one JSON record per cell. Assets are
written under `eval/out/<condition>/` so conditions never overwrite each other.

## Design notes

**One subprocess per cell.** `main.py` and `core/blender.py` resolve their output
roots at *module import time*, so a per-cell output directory only takes effect in a
fresh interpreter. Subprocesses also stop conditions leaking into each other through
module-level config or LLM key-pool state, and keep one hung cell from killing a sweep.

**Resumable.** A full sweep is (30 prompts × N conditions × ~10 min) — days. Completed
cells are skipped on re-run, so it can be stopped and restarted freely. Records are
appended immediately, so an interrupt loses at most one cell.

**Structure memory is disabled** (`ARGUS_AUTO_APPROVE_MEMORY=0`). Otherwise earlier
runs would teach later ones and samples would stop being independent.

**Metrics never re-run the vision model.** Everything except `visual_score` is computed
from artifacts, so re-scoring a finished library is free and a metric cannot disagree
with itself between runs. `visual_score` is read back from `manifest.json` rather than
recomputed — the pipeline has previously had two scoring stages grade the same render
differently (2 vs 4).

## Baseline

Measured over the pre-existing `out/final` library (16 assets, mixed provenance,
several predating the current compiler) — the starting point, not a result:

```
mean_visual_score      6.33      pct_topology_clean      67%
pct_at_or_above_7      56%       pct_topology_critical   11%
pct_assets_textured    62%       taper_uptake_boxes      0.0%
mean_triangles         9,814     taper_uptake_cylinders  8.5%
```

`taper_uptake_*` measures how often the planner reaches for the shape controls it
has. Box taper was added after these assets were built, so 0% is expected there;
cylinders could always taper via `r_top` and were used 8.5% of the time, which is
the gap the prompt changes target.

## Caveats

- `visual_score` is model-judged and noisy; treat single-run differences as weak
  evidence and prefer differences that hold across the benchmark.
- Free-tier API rate limits can make a cell fail for reasons unrelated to the
  condition. `returncode` / `timed_out` / `rejected` are recorded so those cells can
  be excluded rather than silently counted as quality failures.
- The benchmark is deliberately weighted toward ARGUS's domain (hard-surface props).
  It is not a general text-to-3D benchmark and shouldn't be reported as one.
