"""ARGUS Batch Queue — run multiple generation jobs with a configurable thread pool."""
from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

_print_lock = threading.Lock()


def _safe_print(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs)


def load_prompts(source: str | Path) -> list[str]:
    """
    Load prompts from a file path.

    Supported formats:
      - .json  — list of strings, or list of {"prompt": "..."} objects
      - .txt   — one prompt per non-blank line (# lines are comments)
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON prompt file must be a list")
        prompts: list[str] = []
        for item in data:
            if isinstance(item, str):
                prompts.append(item.strip())
            elif isinstance(item, dict):
                p = item.get("prompt") or item.get("text") or ""
                if p:
                    prompts.append(str(p).strip())
        return [p for p in prompts if p]

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def run_batch(
    prompts: list[str],
    *,
    workers: int = 1,
    poly_budget: str | None = None,
    export_format: str = "glb",
    mcp_mode: bool = False,
    use_concept_pipeline: bool = True,
    memory_approval_callback: Optional[Callable] = None,
    report_path: Optional[Path] = None,
) -> list[dict]:
    """
    Run `prompts` concurrently using a thread pool of `workers` threads.

    Parameters
    ----------
    prompts                  : list of natural-language asset prompts
    workers                  : number of parallel generation threads
                               (default 1 — safe for single Blender install;
                                increase only if multiple Blender executables exist)
    poly_budget              : "low" | "medium" | "high"
    export_format            : "glb" | "fbx"
    mcp_mode                 : use live Blender MCP server instead of headless
    use_concept_pipeline     : generate reference images before Blender scripting
    memory_approval_callback : override the default user-approval prompt; pass
                               `lambda _: False` in automated contexts
    report_path              : optional path to write a JSON summary report

    Returns
    -------
    List of result dicts, one per prompt, sorted by job_id:
        job_id      int
        prompt      str
        success     bool
        elapsed_sec float
        error       str   (empty on success)
    """
    from main import run_pipeline  # noqa: PLC0415

    auto_approve = memory_approval_callback or (lambda _: False)
    results: list[dict] = []
    results_lock = threading.Lock()

    def _run_one(job_id: int, prompt: str) -> dict:
        prefix = f"[BATCH {job_id + 1}/{len(prompts)}]"
        _safe_print(f"\n{prefix} Starting: {prompt[:70]}")
        # Each job gets its own cancel scope, so cancelling one does not stop the
        # others. This is what made --workers > 1 unsafe: every job shared one
        # process-wide cancel flag, and since core/blender.py now terminates the
        # Blender subprocess on that signal, a single cancel would have killed
        # every concurrent job's render mid-write.
        from core.cancel import new_scope
        new_scope()
        t0 = time.monotonic()
        error = ""
        try:
            success = run_pipeline(
                prompt=prompt,
                poly_budget=poly_budget,
                export_format=export_format,
                mcp_mode=mcp_mode,
                use_concept_pipeline=use_concept_pipeline,
                memory_approval_callback=auto_approve,
            )
        except Exception as exc:
            success = False
            error = str(exc)
            _safe_print(f"{prefix} Exception: {exc}")

        elapsed = round(time.monotonic() - t0, 1)
        status = "OK" if success else "FAILED"
        _safe_print(f"{prefix} {status} in {elapsed}s — {prompt[:50]}")
        return {
            "job_id":      job_id,
            "prompt":      prompt,
            "success":     bool(success),
            "elapsed_sec": elapsed,
            "error":       error,
        }

    _safe_print(f"\n{'='*60}")
    _safe_print(f"  ARGUS BATCH  —  {len(prompts)} job(s), {workers} worker(s)")
    _safe_print(f"{'='*60}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, i, p): i for i, p in enumerate(prompts)}
        for future in concurrent.futures.as_completed(futures):
            rec = future.result()
            with results_lock:
                results.append(rec)

    results.sort(key=lambda r: r["job_id"])

    passed = sum(1 for r in results if r["success"])
    _safe_print(f"\n[ARGUS BATCH COMPLETE]  {passed}/{len(results)} succeeded")

    if report_path:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "passed":  passed,
            "failed":  len(results) - passed,
            "total":   len(results),
            "results": results,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        _safe_print(f"[ARGUS BATCH] Report saved → {report_path}")

    return results
