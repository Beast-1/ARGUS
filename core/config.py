"""The single source of truth for every environment variable ARGUS reads.

README.md pointed here for years before this file existed; configuration was
actually ~70 scattered `os.getenv()` calls across a dozen modules, with no way
to answer "what can I configure?" short of grepping. This module is that answer.

It is a *registry*, deliberately not a wrapper. Modules keep reading their own
`os.getenv()` at import time (several need the value before this module would
finish importing, and rewriting 70 call sites buys indirection, not safety).
What this module guarantees instead is that the registry is complete:
`tests/test_config_registry.py` walks the codebase for `os.getenv`/
`os.environ.get` calls and fails if any variable is missing from `SETTINGS` or
declares a default that disagrees with the call site. A registry that can
silently fall behind the code is worse than no registry, because it reads as
authoritative while being wrong.

Use `describe()` for a human-readable dump, or `write_env_example()` to
regenerate `.env.example`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Setting:
    name: str
    default: Optional[str]
    group: str
    doc: str
    secret: bool = False

    @property
    def value(self) -> Optional[str]:
        return os.getenv(self.name, self.default)


def _s(name, default, group, doc, secret=False) -> Setting:
    return Setting(name=name, default=default, group=group, doc=doc, secret=secret)


SETTINGS: tuple[Setting, ...] = (
    # -- Credentials ---------------------------------------------------------
    # The planner/codegen chain tries these in order. At least one of Gemini,
    # DeepSeek, Groq or OpenRouter must be set for generation to work at all.
    _s("GOOGLE_API_KEY", "", "credentials",
       "Gemini key (planner, codegen, vision scoring). GOOGLE_API_KEY_2..._10 "
       "are also read and round-robined as a pool.", secret=True),
    _s("DEEPSEEK_API_KEY", "", "credentials", "DeepSeek key (codegen fallback).", secret=True),
    _s("GROQ_API_KEY", "", "credentials", "Groq key (fast codegen fallback).", secret=True),
    _s("GROQ_API_KEY_2", "", "credentials", "Second Groq key for the pool.", secret=True),
    _s("OPENROUTER_API_KEY", "", "credentials", "OpenRouter key (last-resort codegen).", secret=True),
    _s("OPENROUTER_API_KEY_2", "", "credentials", "Second OpenRouter key for the pool.", secret=True),
    _s("HF_API_KEY", "", "credentials", "HuggingFace key (reference-image generation).", secret=True),
    _s("NVIDIA_API_KEY", "", "credentials", "NVIDIA NIM key (reference images).", secret=True),
    _s("NVIDIA_NIM_API_KEY", "", "credentials", "Alias for NVIDIA_API_KEY.", secret=True),
    _s("CLOUDFLARE_ACCOUNT_ID", "", "credentials", "Cloudflare Workers AI account (reference images).", secret=True),
    _s("CLOUDFLARE_API_TOKEN", "", "credentials", "Cloudflare Workers AI token.", secret=True),
    # NOTE: GITHUB_TOKEN* are deliberately absent. GitHub Models was retired
    # 2026-07-30; core/llm.py ignores those tokens on purpose (see the comment
    # there). Registering them would imply they still do something.

    # -- Blender -------------------------------------------------------------
    _s("BLENDER_PATH", "blender", "blender",
       "Path to the Blender executable, or just 'blender' if it's on PATH."),
    _s("BLENDER_EXEC_TIMEOUT", "300", "blender",
       "Seconds a single Blender build may run before it's killed."),
    _s("BLENDER_MCP_TIMEOUT", "120", "blender",
       "Seconds a headless MCP topology-analysis pass may run."),

    # -- Pipeline shape ------------------------------------------------------
    _s("ARGUS_OUTPUT_ROOT", "out", "pipeline",
       "Root directory for runs/ and final/. Read at import time, so one "
       "process per output root (this is why eval/runner.py uses subprocesses)."),
    _s("ARGUS_OUT_ROOT", None, "pipeline",
       "Legacy alias for ARGUS_OUTPUT_ROOT, still honoured."),
    _s("ARGUS_FAST", "0", "pipeline", "1 = skip the expensive quality passes."),
    _s("ARGUS_SEED", "42", "pipeline", "Seed for the compiler's deterministic jitter/greebles."),
    _s("ARGUS_MAX_SECONDS", "0", "pipeline",
       "Hard wall-clock budget for one run; 0 = unlimited."),
    _s("ARGUS_COMPILER", "1", "pipeline",
       "1 = use the deterministic scene-graph compiler; 0 = LLM codegen only."),
    _s("ARGUS_VALIDATE", "1", "pipeline", "1 = run the topology validation pass."),
    _s("ARGUS_KEEP_HISTORY", "0", "pipeline", "1 = keep intermediate per-iteration artifacts."),
    _s("ARGUS_AUTO_APPROVE_MEMORY", "0", "pipeline",
       "1 = save qualifying builds to structure memory without asking."),
    _s("ARGUS_AUDIT_THRESHOLD", "7", "pipeline", "Score at or above which a build is audited."),

    # -- Quality loop --------------------------------------------------------
    _s("ARGUS_VISUAL_TARGET", "7", "quality", "Visual score the repair loop aims for."),
    _s("ARGUS_VISUAL_ITERS", "5", "quality", "Max visual-improvement iterations."),
    _s("ARGUS_OUTER_REGEN", "3", "quality", "Max full-regeneration attempts."),
    _s("ARGUS_INITIAL_CANDIDATES", "3", "quality", "Best-of-N candidate builds."),
    _s("ARGUS_BESTN_EARLY_EXIT", None, "quality",
       "Score that ends best-of-N early. Defaults to ARGUS_VISUAL_TARGET + 1."),
    _s("ARGUS_SPEC_CANDIDATES", "3", "quality", "Parallel plan candidates scored before building."),
    _s("ARGUS_TEXTURE_WORTH", "2", "quality",
       "Cap on the effective score's texture bonus, in score points."),
    _s("ARGUS_MAX_REPAIRS", "4", "quality",
       "DEAD: read here but never used; the live cap is MAX_REPAIR_ATTEMPTS in "
       "core/prompt.py. Registered so the discrepancy is documented, not hidden."),
    _s("ARGUS_CHECKLIST", "1", "quality", "1 = use the checklist critic when scoring."),
    _s("ARGUS_CRITIC_VOTES", "1", "quality", "Critic votes per review round."),
    _s("ARGUS_TWO_STAGE", "1", "quality", "1 = two-stage (brief then plan) planning."),
    _s("ARGUS_MULTIVIEW_REF", "1", "quality", "1 = score against a multi-view render grid."),

    # -- Geometry / finishing ------------------------------------------------
    _s("ARGUS_POLY_ENFORCE", "1", "geometry",
       "1 = enforce the poly budget on the exported GLB (planar dissolve, then "
       "collapse to the _POLY_TARGETS ceiling). The .blend keeps full detail. "
       "Set 0 to restore the old advisory-only behaviour, where the budget was "
       "a prompt request nothing verified and output ran up to 56x over."),
    _s("ARGUS_BEVEL", "1", "geometry", "1 = apply the bevel/weighted-normals pass."),
    _s("ARGUS_GREEBLES", "1", "geometry", "1 = add compiler surface detail."),
    _s("ARGUS_JITTER", "1", "geometry", "1 = apply deterministic imperfection jitter."),
    _s("ARGUS_RIG", "1", "geometry", "1 = auto-rig articulated parts."),
    _s("ARGUS_LODS", "0", "geometry", "1 = export additional LOD meshes."),
    _s("ARGUS_BAKE", "1", "geometry", "1 = bake procedural materials to textures."),
    _s("ARGUS_AO_BAKE", "1", "geometry", "1 = bake ambient occlusion."),
    _s("ARGUS_BAKE_RES", "512", "geometry", "Bake texture resolution in pixels."),

    # -- Texturing -----------------------------------------------------------
    _s("ARGUS_PAINT", "1", "texturing", "1 = run the diffusion texture-paint pass."),
    _s("ARGUS_PAINT_RES", "1024", "texturing", "Paint pass resolution in pixels."),
    _s("ARGUS_PAINT_BRIGHT", "1.4", "texturing", "Brightness multiplier for painted textures."),
    _s("ARGUS_PH_RESOLUTION", "1k", "texturing", "Poly Haven texture resolution to fetch."),
    _s("ARGUS_PH_FETCH_TIMEOUT", "10", "texturing", "Seconds to wait for the Poly Haven catalog."),
    _s("ARGUS_PH_DOWNLOAD_TIMEOUT", "30", "texturing", "Seconds to wait for a texture download."),
    _s("ARGUS_GEMINI_IMAGES", "0", "texturing", "1 = allow Gemini image models for references."),
    _s("ARGUS_IMAGE_PROVIDERS", None, "texturing",
       "Comma-separated reference-image provider order; defaults to the built-in order."),

    # -- LLM transport -------------------------------------------------------
    _s("ARGUS_LLM_MAX_RETRIES", "3", "llm", "Attempts per provider before falling through."),
    _s("ARGUS_LLM_BACKOFF", "2.0", "llm",
       "Cooldown seconds applied to a rate-limited key (times two). Despite the "
       "name this is a flat cooldown, not an exponential backoff base."),
    _s("ARGUS_GEMINI_TIMEOUT", "90", "llm", "Per-request Gemini timeout in seconds."),
    _s("ARGUS_GEMINI_CODE_TIMEOUT", "240", "llm", "Wall-clock cap for a Gemini codegen call."),
    _s("ARGUS_OR_TIMEOUT", "180", "llm", "Per-request OpenRouter/Groq timeout in seconds."),
    _s("ARGUS_USE_GEMINI_PRO", "0", "llm", "1 = include gemini-2.5-pro in the model list."),

    # -- Live Blender (MCP) --------------------------------------------------
    _s("ARGUS_MCP_HOST", "127.0.0.1", "mcp",
       "Bind address for the live-Blender exec server. Changing this off "
       "loopback exposes an code-execution endpoint; don't."),
    _s("ARGUS_MCP_PORT", "9876", "mcp", "Port for the live-Blender exec server."),
    _s("ARGUS_MCP_TIMEOUT", "180", "mcp", "Seconds to wait on a live-MCP response."),
    _s("ARGUS_MCP_AUTOLAUNCH", "1", "mcp", "1 = auto-launch a GUI Blender if none is listening."),
    _s("ARGUS_MCP_TOKEN", "", "mcp",
       "Shared secret required by the live exec server. Generated per launch "
       "when unset; only set it manually to attach to an existing server.", secret=True),
    _s("BLENDER_HOST", None, "mcp", "Legacy alias for ARGUS_MCP_HOST."),

    # -- Service / storage ---------------------------------------------------
    _s("ARGUS_API_TOKEN", None, "service",
       "Bearer token the desktop UI must send. Generated per launch by "
       "run_app.bat and inherited by both processes.", secret=True),
    _s("ARGUS_REQUIRE_USER_KEYS", "0", "service",
       "Set to 1 to reject generation requests that don't carry the caller's own "
       "provider keys. Off by default so the desktop app and CLI keep using .env; "
       "turn it on for any publicly reachable deployment, or a visitor falls "
       "through to the operator's credentials and spends their quota."),
    _s("ARGUS_ALLOWED_ORIGINS", "", "service",
       "Comma-separated extra CORS origins, for serving the UI from somewhere "
       "other than the desktop shell (e.g. https://argus.pages.dev). The "
       "localhost and tauri:// origins are always allowed. An explicit list "
       "rather than '*': a tunnelled backend is reachable from anywhere."),
    _s("REDIS_URL", "redis://localhost:6379/0", "service",
       "Queue backend for the optional worker (service/worker.py)."),
    _s("ARGUS_S3_BUCKET", "", "storage", "Bucket for uploading finished assets. Empty = local only.", secret=False),
    _s("ARGUS_S3_ENDPOINT_URL", "", "storage", "S3-compatible endpoint URL."),
    _s("ARGUS_S3_REGION", "auto", "storage", "S3 region."),
    _s("ARGUS_S3_ACCESS_KEY_ID", "", "storage", "S3 access key id.", secret=True),
    _s("ARGUS_S3_SECRET_ACCESS_KEY", "", "storage", "S3 secret access key.", secret=True),
    _s("ARGUS_S3_PUBLIC_BASE_URL", "", "storage", "Public base URL prefix for uploaded assets."),
    _s("ARGUS_S3_PRESIGN_EXPIRES", "604800", "storage", "Presigned URL lifetime in seconds."),
)

BY_NAME: dict[str, Setting] = {s.name: s for s in SETTINGS}

GROUP_ORDER = (
    "credentials", "blender", "pipeline", "quality", "geometry",
    "texturing", "llm", "mcp", "service", "storage",
)

GROUP_TITLES = {
    "credentials": "API credentials — at least one generation provider is required",
    "blender": "Blender",
    "pipeline": "Pipeline shape",
    "quality": "Quality loop",
    "geometry": "Geometry and finishing",
    "texturing": "Texturing",
    "llm": "LLM transport",
    "mcp": "Live Blender (MCP) — opt-in, executes code in a running Blender",
    "service": "Desktop service",
    "storage": "Remote storage (optional)",
}


def describe() -> str:
    """Human-readable dump of every setting, its default, and its current value."""
    lines: list[str] = []
    for group in GROUP_ORDER:
        lines.append(f"\n# {GROUP_TITLES[group]}")
        for s in (x for x in SETTINGS if x.group == group):
            current = s.value
            if s.secret and current:
                current = f"<set, {len(current)} chars>"
            shown = "" if current in (None, "") else f"  (current: {current})"
            default = "" if s.default in (None, "") else f" [default: {s.default}]"
            lines.append(f"  {s.name}{default}{shown}\n      {s.doc}")
    return "\n".join(lines).lstrip("\n")


def write_env_example(path: str = ".env.example") -> str:
    """Regenerate .env.example from this registry so the two can't drift."""
    out: list[str] = [
        "# ARGUS configuration — generated by `python -m core.config --write-env-example`.",
        "# Every variable ARGUS reads is listed here; see core/config.py for details.",
        "# Copy to .env and fill in at least one generation provider's key.",
    ]
    for group in GROUP_ORDER:
        out.append(f"\n# --- {GROUP_TITLES[group]} ---")
        for s in (x for x in SETTINGS if x.group == group):
            out.append(f"# {s.doc}")
            out.append(f"{s.name}={s.default if s.default is not None else ''}")
    text = "\n".join(out) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


if __name__ == "__main__":
    import sys
    if "--write-env-example" in sys.argv:
        write_env_example()
        print("wrote .env.example")
    else:
        print(describe())
