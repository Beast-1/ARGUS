"""Experimental conditions — each is an env overlay over the default pipeline.

ARGUS already exposes its stages as env vars, so an ablation is a dict, not a code
branch: nothing here modifies the system under test, which keeps the ablation honest
and means these same values reproduce a run from the command line.
"""
from __future__ import annotations

# Applied to every condition: keep runs comparable, non-interactive, and
# non-self-modifying (structure memory would otherwise let earlier runs
# influence later ones and silently break independence between samples).
BASE_ENV = {
    "ARGUS_SEED": "42",
    "ARGUS_AUTO_APPROVE_MEMORY": "0",   # never write to structure memory
    "ARGUS_KEEP_HISTORY": "1",          # don't delete a previous run's folder
    "ARGUS_MAX_SECONDS": "1500",        # hard ceiling so one asset can't stall a sweep
}

CONDITIONS: dict[str, dict[str, str]] = {
    # Reference configuration — what a user gets today.
    "full": {},

    # --- ablations: remove one mechanism at a time ------------------------
    # Does the deterministic compiler beat LLM code generation?
    "no_compiler": {"ARGUS_COMPILER": "0"},

    # Does closed-loop visual repair actually improve the asset, or just churn?
    "no_visual_loop": {"ARGUS_VISUAL_ITERS": "0", "ARGUS_OUTER_REGEN": "0"},

    # Is the outer full-regeneration loop worth its (large) cost?
    "no_outer_regen": {"ARGUS_OUTER_REGEN": "0"},

    # Does sampling several build-specs and scoring them render-free help?
    "single_spec": {"ARGUS_SPEC_CANDIDATES": "1"},

    # Does the plain-English design brief before the JSON spec help?
    "no_two_stage": {"ARGUS_TWO_STAGE": "0"},

    # Does the spec-derived yes/no checklist beat a free-form 1-10 rating?
    "no_checklist": {"ARGUS_CHECKLIST": "0"},

    # Does the reference image ground generation, or is it decoration?
    "no_concept_image": {"__no_concept": "1"},   # handled as a CLI flag, not env

    # Shape-vocabulary ablation for the taper work.
    "no_bevel": {"ARGUS_BEVEL": "0"},

    # Speed preset — the accuracy/latency trade-off users actually face.
    "fast": {"ARGUS_FAST": "1"},
}

# Conditions that need a CLI flag rather than an env var.
CLI_FLAGS = {"no_concept_image": ["--no-concept"]}


def env_for(condition: str) -> dict[str, str]:
    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition {condition!r}; known: {sorted(CONDITIONS)}")
    env = dict(BASE_ENV)
    env.update({k: v for k, v in CONDITIONS[condition].items() if not k.startswith("__")})
    return env


def flags_for(condition: str) -> list[str]:
    return list(CLI_FLAGS.get(condition, []))
