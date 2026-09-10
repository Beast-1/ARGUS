"""core/config.py is a registry, not a wrapper — modules still call os.getenv()
directly. That only works as documentation if the registry cannot fall behind
the code, so these tests walk the actual source for env-var reads and fail when
one is missing or declares a default that disagrees with its call site.

A registry that can silently drift is worse than none: it reads as authoritative
while being wrong. This is what stops that.

Hermetic: reads source files, runs nothing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import BY_NAME, SETTINGS, describe, write_env_example  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

_CALL = re.compile(
    r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Za-z0-9_]+)["']\s*(?:,\s*("[^"]*"|'[^']*'))?"""
)

# BYOK routes single-value provider credentials through
# core/llm.py:_provider_secret("<attr>", "<ENV_VAR>"), which prefers the running
# request's own key and falls back to the environment. The env var is still read
# — it is just no longer inside an os.getenv() call, so the scan above cannot see
# it. Counting these keeps the drift check honest instead of exempting them.
_INDIRECT_CALL = re.compile(r"""_provider_secret\(\s*["'][A-Za-z0-9_]+["']\s*,\s*["']([A-Za-z0-9_]+)["']""")

# Pool members (GOOGLE_API_KEY_2 .. _10) are built by f-string, and the retired
# GitHub Models tokens are read only to warn that they're ignored — neither is a
# separate knob, so neither needs its own registry entry.
_EXEMPT = re.compile(r"^(GOOGLE_API_KEY_\d+|GITHUB_TOKEN(_\d+)?)$")

# Read via an f-string pool builder (core/llm.py:_build_google_pool) rather than
# a literal os.getenv("GOOGLE_API_KEY"), so the source scan can't see it — but
# it is very much read, and is the one key most users will set.
_READ_VIA_FSTRING = {"GOOGLE_API_KEY"}

_SKIP_DIRS = (".venv", "__pycache__", "node_modules", "tests", "eval")


def _source_files():
    for path in sorted(REPO.glob("**/*.py")):
        rel = path.relative_to(REPO).as_posix()
        if any(part in rel for part in _SKIP_DIRS):
            continue
        yield rel, path.read_text(encoding="utf-8")


def _reads_in_source() -> dict[str, set[str]]:
    """{env var name: {files that read it}} across the shipped code."""
    found: dict[str, set[str]] = {}
    for rel, text in _source_files():
        for pattern in (_CALL, _INDIRECT_CALL):
            for match in pattern.finditer(text):
                name = match.group(1)
                if _EXEMPT.match(name):
                    continue
                found.setdefault(name, set()).add(rel)
    return found


def test_every_env_var_the_code_reads_is_registered():
    missing = {
        name: sorted(files)
        for name, files in _reads_in_source().items()
        if name not in BY_NAME
    }
    assert not missing, (
        "env vars read by the code but absent from core/config.py:SETTINGS — "
        "add them there (that file is what README points users at):\n"
        + "\n".join(f"  {n}: {f}" for n, f in sorted(missing.items()))
    )


def test_registry_does_not_document_vars_the_code_never_reads():
    """The other drift direction: a stale entry describes a knob that no longer
    does anything, which is exactly how ARGUS_MAX_REPAIRS became misleading."""
    read = set(_reads_in_source())
    # Read by the service/launcher layer or by generated code rather than a
    # plain os.getenv() in a scanned module.
    launcher_owned = {"ARGUS_API_TOKEN", "ARGUS_MCP_TOKEN", "ARGUS_OUT_ROOT", "BLENDER_HOST"}
    unread = {s.name for s in SETTINGS} - read - launcher_owned - _READ_VIA_FSTRING
    assert not unread, (
        "core/config.py documents settings nothing reads any more:\n  "
        + "\n  ".join(sorted(unread))
    )


def test_registered_defaults_match_the_call_sites():
    """A registry that says 300 where the code says 120 is worse than silence."""
    mismatches: list[str] = []
    for rel, text in _source_files():
        for match in _CALL.finditer(text):
            name, raw_default = match.group(1), match.group(2)
            if _EXEMPT.match(name) or name not in BY_NAME or raw_default is None:
                continue
            actual = raw_default[1:-1]
            declared = BY_NAME[name].default
            if declared is not None and declared != actual:
                mismatches.append(
                    f"  {name} in {rel}: code default {actual!r}, registry says {declared!r}"
                )
    assert not mismatches, "core/config.py defaults disagree with the code:\n" + "\n".join(mismatches)


def test_describe_runs_and_redacts_secrets(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "AIzaSy-super-secret-value")
    text = describe()
    assert "AIzaSy-super-secret-value" not in text, "describe() must never print a key"
    assert "<set, 25 chars>" in text
    assert "BLENDER_PATH" in text


def test_env_example_can_be_regenerated(tmp_path):
    out = tmp_path / ".env.example"
    text = write_env_example(str(out))
    assert out.exists()
    for name in ("GOOGLE_API_KEY", "BLENDER_PATH", "ARGUS_VISUAL_TARGET"):
        assert f"\n{name}=" in text
