"""core/llm.py's pure, network-free helpers — the validation gate that stands
between LLM output and an `exec()` call, the JSON/code extraction it depends on,
the regex-based auto-repair, and the secret-redacting log filter. None of these
need Blender, a network, or an API key, so there was no excuse for them having
zero coverage. Hermetic.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.llm as llm_mod  # noqa: E402
from core.llm import (  # noqa: E402
    _extract_json,
    _gemini_headers,
    _GEMINI_BASE,
    _mark_if_key_invalid,
    _next_github_token,
    _redact_secrets,
    apply_known_repair,
    fingerprint_error,
    reset_repair_state,
    sanitize_generated_code,
    should_abort_repair_loop,
    validate_generated_script,
)


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_in_markdown():
    raw = 'here is the plan:\n```json\n{"a": 1, "b": [1, 2]}\n```\nhope that helps'
    assert _extract_json(raw) == {"a": 1, "b": [1, 2]}


def test_extract_json_finds_braces_in_surrounding_prose():
    raw = 'Sure! {"a": 1} is the object you asked for.'
    assert _extract_json(raw) == {"a": 1}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("not json at all") is None


def test_extract_json_returns_none_on_truncated_object():
    # A common real failure mode: the response got cut off at a token limit.
    assert _extract_json('{"a": 1, "b": [1, 2,') is None


# ---------------------------------------------------------------------------
# sanitize_generated_code
# ---------------------------------------------------------------------------

def test_sanitize_strips_markdown_fences():
    code = "```python\nimport bpy\nx = 1\n```"
    out = sanitize_generated_code(code)
    assert "```" not in out
    assert "import bpy" in out


def test_sanitize_strips_export_selected_kwarg():
    code = "bpy.ops.export_scene.gltf(filepath='x', export_selected=True)\n"
    out = sanitize_generated_code(code)
    assert "export_selected" not in out


def test_sanitize_empty_input_returns_empty():
    assert sanitize_generated_code("") == ""
    assert sanitize_generated_code("   \n  ") == ""


# ---------------------------------------------------------------------------
# validate_generated_script
# ---------------------------------------------------------------------------

VALID_SCRIPT = """
import bpy
import bmesh
assert bpy.app.version >= (4, 0, 0)

def box_obj(name, loc):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = loc
    return obj

box_obj("part_a", (0, 0, 0))
box_obj("part_b", (1, 0, 0))
box_obj("part_c", (2, 0, 0))
"""


def test_valid_script_passes():
    ok, reason = validate_generated_script(VALID_SCRIPT)
    assert ok, reason


def test_rejects_syntax_error():
    ok, reason = validate_generated_script("import bpy\ndef broken(:\n    pass")
    assert not ok
    assert "SyntaxError" in reason or "invalid syntax" in reason.lower()


def test_rejects_missing_bpy_import():
    ok, reason = validate_generated_script(VALID_SCRIPT.replace("import bpy\n", ""))
    assert not ok
    assert "import bpy" in reason


def test_rejects_missing_version_assert():
    ok, reason = validate_generated_script(
        VALID_SCRIPT.replace("assert bpy.app.version >= (4, 0, 0)\n", "")
    )
    assert not ok
    assert "version" in reason.lower()


def test_rejects_too_few_objects():
    minimal = """
import bpy
import bmesh
assert bpy.app.version >= (4, 0, 0)
obj = bpy.data.objects.new("only_one", None)
"""
    ok, reason = validate_generated_script(minimal)
    assert not ok
    assert "too minimal" in reason.lower()


def test_rejects_literal_subprocess_call():
    bad = VALID_SCRIPT.replace(
        'box_obj("part_c", (2, 0, 0))',
        'box_obj("part_c", (2, 0, 0))\nimport subprocess; subprocess.run(["whoami"])',
    )
    ok, reason = validate_generated_script(bad)
    assert not ok
    assert "subprocess" in reason.lower()


def test_catches_aliased_subprocess_import():
    """`import subprocess as sp` never contains the literal substring
    "subprocess." the OLD blocklist checked for, so it used to slip through
    silently. core.script_safety.check_script_safety walks the AST instead, so
    the import is caught regardless of the alias."""
    bad = VALID_SCRIPT.replace(
        'box_obj("part_c", (2, 0, 0))',
        'box_obj("part_c", (2, 0, 0))\nimport subprocess as sp\nsp.run(["whoami"])',
    )
    ok, reason = validate_generated_script(bad)
    assert not ok
    assert "subprocess" in reason.lower()


def test_catches_spaced_dunder_import_call():
    """`__import__ ("os")` (a space before the paren) never contains the literal
    substring "__import__(" the OLD blocklist checked for. AST-based means
    formatting can't hide it."""
    bad = VALID_SCRIPT.replace(
        'box_obj("part_c", (2, 0, 0))',
        'box_obj("part_c", (2, 0, 0))\n__import__ ("os").system("whoami")',
    )
    ok, reason = validate_generated_script(bad)
    assert not ok


def test_catches_os_system_even_though_os_import_itself_is_allowed():
    """os.path/os.makedirs etc. are legitimate and common in export scripts —
    only specific dangerous os.* calls (system, popen, remove, ...) are denied,
    not the module itself."""
    bad = VALID_SCRIPT.replace(
        'box_obj("part_c", (2, 0, 0))',
        'box_obj("part_c", (2, 0, 0))\nimport os\nos.system("whoami")',
    )
    ok, reason = validate_generated_script(bad)
    assert not ok
    assert "os.system" in reason.lower()

    benign = VALID_SCRIPT.replace(
        'box_obj("part_c", (2, 0, 0))',
        'box_obj("part_c", (2, 0, 0))\nimport os\n_p = os.path.join("a", "b")',
    )
    ok2, _ = validate_generated_script(benign)
    assert ok2


def test_catches_eval_and_socket():
    for snippet in ['eval("1+1")', 'import socket as s\ns.socket()']:
        bad = VALID_SCRIPT.replace(
            'box_obj("part_c", (2, 0, 0))',
            f'box_obj("part_c", (2, 0, 0))\n{snippet}',
        )
        ok, _ = validate_generated_script(bad)
        assert not ok, f"should have rejected: {snippet}"


# ---------------------------------------------------------------------------
# apply_known_repair
# ---------------------------------------------------------------------------

def test_apply_known_repair_fixes_diameter_kwarg():
    code = VALID_SCRIPT.replace(
        "bmesh.ops.create_cube(bm, size=1.0)",
        "bmesh.ops.create_cone(bm, diameter1=1.0, diameter2=1.0, depth=2.0)",
    )
    repaired = apply_known_repair(code, {"error_msg": "unexpected keyword diameter1"})
    assert "radius1=" in repaired
    assert "diameter1" not in repaired


def test_apply_known_repair_returns_empty_when_nothing_matches():
    assert apply_known_repair(VALID_SCRIPT, {"error_msg": "totally unrelated error"}) == ""


def test_apply_known_repair_returns_empty_if_repair_still_invalid():
    # The regex fires (removes apply_modifiers=) but the result is still too
    # minimal to pass validate_generated_script, so the repair is discarded.
    tiny = "import bpy\nimport bmesh\nassert bpy.app.version >= (4,0,0)\nx = 1\n" \
           "bpy.ops.object.modifier_apply(apply_modifiers=True)\n"
    assert apply_known_repair(tiny, {"error_msg": "apply_modifiers is not a valid keyword"}) == ""


# ---------------------------------------------------------------------------
# fingerprint_error / should_abort_repair_loop
# ---------------------------------------------------------------------------

def test_fingerprint_normalizes_line_numbers_and_addresses():
    a = fingerprint_error("TypeError at line 42, object 0xdeadbeef")
    b = fingerprint_error("TypeError at line 917, object 0xcafef00d")
    assert a == b


def test_should_abort_repair_loop_flags_repeat_failures():
    reset_repair_state()
    try:
        assert should_abort_repair_loop("KeyError: 'Metallic' at line 10") is False
        assert should_abort_repair_loop("KeyError: 'Metallic' at line 88") is True
    finally:
        reset_repair_state()


def test_should_abort_repair_loop_allows_distinct_errors():
    reset_repair_state()
    try:
        assert should_abort_repair_loop("KeyError: 'Metallic'") is False
        assert should_abort_repair_loop("TypeError: bad argument") is False
    finally:
        reset_repair_state()


# ---------------------------------------------------------------------------
# _redact_secrets / the logging filter that uses it
# ---------------------------------------------------------------------------

def test_redact_secrets_strips_query_param_key():
    leaked = ("ConnectionError: HTTPSConnectionPool(host='generativelanguage.googleapis.com'): "
              "Max retries exceeded with url: /v1beta/models/gemini-2.5-flash:generateContent"
              "?key=AIzaSyC-fake-key-value-000000000")
    redacted = _redact_secrets(leaked)
    assert "AIzaSyC-fake-key-value-000000000" not in redacted
    assert "REDACTED" in redacted


def test_redact_secrets_strips_bare_google_key_without_query_param():
    redacted = _redact_secrets("token dump: AIzaSyDabcdefghijklmnopqrstuvwxyz0123")
    assert "AIzaSyD" not in redacted


def test_redact_secrets_leaves_ordinary_messages_untouched():
    msg = "provider 'groq' raised: rate limited, retrying"
    assert _redact_secrets(msg) == msg


# ---------------------------------------------------------------------------
# Gemini auth — moved off the URL (real incident: 1,026 leaked keys) into a header
# ---------------------------------------------------------------------------

def test_gemini_base_url_has_no_key_placeholder():
    """The key used to travel as ?key=... in the URL, which is what ended up
    embedded in logged exception strings. Confirming the template has no {key}
    slot left is a cheap guardrail against it creeping back in."""
    assert "{key}" not in _GEMINI_BASE
    assert "key=" not in _GEMINI_BASE.format(model="gemini-2.5-flash")


def test_gemini_headers_carries_the_key_instead():
    headers = _gemini_headers("test-key-value")
    assert headers["x-goog-api-key"] == "test-key-value"


def test_mark_if_key_invalid_only_fires_on_the_specific_google_error():
    """400 API_KEY_INVALID means the key is permanently dead (confirmed live:
    one of six configured keys started returning exactly this, most likely
    rotated after the key-leak finding) — must be marked exhausted so it isn't
    retried forever. An ordinary 400 (malformed request) or a 429 (rate limit,
    already handled separately) must NOT trigger this."""
    class _Resp:
        def __init__(self, status_code, text):
            self.status_code = status_code
            self.text = text

    ks = llm_mod._KeyState(key="k", name="TEST_KEY")
    _mark_if_key_invalid(_Resp(400, '{"error":{"status":"INVALID_ARGUMENT",'
                                     '"details":[{"reason":"API_KEY_INVALID"}]}}'), ks)
    assert ks.exhausted is True

    ks2 = llm_mod._KeyState(key="k2", name="TEST_KEY_2")
    _mark_if_key_invalid(_Resp(400, '{"error":{"message":"bad request"}}'), ks2)
    assert ks2.exhausted is False
    _mark_if_key_invalid(_Resp(429, "rate limited"), ks2)
    assert ks2.exhausted is False


# ---------------------------------------------------------------------------
# GitHub Models — retired 2026-07-30, must never be called regardless of config
# ---------------------------------------------------------------------------

def test_github_tokens_are_never_populated_even_if_configured():
    """GitHub Models (both the old azure endpoint and its models.github.ai
    replacement) is permanently retired. This repo's own .env has GITHUB_TOKEN(_2..4)
    set, which — before this fix — meant every generation call wasted a full
    connection attempt against a dead hostname; discovered when it flooded a real
    ablation sweep with NameResolutionError on every single cell."""
    assert llm_mod._GITHUB_TOKENS == []
    assert _next_github_token() == ""


def test_llm_logger_filter_redacts_warning_args(caplog):
    import core.llm as llm_mod

    exc = Exception(
        "Max retries exceeded with url: /v1beta/models/x:generateContent"
        "?key=AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
    )
    with caplog.at_level(logging.WARNING, logger="ARGUS.llm"):
        llm_mod.logger.warning("[GEMINI_TEXT] %s failed: %s", "gemini-2.5-flash", exc)

    assert "AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKE" not in caplog.text
    assert "REDACTED" in caplog.text
