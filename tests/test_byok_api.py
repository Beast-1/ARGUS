"""The /api/generate boundary for bring-your-own-key requests.

Proves the wire payload becomes the domain object the pipeline reads, that a
provider the caller didn't supply stays off, and that the operator's own keys are
never substituted in. Hermetic: FastAPI's TestClient with run_manager.start
stubbed, so no pipeline, no Blender, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import service.api as api_mod  # noqa: E402
from service.auth import API_TOKEN  # noqa: E402

client = TestClient(api_mod.app)
AUTH = {"Authorization": f"Bearer {API_TOKEN}"}

GOOGLE = "AIzaSyD" + "K" * 30
GROQ = "gsk_" + "M" * 44


@pytest.fixture
def captured(monkeypatch):
    """Swap out the real run so nothing executes; keep what it was handed."""
    seen = {}

    def fake_start(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(api_mod.run_manager, "start", fake_start)
    return seen


def _post(body, **kw):
    return client.post("/api/generate", json=body, headers=AUTH, **kw)


def test_supplied_keys_reach_the_run_as_a_domain_object(captured):
    res = _post({
        "prompt": "a cast iron fire hydrant",
        "provider_keys": {"google": [GOOGLE], "groq": [GROQ]},
    })
    assert res.status_code == 202

    keys = captured["provider_keys"]
    assert keys is not None
    assert keys.google == (GOOGLE,)
    assert keys.groq == (GROQ,)
    # Sorted for a stable assertion; order isn't part of the contract.
    assert sorted(keys.providers()) == ["google", "groq"]


def test_a_request_without_keys_passes_none_so_the_env_is_used(captured):
    res = _post({"prompt": "a wooden crate"})
    assert res.status_code == 202
    # None, not an empty ProviderKeys — an empty object would gate every provider
    # off and produce a run that cannot do anything.
    assert captured["provider_keys"] is None


def test_an_all_blank_payload_is_treated_as_no_keys(captured):
    res = _post({
        "prompt": "a wooden crate",
        "provider_keys": {"google": [], "deepseek": "", "nvidia": None},
    })
    assert res.status_code == 202
    assert captured["provider_keys"] is None


def test_blank_and_whitespace_entries_are_dropped(captured):
    _post({
        "prompt": "a wooden crate",
        "provider_keys": {"google": [f"  {GOOGLE}  ", "", "   "]},
    })
    keys = captured["provider_keys"]
    assert keys.google == (GOOGLE,), "entries should be trimmed and blanks dropped"


def test_unsupplied_providers_stay_off(captured):
    _post({"prompt": "a wooden crate", "provider_keys": {"google": [GOOGLE]}})
    keys = captured["provider_keys"]
    assert keys.providers() == ["google"]
    # The point of the whole feature: nothing fills these in from the operator's
    # environment behind the caller's back.
    assert keys.groq == () and keys.openrouter == () and keys.deepseek == ""


def test_cloudflare_needs_both_halves_to_count(captured):
    _post({
        "prompt": "a wooden crate",
        "provider_keys": {"google": [GOOGLE], "cloudflare_account_id": "acct-1234"},
    })
    assert "cloudflare" not in captured["provider_keys"].providers()


def test_require_user_keys_rejects_a_request_with_none(captured, monkeypatch):
    # The setting for a publicly reachable deployment: without it a visitor who
    # supplies nothing silently spends the operator's quota.
    monkeypatch.setenv("ARGUS_REQUIRE_USER_KEYS", "1")
    res = _post({"prompt": "a wooden crate"})
    assert res.status_code == 400
    assert "your own API key" in res.json()["detail"]
    assert "provider_keys" not in captured, "the run must not have been started"


def test_require_user_keys_allows_a_request_that_brings_them(captured, monkeypatch):
    monkeypatch.setenv("ARGUS_REQUIRE_USER_KEYS", "1")
    res = _post({"prompt": "a wooden crate", "provider_keys": {"google": [GOOGLE]}})
    assert res.status_code == 202
    assert captured["provider_keys"].google == (GOOGLE,)


def test_keys_are_not_echoed_in_a_validation_error():
    # SecretStr's repr is "**********", so a 422 for some *other* field cannot
    # spill a key that was present in the same body.
    res = _post({
        "prompt": None,  # wrong type -> validation error mentioning the body
        "provider_keys": {"google": [GOOGLE]},
    })
    assert res.status_code == 422
    assert GOOGLE not in res.text
