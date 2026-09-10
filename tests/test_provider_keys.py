"""Per-run provider credentials.

The properties worth pinning are the ones whose failure modes are silent: a run
quietly spending the operator's quota, a key surviving into a repr, or a pool
being rebuilt so often that rate-limit state never sticks.

Hermetic — no network, no Blender.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.llm as llm  # noqa: E402
from core.provider_keys import (  # noqa: E402
    ProviderKeys,
    adopt_keys,
    clear_keys,
    current_keys,
    new_keys,
)

GOOGLE = "AIzaSyD" + "G" * 30
GROQ = "gsk_" + "R" * 48


def _keys(**kw) -> ProviderKeys:
    return ProviderKeys(**kw)


def teardown_function():
    clear_keys()


# --- isolation -------------------------------------------------------------

def test_threads_do_not_see_each_others_keys():
    a = _keys(google=("key-a-aaaaaaaaaaaa",))
    b = _keys(google=("key-b-bbbbbbbbbbbb",))
    seen: dict[str, tuple] = {}
    barrier = threading.Barrier(2)

    def worker(name, keys):
        adopt_keys(keys)
        barrier.wait()  # both set before either reads
        seen[name] = current_keys().google

    threads = [
        threading.Thread(target=worker, args=("a", a)),
        threading.Thread(target=worker, args=("b", b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen["a"] == a.google
    assert seen["b"] == b.google


def test_a_thread_that_adopts_nothing_falls_back_to_the_env_pool():
    # This is the CLI and desktop path: no per-run keys, so the module-level
    # pools built from .env are used exactly as before.
    result = {}

    def worker():
        result["keys"] = current_keys()
        result["pool_is_env"] = llm._google_pool() is llm._GOOGLE_POOL

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result["keys"] is None
    assert result["pool_is_env"] is True


def test_new_keys_returns_the_object_so_another_thread_can_adopt_it():
    # Mirrors cancel.new_scope(): created on the accepting thread, handed to the
    # worker. Without adopt_keys on the worker it would fall back to the env.
    keys = new_keys(_keys(google=(GOOGLE,)))
    observed = {}

    def worker():
        observed["before"] = current_keys()
        adopt_keys(keys)
        observed["after"] = current_keys()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert observed["before"] is None
    assert observed["after"] is keys


# --- no key material escapes ----------------------------------------------

def test_repr_reports_counts_not_values():
    keys = _keys(google=(GOOGLE, GOOGLE), deepseek="sk-" + "9" * 40)
    text = f"{keys!r} {keys}"
    assert GOOGLE not in text
    assert "sk-" + "9" * 40 not in text
    assert "google=2" in text
    assert "deepseek=set" in text


def test_secret_values_collects_credentials_but_not_the_cloudflare_account_id():
    keys = _keys(
        google=(GOOGLE,),
        cloudflare_account_id="0123456789abcdef",
        cloudflare_api_token="tok-" + "z" * 30,
    )
    values = keys.secret_values()
    assert GOOGLE in values
    assert "tok-" + "z" * 30 in values
    # An account id is not a credential, and it is short and unremarkable enough
    # that registering it would blanket-replace ordinary text in every log line.
    assert "0123456789abcdef" not in values


# --- gating ----------------------------------------------------------------

def test_only_supplied_providers_resolve_to_a_usable_pool():
    adopt_keys(_keys(google=(GOOGLE,)))
    assert llm._google_pool().available_keys(), "supplied provider should be usable"
    # Everything the user did not supply is unavailable, so the existing fallback
    # cascade skips it. Crucially it must NOT fall back to the operator's env pool.
    assert llm._groq_pool().available_keys() == []
    assert llm._openrouter_pool().available_keys() == []
    assert llm._groq_pool() is not llm._GROQ_POOL
    assert llm._openrouter_pool() is not llm._OPENROUTER_POOL


def test_unsupplied_single_value_credentials_do_not_fall_back_to_the_env(monkeypatch):
    monkeypatch.setenv("HF_API_KEY", "hf_" + "operator" * 4)
    # No keys adopted: the operator's env value is used, as today.
    clear_keys()
    assert llm._provider_secret("huggingface", "HF_API_KEY").startswith("hf_")
    # A run that brought its own keys but no HF key gets nothing, rather than
    # silently spending the operator's HuggingFace quota.
    adopt_keys(_keys(google=(GOOGLE,)))
    assert llm._provider_secret("huggingface", "HF_API_KEY") == ""


def test_providers_lists_only_what_was_supplied():
    keys = _keys(google=(GOOGLE,), groq=(GROQ,), cloudflare_account_id="acct")
    # Cloudflare needs both halves to be usable, so a lone account id doesn't count.
    assert keys.providers() == ["google", "groq"]


# --- pool state ------------------------------------------------------------

def test_the_same_pool_object_is_reused_within_a_run():
    # A KeyPool carries cooldown and exhaustion state. Rebuilding it per call
    # would discard that the moment a key was marked, and the run would keep
    # retrying a key it already knows is dead.
    adopt_keys(_keys(google=(GOOGLE,)))
    first = llm._google_pool()
    assert llm._google_pool() is first

    state = first.next_available()
    state.mark_exhausted()
    assert llm._google_pool().available_keys() == [], "exhaustion did not persist"


def test_switching_key_sets_rebuilds_the_pool():
    adopt_keys(_keys(google=(GOOGLE,)))
    first = llm._google_pool()
    adopt_keys(_keys(google=("AIzaSyD" + "H" * 30,)))
    assert llm._google_pool() is not first
