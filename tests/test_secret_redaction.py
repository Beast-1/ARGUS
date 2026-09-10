"""Redaction has to hold for keys users supply, not just the operator's own.

Every assertion here checks that the literal secret is ABSENT rather than that
"***REDACTED***" is present. A pattern that silently stops matching would still
satisfy the second form (the surrounding text is unchanged, so some other part of
the line could contain the marker) but fails the first, which is the property
that actually matters.

None of these strings is a real credential; they are shaped like real ones so the
patterns are exercised.
"""
from __future__ import annotations

import json
import logging
import pytest

from core.secrets import (
    RedactingFormatter,
    clear_run_secrets,
    install_redaction,
    redact,
    register_run_secrets,
)

# Shaped like the real thing, deliberately not real.
FAKE = {
    "google": "AIzaSyD" + "F" * 30,
    "openrouter": "sk-or-v1-" + "a1b2c3d4" * 8,
    "groq": "gsk_" + "Z" * 48,
    "huggingface": "hf_" + "Q" * 34,
    "nvidia": "nvapi-" + "k" * 40,
    "deepseek": "sk-" + "9" * 40,
}


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_run_secrets()
    yield
    clear_run_secrets()


@pytest.mark.parametrize("provider,secret", sorted(FAKE.items()))
def test_every_provider_key_shape_is_redacted(provider, secret):
    leaked = f"POST failed for {provider}: 401 Unauthorized (key {secret})"
    assert secret not in redact(leaked), f"{provider} key survived redaction"


def test_url_query_form_is_redacted():
    # The exact shape that leaked 1,026 keys into logs/argus.log.
    url = "https://generativelanguage.googleapis.com/v1beta/models:x?key=" + "A" * 39
    assert "A" * 39 not in redact(url)


@pytest.mark.parametrize(
    "line",
    [
        "headers={'x-goog-api-key': 'no-recognisable-prefix-here-12345'}",
        'Authorization: Bearer no-recognisable-prefix-here-12345',
    ],
)
def test_header_forms_are_redacted_without_a_known_prefix(line):
    assert "no-recognisable-prefix-here-12345" not in redact(line)


def test_registered_exact_value_is_redacted_with_no_matching_pattern():
    # A Cloudflare API token has no distinctive prefix, so shape patterns can
    # never catch it. This is precisely why exact registration exists.
    token = "8f3b9c1d2e4a6b7c8d9e0f1a2b3c4d5e6f7a8b9c"
    assert token in redact(f"cf error {token}")  # not yet registered
    register_run_secrets([token])
    assert token not in redact(f"cf error {token}")


def test_short_values_are_not_registered():
    # Registering a 3-character string would blanket-replace it across every log
    # line. The length floor keeps the registry from mangling ordinary output.
    register_run_secrets(["abc"])
    assert redact("abc is a common substring") == "abc is a common substring"


def test_non_strings_pass_through_untouched():
    # Log args must keep their type or %-formatting breaks.
    assert redact(42) == 42
    assert redact(None) is None


def test_openrouter_key_is_not_half_redacted_by_the_generic_sk_rule():
    # sk-or-v1- must be matched before the bare sk- pattern, or the leading
    # fragment survives.
    out = redact(FAKE["openrouter"])
    assert "sk-or-v1-" not in out and FAKE["openrouter"] not in out


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(self.format(record))


def _capturing_logger(name):
    handler = _Capture()
    handler.setFormatter(logging.Formatter("%(message)s"))
    log = logging.getLogger(name)
    log.handlers = [handler]
    log.propagate = False
    log.setLevel(logging.DEBUG)
    install_redaction(log)
    return log, handler


def test_key_in_a_log_arg_does_not_reach_the_handler():
    log, cap = _capturing_logger("test.redaction.args")
    log.warning("request failed: %s", f"401 for {FAKE['google']}")
    assert FAKE["google"] not in "\n".join(cap.lines)


def test_key_inside_a_traceback_does_not_reach_the_handler():
    # The case a logging.Filter cannot cover: the traceback is rendered from
    # exc_info at format time, after every filter has run.
    log, cap = _capturing_logger("test.redaction.traceback")
    try:
        raise RuntimeError(f"provider rejected {FAKE['groq']}")
    except RuntimeError:
        log.exception("generation failed")
    assert FAKE["groq"] not in "\n".join(cap.lines)


def test_dict_args_survive_formatting():
    # The previous filter turned a mapping arg into a tuple of its keys, which
    # breaks %(name)s formatting outright.
    log, cap = _capturing_logger("test.redaction.dictargs")
    log.warning("%(what)s failed with %(key)s", {"what": "gemini", "key": FAKE["google"]})
    out = "\n".join(cap.lines)
    assert "gemini failed with" in out
    assert FAKE["google"] not in out


def test_install_redaction_is_idempotent():
    log, cap = _capturing_logger("test.redaction.idempotent")
    install_redaction(log)
    install_redaction(log)
    handler = log.handlers[0]
    assert isinstance(handler.formatter, RedactingFormatter)
    assert not isinstance(handler.formatter._inner, RedactingFormatter), "double-wrapped"
    assert sum(1 for f in handler.filters if f.__class__.__name__ == "SecretRedactingFilter") == 1


# ---------------------------------------------------------------------------
# The service-layer exits: the SSE firehose and the persisted run state.
# These are integration checks — a user's key travelling the real path a run
# would take, asserted absent at the far end.
# ---------------------------------------------------------------------------

def test_a_key_printed_by_the_pipeline_does_not_reach_the_sse_stream():
    from service.pipeline_events import PipelineEventWriter

    secret = FAKE["google"]
    register_run_secrets([secret])

    events = []
    writer = PipelineEventWriter(lambda kind, payload: events.append((kind, payload)))
    # Exactly how a failure surfaces today: main.py print()s the provider error,
    # stdout is redirected into this writer, and every line is forwarded to the
    # browser as a log_line.
    writer.write(f"[GEMINI] request failed for {secret}\n")
    writer.flush()

    streamed = " ".join(str(p) for _, p in events)
    assert streamed, "writer produced no events — test would pass vacuously"
    assert secret not in streamed


def test_run_state_file_never_carries_provider_keys(tmp_path, monkeypatch):
    import service.run_manager as rm
    from core.provider_keys import ProviderKeys

    secret = FAKE["groq"]
    manager = rm.RunManager()
    manager._provider_keys = ProviderKeys(groq=(secret,))
    manager.prompt = "a cast iron fire hydrant"

    # snapshot() is what _persist_state writes to out/.run_state.json. The keys
    # must not be reachable from it, which is why they live on a private field
    # snapshot() does not read.
    assert secret not in json.dumps(manager.snapshot())
