"""Secret redaction for logs, streams and on-disk artifacts.

This exists because it already failed once: Gemini used to carry its key in the
URL query string, failed-request exceptions were logged verbatim, and 1,026 real
API keys ended up in logs/argus.log. The key moved to a header and the log was
sanitised, but the redaction that was added alongside covered one logger and two
Google-specific patterns, which is not enough once users supply their own keys.

Two layers, in order of strength:

1. **Exact values.** When a run is handed keys we know the literal strings, so we
   can redact exactly those wherever they appear, in any shape, in any provider's
   error envelope. This is the primary defence and the only one that covers a
   credential with no distinctive prefix (a Cloudflare API token, say).

2. **Shape patterns.** The backstop for keys nobody registered — the operator's
   own env-provided keys, or anything a provider echoes back at us.

Over-redaction is safe here and under-redaction is not, so the bar for adding a
pattern is low and there is deliberately no fast-path that skips the scan.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any, Iterable, Optional

REDACTED = "***REDACTED***"

# Registered exact values. A plain module-level set rather than a ContextVar: a
# log record can be formatted on any thread (and by a handler on another), so the
# redaction list has to be visible process-wide. Redacting one run's key while
# another run is logging is harmless; missing it is not.
_exact_lock = threading.Lock()
_exact_secrets: set[str] = set()

# Below this length a "secret" is more likely to be a fragment that appears in
# ordinary log text, and blanket-replacing it would mangle every line.
_MIN_SECRET_LEN = 12

# Ordered: the more specific prefix has to win before the generic one. sk-or-v1-
# must precede the bare sk- rule or OpenRouter keys get half-redacted.
_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?<=[?&]key=)[^&\s'\"]+", re.IGNORECASE),          # legacy URL form
    re.compile(r"AIzaSy[A-Za-z0-9_\-]{10,}"),                        # Google / Gemini
    re.compile(r"sk-or-v1-[A-Za-z0-9]{16,}"),                        # OpenRouter
    re.compile(r"gsk_[A-Za-z0-9]{16,}"),                             # Groq
    re.compile(r"hf_[A-Za-z0-9]{16,}"),                              # HuggingFace
    re.compile(r"nvapi-[A-Za-z0-9_\-]{16,}"),                        # NVIDIA NIM
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                              # DeepSeek / OpenAI shape
]

# Header forms, where the value itself may have no recognisable prefix. The
# capture keeps the header name so the line still reads sensibly once redacted.
_HEADER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)(x-goog-api-key['\"]?\s*[:=]\s*['\"]?)([^\s'\",}\]]+)"),
    re.compile(r"(?i)(authorization['\"]?\s*[:=]\s*['\"]?bearer\s+)([^\s'\",}\]]+)"),
]


def register_run_secrets(values: Iterable[Optional[str]]) -> None:
    """Add literal secret values to the redaction set for the life of a run."""
    with _exact_lock:
        for v in values:
            if v and len(v) >= _MIN_SECRET_LEN:
                _exact_secrets.add(v)


def clear_run_secrets(values: Optional[Iterable[Optional[str]]] = None) -> None:
    """Drop registered values. Passing None clears everything."""
    with _exact_lock:
        if values is None:
            _exact_secrets.clear()
            return
        for v in values:
            _exact_secrets.discard(v or "")


def redact(value: Any) -> Any:
    """Strip anything key-shaped from `value`.

    Non-strings are returned untouched rather than stringified, so a log record's
    numeric or structured args survive `%`-formatting intact.
    """
    if not isinstance(value, str) or not value:
        return value

    text = value
    # Exact values first: they are certain, and doing them first means a key that
    # also matches a shape pattern is replaced once rather than twice.
    with _exact_lock:
        known = tuple(_exact_secrets)
    for secret in known:
        if secret in text:
            text = text.replace(secret, REDACTED)

    for pattern in _PATTERNS:
        text = pattern.sub(REDACTED, text)
    for pattern in _HEADER_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}{REDACTED}", text)
    return text


def redact_arg(value: Any) -> Any:
    """Redact a single logging arg, which is very often not a string.

    `logger.warning("failed: %s", exc)` passes an exception *object*; its text is
    produced later by %-formatting, so a string-only redact would let a key in
    the exception message straight through. Numbers and None are returned as-is
    so %d/%f formatting still works, and any other object is only replaced when
    redacting its text actually changed something — otherwise the original is
    kept and its own __str__ still runs at format time.
    """
    if isinstance(value, str):
        return redact(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    cleaned = redact(text)
    return cleaned if cleaned != text else value


class SecretRedactingFilter(logging.Filter):
    """Redacts a record's message and args in place.

    Kept alongside the formatter below because some code reads `record.msg`
    directly; the formatter is what actually guarantees coverage.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            if isinstance(record.args, dict):
                # logging keeps a single mapping arg as a dict for %(name)s-style
                # formatting. Iterating it as a tuple yields the *keys* and breaks
                # the format — the previous implementation had exactly that bug.
                record.args = {k: redact_arg(v) for k, v in record.args.items()}
            else:
                record.args = tuple(redact_arg(a) for a in record.args)
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        return True


class RedactingFormatter(logging.Formatter):
    """Wraps another formatter and redacts its finished output.

    This is the real choke point. A filter can only reach `msg` and `args`, but a
    traceback is rendered from `exc_info` at format time — so `logger.exception`
    on an error whose text contains a key would slip straight past a filter.
    Redacting the formatted string covers message, args, traceback and stack info
    in one place.
    """

    def __init__(self, inner: Optional[logging.Formatter] = None) -> None:
        super().__init__()
        self._inner = inner or logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._inner.format(record))


def install_redaction(logger: Optional[logging.Logger] = None) -> None:
    """Attach redaction to a logger's handlers — the root logger by default.

    Handler-level, not logger-level, and this is the whole point: a
    `logging.Filter` added to one logger never sees records emitted through its
    siblings, which is why the previous `ARGUS.llm`-only filter left every other
    logger writing unfiltered into logs/argus.log. Every record that reaches a
    handler passes through its formatter, so wrapping the formatter covers them
    all. Idempotent, so repeated calls are safe.
    """
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if not isinstance(handler.formatter, RedactingFormatter):
            handler.setFormatter(RedactingFormatter(handler.formatter))
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(SecretRedactingFilter())
