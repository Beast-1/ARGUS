"""Per-run provider credentials, for runs that bring their own keys.

core/llm.py builds every provider pool once at import time from os.environ, which
is right for the CLI and for a desktop app run by the person who owns the keys. It
is wrong for the web app: a visitor supplies their own credentials and the run
must use exactly those, spending their quota rather than the operator's.

The threading problem is identical to the one core/cancel.py solves, so the shape
is identical:

  1. A run's keys must not be visible to another run   -> per-run state
  2. The keys arrive on the FastAPI request thread but are needed on RunManager's
     worker thread                                     -> a plain thread-local
     would put them out of reach of the thread that actually runs the pipeline

So a ContextVar records *which* key set this thread's pipeline belongs to, while
the ProviderKeys object is an ordinary value the accepting thread can hand to the
worker. When no set is current, core/llm.py falls back to its env-built pools and
behaves exactly as it always has.

Two deliberate non-features:

  * These never go into os.environ. eval/runner.py and core/mcp_server.py both do
    `dict(os.environ)` to build a child process's environment, so anything placed
    there would propagate into Blender and into eval subprocesses whose full
    stdout is written to git-tracked log files.
  * ProviderKeys does not render its values. Its repr reports counts, so an
    accidental f-string or a logged dataclass prints nothing useful.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

# The key set belonging to the pipeline running on this thread. None = "this run
# brought no keys", which falls back to core/llm.py's env-built pools.
_current_keys: ContextVar[Optional["ProviderKeys"]] = ContextVar(
    "argus_provider_keys", default=None
)


@dataclass(frozen=True)
class ProviderKeys:
    """Credentials supplied for a single run.

    Google, Groq and OpenRouter are tuples because core/llm.py pools those and
    rotates on rate limits; the rest take a single value, matching how llm.py
    reads them today.
    """

    google: tuple[str, ...] = ()
    groq: tuple[str, ...] = ()
    openrouter: tuple[str, ...] = ()
    deepseek: str = ""
    huggingface: str = ""
    nvidia: str = ""
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""

    def __repr__(self) -> str:
        """Counts, never values — see the module docstring."""
        parts = [
            f"google={len(self.google)}",
            f"groq={len(self.groq)}",
            f"openrouter={len(self.openrouter)}",
        ]
        for name in (
            "deepseek",
            "huggingface",
            "nvidia",
            "cloudflare_account_id",
            "cloudflare_api_token",
        ):
            if getattr(self, name):
                parts.append(f"{name}=set")
        return f"ProviderKeys({', '.join(parts)})"

    __str__ = __repr__

    def secret_values(self) -> list[str]:
        """Every literal value, for registering with core.secrets.

        Cloudflare's account id is not a credential and is excluded: it is short
        and unremarkable, and registering it would blanket-replace an ordinary
        hex string across every log line.
        """
        values = [*self.google, *self.groq, *self.openrouter]
        values += [self.deepseek, self.huggingface, self.nvidia, self.cloudflare_api_token]
        return [v for v in values if v]

    def any_present(self) -> bool:
        return bool(self.secret_values() or self.cloudflare_account_id)

    def providers(self) -> list[str]:
        """Names of the providers this run can actually use — what the UI shows
        and what the gating in core/llm.py enforces."""
        present = []
        if self.google:
            present.append("google")
        if self.groq:
            present.append("groq")
        if self.openrouter:
            present.append("openrouter")
        if self.deepseek:
            present.append("deepseek")
        if self.huggingface:
            present.append("huggingface")
        if self.nvidia:
            present.append("nvidia")
        if self.cloudflare_account_id and self.cloudflare_api_token:
            present.append("cloudflare")
        return present


def new_keys(keys: ProviderKeys) -> ProviderKeys:
    """Make `keys` current on this thread and return them.

    Called on the thread that accepts the run, mirroring cancel.new_scope(), so
    the returned object can be handed to the worker thread that will run it.
    """
    _current_keys.set(keys)
    return keys


def adopt_keys(keys: Optional[ProviderKeys]) -> None:
    """Make an already-created key set current on *this* thread.

    A ContextVar set on the request thread is invisible to the worker thread, so
    without this the pipeline would fall back to the operator's env keys and
    silently spend their quota — the exact outcome this feature exists to stop.
    """
    _current_keys.set(keys)


def current_keys() -> Optional[ProviderKeys]:
    """The key set this thread's pipeline should use, or None to use the env."""
    return _current_keys.get()


def clear_keys() -> None:
    _current_keys.set(None)
