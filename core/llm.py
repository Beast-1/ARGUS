from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Optional

import requests
from dotenv import load_dotenv
from core.structure_library import (
    apply_structure_blueprint,
    blueprint_count,
    retrieve_exemplar_script,
)
# The keyword-matched procedural fallbacks used to live in this file (~2,300
# lines of static Blender geometry inside the LLM module). They are the last
# stop before the generic-cube fallback, and _procedural_blueprint_fallback is
# the only entry point the rest of this module needs — the individual scripts
# it dispatches to are resolved inside handcrafted_assets.py, not here.
from core.handcrafted_assets import _procedural_blueprint_fallback
from core.texture_sources import build_material_catalog_block
from core.spec import (
    legacy_scene_graph as _spec_legacy_scene_graph,
    spec_summary as _spec_summary,
    validate_spec as _validate_build_spec,
)
from core.script_safety import (
    check_script_safety,
    sanitize_generated_code,
    validate_python_syntax,
)

load_dotenv()

logger = logging.getLogger("ARGUS.llm")

# Gemini calls embed the API key in the request URL (_GEMINI_BASE below). When a
# request fails, str(exc) from `requests` typically includes that URL verbatim, and
# several call sites below log it via logger.warning("... %s", exc). Redacting at
# the logger level (rather than fixing each call site) means every current AND
# future log line through this logger is covered, not just the ones caught by a
# manual audit — this exact leak already put 1,000+ real keys into logs/argus.log.
# The implementation moved to core/secrets.py so service/ and eval/ can import it
# without pulling in this module, and so it could be widened past the two
# Google-only patterns that were all it covered. It was also attached to this
# logger alone, which meant every other logger reached logs/argus.log unfiltered;
# install_redaction() now wraps the root handlers in main.py instead. Re-exported
# under the old names because the existing tests and call sites use them.
from core.secrets import SecretRedactingFilter as _SecretRedactingFilter  # noqa: E402
from core.secrets import redact as _redact_secrets  # noqa: E402

logger.addFilter(_SecretRedactingFilter())

DEFAULT_MAX_RETRIES = int(os.getenv("ARGUS_LLM_MAX_RETRIES", "3"))
DEFAULT_BACKOFF_BASE = float(os.getenv("ARGUS_LLM_BACKOFF", "2.0"))
REQUEST_TIMEOUT_GEMINI = int(os.getenv("ARGUS_GEMINI_TIMEOUT", "90"))
REQUEST_TIMEOUT_OR = int(os.getenv("ARGUS_OR_TIMEOUT", "180"))


@dataclass
class _KeyState:
    key: str
    name: str
    request_count: int = 0
    exhausted: bool = False
    cooldown_until: float = 0.0

    @property
    def available(self) -> bool:
        return not self.exhausted and time.monotonic() >= self.cooldown_until

    def mark_rate_limited(self, backoff_sec: float) -> None:
        self.cooldown_until = time.monotonic() + backoff_sec
        logger.warning("[KEY_POOL] %s cooling %.0fs", self.name, backoff_sec)

    def mark_exhausted(self) -> None:
        self.exhausted = True
        logger.error("[KEY_POOL] %s exhausted", self.name)

    def record_request(self) -> None:
        self.request_count += 1


class KeyPool:
    def __init__(self, keys: list[tuple[str, str]]) -> None:
        self._pool: list[_KeyState] = [
            _KeyState(key=value, name=name)
            for value, name in keys
            if value
        ]
        self._index = 0

    def available_keys(self) -> list[_KeyState]:
        return [s for s in self._pool if s.available]

    def next_available(self) -> Optional[_KeyState]:
        if not self._pool:
            return None

        available = self.available_keys()
        if available:
            state = available[self._index % len(available)]
            self._index = (self._index + 1) % len(available)
            return state

        cooling = [s for s in self._pool if not s.exhausted]
        if not cooling:
            return None

        wait_until = min(s.cooldown_until for s in cooling)
        wait_sec = max(0.0, wait_until - time.monotonic())
        if wait_sec > 0:
            logger.warning("[KEY_POOL] Waiting %.1fs for cooldown", wait_sec)
            time.sleep(wait_sec + 0.25)

        available = self.available_keys()
        return available[0] if available else None

    def report(self) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "name": s.name,
                "requests": s.request_count,
                "exhausted": s.exhausted,
                "cooling_for_sec": round(max(0.0, s.cooldown_until - now), 1),
            }
            for s in self._pool
        ]


def _build_google_pool() -> KeyPool:
    _google_keys = [
        (os.getenv(f"GOOGLE_API_KEY{'_' + str(i) if i > 1 else ''}", ""),
         f"GOOGLE_API_KEY{'_' + str(i) if i > 1 else ''}")
        for i in range(1, 11)
    ]
    pool = KeyPool(_google_keys)
    if not pool.available_keys():
        logger.warning("[KEY_POOL] Google keys missing; planner may fallback.")
    return pool


def _build_openrouter_pool() -> KeyPool:
    pool = KeyPool(
        [
            (os.getenv("OPENROUTER_API_KEY", ""), "OPENROUTER_API_KEY"),
            (os.getenv("OPENROUTER_API_KEY_2", ""), "OPENROUTER_API_KEY_2"),
        ]
    )
    if not pool.available_keys():
        logger.warning("[KEY_POOL] OpenRouter keys missing; generation may fallback.")
    return pool


def _build_groq_pool() -> KeyPool:
    pool = KeyPool(
        [
            (os.getenv("GROQ_API_KEY", ""), "GROQ_API_KEY"),
            (os.getenv("GROQ_API_KEY_2", ""), "GROQ_API_KEY_2"),
        ]
    )
    if not pool.available_keys():
        logger.warning("[KEY_POOL] Groq keys missing; Groq fallback unavailable.")
    return pool


_GOOGLE_POOL = _build_google_pool()
_OPENROUTER_POOL = _build_openrouter_pool()
_GROQ_POOL = _build_groq_pool()


# ---------------------------------------------------------------------------
# Per-run credentials (core/provider_keys.py)
#
# The pools above are built once at import from os.environ, which is correct for
# the CLI and for a desktop app run by whoever owns the keys. The web app has to
# support a visitor supplying their own, so every read below goes through an
# accessor that prefers this run's key set and falls back to the env pools.
#
# The gating rule is the whole feature: when a run brings its own keys, a
# provider it did NOT supply resolves to an empty pool. Every call site already
# treats an empty pool as "provider unavailable, try the next one", so the
# existing fallback cascade skips it with no new branching — and the operator's
# env keys are never substituted in, which is what stops their quota being spent
# on someone else's generation.
# ---------------------------------------------------------------------------

_EMPTY_POOL = KeyPool([])

# A KeyPool carries rate-limit state — mark_rate_limited/mark_exhausted and the
# cooldown timers only mean anything if the same object is reused for the whole
# run. That is why the env pools above are module singletons, and why a run's
# pools are built once and cached here rather than rebuilt on each access: a
# fresh pool per call would forget an exhausted key the instant it was marked,
# and the run would hammer a dead key until the retry budget ran out.
#
# Keyed on the ProviderKeys object itself (frozen, so identity is stable for the
# life of a run) and held in a ContextVar, so concurrent batch workers with
# different keys cannot see each other's pools.
_run_pools: ContextVar[Optional[tuple]] = ContextVar("argus_run_pools", default=None)


def _run_keys():
    from core.provider_keys import current_keys

    return current_keys()


def _pools_for(keys) -> dict:
    cached = _run_pools.get()
    if cached is not None and cached[0] is keys:
        return cached[1]

    def build(values, label):
        if not values:
            return _EMPTY_POOL
        return KeyPool([(v, f"user_{label}_{i + 1}") for i, v in enumerate(values)])

    pools = {
        "google": build(keys.google, "google"),
        "openrouter": build(keys.openrouter, "openrouter"),
        "groq": build(keys.groq, "groq"),
    }
    _run_pools.set((keys, pools))
    return pools


def _google_pool() -> KeyPool:
    keys = _run_keys()
    return _GOOGLE_POOL if keys is None else _pools_for(keys)["google"]


def _openrouter_pool() -> KeyPool:
    keys = _run_keys()
    return _OPENROUTER_POOL if keys is None else _pools_for(keys)["openrouter"]


def _groq_pool() -> KeyPool:
    keys = _run_keys()
    return _GROQ_POOL if keys is None else _pools_for(keys)["groq"]


def _provider_secret(attr: str, env_var: str) -> str:
    """A single-value credential: this run's, or the operator's from the env.

    Note the asymmetry with the pools — it is deliberate. When a run brings keys,
    an unsupplied provider returns "" rather than the env value.
    """
    keys = _run_keys()
    if keys is None:
        return os.getenv(env_var, "")
    return getattr(keys, attr, "") or ""


def _deepseek_key() -> str:
    """Reads the module global rather than the env so a test that patches
    _DEEPSEEK_KEY still works, matching how the rest of this module is tested."""
    keys = _run_keys()
    if keys is None:
        return _DEEPSEEK_KEY
    return keys.deepseek

_github_tokens_configured = [
    t for t in (
        os.getenv(f"GITHUB_TOKEN{'_' + str(i) if i > 1 else ''}", "")
        for i in range(1, 11)
    ) if t and not t.startswith("paste-")
]
if _github_tokens_configured:
    # GitHub Models was fully retired 2026-07-30 (playground, catalog, both the
    # old models.inference.ai.azure.com endpoint AND its models.github.ai
    # replacement — everything). There is no working endpoint to point this at.
    # Deliberately never populating _GITHUB_TOKENS rather than updating the URL:
    # every call site already treats "no token" as "provider unavailable, fall
    # through to the next one" (a normal, fully-supported configuration), so this
    # needed no other changes anywhere in the fallback cascade. Discovered via a
    # real ablation sweep where every cell logged NameResolutionError against the
    # dead hostname and wasted a full connection-timeout per attempt.
    logger.warning(
        "[KEY_POOL] %d GITHUB_TOKEN(s) configured but GitHub Models is retired "
        "(2026-07-30) — ignoring them; falling back to other providers.",
        len(_github_tokens_configured),
    )
_GITHUB_TOKENS: list[str] = []
_GITHUB_TOKEN_IDX: int = 0

def _next_github_token() -> str:
    """DEAD PATH — always returns "" (see _GITHUB_TOKENS above)."""
    global _GITHUB_TOKEN_IDX
    if not _GITHUB_TOKENS:
        return ""
    tok = _GITHUB_TOKENS[_GITHUB_TOKEN_IDX % len(_GITHUB_TOKENS)]
    _GITHUB_TOKEN_IDX += 1
    return tok

_DEEPSEEK_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
_DEEPSEEK_URL: str = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODELS = [
    "deepseek-coder",
    "deepseek-chat",
]

_USE_PRO = os.getenv("ARGUS_USE_GEMINI_PRO", "0") == "1"
GEMINI_PLANNER_MODELS = (
    (["gemini-2.5-pro"] if _USE_PRO else [])
    + ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
)

OPENROUTER_GENERATION_MODELS = [
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
]

OPENROUTER_REPAIR_MODELS = [
    "openai/gpt-oss-120b:free",
]

OPENROUTER_CRITIC_MODELS = [
    "openai/gpt-oss-120b:free",
]

OPENROUTER_NAMING_MODELS = [
    "deepseek/deepseek-chat-v3-0324",
    "openai/gpt-oss-20b:free",
]

GROQ_GENERATION_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]

GROQ_REPAIR_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
]

_GEMINI_BASE = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/{model}:generateContent"
)
_OR_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def _mark_if_key_invalid(res: "requests.Response", key_state: "_KeyState") -> None:
    """A permanently-revoked/invalid key returns 400 API_KEY_INVALID, not 429 —
    left unhandled, that key gets retried (and fails) on every single future
    call forever instead of once. Found live: one of six configured Google keys
    started returning this (most likely rotated after the earlier key-leak
    finding), and confirming the header-auth migration below was correct required
    noticing this was key-specific, not a header-format regression."""
    if res.status_code == 400 and "API_KEY_INVALID" in res.text:
        key_state.mark_exhausted()


def _gemini_headers(key: str) -> dict:
    """The key used to travel in the URL query string (?key=...), which meant
    every failed-request exception's string form — logged verbatim in several
    places — carried it in plaintext (the actual, already-manifested incident:
    1,026 leaked keys found in a real log file). The redaction filter on the
    logger stays as defense in depth, but the key no longer belongs in a URL at
    all — the Gemini API accepts it as a header instead."""
    return {"x-goog-api-key": key, "Content-Type": "application/json"}

GEMINI_VISION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]

GEMINI_CODE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
_NAME_INVALID_CHARS = re.compile(r"[^a-z0-9_]")
_RECENT_REPAIR_FAILURES: set[str] = set()
_BMESH_OP_RE = re.compile(r"\bbmesh\.ops\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_BPY_OP_RE = re.compile(
    r"\bbpy\.ops\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)

_BLOCKED_BMESH_OPS = {
    "create_cylinder",
    "create_torus",
    "create_capsule",
    "create_plane",
    "extrude_edge_region",
    "extrude_loop_fill",
}

_BLOCKED_BPY_OPS = {
    ("object", "parent_set"),
    ("wm", "collada_export"),
    ("mesh", "uv_texture_add"),
    ("mesh", "primitive_cube_add"),
    ("mesh", "primitive_uv_sphere_add"),
    ("mesh", "primitive_cylinder_add"),
    ("mesh", "primitive_cone_add"),
    ("mesh", "primitive_plane_add"),
    ("mesh", "primitive_torus_add"),
    ("mesh", "primitive_circle_add"),
    ("mesh", "primitive_grid_add"),
    ("mesh", "primitive_ico_sphere_add"),
}


@dataclass
class LLMCallRecord:
    role: str
    model: str
    api_key_name: str
    prompt_chars: int
    response_chars: int
    success: bool
    latency_sec: float
    attempt: int
    error: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _log_call(record: LLMCallRecord) -> None:
    logger.debug(
        "[LLM_CALL] role=%s model=%s key=%s success=%s latency=%.2fs attempt=%d%s",
        record.role,
        record.model,
        record.api_key_name,
        record.success,
        record.latency_sec,
        record.attempt,
        f" err={record.error}" if record.error else "",
    )


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _groq_chat(
    model: str,
    messages: list[dict],
    role: str,
    temperature: float = 0.15,
    max_tokens: int = 8000,
) -> Optional[str]:
    """Groq OpenAI-compatible chat endpoint — used as fallback after OpenRouter."""
    key_state = _groq_pool().next_available()
    if not key_state:
        return None

    headers = {
        "Authorization": f"Bearer {key_state.key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    t0 = time.monotonic()
    try:
        res = requests.post(_GROQ_ENDPOINT, headers=headers, json=payload,
                            timeout=REQUEST_TIMEOUT_OR)
        key_state.record_request()
        if res.status_code == 429:
            key_state.mark_rate_limited(DEFAULT_BACKOFF_BASE * 2)
            return None
        if res.status_code >= 500:
            return None
        res.raise_for_status()
        text = (
            res.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
        _log_call(LLMCallRecord(
            role=role, model=f"groq/{model}", api_key_name=key_state.name,
            prompt_chars=sum(len(m.get("content","")) for m in messages),
            response_chars=len(text), success=bool(text),
            latency_sec=time.monotonic() - t0, attempt=1,
        ))
        return _strip_thinking(text) if text else None
    except Exception as exc:
        _log_call(LLMCallRecord(
            role=role, model=f"groq/{model}", api_key_name=key_state.name,
            prompt_chars=0, response_chars=0, success=False,
            latency_sec=time.monotonic() - t0, attempt=1, error=str(exc),
        ))
        return None


def _extract_json(raw: str) -> Optional[dict]:
    clean = _strip_thinking(raw).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    m = _JSON_FENCE_RE.search(clean)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(clean[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _extract_code(raw: str) -> str:
    raw = _strip_thinking(raw)
    m = _CODE_FENCE_RE.search(raw)
    code = m.group(1) if m else raw
    code = code.replace("\ufeff", "").replace("\t", "    ")
    code = textwrap.dedent(code).lstrip("\n")
    return code.strip()


def validate_generated_script(code: str) -> tuple[bool, str]:
    code = sanitize_generated_code(code)
    valid, reason = validate_python_syntax(code)
    if not valid:
        return False, reason

    required_imports = ["import bpy", "import bmesh"]
    for imp in required_imports:
        if imp not in code:
            return False, f"Missing required import: {imp}"

    if "bpy.app.version" not in code:
        return False, "Missing Blender version contract assert"

    # The real security gate: an AST-based denylist (core/script_safety.py),
    # not the two literal substrings ("subprocess.", "__import__(") this used
    # to check — both trivially bypassed by `import subprocess as sp` or
    # `__import__ ("os")`, and neither one ever looked for os.system, eval,
    # socket, or shutil at all.
    safe, safety_reason = check_script_safety(code)
    if not safe:
        return False, safety_reason

    forbidden_patterns = [
        "apply_modifiers=",
        "mesh_smooth_type=",
        "extrude_edge_region",
        "bpy.context.scene.objects.link",
        "mesh.update(calc_edges",
        "bpy.ops.mesh.uv_texture_add",
        "show_double_sided",
        "bpy.ops.wm.collada_export",
        "bpy.ops.object.parent_set",
        "bpy.context.window_manager.windows[0]",
        'inputs["Transmission"]',
        'inputs["Transmission Weight"]',
        "inputs['Transmission']",
        "inputs['Transmission Weight']",
        "bpy.app.version >= (3, 0, 0)",
        "requires Blender >= 3.0.0",
    ]
    for pattern in forbidden_patterns:
        if pattern in code:
            return False, f"Forbidden pattern detected: {pattern}"

    used_bmesh_ops = set(_BMESH_OP_RE.findall(code))
    bad_bmesh = sorted(used_bmesh_ops & _BLOCKED_BMESH_OPS)
    if bad_bmesh:
        return False, (
            "Removed/nonexistent bmesh operator(s): "
            + ", ".join(f"bmesh.ops.{name}" for name in bad_bmesh)
        )

    used_bpy_ops = set(_BPY_OP_RE.findall(code))
    bad_bpy = sorted(used_bpy_ops & _BLOCKED_BPY_OPS)
    if bad_bpy:
        return False, (
            "Removed/unsafe bpy operator(s): "
            + ", ".join(f"bpy.ops.{module}.{name}" for module, name in bad_bpy)
        )

    obj_creations = len(re.findall(r"bpy\.data\.objects\.new\s*\(", code))
    helper_calls = len(re.findall(
        r"\b(create_cylinder|create_cone|cyl|cube_part|cyl_part|sphere_part|"
        r"box_obj|cone_part|mesh_obj|create_bolt|create_mesh|ellipsoid|"
        r"plank|bracket|nail|rope_loop|vcyl|obj_from_bm|ring|pentagon_cap|bolt_ring|"
        r"argus_box|argus_cylinder|argus_sphere|argus_wheel|argus_bolt|"
        r"argus_panel|argus_lathe|argus_tube|argus_ring|argus_leaf_card)\s*\(",
        code,
    ))
    total_objects = max(obj_creations, helper_calls)
    if total_objects < 3:
        return False, (
            f"Script too minimal: only {total_objects} apparent mesh object(s) "
            "(need at least 3 — use create_cone/create_cylinder helpers or "
            "direct bpy.data.objects.new() calls for each named part)"
        )

    return True, ""


def apply_known_repair(code: str, traceback_obj: dict) -> str:
    message = " ".join(
        str(traceback_obj.get(key, ""))
        for key in ("error_msg", "message", "raw")
    )

    repaired = code

    if "create_cylinder" in code or "create_cylinder" in message:
        repaired = re.sub(
            r"bmesh\.ops\.create_cylinder\(",
            "bmesh.ops.create_cone(",
            repaired,
        )
        repaired = re.sub(
            r"bmesh\.ops\.create_cone\(([^)]*?)radius\s*=\s*([^,\)\n]+)",
            r"bmesh.ops.create_cone(\1radius1=\2, radius2=\2",
            repaired,
            flags=re.DOTALL,
        )

    if "diameter1" in code or "diameter2" in code or "diameter1" in message or "diameter2" in message:
        repaired = re.sub(r"\bdiameter1\s*=\s*([^,\)\n]+)", r"radius1=\1", repaired)
        repaired = re.sub(r"\bdiameter2\s*=\s*([^,\)\n]+)", r"radius2=\1", repaired)

    if "create_cone" in repaired and ("size=" in message or "size= is invalid" in message):
        repaired = re.sub(
            r"(bmesh\.ops\.create_cone\([^)]*?)\bsize\s*=\s*([^,\)\n]+)",
            r"\1depth=\2",
            repaired,
            flags=re.DOTALL,
        )

    if "use_auto_smooth" in message or "auto_smooth_angle" in message or \
       "use_auto_smooth" in code or "auto_smooth_angle" in code:
        repaired = re.sub(r"[ \t]*mesh\.use_auto_smooth\s*=\s*[^\n]*\n?", "", repaired)
        repaired = re.sub(r"[ \t]*mesh\.auto_smooth_angle\s*=\s*[^\n]*\n?", "", repaired)
        repaired = re.sub(
            r"(mesh\.update\(\))\n",
            r"\1\n"
            r"    bpy.context.view_layer.objects.active = obj\n"
            r"    obj.select_set(True)\n"
            r"    try:\n"
            r"        bpy.ops.object.shade_smooth_by_angle(angle=__import__('math').radians(30))\n"
            r"    except Exception:\n"
            r"        pass\n",
            repaired,
            count=1,
        )

    if "'Material' object has no attribute 'inputs'" in message or (
        "has no attribute 'inputs'" in message and "Material" in message
    ):
        repaired = re.sub(
            r"\b(\w+)\.inputs\[(['\"][A-Za-z ]+['\"])\]\s*=",
            lambda m: (
                f"{m.group(1)}.node_tree.nodes['Principled BSDF'].inputs[{m.group(2)}] ="
                if not m.group(1).startswith("node") and not m.group(1).startswith("principled")
                else m.group(0)
            ),
            repaired,
        )
        repaired = re.sub(
            r"\b(\w*mat\w*)\s*=\s*bpy\.data\.materials\.new\([^)]+\)\s*\n"
            r"(.*?)"
            r"\1\.inputs\[",
            lambda m: m.group(0).replace(
                m.group(1) + ".inputs[",
                m.group(1) + ".node_tree.nodes['Principled BSDF'].inputs["
            ),
            repaired,
            flags=re.DOTALL,
        )

    if "no attribute 'metallic'" in message or "no attribute 'roughness'" in message \
            or "no attribute 'base_color'" in message:
        for attr, socket in [
            ("metallic",   "Metallic"),
            ("roughness",  "Roughness"),
            ("base_color", "Base Color"),
            ("emission",   "Emission Color"),
        ]:
            repaired = re.sub(
                rf"\b([\w]+)\.{attr}\s*=\s*([^\n]+)",
                rf'\1.inputs["{socket}"].default_value = \2',
                repaired,
            )

    if "apply_modifiers" in message or "apply_modifiers" in code:
        repaired = re.sub(r",?\s*apply_modifiers\s*=\s*(True|False)", "", repaired)
    if "mesh_smooth_type" in message or "mesh_smooth_type" in code:
        repaired = re.sub(r",?\s*mesh_smooth_type\s*=\s*['\"][^'\"]*['\"]", "", repaired)

    if "min() arg is an empty sequence" in message or "max() arg is an empty sequence" in message:
        repaired = re.sub(
            r"(min|max)\(point\.([xyz]) for point in world_points\)",
            r"\1((point.\2 for point in world_points), default=0.0)",
            repaired,
        )
        repaired = re.sub(
            r"for corner in obj\.bound_box",
            "for corner in (obj.bound_box if obj.data and obj.data.vertices else [])",
            repaired,
        )

    if "bpy_prop_collection.__contains__" in message or \
       "expected a string or a tuple of strings" in message or \
       "not in obj.data.materials" in code or \
       " in obj.data.materials" in code:
        repaired = re.sub(
            r"\bif\s+(\w+)\s+not\s+in\s+(\w+(?:\.\w+)*\.materials)\s*:",
            r"if \1.name not in [m.name for m in \2]:",
            repaired,
        )
        repaired = re.sub(
            r"\bif\s+(\w+)\s+in\s+(\w+(?:\.\w+)*\.materials)\s*:",
            r"if \1.name in [m.name for m in \2]:",
            repaired,
        )

    _bmesh_replacements = {
        r"bmesh\.ops\.create_torus\s*\(": "bmesh.ops.create_cone(",
        r"bmesh\.ops\.create_cylinder\s*\(": "bmesh.ops.create_cone(",
        r"bmesh\.ops\.create_capsule\s*\(": "bmesh.ops.create_uvsphere(",
        r"bmesh\.ops\.create_plane\s*\(": "bmesh.ops.create_grid(",
        r"bmesh\.ops\.extrude_edge_region\s*\(": "bmesh.ops.extrude_face_region(",
    }
    for bad_pattern, good_op in _bmesh_replacements.items():
        if re.search(bad_pattern, repaired):
            repaired = re.sub(bad_pattern, good_op, repaired)

    if repaired == code:
        return ""

    valid, reason = validate_generated_script(repaired)
    if valid:
        logger.info("[KNOWN_REPAIR] applied regex fix(es) successfully")
        return repaired
    logger.warning("[KNOWN_REPAIR] regex-repaired script rejected: %s", reason)
    return ""


def fingerprint_error(error_text: str) -> str:
    error_text = error_text.lower()
    error_text = re.sub(r"line\s+\d+", "line", error_text)
    error_text = re.sub(r"0x[a-f0-9]+", "0x", error_text)
    return error_text.strip()


def should_abort_repair_loop(error_text: str) -> bool:
    fp = fingerprint_error(error_text)
    if fp in _RECENT_REPAIR_FAILURES:
        return True
    _RECENT_REPAIR_FAILURES.add(fp)
    return False


def reset_repair_state() -> None:
    """Clear per-asset repair state.  Call once at the start of each pipeline run
    so errors from a previous asset don't block the repair loop of the next one."""
    _RECENT_REPAIR_FAILURES.clear()


def _gemini_generate(model: str, prompt: str, role: str,
                     thinking_budget: int = 0,
                     max_output_tokens: int = 8192,
                     temperature: float = 0.2,
                     json_mode: bool = False) -> Optional[str]:
    key_state = _google_pool().next_available()
    if not key_state:
        return None

    url = _GEMINI_BASE.format(model=model)
    gen_cfg = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    if "2.5" in model:
        gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }

    t0 = time.monotonic()
    try:
        res = requests.post(url, json=payload, headers=_gemini_headers(key_state.key),
                            timeout=REQUEST_TIMEOUT_GEMINI)
        key_state.record_request()
        _mark_if_key_invalid(res, key_state)
        if res.status_code == 429:
            key_state.mark_rate_limited(DEFAULT_BACKOFF_BASE * 2)
            return None
        if res.status_code >= 500:
            return None
        res.raise_for_status()
        data = res.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        _log_call(
            LLMCallRecord(
                role=role,
                model=model,
                api_key_name=key_state.name,
                prompt_chars=len(prompt),
                response_chars=len(text),
                success=bool(text),
                latency_sec=time.monotonic() - t0,
                attempt=1,
            )
        )
        return text.strip() if text else None
    except Exception as exc:  # noqa: BLE001
        _log_call(
            LLMCallRecord(
                role=role,
                model=model,
                api_key_name=key_state.name,
                prompt_chars=len(prompt),
                response_chars=0,
                success=False,
                latency_sec=time.monotonic() - t0,
                attempt=1,
                error=str(exc),
            )
        )
        return None


def generate_reference_image(part_data: dict, user_prompt: str) -> Optional[bytes]:
    """
    Generate a technical reference image for the asset.
    Requests an orthographic-style multi-view sheet rather than a photorealistic
    render — cleaner silhouettes and better proportions help the vision model
    extract geometry for the Blender script.

    Chain: Gemini image gen → Pollinations.ai FLUX
    Returns image bytes on success, None on failure.
    """
    category  = part_data.get("category", "object")
    style     = part_data.get("style", "")
    material  = part_data.get("material", "")
    parts_str = ", ".join((part_data.get("parts") or [])[:10])
    geo_mode  = part_data.get("geometry_mode", "hard_surface")

    if geo_mode == "organic":
        style_directive = (
            "clay render, smooth organic forms, clear silhouette, "
            "matte grey material, soft rim light, no specular highlights"
        )
    else:
        style_directive = (
            "technical reference sheet, orthographic perspective, "
            "flat even lighting, crisp hard edges, matte surface, "
            "clean silhouette, slight three-quarter angle, "
            "no reflections, no shadows on background"
        )

    if os.getenv("ARGUS_MULTIVIEW_REF", "1") != "0":
        layout_directive = (
            "Orthographic multi-view reference sheet of the SAME single object, "
            "three angles arranged left-to-right: front view, side view, and a "
            "three-quarter view. Identical proportions, scale and details across "
            "all three views, evenly spaced, fully visible, not overlapping."
        )
    else:
        layout_directive = (
            "Single object centred in frame, full body visible, "
            "slight three-quarter angle, correct proportions."
        )

    image_prompt = (
        f"3D asset reference: {user_prompt}. "
        f"Type: {style} {category}. "
        f"Key parts visible: {parts_str}. "
        f"Material: {material}. "
        f"{style_directive}. {layout_directive} "
        "Pure white background. No text labels, no dimension lines, no people, no UI."
    )

    # Provider fallback chain — first one that returns bytes wins. NVIDIA NIM
    # (FLUX.1-dev) and Cloudflare Workers AI (FLUX.1-schnell) lead because they
    # have the strongest prompt adherence on free tiers; Gemini/Pollinations
    # remain as last resorts. Order is overridable via ARGUS_IMAGE_PROVIDERS.
    providers = {
        "nim": _nvidia_nim_generate_image,
        "cloudflare": _cloudflare_generate_image,
        "gemini": _gemini_generate_image,
        "pollinations": _pollinations_generate_image,
    }
    default_order = "nim,cloudflare,pollinations,gemini"
    if os.getenv("ARGUS_GEMINI_IMAGES", "0") == "1":
        default_order = "gemini,nim,cloudflare,pollinations"
    order = [p.strip() for p in
             os.getenv("ARGUS_IMAGE_PROVIDERS", default_order).split(",") if p.strip()]
    for name in order:
        fn = providers.get(name)
        if fn is None:
            continue
        try:
            img = fn(image_prompt)
        except Exception as exc:  # noqa: BLE001 — try the next provider
            logger.warning("[REF_IMG] provider '%s' raised: %s", name, exc)
            img = None
        if img:
            logger.info("[REF_IMG] reference image from '%s' (%.1fKB)",
                        name, len(img) / 1024)
            return img
    logger.warning("[REF_IMG] all image providers failed — text-only fallback")
    return None


_NVIDIA_NIM_URL = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-dev"


def _nvidia_nim_generate_image(prompt: str) -> Optional[bytes]:
    """NVIDIA NIM FLUX.1-dev — strongest free prompt adherence. Needs a free
    NVIDIA_API_KEY (build.nvidia.com). Returns image bytes or None."""
    import base64

    key = _provider_secret("nvidia", "NVIDIA_API_KEY") or _provider_secret("nvidia", "NVIDIA_NIM_API_KEY")
    if not key:
        logger.debug("[NIM] NVIDIA_API_KEY not set — skipping")
        return None
    payload = {
        "prompt": prompt[:9000],
        "width": 1024, "height": 1024,
        "cfg_scale": 3.5, "steps": 40, "seed": 42, "mode": "base",
    }
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    t0 = time.monotonic()
    try:
        res = requests.post(_NVIDIA_NIM_URL, json=payload, headers=headers, timeout=120)
        if res.status_code >= 400:
            logger.warning("[NIM] error %s: %s", res.status_code, res.text[:200])
            return None
        data = res.json()
        raw = None
        if isinstance(data.get("artifacts"), list) and data["artifacts"]:
            raw = data["artifacts"][0].get("base64") or data["artifacts"][0].get("b64_json")
        raw = raw or data.get("image") or data.get("b64_json")
        if isinstance(raw, str) and raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        if not raw:
            logger.warning("[NIM] response had no image field: %s", str(data)[:160])
            return None
        img = base64.b64decode(raw)
        logger.info("[NIM] FLUX.1-dev %.1fKB in %.1fs", len(img) / 1024, time.monotonic() - t0)
        return img if len(img) > 1024 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[NIM] failed: %s", exc)
        return None


_CLOUDFLARE_URL = ("https://api.cloudflare.com/client/v4/accounts/"
                   "{account}/ai/run/@cf/black-forest-labs/flux-1-schnell")


def _cloudflare_generate_image(prompt: str) -> Optional[bytes]:
    """Cloudflare Workers AI FLUX.1-schnell — ~2000 free images/day. Needs free
    CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN. Returns image bytes or None."""
    import base64

    account = _provider_secret("cloudflare_account_id", "CLOUDFLARE_ACCOUNT_ID")
    token = _provider_secret("cloudflare_api_token", "CLOUDFLARE_API_TOKEN")
    if not account or not token:
        logger.debug("[CF_AI] CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN not set — skipping")
        return None
    url = _CLOUDFLARE_URL.format(account=account)
    payload = {"prompt": prompt[:2000], "steps": 8}
    headers = {"Authorization": f"Bearer {token}"}
    t0 = time.monotonic()
    try:
        res = requests.post(url, json=payload, headers=headers, timeout=90)
        if res.status_code >= 400:
            logger.warning("[CF_AI] error %s: %s", res.status_code, res.text[:200])
            return None
        data = res.json()
        raw = (data.get("result") or {}).get("image")
        if not raw:
            logger.warning("[CF_AI] response had no result.image: %s", str(data)[:160])
            return None
        img = base64.b64decode(raw)
        logger.info("[CF_AI] FLUX-schnell %.1fKB in %.1fs", len(img) / 1024, time.monotonic() - t0)
        return img if len(img) > 1024 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[CF_AI] failed: %s", exc)
        return None


_GEMINI_IMAGE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-2.0-flash-exp-image-generation",
    "gemini-2.0-flash-preview-image-generation",
]

def _gemini_generate_image(prompt: str) -> Optional[bytes]:
    """
    Use Gemini's built-in image output (responseModalities=IMAGE).
    Tries every available Google key + every model until one succeeds,
    then falls back to Pollinations.ai if all keys are exhausted.
    """
    import base64

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    all_keys: list = [ks for ks in _google_pool()._pool if ks.available]

    if not all_keys:
        logger.warning("[GEMINI_IMG] No Google API key available — falling back to Pollinations.ai")
        return _pollinations_generate_image(prompt)

    for key_state in all_keys:
        for model_id in _GEMINI_IMAGE_MODELS:
            url = _GEMINI_BASE.format(model=model_id)
            try:
                res = requests.post(url, json=payload, headers=_gemini_headers(key_state.key),
                                    timeout=60)
                key_state.record_request()
                _mark_if_key_invalid(res, key_state)
                if res.status_code == 429:
                    key_state.mark_rate_limited(DEFAULT_BACKOFF_BASE * 2)
                    break
                if res.status_code in (404, 400):
                    logger.debug("[GEMINI_IMG] %s not available (%s)", model_id, res.status_code)
                    continue
                if res.status_code >= 400:
                    logger.warning("[GEMINI_IMG] error %s (%s): %s",
                                   res.status_code, model_id, res.text[:200])
                    continue
                parts = (
                    (res.json().get("candidates") or [{}])[0]
                    .get("content", {})
                    .get("parts", [])
                )
                for part in parts:
                    inline = part.get("inlineData") or part.get("inline_data") or {}
                    raw = inline.get("data") or inline.get("b64_json")
                    if raw:
                        img_bytes = base64.b64decode(raw)
                        logger.info("[GEMINI_IMG] Generated %.1fKB via %s",
                                    len(img_bytes) / 1024, model_id)
                        return img_bytes
                logger.debug("[GEMINI_IMG] %s returned OK but no image part", model_id)
            except Exception as exc:
                logger.warning("[GEMINI_IMG] %s failed: %s", model_id, exc)

    logger.warning("[GEMINI_IMG] All Gemini keys/models exhausted — falling back to HuggingFace FLUX")
    return _huggingface_generate_image(prompt)


def nim_edit_image(prompt: str, image_bytes: bytes) -> Optional[bytes]:
    """Structure-preserving repaint via NVIDIA NIM FLUX.1-dev canny mode: a
    canny edge map of the render conditions the diffusion so the repaint keeps
    the asset's silhouette and part boundaries while restyling the surface.
    Needs NVIDIA_API_KEY. Returns image bytes or None."""
    import base64
    import io

    key = _provider_secret("nvidia", "NVIDIA_API_KEY") or _provider_secret("nvidia", "NVIDIA_NIM_API_KEY")
    if not key:
        return None
    try:
        from PIL import Image, ImageFilter, ImageOps
        src = Image.open(io.BytesIO(image_bytes)).convert("L")
        edges = ImageOps.autocontrast(src.filter(ImageFilter.FIND_EDGES)).convert("RGB")
        buf = io.BytesIO()
        edges.save(buf, "PNG")
        ctrl = base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        logger.warning("[NIM_EDIT] edge map failed: %s", exc)
        return None

    payload = {
        "prompt": prompt[:9000],
        "mode": "canny",
        "image": "data:image/png;base64," + ctrl,
        "cfg_scale": 3.5, "steps": 40, "seed": 42,
        "width": 1024, "height": 1024,
    }
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    try:
        res = requests.post(_NVIDIA_NIM_URL, json=payload, headers=headers, timeout=120)
        if res.status_code >= 400:
            logger.warning("[NIM_EDIT] error %s: %s", res.status_code, res.text[:160])
            return None
        data = res.json()
        raw = None
        if isinstance(data.get("artifacts"), list) and data["artifacts"]:
            raw = data["artifacts"][0].get("base64") or data["artifacts"][0].get("b64_json")
        raw = raw or data.get("image")
        if isinstance(raw, str) and raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        if not raw:
            return None
        out = base64.b64decode(raw)
        logger.info("[NIM_EDIT] repainted %.1fKB (canny)", len(out) / 1024)
        return out if len(out) > 1024 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[NIM_EDIT] failed: %s", exc)
        return None


def cloudflare_edit_image(prompt: str, image_bytes: bytes,
                          strength: float = 0.6) -> Optional[bytes]:
    """Structure-preserving repaint via Cloudflare Workers AI SD-1.5 img2img:
    the source render is kept (strength controls how much is restyled) while
    the prompt drives the new surface. Takes the image INLINE (no asset upload,
    unlike NIM), returns the repainted PNG bytes directly. Needs free
    CLOUDFLARE_ACCOUNT_ID + CLOUDFLARE_API_TOKEN."""
    account = _provider_secret("cloudflare_account_id", "CLOUDFLARE_ACCOUNT_ID")
    token = _provider_secret("cloudflare_api_token", "CLOUDFLARE_API_TOKEN")
    if not account or not token:
        return None
    url = (f"https://api.cloudflare.com/client/v4/accounts/{account}"
           "/ai/run/@cf/runwayml/stable-diffusion-v1-5-img2img")
    # Blender renders views with an alpha channel; SD img2img wants flat RGB,
    # so normalise (drop alpha over white) before sending — otherwise the call
    # silently returns nothing and the painter no-ops.
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(image_bytes))
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        image_bytes = buf.getvalue()
    except Exception:
        pass
    payload = {
        "prompt": prompt[:1500],
        "image": list(image_bytes),
        "strength": max(0.1, min(strength, 0.9)),
        "num_steps": 20,
        "guidance": 7.5,
    }
    try:
        res = requests.post(url, headers={"Authorization": f"Bearer {token}"},
                            json=payload, timeout=110)
        ctype = res.headers.get("content-type", "")
        if res.status_code == 200 and ctype.startswith("image"):
            out = res.content
            logger.info("[CF_EDIT] repainted %.1fKB (img2img)", len(out) / 1024)
            return out if len(out) > 1024 else None
        # Some accounts return JSON base64 instead of binary.
        if res.status_code == 200 and "json" in ctype:
            import base64
            raw = (res.json().get("result") or {}).get("image")
            if raw:
                return base64.b64decode(raw)
        logger.warning("[CF_EDIT] error %s: %s", res.status_code, res.text[:160])
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("[CF_EDIT] failed: %s", exc)
        return None


def repaint_image(prompt: str, image_bytes: bytes) -> Optional[bytes]:
    """Img2img repaint via the first available provider. Cloudflare SD-1.5
    img2img leads (inline, reliable on the free tier); NIM canny and Gemini
    remain as fallbacks."""
    return (cloudflare_edit_image(prompt, image_bytes)
            or nim_edit_image(prompt, image_bytes)
            or gemini_edit_image(prompt, image_bytes))


def gemini_edit_image(prompt: str, image_bytes: bytes) -> Optional[bytes]:
    """Image-to-image edit via Gemini image models (input render + instruction
    → repainted image). Returns None when no model/key can do it."""
    import base64
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": _image_mime(image_bytes),
                                "data": base64.b64encode(image_bytes).decode()}},
            ]
        }],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }
    for key_state in [ks for ks in _google_pool()._pool if ks.available]:
        for model_id in _GEMINI_IMAGE_MODELS:
            url = _GEMINI_BASE.format(model=model_id)
            try:
                res = requests.post(url, json=payload, headers=_gemini_headers(key_state.key),
                                    timeout=90)
                key_state.record_request()
                _mark_if_key_invalid(res, key_state)
                if res.status_code == 429:
                    key_state.mark_rate_limited(DEFAULT_BACKOFF_BASE * 2)
                    break
                if res.status_code >= 400:
                    logger.debug("[GEMINI_EDIT] %s -> %s", model_id, res.status_code)
                    continue
                parts = ((res.json().get("candidates") or [{}])[0]
                         .get("content", {}).get("parts", []))
                for part in parts:
                    inline = part.get("inlineData") or part.get("inline_data") or {}
                    raw = inline.get("data") or inline.get("b64_json")
                    if raw:
                        out = base64.b64decode(raw)
                        logger.info("[GEMINI_EDIT] repainted %.1fKB via %s",
                                    len(out) / 1024, model_id)
                        return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("[GEMINI_EDIT] %s failed: %s", model_id, exc)
    logger.warning("[GEMINI_EDIT] no image-edit model available")
    return None


_HF_IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.1-dev",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

def _huggingface_generate_image(prompt: str) -> Optional[bytes]:
    """
    Generate a reference image via HuggingFace Inference API using FLUX.1.
    Much more reliable than Pollinations — uses your own HF API key so no
    anonymous rate limits.
    """
    hf_key = _provider_secret("huggingface", "HF_API_KEY")
    if not hf_key:
        logger.warning("[HF_IMG] No HF_API_KEY — falling back to Pollinations")
        return _pollinations_generate_image(prompt)

    headers = {"Authorization": f"Bearer {hf_key}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "width": 768, "height": 768,
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
        }
    }

    for model_id in _HF_IMAGE_MODELS:
        url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        t0 = time.monotonic()
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=90)
            if res.status_code == 503:
                logger.debug("[HF_IMG] %s loading (503), trying next", model_id)
                continue
            if res.status_code == 429:
                logger.warning("[HF_IMG] %s rate limited", model_id)
                continue
            if res.status_code != 200:
                logger.warning("[HF_IMG] %s HTTP %s", model_id, res.status_code)
                continue
            img_bytes = res.content
            if len(img_bytes) < 1024:
                logger.warning("[HF_IMG] %s response too small (%dB)", model_id, len(img_bytes))
                continue
            logger.info("[HF_IMG] Generated %.1fKB via %s in %.1fs",
                        len(img_bytes) / 1024, model_id, time.monotonic() - t0)
            return img_bytes
        except Exception as exc:
            logger.warning("[HF_IMG] %s failed: %s", model_id, exc)

    logger.warning("[HF_IMG] All HF models failed — falling back to Pollinations")
    return _pollinations_generate_image(prompt)


_POLLINATIONS_URL = "https://image.pollinations.ai/prompt/{prompt}"

def _pollinations_generate_image(prompt: str) -> Optional[bytes]:
    """Last-resort image gen — no key required, but rate-limited and unreliable."""
    import urllib.parse
    encoded = urllib.parse.quote(prompt, safe="")
    url = _POLLINATIONS_URL.format(prompt=encoded)
    params = {"width": 768, "height": 768, "nologo": "true", "model": "flux", "seed": 42}
    t0 = time.monotonic()
    try:
        res = requests.get(url, params=params, timeout=90, stream=True)
        if res.status_code != 200:
            return None
        img_bytes = res.content
        if len(img_bytes) < 1024:
            return None
        logger.info("[POLLINATIONS] Generated %.1fKB in %.2fs",
                    len(img_bytes) / 1024, time.monotonic() - t0)
        return img_bytes
    except Exception as exc:
        logger.warning("[POLLINATIONS] failed: %s", exc)
        return None


_GEMINI_CODE_WALL_TIMEOUT = int(os.getenv("ARGUS_GEMINI_CODE_TIMEOUT", "240"))


def _gemini_code_request(url: str, payload: dict, key: str) -> requests.Response:
    """Synchronous POST used inside a thread so we can enforce a wall-clock timeout."""
    return requests.post(url, json=payload, headers=_gemini_headers(key),
                         timeout=REQUEST_TIMEOUT_GEMINI)


def _gemini_generate_code(prompt: str, role: str = "code_gen") -> Optional[str]:
    """
    Generate a Blender Python script using Gemini text API.

    Uses a thread-level wall-clock timeout (_GEMINI_CODE_WALL_TIMEOUT seconds) so
    that long streaming responses don't block the pipeline indefinitely.
    """
    import concurrent.futures

    full_prompt = (
        "Return ONLY executable Python code for Blender 4.1+. No markdown fences, "
        "no explanations, no prose — just the code.\n\n"
        + prompt
    )
    _code_models = (["gemini-2.5-pro"] if _USE_PRO else []) + [m for m in GEMINI_CODE_MODELS if "pro" not in m]
    _attempt_idx = 0
    for model in _code_models:
        for key_state in [ks for ks in _google_pool()._pool if ks.available]:
            url = _GEMINI_BASE.format(model=model)
            gen_cfg = {
                "temperature": 0.15,
                "maxOutputTokens": 32000,
            }
            if "2.5" in model and _attempt_idx == 0:
                gen_cfg["thinkingConfig"] = {"thinkingBudget": 8000}
            _attempt_idx += 1
            payload = {
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": gen_cfg,
            }
            t0 = time.monotonic()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _exe:
                    _future = _exe.submit(_gemini_code_request, url, payload, key_state.key)
                    try:
                        res = _future.result(timeout=_GEMINI_CODE_WALL_TIMEOUT)
                    except concurrent.futures.TimeoutError:
                        logger.warning(
                            "[GEMINI_CODE] %s wall-clock timeout (%ds) — skipping",
                            model, _GEMINI_CODE_WALL_TIMEOUT,
                        )
                        continue
                key_state.record_request()
                _mark_if_key_invalid(res, key_state)
                if res.status_code == 429:
                    logger.debug("[GEMINI_CODE] %s 429 — trying next model", model)
                    break
                if res.status_code in (400, 404):
                    logger.debug("[GEMINI_CODE] %s unavailable (%s)", model, res.status_code)
                    break
                if res.status_code >= 500:
                    continue
                res.raise_for_status()
                text = (
                    res.json().get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                ) or ""
                _log_call(LLMCallRecord(
                    role=role, model=model, api_key_name=key_state.name,
                    prompt_chars=len(full_prompt), response_chars=len(text),
                    success=bool(text), latency_sec=time.monotonic() - t0, attempt=1,
                ))
                if text:
                    return _strip_thinking(text).strip()
            except Exception as exc:
                logger.warning("[GEMINI_CODE] %s failed: %s", model, exc)
    return None


def _image_mime(image_bytes: bytes) -> str:
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    if image_bytes[:4] == b"\x89PNG":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _gemini_generate_multimodal(
    model: str, prompt: str, image_bytes, role: str,
    temperature: float = 0.15, json_mode: bool = False,
) -> Optional[str]:
    """Gemini generateContent call with text + one or more inline images.
    image_bytes may be a single bytes blob or a list of them (in order)."""
    import base64
    key_state = _google_pool().next_available()
    if not key_state:
        return None

    url = _GEMINI_BASE.format(model=model)
    images = image_bytes if isinstance(image_bytes, (list, tuple)) else [image_bytes]
    parts = [{"text": prompt}]
    for img in images:
        if not img:
            continue
        parts.append({"inlineData": {
            "mimeType": _image_mime(img),
            "data": base64.b64encode(img).decode(),
        }})
    gen_cfg = {"temperature": temperature}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": gen_cfg,
    }

    t0 = time.monotonic()
    try:
        res = requests.post(url, json=payload, headers=_gemini_headers(key_state.key),
                            timeout=REQUEST_TIMEOUT_GEMINI)
        key_state.record_request()
        _mark_if_key_invalid(res, key_state)
        if res.status_code == 429:
            key_state.mark_rate_limited(DEFAULT_BACKOFF_BASE * 2)
            return None
        if res.status_code >= 500:
            return None
        res.raise_for_status()
        data = res.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )
        _log_call(LLMCallRecord(
            role=role, model=model, api_key_name=key_state.name,
            prompt_chars=len(prompt), response_chars=len(text),
            success=bool(text), latency_sec=time.monotonic()-t0, attempt=1,
        ))
        return text.strip() if text else None
    except Exception as exc:
        _log_call(LLMCallRecord(
            role=role, model=model, api_key_name=key_state.name,
            prompt_chars=len(prompt), response_chars=0,
            success=False, latency_sec=time.monotonic()-t0, attempt=1,
            error=str(exc),
        ))
        return None


def _gemini_multimodal_code(
    prompt: str,
    images: list[bytes],
    role: str,
    thinking_budget: int = 10000,
    max_output_tokens: int = 50000,
) -> Optional[str]:
    """Gemini code generation that can SEE one or more images (the broken render
    and/or the reference) while writing/fixing the script. Uses thinking mode +
    a large token budget. Tries 2.5-pro first, then 2.5-flash."""
    import base64
    def _mime(b):
        if b[:2] == b"\xff\xd8": return "image/jpeg"
        if b[:4] == b"\x89PNG": return "image/png"
        if b[:4] == b"RIFF" and b[8:12] == b"WEBP": return "image/webp"
        return "image/png"

    parts = [{"text": prompt}]
    for img in images:
        if img:
            parts.append({"inlineData": {"mimeType": _mime(img),
                                         "data": base64.b64encode(img).decode()}})

    _models = (["gemini-2.5-pro"] if _USE_PRO else []) + ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    for model in _models:
        key_state = _google_pool().next_available()
        if not key_state:
            return None
        url = _GEMINI_BASE.format(model=model)
        gen_cfg = {"temperature": 0.15, "maxOutputTokens": max_output_tokens}
        if thinking_budget > 0:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        payload = {"contents": [{"parts": parts}], "generationConfig": gen_cfg}
        t0 = time.monotonic()
        try:
            res = requests.post(url, json=payload, headers=_gemini_headers(key_state.key),
                                timeout=max(REQUEST_TIMEOUT_GEMINI, 240))
            key_state.record_request()
            _mark_if_key_invalid(res, key_state)
            if res.status_code == 429:
                logger.debug("[GEMINI_MM_CODE] %s 429 (model quota) — next model", model)
                continue
            if res.status_code in (400, 404):
                continue
            if res.status_code >= 500:
                continue
            res.raise_for_status()
            text = (res.json().get("candidates", [{}])[0]
                    .get("content", {}).get("parts", [{}])[0].get("text", "")) or ""
            _log_call(LLMCallRecord(
                role=role, model=f"{model}+vision", api_key_name=key_state.name,
                prompt_chars=len(prompt), response_chars=len(text),
                success=bool(text), latency_sec=time.monotonic()-t0, attempt=1,
            ))
            if text:
                logger.info("[GEMINI_MM_CODE] %s saw %d image(s) → %d chars",
                            model, len([i for i in images if i]), len(text))
                return text.strip()
        except Exception as exc:
            logger.warning("[GEMINI_MM_CODE] %s failed: %s", model, exc)
    return None


_GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com/chat/completions"

GITHUB_CODEGEN_MODELS = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "Meta-Llama-3.1-405B-Instruct",
]


def _github_chat(
    model: str,
    messages: list[dict],
    role: str,
    temperature: float = 0.15,
    max_tokens: int = 16000,
) -> Optional[str]:
    """DEAD PATH — GitHub Models was retired 2026-07-30 and _GITHUB_TOKENS is
    permanently empty, so this returns None on the first line every time,
    without a network call. The ~16 call sites through the provider cascade
    are therefore no-ops kept only because removing them means editing nine
    different functions in the least-tested code here; do that when there is
    API quota to verify generation afterwards, not before.

    Was: text-only chat via GitHub Models, round-robining across tokens."""
    tok = _next_github_token()
    if not tok:
        return None
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    t0 = time.monotonic()
    try:
        res = requests.post(_GITHUB_MODELS_ENDPOINT, headers=headers,
                            json=payload, timeout=90)
        if res.status_code == 429:
            logger.warning("[GITHUB_CHAT] %s rate-limited (token rotated)", model)
            return None
        if res.status_code != 200:
            logger.debug("[GITHUB_CHAT] %s HTTP %d", model, res.status_code)
            return None
        text = (res.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "") or "")
        _log_call(LLMCallRecord(role=role, model=f"github/{model}",
            api_key_name=f"GITHUB_TOKEN(pool/{len(_GITHUB_TOKENS)})",
            prompt_chars=sum(len(m.get("content","")) for m in messages),
            response_chars=len(text), success=bool(text),
            latency_sec=time.monotonic()-t0, attempt=1))
        if text:
            logger.info("[GITHUB_CHAT] %s → %d chars", model, len(text))
        return _strip_thinking(text) if text else None
    except Exception as exc:
        logger.warning("[GITHUB_CHAT] %s failed: %s", model, exc)
        return None


def _deepseek_chat(
    model: str,
    messages: list[dict],
    role: str,
    temperature: float = 0.15,
    max_tokens: int = 16000,
) -> Optional[str]:
    """DeepSeek API — OpenAI-compatible, excellent coder, free credits."""
    if not _deepseek_key():
        return None
    headers = {"Authorization": f"Bearer {_deepseek_key()}",
               "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    t0 = time.monotonic()
    try:
        res = requests.post(_DEEPSEEK_URL, headers=headers, json=payload, timeout=90)
        if res.status_code == 429:
            logger.warning("[DEEPSEEK] rate-limited")
            return None
        if res.status_code != 200:
            logger.debug("[DEEPSEEK] HTTP %d", res.status_code)
            return None
        text = (res.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "") or "")
        _log_call(LLMCallRecord(role=role, model=f"deepseek/{model}",
            api_key_name="DEEPSEEK_API_KEY",
            prompt_chars=sum(len(m.get("content","")) for m in messages),
            response_chars=len(text), success=bool(text),
            latency_sec=time.monotonic()-t0, attempt=1))
        if text:
            logger.info("[DEEPSEEK] %s → %d chars", model, len(text))
        return _strip_thinking(text) if text else None
    except Exception as exc:
        logger.warning("[DEEPSEEK] %s failed: %s", model, exc)
        return None


def _github_vision_chat(
    prompt: str,
    image_bytes: bytes,
    role: str,
    model: str = "gpt-4o",
) -> Optional[str]:
    """DEAD PATH — GitHub Models was retired 2026-07-30 and _GITHUB_TOKENS is
    permanently empty, so this returns None on the first line every time,
    without a network call. The ~16 call sites through the provider cascade
    are therefore no-ops kept only because removing them means editing nine
    different functions in the least-tested code here; do that when there is
    API quota to verify generation afterwards, not before.

    Was: vision call via GitHub Models, round-robining tokens."""
    import base64
    tok = _next_github_token()
    if not tok or not image_bytes:
        return None
    if image_bytes[:2] == b"\xff\xd8": mime = "image/jpeg"
    elif image_bytes[:4] == b"RIFF": mime = "image/webp"
    else: mime = "image/png"
    b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]}],
        "max_tokens": 1500,
    }
    t0 = time.monotonic()
    try:
        res = requests.post(_GITHUB_MODELS_ENDPOINT,
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"},
            json=payload, timeout=60)
        if res.status_code == 429:
            logger.warning("[GITHUB_VISION] Rate limited")
            return None
        if res.status_code != 200:
            logger.warning("[GITHUB_VISION] HTTP %s", res.status_code)
            return None
        text = (res.json().get("choices", [{}])[0]
                .get("message", {}).get("content", "") or "")
        _log_call(LLMCallRecord(role=role, model=f"github/{model}-vision",
            api_key_name="GITHUB_TOKEN",
            prompt_chars=len(prompt), response_chars=len(text),
            success=bool(text), latency_sec=time.monotonic() - t0, attempt=1,
        ))
        return text.strip() if text else None
    except Exception as exc:
        logger.warning("[GITHUB_VISION] failed: %s", exc)
        return None


def _openrouter_chat(
    model: str,
    messages: list[dict[str, str]],
    role: str,
    temperature: float = 0.2,
    max_tokens: int = 4096,
) -> Optional[str]:
    key_state = _openrouter_pool().next_available()
    if not key_state:
        return None

    headers = {
        "Authorization": f"Bearer {key_state.key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://argus.local",
        "X-Title": "ARGUS",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    prompt_chars = sum(len(m.get("content", "")) for m in messages)
    t0 = time.monotonic()
    try:
        res = requests.post(_OR_ENDPOINT, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_OR)
        key_state.record_request()
        if res.status_code == 429:
            key_state.mark_rate_limited(DEFAULT_BACKOFF_BASE * 2)
            return None
        if res.status_code in {401, 403}:
            key_state.mark_exhausted()
            return None
        if res.status_code >= 500:
            return None
        res.raise_for_status()
        data = res.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        _log_call(
            LLMCallRecord(
                role=role,
                model=model,
                api_key_name=key_state.name,
                prompt_chars=prompt_chars,
                response_chars=len(content),
                success=bool(content),
                latency_sec=time.monotonic() - t0,
                attempt=1,
            )
        )
        return content.strip() if content else None
    except Exception as exc:  # noqa: BLE001
        _log_call(
            LLMCallRecord(
                role=role,
                model=model,
                api_key_name=key_state.name,
                prompt_chars=prompt_chars,
                response_chars=0,
                success=False,
                latency_sec=time.monotonic() - t0,
                attempt=1,
                error=str(exc),
            )
        )
        return None


def key_pool_status() -> dict:
    return {
        "google": _google_pool().report(),
        "openrouter": _openrouter_pool().report(),
    }


def _sanitize_scene_graph(raw) -> list:
    """T2.8 — validate/clean the planner's scene-graph layout.

    Keeps only well-formed nodes ({name, pos[3], size[3], attach_to?}) and
    rebases the layout so the lowest point sits at Z=0.
    """
    if not isinstance(raw, list) or not raw:
        return []
    clean: list[dict] = []
    for node in raw:
        if not isinstance(node, dict):
            continue
        name = str(node.get("name", "")).strip()
        if not name:
            continue
        pos = node.get("pos") or node.get("position") or [0, 0, 0]
        size = node.get("size") or node.get("dims") or [0.2, 0.2, 0.2]
        try:
            pos = [float(v) for v in list(pos)[:3]]
            size = [abs(float(v)) for v in list(size)[:3]]
        except (TypeError, ValueError):
            continue
        if len(pos) != 3 or len(size) != 3:
            continue
        clean.append({
            "name": name,
            "pos": pos,
            "size": size,
            "attach_to": str(node.get("attach_to") or node.get("parent") or "").strip(),
        })
    if not clean:
        return []
    min_z = min(n["pos"][2] - n["size"][2] * 0.5 for n in clean)
    if abs(min_z) > 1e-6:
        for n in clean:
            n["pos"][2] -= min_z
    return clean


def generate_design_brief(user_prompt: str) -> str:
    """Stage-A planning: a plain-language engineering brief with real-world
    dimensions. Thinking in words first makes the JSON translation step far
    easier for small models. Returns "" when no model responds."""
    brief_prompt = (
        "You are a senior game prop artist planning a 3D model.\n"
        f"Object requested: {user_prompt}\n\n"
        "Write a SHORT engineering brief (max 250 words, plain text, NO JSON):\n"
        "1. Identify the object and its most recognizable silhouette features.\n"
        "2. State real-world overall dimensions in metres as named constants "
        "(e.g. SEAT_H=0.45, TOP_W=1.2).\n"
        "3. List every visible part (6-16) bottom-up: name, the primitive it is "
        "closest to (box / cylinder / sphere / lathe-profile / tube / wheel / "
        "ring), its size in metres derived from the constants, and what it "
        "attaches to.\n"
        "4. Note symmetry (what can be mirrored across X or Y) and repetition "
        "(what can be arrayed: bolts, slats, rungs).\n"
        "Keep functional parts at working size: a graspable handle spans "
        "50-70% of the body height and stands off the body by 0.4-0.7x the "
        "body radius; spouts, knobs and feet must be clearly visible at arm's "
        "length, never vestigial.\n"
        "5. Give THIS design one or two distinctive, plausible signature "
        "features that set it apart from the generic catalog version — an "
        "unusual proportion, a characterful silhouette curve, an asymmetric "
        "functional detail. Name them concretely with dimensions; do not "
        "sacrifice recognizability.\n"
        "Be concrete and dimensional, not poetic."
    )
    for model in GEMINI_PLANNER_MODELS:
        raw = _gemini_generate(model, brief_prompt, "design_brief",
                               max_output_tokens=1024, temperature=0.85)
        if raw:
            return raw.strip()[:4000]
    _msgs = [{"role": "system", "content": "You are a senior game prop artist."},
             {"role": "user", "content": brief_prompt}]
    raw = _github_chat("gpt-4o", _msgs, "design_brief",
                       temperature=0.8, max_tokens=1024)
    return raw.strip()[:4000] if raw else ""


def _plan_once(user_prompt: str, brief: str = "",
               temperature: float = 0.3,
               thinking_budget: int = 6000) -> Optional[dict]:
    _hint_keywords = [
        "metal", "steel", "iron", "rust", "copper", "gold",
        "wood", "timber", "plank", "concrete", "cement",
        "stone", "rock", "brick", "tile", "fabric", "cloth",
        "leather", "plastic", "rubber", "ground", "soil",
        "dirt", "grass", "sand", "bark", "organic", "marble",
        "asphalt", "paint", "plaster", "wall", "ceramic",
        "porcelain", "glass", "chrome", "aluminum", "brass",
        "bronze", "glazed",
    ]
    prompt_lower = user_prompt.lower()
    # No default hints: an unrelated catalog ("metal, concrete" for a ceramic
    # mug) invites off-topic texture picks. No hints → measured/procedural.
    material_hints = [kw for kw in _hint_keywords if kw in prompt_lower]

    catalog_block = build_material_catalog_block(material_hints, user_prompt)

    from core.blueprints import fewshot_block as _bp_fewshots
    try:
        _fewshot = _bp_fewshots(user_prompt)
    except Exception as _bp_exc:  # noqa: BLE001 — planner must never crash
        logger.warning("[PLANNER] blueprint retrieval failed: %s", _bp_exc)
        _fewshot = ""

    _brief_block = (
        "\nDESIGN BRIEF — a senior artist already analysed this request; follow "
        f"its dimensions and part breakdown closely:\n{brief}\n"
        if brief else ""
    )

    planner_prompt = f"""
You are a 3D asset construction planner. Before writing any JSON, think through
the object step by step:

THINK FIRST (do this mentally before writing JSON):
1. What is the load-bearing base? What part touches the ground first?
2. Work upward — what attaches to the base? What attaches to that?
3. For each major part: what is its real-world size in metres?
   Where exactly is its centre (x, y, z) with Z=0 at ground level?
4. How do parts physically contact each other — overlap, bolt, slot, weld, hinge?
5. What materials are clearly visible and distinct from each other?

Then output ONLY valid JSON with these keys:
rejected (boolean — true if the request is too complex to generate reliably, false otherwise),
rejection_reason (string — plain-English explanation shown to the user if rejected is true, empty string otherwise),
category (string),
parts (array of strings),
style (string),
material (string),
poly_budget ("low"|"medium"|"high"),
export_format ("glb"|"fbx"),
asset_scale (number),
structure_summary (string, 1-2 concise sentences),
primary_forms (array of strings describing main volumes/shapes),
assembly_order (array of strings from load-bearing core to details),
attachment_points (array of strings describing visible joints, hubs, brackets, welds, bolts),
orientation_notes (array of strings describing axes: vertical, horizontal, wheel/ring plane, front/back),
support_strategy (string describing base/feet/stance/center-of-mass stability),
texture_plan (array of strings describing materials, roughness, color variation, wear, seams, labels),
shape_refinement_plan (array of strings describing bevels, tapers, rounded forms, contours, seams, and non-blocky silhouette improvements),
critical_constraints (array of strings for details that must not be wrong),
poly_haven_textures (object mapping short material purpose names to Poly Haven slugs, or empty object if no good match),
geometry_mode ("hard_surface" | "organic" | "hybrid" — hard_surface for man-made props, organic for botanical/creature-like forms, hybrid when the asset has both),
organic_parts (array of part name strings that should use metaballs when geometry_mode is "hybrid"; empty array otherwise),
materials_palette (object mapping material names to {{"color": [r,g,b] each 0-1, "metallic": 0-1, "roughness": 0-1, "texture": "<key from poly_haven_textures, or omit>"}} — 2-5 distinct visible materials),
scene_graph (array of objects, one per part — this IS the build recipe, a deterministic
  compiler turns it directly into geometry, so dimensions must be real and complete.
  Every node has:
  "name" (string, unique part name),
  "primitive" (one of: "box" | "cylinder" | "sphere" | "wheel" | "bolt" | "panel" | "lathe" | "tube" | "ring" | "leaf_card"),
  "pos" (array of 3 numbers — placement x,y,z in metres; Z is up; the LOWEST point of the whole asset sits at Z=0),
  "material" (string — a key from materials_palette),
  "attach_to" (string — the name of the part this one physically connects to, or "" for the root/base),
  plus the primitive's own dimensions:
    box/panel/sphere → "size": [sx,sy,sz] full dimensions, pos = centre
    box may also set optional rounding controls — pick ONE style per part:
            "smooth": 1-3 — subdivision smoothing, THE tool for soft rounded
            forms. Cushions/pillows/seat pads/mattresses/sofa arms →
            "smooth": 2 with "crease": 0-0.2 (puffy). Rounded plastic or
            appliance shells → "smooth": 2 with "crease": 0.6-0.9
            (higher crease = sharper edges, 1.0 = fully sharp).
            "bevel" (metres, default 0.012) — small edge chamfer for crisp
            hard-surface boxes (crates, tables, machinery); 0.005-0.02 typical.
            Smoothing pulls the surface inward ~15%, so give smooth parts a
            slightly LARGER size than the intended final form.
            "taper": 0-2 (+ optional "taper_axis": "X"|"Y"|"Z", default "Z") —
            scales the far face, turning a prism into a wedge or frustum.
            0.85 = subtly narrower at the top, 0.0 = a point, 1.15 = flared.
            Real objects are rarely perfect prisms: table/chair legs narrow
            downward, chimneys and plinths narrow upward, hoppers and buckets
            flare. Reach for a small taper (0.8-0.95) on any upright box part
            unless it is genuinely a machined rectangular block.
    cylinder → "radius", "depth", "axis": "X"|"Y"|"Z" (+ optional "r_top" for taper), pos = centre
               USE "r_top" LIBERALLY — a straight tube reads as a stock pipe.
               Legs, posts, spouts, necks, nozzles, trunks and stacks all taper;
               r_top of 0.6-0.9 x radius is the common case.
    wheel → "radius", "width", "axis": "X"|"Y" (UPRIGHT, horizontal axle), pos = hub centre
    bolt → "radius", "height", "axis" (hex detail), pos = centre
    ring → "radius" (centreline), "thickness" (cross-section radius), "axis" = hole direction, pos = centre
    lathe → "profile": [[radius,z],...] bottom-to-top revolve profile (radius 0 closes the end);
            pos = where profile z=0 sits. USE THIS for any round-bodied form:
            hydrants, barrels, bottles, vases, lamp posts, cones, extinguishers.
    tube → "path": [[x,y,z],...] polyline relative to pos, "radius". USE THIS for
           hoses, pipes, cables, curved handles.
    leaf_card → "length", "width", optional "yaw"/"pitch" degrees (foliage blade)
  and optional placement helpers:
  "rot": [rx,ry,rz] degrees (only when a part is genuinely tilted),
  "mirror": "X" or "Y" — emit one side only, the compiler mirrors it (valves, handles, lights),
  "array": {{"type":"ring","count":N,"ring_radius":R,"z":Z}} or {{"type":"linear","count":N,"step":[dx,dy,dz]}}
           — emit ONE prototype bolt/plank/rung, the compiler replicates it.
  "fuse": true — smoothly MELT this part into its attach_to parent (blended
           fillet joint, like pulled clay or a weld). Use for mug/jug handles,
           spouts, teapot necks, branches, and any organic or seamless join.
           The two parts must overlap slightly. Don't fuse mechanical bolted
           parts (wheels, brackets, lids) — those stay separate.
  Lay parts out so they occupy their real relative positions and physically touch where
  they attach (small gaps are auto-snapped closed). Provide 4-25 entries covering EVERY
  visible part of the object — with mirror/array a complex asset still only needs ~10 nodes.
  GROUND-CONTACT & SIZING: wheels, casters, legs and feet sit UNDER the body — set a
  wheel's pos.z equal to its radius so its bottom touches the ground (z=0), and keep its
  radius small enough that its TOP stays at or below the body's underside. Never let a
  wheel, dome or other round part sink more than halfway into a larger part; round
  features should protrude and read as separate, not be swallowed by the body.
  SECONDARY DETAIL (read as a real built object, not a block-out): after the main forms,
  add the characteristic working details for this object class using small tubes, rings,
  bolts and arrays — e.g. a locomotive needs connecting/coupling rods (tube) linking the
  driving wheels, piping/handrails (tube), a chimney and steam/sand domes (lathe), buffers
  and rivet rows (bolt array); machinery needs bolts, vents, hinges, brackets; furniture
  needs stretchers and fasteners. Aim for the richer end of the 4-25 parts on mechanical
  or vehicle subjects — a bare silhouette of primitives scores poorly.

{_fewshot}),
reasoning (string — PRECISION DIMENSIONAL SPEC. Start with named key constants
  (e.g. HULL_LEN=9.0, HULL_BEAM=3.1, DECK_Z=1.4), then give each major part's
  exact centre position derived from those constants, and explain every join point.
  Example: "HULL_LEN=9.0m, HULL_BEAM=3.1m, HULL_DEPTH=1.4m, DECK_Z=1.4.
  Hull centred at (0,0,0.7), keel at Z=0, deck at Z=DECK_Z=1.4.
  Cabin at pos=(0, 1.2, DECK_Z+0.67), extends Z to 2.74.
  Flybridge at pos=(0, -0.8, 2.74+0.1), 2.55m wide × 2.5m long.
  Twin motors at pos=(±0.72, -4.6, 0.88), attached to stern transom at Y=-4.5."
  This spec is injected directly into the code generator as dimensional constants.).

COMPLEXITY REJECTION RULES — set rejected=true ONLY if:
- The asset is an entire building interior, city block, landscape, or scene (not a single prop)
- The asset requires animated rigs, cloth simulation, fluid, or hair/particles
- The asset is a humanoid, creature with detailed anatomy, or character with face/hands
- The asset requires readable fine text, logos, or decals baked into the geometry
- The asset is inherently defined by interior structure only (e.g. full gearbox cross-section)
- The prompt is completely ambiguous — no single plausible prop can be identified

IMPORTANT: Do NOT reject vehicles (trucks, cars, bikes, trains), boats, furniture,
machinery, or any single game-ready prop regardless of part count. These are all
achievable using Mirror + Array modifiers which reduce unique mesh count dramatically.
A semi-truck with 18 wheels only needs 2-3 unique wheel meshes + Mirror/Array.
A steam locomotive with many rods only needs 4-6 unique primitives + instancing.
Set rejected=false for all such assets. Only reject truly impossible requests
(full building interiors, animated characters, etc.).

Be concrete. Mention if wheels/tires/handwheels must stand upright and need
horizontal axles. Mention how each major part is attached. For botanical
requests, include trunk/stem, branches or crown, leaves, fruit, roots, bark,
and natural attachment points. Do not invent humans, creatures, anatomy, or
character parts.

{catalog_block}
{_brief_block}
User request: {user_prompt}
"""
    _planner_attempts: list[tuple] = []
    for model in GEMINI_PLANNER_MODELS:
        _planner_attempts.append(("gemini", model))
    for gh_model in ["gpt-4.1", "gpt-4o"]:
        _planner_attempts.append(("github", gh_model))

    for _src, model in _planner_attempts:
        if _src == "gemini":
            raw = _gemini_generate(model, planner_prompt, "planner",
                                   thinking_budget=thinking_budget,
                                   max_output_tokens=20000,
                                   temperature=temperature, json_mode=True)
        else:
            _msgs = [{"role": "system", "content": "Return only valid JSON. No markdown."},
                     {"role": "user", "content": planner_prompt}]
            raw = _github_chat(model, _msgs, "planner", temperature=0.1, max_tokens=4096)
        if not raw:
            continue
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            continue
        parsed.setdefault("category", "object")
        parsed.setdefault("rejected", False)
        parsed.setdefault("rejection_reason", "")
        if parsed.get("rejected") is True:
            return parsed
        parsed.setdefault("parts", ["body"])
        parsed.setdefault("style", "generic")
        parsed.setdefault("material", "metal")
        parsed.setdefault("poly_budget", "medium")
        parsed.setdefault("export_format", "glb")
        parsed.setdefault("asset_scale", 1.0)
        parsed.setdefault("structure_summary", "A stable hard-surface object with connected parts.")
        parsed.setdefault("primary_forms", ["main body volume"])
        parsed.setdefault("assembly_order", ["build main body", "attach details"])
        parsed.setdefault("attachment_points", ["visible structural joints between parts"])
        parsed.setdefault("orientation_notes", ["Z is up; align circular mechanical parts to their real axis"])
        parsed.setdefault("support_strategy", "Keep the asset grounded with a stable support footprint.")
        parsed.setdefault("texture_plan", [f"{parsed['material']} base material with subtle roughness variation"])
        parsed.setdefault(
            "shape_refinement_plan",
            ["beveled edges, tapered supports, rounded pads, and non-blocky silhouettes"],
        )
        parsed.setdefault("critical_constraints", ["no floating parts", "all major parts visibly connected"])
        if not isinstance(parsed.get("poly_haven_textures"), dict):
            parsed["poly_haven_textures"] = {}
        if parsed.get("geometry_mode") not in ("hard_surface", "organic", "hybrid"):
            parsed["geometry_mode"] = "hard_surface"
        if parsed.get("geometry_mode") in ("organic", "hybrid") and not is_botanical_prompt(user_prompt):
            logger.warning(
                "[PLANNER] geometry_mode='%s' rejected for non-botanical prompt — forcing hard_surface",
                parsed["geometry_mode"],
            )
            parsed["geometry_mode"] = "hard_surface"
            parsed["organic_parts"] = []
        if not isinstance(parsed.get("organic_parts"), list):
            parsed["organic_parts"] = []
        parsed["build_spec"] = None
        try:
            _spec_report = _validate_build_spec({
                "name": str(parsed.get("category") or "asset"),
                "materials": parsed.get("materials_palette"),
                "parts": parsed.get("scene_graph"),
            })
            if _spec_report.ok:
                parsed["build_spec"] = _spec_report.spec
                parsed["build_spec_warnings"] = _spec_report.warnings
                parsed["scene_graph"] = _spec_legacy_scene_graph(_spec_report.spec)
                logger.info("[PLANNER] build spec OK: %s",
                            _spec_summary(_spec_report.spec))
            else:
                logger.warning("[PLANNER] build spec rejected: %s",
                               "; ".join(_spec_report.errors))
                parsed["scene_graph"] = _sanitize_scene_graph(parsed.get("scene_graph"))
        except Exception as _spec_exc:  # noqa: BLE001 — planner must never crash
            logger.warning("[PLANNER] build spec assembly failed: %s", _spec_exc)
            parsed["scene_graph"] = _sanitize_scene_graph(parsed.get("scene_graph"))
        parsed.setdefault("reasoning", "")
        if not isinstance(parsed["parts"], list) or not parsed["parts"]:
            parsed["parts"] = ["body"]
        for key in (
            "primary_forms",
            "assembly_order",
            "attachment_points",
            "orientation_notes",
            "texture_plan",
            "shape_refinement_plan",
            "critical_constraints",
        ):
            if not isinstance(parsed.get(key), list) or not parsed[key]:
                parsed[key] = ["unspecified"]
        return parsed

    return None


def generate_part_list(user_prompt: str) -> dict:
    """Plan the asset: optional design brief (stage A), then N spec candidates
    (stage B) scored render-free, best one wins.

    Knobs: ARGUS_TWO_STAGE (default 1), ARGUS_SPEC_CANDIDATES (default 3)."""
    brief = ""
    if os.environ.get("ARGUS_TWO_STAGE", "1") != "0":
        brief = generate_design_brief(user_prompt)
        if brief:
            logger.info("[PLANNER] design brief ready (%d chars)", len(brief))
        else:
            logger.info("[PLANNER] design brief unavailable — planning directly")

    try:
        n = max(1, min(5, int(os.environ.get("ARGUS_SPEC_CANDIDATES", "3"))))
    except ValueError:
        n = 3

    from concurrent.futures import ThreadPoolExecutor
    from core.spec_scorer import score_spec

    def _candidate(i: int) -> Optional[dict]:
        # All candidates get the full thinking budget — they run in parallel,
        # so extra thinking costs almost no wall time. A temperature ladder
        # provides the diversity: one conservative anchor, the rest explore
        # progressively hotter designs. The deterministic scorer (grounding,
        # connectivity, islands, default-dims) keeps whichever lands best, so
        # hot candidates only win when they are also structurally coherent.
        ladder = (0.3, 0.9, 1.15, 1.25, 1.3)
        return _plan_once(user_prompt, brief=brief,
                          temperature=ladder[min(i - 1, len(ladder) - 1)])

    if n == 1:
        results = [_candidate(1)]
    else:
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(_candidate, range(1, n + 1)))

    candidates: list[dict] = []
    for i, parsed in enumerate(results, 1):
        if parsed is None:
            continue
        if parsed.get("rejected") is True:
            return parsed
        spec = parsed.get("build_spec")
        if spec:
            try:
                s, why = score_spec(spec, user_prompt)
            except Exception as _sc_exc:  # noqa: BLE001
                logger.warning("[PLANNER] spec scoring failed: %s", _sc_exc)
                s, why = 0.0, []
            parsed["spec_score"] = s
            parsed["spec_score_reasons"] = why
            logger.info("[PLANNER] candidate %d/%d spec score %.0f%s", i, n, s,
                        (" — " + "; ".join(why[:3])) if why else "")
        else:
            parsed["spec_score"] = -1.0
            logger.info("[PLANNER] candidate %d/%d has no valid build spec", i, n)
        candidates.append(parsed)

    if not candidates:
        return enhance_part_data_for_prompt(user_prompt, {
            "category": "object",
            "parts": ["body", "detail"],
            "style": "generic",
            "material": "metal",
            "poly_budget": "medium",
            "export_format": "glb",
            "asset_scale": 1.0,
            "structure_summary": "A stable hard-surface object with connected parts.",
            "primary_forms": ["main body volume", "secondary details"],
            "assembly_order": ["build grounded main body", "attach secondary details"],
            "attachment_points": ["visible joints connecting all major pieces"],
            "orientation_notes": ["Z is up", "circular mechanical parts follow their real-world axis"],
            "support_strategy": "Ground the asset on a stable base or support footprint.",
            "texture_plan": ["metal material", "subtle roughness and edge wear"],
            "shape_refinement_plan": [
                "beveled hard-surface edges",
                "rounded or tapered forms where the object is not naturally box-shaped",
                "surface details that break up large plain faces",
            ],
            "critical_constraints": ["no floating parts", "no flat wheels unless explicitly horizontal"],
            "poly_haven_textures": {},
            "geometry_mode": "hard_surface",
            "organic_parts": [],
            "scene_graph": [],
            "build_spec": None,
            "reasoning": "",
        })

    best = max(candidates, key=lambda c: c.get("spec_score", -1.0))
    if len(candidates) > 1:
        others = ", ".join(f"{c.get('spec_score', -1):.0f}"
                           for c in candidates if c is not best)
        logger.info("[PLANNER] best-of-%d: picked spec score %.0f (others: %s)",
                    len(candidates), best.get("spec_score", -1), others)
    return enhance_part_data_for_prompt(user_prompt, best)


def is_gaming_chair_prompt(user_prompt: str) -> bool:
    text = user_prompt.lower()
    chair_terms = ("gaming chair", "gamer chair", "racing chair", "ergo chair", "ergonomic chair")
    return any(term in text for term in chair_terms) or ("chair" in text and "gaming" in text)


def is_botanical_prompt(user_prompt: str) -> bool:
    text = user_prompt.lower()
    return any(
        term in text
        for term in (
            "tree",
            "plant",
            "banana",
            "fruit",
            "leaf",
            "leaves",
            "palm",
            "trunk",
            "branch",
            "foliage",
            "flower",
            "banyan",
            "aerial root",
            "prop root",
            "buttress root",
        )
    )


def _append_unique(values: list, additions: list[str]) -> list[str]:
    existing = {str(value).strip().lower() for value in values}
    for addition in additions:
        key = addition.strip().lower()
        if key and key not in existing:
            values.append(addition)
            existing.add(key)
    return values


def enhance_part_data_for_prompt(user_prompt: str, part_data: dict) -> dict:
    part_data = apply_structure_blueprint(user_prompt, part_data)
    prompt_text = user_prompt.lower()
    blueprint_text = str(part_data.get("structure_blueprint", "")).lower()
    category_text = str(part_data.get("category", "")).lower()
    combined_text = " ".join([prompt_text, blueprint_text, category_text])

    part_data["shape_refinement_plan"] = _append_unique(
        list(part_data.get("shape_refinement_plan") or []),
        [
            "use part-appropriate geometry instead of one generic cube helper for every visible part",
            "keep meaningful visible parts as separate semantically named mesh objects",
            "use clean object and material names without ARGUS_ prefixes except ARGUS_ROOT",
        ],
    )
    part_data["critical_constraints"] = _append_unique(
        list(part_data.get("critical_constraints") or []),
        [
            "visible mesh objects and materials should use clean semantic names, not ARGUS_ debug-style prefixes",
            "soft, round, wheeled, tubular, and shell-like parts must not be implemented as plain scaled cubes",
        ],
    )

    if any(term in combined_text for term in ("car", "sedan", "truck", "van", "bus", "vehicle_four_wheel", "go kart", "rc car")):
        part_data["shape_refinement_plan"] = _append_unique(
            list(part_data.get("shape_refinement_plan") or []),
            [
                "build vehicle body as a bevelled/tapered exterior shell with hood, cabin, trunk/bed, bumpers, fenders, and glass",
                "make tires, rims, and hubs separate upright circular parts named by corner such as front_left and rear_right",
                "add windows, lights, door seams, wheel arches, bumpers, and material separation for paint, rubber, metal, and glass",
            ],
        )
        part_data["critical_constraints"] = _append_unique(
            list(part_data.get("critical_constraints") or []),
            [
                "four-wheel vehicles need four vertical tires with separate rims and hubs",
                "vehicle wheels must connect to the body/chassis through hubs, axles, or wheel wells",
                "vehicle body must not be a single plain rectangular cube",
            ],
        )

    if any(term in combined_text for term in ("bike", "bicycle", "motorcycle", "motorbike", "scooter", "vehicle_two_wheel")):
        part_data["shape_refinement_plan"] = _append_unique(
            list(part_data.get("shape_refinement_plan") or []),
            [
                "use cylindrical tubes for frame members, forks, handlebars, seat posts, and supports",
                "make two separate vertical wheel assemblies with tire, rim, hub, fork/dropout connections, and drive detail",
                "add saddle, handlebar grips, pedals/crank or engine/exhaust details depending on vehicle type",
            ],
        )
        part_data["critical_constraints"] = _append_unique(
            list(part_data.get("critical_constraints") or []),
            [
                "two-wheel vehicles need exactly two main wheels aligned in one vertical plane",
                "bike and motorcycle frames must be tubular/cylindrical, not flat cube bars",
            ],
        )

    if any(term in combined_text for term in ("backpack", "school bag", "rucksack", "luggage", "suitcase")):
        part_data["shape_refinement_plan"] = _append_unique(
            list(part_data.get("shape_refinement_plan") or []),
            [
                "make bag bodies soft rounded bevelled/tapered volumes, not cubes",
                "add front pocket, zipper tracks, zipper pulls, shoulder straps, buckles, seams, carry handle, and fabric panels",
                "straps must attach at both ends and remain separately editable",
            ],
        )
        part_data["critical_constraints"] = _append_unique(
            list(part_data.get("critical_constraints") or []),
            [
                "backpack or bag body must not be a plain box",
                "straps, handles, pockets, zippers, and buckles must be visible named parts",
            ],
        )

    if any(term in combined_text for term in ("shoe", "sneaker", "trainer", "boot")):
        part_data["shape_refinement_plan"] = _append_unique(
            list(part_data.get("shape_refinement_plan") or []),
            [
                "build layered outsole/midsole, rounded upper, toe box, heel counter, tongue, laces, eyelets, collar, and tread blocks",
                "use rounded shell-like forms for upper and toe box; laces and eyelets should be separate details",
                "separate materials for rubber sole, fabric/leather upper, lace, and metal/plastic eyelets",
            ],
        )
        part_data["critical_constraints"] = _append_unique(
            list(part_data.get("critical_constraints") or []),
            [
                "shoe upper and toe box must not be raw cubes",
                "shoe must include editable sole, upper, tongue, lace, eyelet, heel, and tread parts",
            ],
        )

    if is_botanical_prompt(user_prompt):
        part_data["geometry_mode"] = "organic"
        part_data["organic_parts"] = list(part_data.get("parts") or [])
        part_data["poly_budget"] = "high"
        part_data["parts"] = _append_unique(
            list(part_data.get("parts") or []),
            [
                "root flare",
                "tapered organic trunk or pseudostem",
                "layered leaf crown",
                "individual broad leaf blades",
                "leaf center veins",
                "hanging fruit bunch if requested",
                "surface bark or fibrous stem strips",
                "aerial roots and prop roots for banyan trees",
            ],
        )
        part_data["shape_refinement_plan"] = _append_unique(
            list(part_data.get("shape_refinement_plan") or []),
            [
                "use organic curvature and taper instead of hard-surface boxes",
                "make leaves broad tapered blades with center veins and varied angles",
                "layer foliage in overlapping clusters with natural asymmetry",
                "attach fruit to visible stems or bunch stalks",
                "for banyan trees, build a wide trunk cluster, buttress roots, hanging aerial roots, and broad tiered canopy",
            ],
        )
        part_data["critical_constraints"] = _append_unique(
            list(part_data.get("critical_constraints") or []),
            [
                "must read as botanical vegetation, not a mechanical prop",
                "leaves must not be plain rectangles or disconnected cards",
                "fruit must attach to the plant through visible stalks",
                "root or trunk base must touch the ground",
                "banyan prompts must include aerial root curtains and grounded prop/buttress roots",
            ],
        )
    if not is_gaming_chair_prompt(user_prompt):
        return part_data

    part_data["category"] = "Gaming Chair"
    part_data["structure_blueprint"] = part_data.get("structure_blueprint") or "gaming_chair"
    part_data["blueprint_count"] = blueprint_count()
    part_data["style"] = "Ergonomic racing-style gaming chair with bucket seat and high contoured back"
    part_data["asset_scale"] = max(float(part_data.get("asset_scale", 1.0) or 1.0), 1.35)

    part_data["parts"] = _append_unique(
        list(part_data.get("parts") or []),
        [
            "five-star radial wheeled base",
            "five caster wheels",
            "central gas lift cylinder",
            "tilt mechanism under seat",
            "bucket seat cushion with raised side bolsters",
            "tall contoured backrest with shoulder wings",
            "lumbar pillow on front of backrest",
            "headrest pillow near top of backrest",
            "left adjustable armrest with vertical posts and rounded pad",
            "right adjustable armrest with vertical posts and rounded pad",
            "visible rear backrest brackets",
            "stitching seams and upholstery panels",
        ],
    )
    part_data["primary_forms"] = _append_unique(
        list(part_data.get("primary_forms") or []),
        [
            "rounded bucket seat pan with raised left/right bolsters",
            "tall reclined shield-shaped backrest with side bolsters",
            "five long tapered spokes radiating from central hub",
            "small vertical caster wheels at each spoke end",
            "soft ellipsoid headrest and lumbar cushions attached to backrest front",
        ],
    )
    part_data["assembly_order"] = [
        "five caster wheels attached to the ends of five radial base spokes",
        "central hub and gas lift cylinder rising vertically from the base",
        "tilt mechanism plate attached below the seat cushion",
        "bucket seat cushion mounted on mechanism",
        "backrest connected to rear of seat with visible side brackets",
        "armrest posts mounted to left and right seat sides",
        "rounded arm pads mounted horizontally on posts",
        "lumbar and headrest pillows strapped/contacted to front of backrest",
        "stitching seams, panel grooves, bolts, and upholstery detail",
    ]
    part_data["attachment_points"] = _append_unique(
        list(part_data.get("attachment_points") or []),
        [
            "gas lift penetrates both central base hub and underside seat mechanism",
            "backrest side brackets overlap the rear seat frame",
            "armrest posts touch the seat side frame and support rounded pads",
            "caster fork/yoke at each spoke end holds a vertical wheel with horizontal axle",
            "headrest and lumbar pillows visibly touch the front face of the backrest",
        ],
    )
    part_data["shape_refinement_plan"] = _append_unique(
        list(part_data.get("shape_refinement_plan") or []),
        [
            "avoid simple stool/chair silhouette; must read as a racing-style gaming chair",
            "use bevels on all hard cube/prism edges",
            "use scaled UV spheres/ellipsoids for cushions, headrest, lumbar pillow, and rounded arm pads",
            "add raised seat side bolsters and tall backrest shoulder wings",
            "make base spokes long and tapered, not one straight bar",
            "add seam grooves/stitch lines on seat and back upholstery panels",
        ],
    )
    part_data["critical_constraints"] = _append_unique(
        list(part_data.get("critical_constraints") or []),
        [
            "must have exactly five radial base spokes or a clear five-star base",
            "must have five caster wheels at the spoke ends",
            "must include headrest and lumbar pillows attached to the backrest",
            "must include left and right armrests with pads supported by posts",
            "must not look like a stool, dining chair, or generic office chair",
            "wheels must not float detached from the base",
        ],
    )
    return part_data


_NAME_FILLER_WORDS = {"a", "an", "the", "some", "of", "with", "and"}
_NAME_PREAMBLE = re.compile(
    r"^\s*(here(?:'s| is)?(?: the| a)?(?: name)?|name|asset(?: name)?|"
    r"base name|suggestion|answer)\s*[:\-]\s*", re.IGNORECASE)


def _clean_asset_name(raw: str) -> str:
    """Collapse any model output into a clean 2-4 word snake_case base name.

    Small naming models routinely ignore 'return only the name' and emit a
    list of alternatives, a sentence, markdown, or a preamble. Taking only the
    first candidate and the first few content words makes the result robust to
    all of that (a comma/newline list of three names previously got glued into
    one 48-char blob)."""
    if not raw:
        return ""
    # First non-empty line, minus code fences / bullets / quotes.
    line = ""
    for ln in raw.replace("`", "").splitlines():
        ln = ln.strip().lstrip("-*0123456789. ").strip().strip('"\'')
        if ln:
            line = ln
            break
    line = _NAME_PREAMBLE.sub("", line)
    # Keep only the FIRST candidate when the model returns a list/choices.
    line = re.split(r"[,;/\n]|\bor\b", line, maxsplit=1, flags=re.IGNORECASE)[0]
    # Words = alphanumeric tokens; drop leading filler ("a", "the", ...).
    words = re.findall(r"[a-z0-9]+", line.lower())
    while words and words[0] in _NAME_FILLER_WORDS:
        words.pop(0)
    words = [w for w in words if w not in _NAME_FILLER_WORDS] or words
    if not words:
        return ""
    return "_".join(words[:4])[:48].strip("_")


def generate_asset_name(user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Generate a clean, human-readable asset base name in snake_case. "
                "Return ONLY the single name on one line — no list, no "
                "alternatives, no explanation. Use 2 to 4 meaningful words. "
                "Do not include dates, timestamps, version numbers, counters, IDs, "
                "file extensions, or decorative punctuation."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]
    for model in OPENROUTER_NAMING_MODELS:
        raw = _openrouter_chat(model, messages, "naming", temperature=0.1, max_tokens=32)
        name = _clean_asset_name(raw or "")
        if name:
            return name
    return _clean_asset_name(user_prompt) or "generated_asset"


def generate_blender_script(
    generation_prompt: str,
    part_data: dict,
    reference_image: Optional[bytes] = None,
    user_prompt: str = "",
) -> str:
    exemplar = retrieve_exemplar_script(user_prompt or "", part_data)
    if exemplar.get("script"):
        logger.info(
            "[SCRIPT_GEN] RAG exemplar '%s' (score=%d, %d approved run(s))",
            exemplar.get("blueprint"), exemplar.get("score"), exemplar.get("approved_runs"),
        )
        _exemplar_block = (
            "REFERENCE EXEMPLAR — a previously SUCCESSFUL Blender script for a "
            f"similar asset (blueprint '{exemplar.get('blueprint')}', "
            f"{exemplar.get('approved_runs')} approved run(s)). Use it as a "
            "structural and API-style guide for proportions, part decomposition, "
            "material setup, and parenting. Do NOT copy it verbatim — adapt it to "
            "the current request below.\n\n```python\n"
            + exemplar["script"][:14000]
            + "\n```\n\n=== NOW BUILD THE CURRENT REQUEST ===\n"
        )
        generation_prompt = _exemplar_block + generation_prompt

    if reference_image:
        direct_vision_prompt = (
            "You are a Blender 4.1+ Python expert. The attached image is a "
            "reference render of the 3D asset you must build. Study its shape, "
            "proportions, parts, and how they connect. Build ONLY the asset — "
            "ignore the background/floor/shadows entirely.\n\n"
            "Match the silhouette and proportions you SEE in the image as closely "
            "as possible. Return ONLY executable Python code, no markdown.\n\n"
            + generation_prompt
        )
        raw = _gemini_multimodal_code(
            direct_vision_prompt, [reference_image], "generation_direct_vision",
            thinking_budget=6000, max_output_tokens=32000,
        )
        if raw:
            code = sanitize_generated_code(_extract_code(raw))
            valid, _reason = validate_generated_script(code)
            if valid:
                logger.info("[SCRIPT_GEN] DIRECT VISION codegen OK (Gemini saw reference)")
                return code
            logger.warning("[SCRIPT_GEN] direct vision codegen failed validation: %s", _reason)

    if reference_image:
        parts_hint = ", ".join((part_data.get("parts") or [])[:12])

        describe_prompt = (
            "You are a 3D geometry analyst. Look at this reference image carefully.\n\n"
            f"This is a reference for a 3D asset. Expected parts: {parts_hint}.\n\n"
            "IMPORTANT: Describe ONLY the main object. "
            "Completely ignore the background, floor plane, ground surface, shadows, and environment — "
            "those are studio lighting artifacts, NOT part of the 3D asset.\n\n"
            "Describe the object's geometry in structured detail:\n"
            "1. OVERALL SHAPE: Dominant silhouette? Proportions (width:height:depth ratio)?\n"
            "2. MAIN VOLUMES: Each major component — shape primitive "
            "(box/cylinder/sphere/cone/wedge), relative size, position.\n"
            "3. SURFACE DETAIL: Bevels, panel lines, holes, extrusions, insets.\n"
            "4. CONNECTIVITY: How do parts attach?\n"
            "5. SYMMETRY: Left-right? Top-bottom? Radially symmetric?\n\n"
            "Be precise about proportions — e.g. 'hull is 4× longer than it is tall'. "
            "Output plain text only, no code, no mention of background or environment."
        )
        geometry_description = ""
        for model in GEMINI_VISION_MODELS[:2]:
            desc = _gemini_generate_multimodal(model, describe_prompt, reference_image, "vision_describe")
            if desc and len(desc) > 100:
                geometry_description = desc.strip()
                logger.info("[SCRIPT_GEN] Geometry extracted via %s (%d chars)", model, len(geometry_description))
                break
        if not geometry_description:
            for _gh_m in ("gpt-4o", "openai/gpt-4.1"):
                desc = _github_vision_chat(describe_prompt, reference_image, "vision_describe", model=_gh_m)
                if desc and len(desc) > 100:
                    break
            if desc and len(desc) > 100:
                geometry_description = desc.strip()
                logger.info("[SCRIPT_GEN] Geometry extracted via GitHub GPT-4o (%d chars)", len(geometry_description))

        if geometry_description:
            grounded_prompt = (
                "You are a Blender 5.0 Python expert.\n\n"
                "=== VISUAL GEOMETRY ANALYSIS (extracted from reference image) ===\n"
                f"{geometry_description}\n\n"
                "=== TASK ===\n"
                "Using the geometry analysis above as your primary shape reference, "
                "write a complete Blender script that matches those exact proportions "
                "and structural details. The full specification follows — obey all rules "
                "in it, and let the geometry analysis drive your scale, silhouette, and "
                "part decisions.\n\n"
                "CRITICAL: Do NOT add a background plane, floor mesh, ground surface, "
                "or any environment object. Build ONLY the asset itself.\n\n"
                "Return ONLY executable Python code. No markdown. No explanation.\n\n"
                + generation_prompt
            )
        else:
            grounded_prompt = (
                "You are a Blender 5.0 Python expert. "
                "The attached image is a reference render of the 3D asset you must build. "
                "Match its proportions, silhouette, and part shapes as closely as possible. "
                "Ignore the background completely — do NOT create a background plane, floor mesh, "
                "or any environment object. Build ONLY the asset itself.\n\n"
                "Return ONLY executable Python code. No markdown. No explanation.\n\n"
                + generation_prompt
            )

        if geometry_description:
            grounded_messages = [
                {"role": "system", "content": "Return only executable Python code for Blender. No markdown."},
                {"role": "user",   "content": grounded_prompt},
            ]
            for _attempt in range(2):
                raw = _gemini_generate_code(grounded_prompt, "generation_grounded")
                if not raw:
                    break
                code = sanitize_generated_code(_extract_code(raw))
                valid, _reason = validate_generated_script(code)
                if valid:
                    logger.info("[SCRIPT_GEN] Vision+grounded OK via Gemini (primary)")
                    return code
            for _gh_m in GITHUB_CODEGEN_MODELS:
                raw = _github_chat(_gh_m, grounded_messages, "generation_grounded",
                                   temperature=0.15, max_tokens=20000)
                if not raw: continue
                code = sanitize_generated_code(_extract_code(raw))
                valid, _r = validate_generated_script(code)
                if valid:
                    logger.info("[SCRIPT_GEN] Vision+grounded OK via github/%s", _gh_m)
                    return code
            if _deepseek_key():
                for _ds_m in DEEPSEEK_MODELS:
                    raw = _deepseek_chat(_ds_m, grounded_messages, "generation_grounded",
                                        temperature=0.15, max_tokens=20000)
                    if not raw: continue
                    code = sanitize_generated_code(_extract_code(raw))
                    valid, _ = validate_generated_script(code)
                    if valid:
                        logger.info("[SCRIPT_GEN] Vision+grounded OK via deepseek/%s", _ds_m)
                        return code
            for model in GROQ_GENERATION_MODELS:
                raw = _groq_chat(model, grounded_messages, "generation_grounded",
                                 temperature=0.15, max_tokens=16000)
                if not raw:
                    continue
                code = sanitize_generated_code(_extract_code(raw))
                valid, reason = validate_generated_script(code)
                if valid:
                    logger.info("[SCRIPT_GEN] Vision+grounded OK via Groq/%s", model)
                    return code
                logger.warning("[SCRIPT_GEN] Groq/%s grounded failed validation: %s", model, reason)
            for _attempt in range(DEFAULT_MAX_RETRIES):
                for model in OPENROUTER_GENERATION_MODELS:
                    raw = _openrouter_chat(model, grounded_messages, "generation_grounded",
                                          temperature=0.15, max_tokens=16000)
                    if not raw:
                        continue
                    code = sanitize_generated_code(_extract_code(raw))
                    valid, _reason = validate_generated_script(code)
                    if valid:
                        logger.info("[SCRIPT_GEN] Vision+grounded OK via OpenRouter/%s", model)
                        return code
        else:
            for model in GEMINI_VISION_MODELS[:1]:
                raw = _gemini_generate_multimodal(model, grounded_prompt, reference_image, "generation_vision")
                if raw:
                    code = sanitize_generated_code(_extract_code(raw))
                    valid, _reason = validate_generated_script(code)
                    if valid:
                        logger.info("[SCRIPT_GEN] Direct vision OK via Gemini/%s", model)
                        return code

        logger.warning("[SCRIPT_GEN] Vision-guided generation failed, falling back to text-only")

    _code_messages = [
        {"role": "system", "content": "Return only executable Python code for Blender 4.1+. No markdown fences. No explanations."},
        {"role": "user", "content": generation_prompt},
    ]

    for _attempt in range(2):
        raw = _gemini_generate_code(generation_prompt, "generation")
        if not raw:
            break
        code = sanitize_generated_code(_extract_code(raw))
        valid, reason = validate_generated_script(code)
        if valid:
            logger.info("[SCRIPT_GEN] Text-only OK via Gemini (primary)")
            return code
        logger.warning("[SCRIPT_GEN] Gemini code failed validation: %s", reason)

    _gh_messages = [
        {"role": "system", "content": "Return only executable Python code for Blender 4.1+. No markdown fences. No explanations."},
        {"role": "user", "content": generation_prompt},
    ]
    for _gh_model in GITHUB_CODEGEN_MODELS:
        raw = _github_chat(_gh_model, _gh_messages, "generation",
                           temperature=0.15, max_tokens=20000)
        if not raw:
            continue
        code = sanitize_generated_code(_extract_code(raw))
        valid, reason = validate_generated_script(code)
        if valid:
            logger.info("[SCRIPT_GEN] Text-only OK via github/%s", _gh_model)
            return code
        logger.warning("[SCRIPT_GEN] github/%s failed validation: %s", _gh_model, reason[:60])

    if _deepseek_key():
        for _ds_model in DEEPSEEK_MODELS:
            raw = _deepseek_chat(_ds_model, _gh_messages, "generation",
                                 temperature=0.15, max_tokens=20000)
            if not raw: continue
            code = sanitize_generated_code(_extract_code(raw))
            valid, reason = validate_generated_script(code)
            if valid:
                logger.info("[SCRIPT_GEN] Text-only OK via deepseek/%s", _ds_model)
                return code

    for model in GROQ_GENERATION_MODELS:
        raw = _groq_chat(model, _code_messages, "generation", temperature=0.15, max_tokens=16000)
        if not raw:
            continue
        code = sanitize_generated_code(_extract_code(raw))
        valid, reason = validate_generated_script(code)
        if valid:
            logger.info("[SCRIPT_GEN] Text-only OK via Groq/%s", model)
            return code
        logger.warning("[SCRIPT_GEN] Groq/%s code failed validation: %s", model, reason)

    messages = _code_messages
    for _attempt in range(DEFAULT_MAX_RETRIES):
        for model in OPENROUTER_GENERATION_MODELS:
            raw = _openrouter_chat(model, messages, "generation", temperature=0.15, max_tokens=16000)
            if not raw:
                continue
            code = sanitize_generated_code(_extract_code(raw))
            valid, _reason = validate_generated_script(code)
            if valid:
                return code

    procedural_fallback = _procedural_blueprint_fallback(part_data, user_prompt)
    if procedural_fallback:
        return procedural_fallback

    fallback = """
import bpy
import bmesh
import math
import mathutils
import random
import os
assert bpy.app.version[0] >= 4, f"ARGUS requires Blender 5.0.x, got {bpy.app.version}"
for _obj in list(bpy.data.objects):
    bpy.data.objects.remove(_obj, do_unlink=True)
for _mesh in list(bpy.data.meshes):
    bpy.data.meshes.remove(_mesh)
for _mat in list(bpy.data.materials):
    bpy.data.materials.remove(_mat)
bpy.context.view_layer.update()
root_obj = bpy.data.objects.new("ARGUS_ROOT", None)
bpy.context.collection.objects.link(root_obj)
mesh = bpy.data.meshes.new("unsupported_body")
obj = bpy.data.objects.new("unsupported_body", mesh)
bpy.context.collection.objects.link(obj)
obj.parent = root_obj
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bmesh.ops.triangulate(bm, faces=bm.faces[:])
bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
bm.to_mesh(mesh)
bm.free()
mesh.update()
mat = bpy.data.materials.new(name="generic_metal")
mat.use_nodes = True
obj.data.materials.append(mat)
"""
    return sanitize_generated_code(fallback)


def critique_blender_script(script: str, user_prompt: str, part_data: Optional[dict] = None,
                            reference_image: "Optional[bytes]" = None) -> dict:
    local_valid, local_reason = validate_generated_script(script)
    if not local_valid:
        return {"repair_required": True, "summary": local_reason, "issues": [local_reason]}

    local_report = local_structural_critique(script, user_prompt, part_data)
    if local_report.get("repair_required"):
        return local_report

    part_data = part_data or {}
    structural_brief = json.dumps(
        {
            "category": part_data.get("category"),
            "parts": part_data.get("parts"),
            "structure_summary": part_data.get("structure_summary"),
            "primary_forms": part_data.get("primary_forms"),
            "assembly_order": part_data.get("assembly_order"),
            "attachment_points": part_data.get("attachment_points"),
            "orientation_notes": part_data.get("orientation_notes"),
                "support_strategy": part_data.get("support_strategy"),
                "shape_refinement_plan": part_data.get("shape_refinement_plan"),
                "critical_constraints": part_data.get("critical_constraints"),
        },
        ensure_ascii=True,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Review Blender 5.0 Python code for both runtime correctness and "
                "physical/visual assembly correctness. Return strict JSON with keys "
                "repair_required(bool), summary(str), issues(array). "
                "Mark repair_required true if major parts are likely disconnected, "
                "floating, side-by-side instead of assembled, wrongly oriented, "
                "unsupported, ungrounded, or using banned Blender APIs. "
                "For chairs: seat, backrest, column/base, arms, casters, and cushions "
                "must be in real relative positions and visibly attached. "
                "Mark repair_required true if the object is mostly raw cubes or blocky "
                "slabs where rounded, beveled, tapered, or contoured forms are required."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Prompt:\n{user_prompt}\n\n"
                f"Structural brief:\n{structural_brief}\n\n"
                f"Script:\n{script}"
            ),
        },
    ]
    if reference_image:
        vision_critic_prompt = (
            messages[0]["content"] + "\n\n"
            "IMPORTANT: The attached image is the REFERENCE showing what this asset "
            "should look like. Review the script and predict which visual problems the "
            "rendered output will have compared to this reference. "
            "Flag: wrong orientations (flat wheels), boxes where curves needed, "
            "missing signature parts, wrong proportions, floating parts.\n\n"
            + messages[1]["content"]
        )
        for _gh_m in ("gpt-4.1", "gpt-4o"):
            raw = _github_vision_chat(vision_critic_prompt, reference_image,
                                      "critic_vision", model=_gh_m)
            if not raw: continue
            parsed = _extract_json(raw)
            if isinstance(parsed, dict):
                parsed.setdefault("repair_required", False)
                parsed.setdefault("summary", "Vision critic completed.")
                parsed.setdefault("issues", [])
                logger.info("[CRITIC] github/%s vision critique OK", _gh_m)
                return parsed

    critic_text_prompt = messages[0]["content"] + "\n\n" + messages[1]["content"]
    for model in GEMINI_PLANNER_MODELS[:2]:
        raw = _gemini_generate(model, critic_text_prompt, "critic",
                               thinking_budget=4000, max_output_tokens=8000)
        if not raw:
            continue
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            parsed.setdefault("repair_required", False)
            parsed.setdefault("summary", "Critic completed.")
            parsed.setdefault("issues", [])
            return parsed
    for _gh_m in ("gpt-4.1", "gpt-4o"):
        raw = _github_chat(_gh_m, messages, "critic", temperature=0.1, max_tokens=3000)
        if not raw: continue
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            parsed.setdefault("repair_required", False)
            parsed.setdefault("summary", "Critic completed.")
            parsed.setdefault("issues", [])
            return parsed
    if _deepseek_key():
        raw = _deepseek_chat("deepseek-chat", messages, "critic", temperature=0.1, max_tokens=3000)
        if raw:
            parsed = _extract_json(raw)
            if isinstance(parsed, dict):
                parsed.setdefault("repair_required", False)
                parsed.setdefault("summary", "Critic completed.")
                parsed.setdefault("issues", [])
                return parsed
    for model in OPENROUTER_CRITIC_MODELS:
        raw = _openrouter_chat(model, messages, "critic", temperature=0.1, max_tokens=2000)
        if not raw:
            continue
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            parsed.setdefault("repair_required", False)
            parsed.setdefault("summary", "Critic completed.")
            parsed.setdefault("issues", [])
            return parsed
    return {"repair_required": False, "summary": "Critic unavailable; proceeding.", "issues": []}


def local_structural_critique(script: str, user_prompt: str, part_data: Optional[dict] = None) -> dict:
    if not is_gaming_chair_prompt(user_prompt):
        return local_blueprint_critique(script, part_data or {})

    lower = script.lower()
    issues: list[str] = []
    required_name_groups = {
        "bucket seat cushion": ("seat",),
        "tall backrest": ("backrest", "back_rest", "back"),
        "left/right armrests": ("armrest", "arm_pad", "armpad"),
        "central gas lift/column": ("gas", "column", "lift"),
        "five-star base": ("five", "star", "spoke"),
        "caster wheels": ("caster", "wheel"),
        "headrest pillow": ("headrest", "head_rest"),
        "lumbar pillow": ("lumbar",),
    }
    for label, terms in required_name_groups.items():
        if not any(term in lower for term in terms):
            issues.append(f"Missing required gaming-chair feature: {label}")

    wheel_mentions = len(re.findall(r"wheel|caster", lower))
    if wheel_mentions < 5:
        issues.append("Gaming chair needs five caster wheels; script appears to describe fewer.")

    cube_count = len(re.findall(r"bmesh\.ops\.create_cube", lower))
    soft_count = len(re.findall(r"create_uvsphere|bevel|inset_region", lower))
    if cube_count >= 4 and soft_count < 3:
        issues.append("Chair is likely too blocky; use rounded cushions, bevels, bolsters, and pillows.")

    if any(term in lower for term in ("headrest", "lumbar", "cushion", "pillow", "bolster", "arm_pad")):
        if "create_uvsphere" not in lower and "bevel" not in lower:
            issues.append("Soft chair parts are present but no rounded/soft geometry operation is detectable.")
    if any(term in lower for term in ("caster", "wheel", "gas", "lift", "column")):
        if "create_cone" not in lower:
            issues.append("Chair wheels/gas lift/columns need cylindrical geometry, but no cylinder/cone primitive is detectable.")
    if "argus_chair_" in lower:
        issues.append("Visible chair mesh names still use ARGUS_ debug prefixes; use clean semantic names under ARGUS_ROOT.")

    if issues:
        return {
            "repair_required": True,
            "summary": "Generated gaming chair is structurally incomplete or too generic.",
            "issues": issues,
        }
    return {"repair_required": False, "summary": "Local gaming-chair checks passed.", "issues": []}


def local_blueprint_critique(script: str, part_data: dict) -> dict:
    signature_parts = part_data.get("blueprint_signature_parts") or []
    blueprint = part_data.get("structure_blueprint")
    if not blueprint or not signature_parts:
        return {"repair_required": False, "summary": "Local structural checks passed.", "issues": []}

    lower = script.lower()
    compiled = "[argus compiled]" in lower
    issues: list[str] = []
    blueprint_lower = str(blueprint).lower()
    category_lower = str(part_data.get("category", "")).lower()
    style_lower = str(part_data.get("style", "")).lower()
    asset_text = " ".join([blueprint_lower, category_lower, style_lower])
    is_botanical_blueprint = any(
        term in blueprint_lower
        for term in ("tree", "plant", "banana", "organic")
    )
    object_creations = len(re.findall(r"bpy\.data\.objects\.new\s*\(", lower))
    helper_object_calls = len(re.findall(r"\bbox_obj\s*\(", lower))
    mesh_builder_calls = len(re.findall(r"\bcreate_mesh_object\s*\(", lower))
    hard_surface_builder_calls = len(re.findall(
        r"\b(cube_part|cylinder_part|cyl_part|sphere_part|vcyl|obj_from_bm|ring|"
        r"create_cylinder|create_bolt|ellipsoid|pentagon_cap|bolt_ring|"
        r"plank|bracket|nail|rope_loop)\s*\(",
        lower,
    ))
    organic_builder_calls = len(re.findall(r"\b(cone_part|leaf|banana_finger|mesh_obj|sphere_part)\s*\(", lower))
    material_creations = len(re.findall(r"bpy\.data\.materials\.new\s*\(", lower))
    helper_material_calls = len(re.findall(r"\bmake_mat\s*\(", lower))
    apparent_objects = max(
        object_creations,
        helper_object_calls + mesh_builder_calls + hard_surface_builder_calls + organic_builder_calls,
    )
    apparent_materials = max(material_creations, helper_material_calls)
    if compiled:
        # Compiler-emitted scripts build parts via argus_* component calls and
        # materials via _argus_smart_mat — none of the LLM-codegen patterns
        # above match them, which made every compiled run look "too sparse"
        # and permanently blocked the structure-memory gate.
        _m = re.search(r"\[argus compiled\][^:]*:\s*(\d+)\s+parts built", lower)
        if _m:
            apparent_objects = max(apparent_objects, int(_m.group(1)))
        apparent_materials = max(
            apparent_materials,
            len(re.findall(r"=\s*_argus_smart_mat\s*\(", lower)),
        )
    if is_botanical_blueprint:
        detail_terms = (
            "root",
            "trunk",
            "pseudostem",
            "leaf",
            "leaves",
            "vein",
            "fruit",
            "banana",
            "bunch",
            "finger",
            "stalk",
            "bark",
            "strip",
            "crown",
            "branch",
            "foliage",
        )
    else:
        detail_terms = (
            "bolt",
            "screw",
            "rivet",
            "vent",
            "slat",
            "hinge",
            "latch",
            "handle",
            "bracket",
            "guard",
            "strap",
            "rail",
            "rib",
            "panel",
            "seam",
            "gasket",
            "collar",
            "foot",
            "feet",
        )
    detail_hits = sum(1 for term in detail_terms if term in lower)
    detail_floor = 2 if compiled else 4

    if apparent_objects < (4 if compiled else 8):
        issues.append(
            f"Too few named mesh/detail objects for a blueprint asset ({apparent_objects} object creations found)."
        )
    if apparent_materials < 2:
        issues.append(
            f"Too few distinct materials for a detailed hard-surface asset ({apparent_materials} material creation found)."
        )
    if detail_hits < detail_floor:
        if is_botanical_blueprint:
            issues.append("Insufficient visible botanical detail: add roots, trunk/stem sections, leaf blades, veins, bark strips, fruit stalks, or attached fruit clusters.")
        else:
            issues.append("Insufficient visible hard-surface detail: add bolts, seams, panels, brackets, vents, handles, feet, or similar attached parts.")

    clean_name_exemptions = ("argus_root", "argus_export", "argus_ok", "argus_path", "argus_uv")
    visible_argus_names = [
        match
        for match in re.findall(r"['\"](argus_[a-z0-9_]+)['\"]", lower)
        if not any(match.startswith(exempt) for exempt in clean_name_exemptions)
    ]
    if visible_argus_names and not compiled:
        # Compiled scripts intentionally prefix material names with ARGUS_;
        # the bake step rebuilds and renames them before export.
        issues.append("Visible mesh/material names should be clean semantic names without ARGUS_ prefixes.")

    cube_count = len(re.findall(r"bmesh\.ops\.create_cube", lower))
    sphere_count = len(re.findall(r"create_uvsphere", lower))
    cone_count = len(re.findall(r"create_cone", lower))
    bevel_count = len(re.findall(r"\bbevel\s*\(", lower))
    soft_asset = any(term in asset_text for term in ("backpack", "school bag", "rucksack", "luggage", "shoe", "sneaker", "trainer", "boot"))
    vehicle_asset = any(term in asset_text for term in ("vehicle", "car", "truck", "van", "bus", "bike", "bicycle", "motorcycle", "scooter"))

    if soft_asset and sphere_count == 0 and bevel_count < 2:
        issues.append("Soft-good asset appears cube-only; use rounded or bevelled soft forms for bag/shoe body parts.")
    if vehicle_asset and cone_count == 0:
        issues.append("Vehicle asset needs cylindrical wheel/tire/rim geometry, but no cone/cylinder primitive is detectable.")
    if vehicle_asset and len(re.findall(r"tire|wheel|rim|hub", lower)) < 4:
        issues.append("Vehicle asset has too few detectable wheel/rim/hub semantic parts.")
    if any(term in asset_text for term in ("car", "truck", "van", "bus", "vehicle_four_wheel")):
        for corner in ("front_left", "front_right", "rear_left", "rear_right"):
            if corner not in lower:
                issues.append(f"Four-wheel vehicle is missing semantic corner naming for {corner}.")
                break
        for feature in ("window", "light", "bumper"):
            if feature not in lower:
                issues.append(f"Vehicle asset is missing visible {feature} detail.")
        if "fender" not in lower and "wheel_arch" not in lower and "wheel arch" not in lower:
            issues.append("Vehicle asset is missing wheel-arch/fender forms around tires.")
        if "axle" not in lower and "hub" not in lower:
            issues.append("Vehicle wheels need explicit hub/axle connection geometry.")
        if cube_count >= 8 and sphere_count < 2 and bevel_count < 4:
            issues.append("Car body appears too blocky; add curved canopy/nose/deck and fender forms.")
    if any(term in asset_text for term in ("bike", "bicycle", "motorcycle", "scooter", "vehicle_two_wheel")):
        if len(re.findall(r"tire|wheel", lower)) < 2:
            issues.append("Two-wheel vehicle needs two clearly named wheel/tire assemblies.")
        if "handlebar" not in lower or "fork" not in lower:
            issues.append("Two-wheel vehicle is missing handlebar or fork detail.")
    if any(term in asset_text for term in ("backpack", "school bag", "rucksack")):
        for feature in ("strap", "zipper", "pocket", "buckle"):
            if feature not in lower:
                issues.append(f"Backpack/school bag asset is missing visible {feature} detail.")
    if any(term in asset_text for term in ("shoe", "sneaker", "trainer", "boot")):
        for feature in ("sole", "upper", "tongue", "lace", "eyelet", "tread"):
            if feature not in lower:
                issues.append(f"Shoe/sneaker asset is missing visible {feature} detail.")
    if cube_count >= 6 and sphere_count == 0 and cone_count == 0 and bevel_count < 2:
        issues.append("Asset appears to be mostly raw cube primitives; use bevels, cylinders, ellipsoids, and part-appropriate forms.")

    generic_terms = {
        "main",
        "body",
        "base",
        "frame",
        "support",
        "supports",
        "detail",
        "details",
        "small",
        "large",
        "thin",
        "thick",
        "front",
        "rear",
        "side",
        "left",
        "right",
        "top",
        "bottom",
    }
    missing: list[str] = []
    hits = 0
    for part in signature_parts[:8]:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(part).lower()).strip()
        terms = [
            term
            for term in normalized.split()
            if len(term) > 2 and term not in generic_terms
        ]
        matched = normalized in lower or any(term in lower for term in terms)
        if matched:
            hits += 1
        else:
            missing.append(str(part))

    required_hits = max(2, min(4, len(signature_parts[:8]) // 2))
    if hits < required_hits:
        issues.extend(
            [
                f"Only {hits}/{len(signature_parts[:8])} signature blueprint parts were detectable.",
                "Missing or unnamed signature parts: " + ", ".join(missing[:6]),
            ]
        )

    if issues:
        return {
            "repair_required": True,
            "summary": f"Generated script does not satisfy structure blueprint '{blueprint}'.",
            "issues": issues,
        }

    return {"repair_required": False, "summary": "Local blueprint checks passed.", "issues": []}


def generate_repair_patch(
    broken_script: str,
    traceback_obj: dict,
    repair_class: str,
    critic_report: Optional[dict] = None,
    user_prompt: str = "",
    part_data: Optional[dict] = None,
) -> str:
    if repair_class == "CriticFailure":
        procedural = _procedural_blueprint_fallback(part_data, user_prompt)
        if procedural:
            return procedural

    if repair_class in ("EmptyExport", "ExportOperatorFailed"):
        procedural = _procedural_blueprint_fallback(part_data, user_prompt)
        if procedural:
            logger.info("[REPAIR] EmptyExport/ExportFailed: using procedural blueprint fallback")
            return procedural

    known_repair = apply_known_repair(broken_script, traceback_obj)
    if known_repair:
        return known_repair

    err_text = f"{repair_class}: {traceback_obj.get('error_msg') or traceback_obj.get('message', '')}"
    if should_abort_repair_loop(err_text):
        logger.warning("[REPAIR] Duplicate repair failure; refusing to replay broken script")
        return ""

    critic_text = json.dumps(critic_report or {}, ensure_ascii=True)
    part_text = json.dumps(part_data or {}, ensure_ascii=True)

    texture_paths_block = ""
    if part_data and part_data.get("poly_haven_texture_paths"):
        texture_paths_block = (
            "\n\nCRITICAL — POLY HAVEN TEXTURE PATHS (DO NOT CHANGE THESE):\n"
            + json.dumps(part_data["poly_haven_texture_paths"], indent=2, ensure_ascii=True)
            + "\nUse ONLY these exact absolute paths in bpy.data.images.load() calls.\n"
            "Wrap every image load in try/except so a missing file does NOT crash Blender:\n"
            "  try:\n"
            "      img = bpy.data.images.load(r\"<exact_path_from_above>\")\n"
            "  except Exception:\n"
            "      img = None  # fall back to procedural color\n"
            "  if img: ... assign tex node ...\n"
            "NEVER invent, guess, or modify texture file paths.\n"
        )

    repair_prompt = f"""
Repair this Blender Python script and return only corrected code.
Error context: {json.dumps(traceback_obj, ensure_ascii=True)}
Repair class: {repair_class}
User prompt: {user_prompt}
Structural plan: {part_text}
Critic report: {critic_text}
{texture_paths_block}
Script:
{broken_script}
"""
    messages = [
        {
            "role": "system",
            "content": (
                "Return only corrected Blender Python code, no markdown. "
                "Fix ONLY the reported error. Do NOT change texture file paths — "
                "use the exact absolute paths supplied in the repair prompt. "
                "Wrap every bpy.data.images.load() call in try/except."
            ),
        },
        {"role": "user", "content": repair_prompt},
    ]
    raw = _gemini_generate_code(repair_prompt, "repair")
    if raw:
        repaired = sanitize_generated_code(_extract_code(raw))
        valid, _reason = validate_generated_script(repaired)
        if valid:
            logger.info("[REPAIR] Gemini repair succeeded")
            return repaired

    _repair_gh_messages = [
        {"role": "system", "content": "Return only corrected Blender Python code, no markdown. Fix ONLY the reported error."},
        {"role": "user", "content": repair_prompt},
    ]
    for _gh_m in ("gpt-4.1", "gpt-4o"):
        raw = _github_chat(_gh_m, _repair_gh_messages, "repair",
                           temperature=0.0, max_tokens=20000)
        if not raw: continue
        repaired = sanitize_generated_code(_extract_code(raw))
        valid, _ = validate_generated_script(repaired)
        if valid:
            logger.info("[REPAIR] github/%s succeeded", _gh_m)
            return repaired
    if _deepseek_key():
        for _ds_m in DEEPSEEK_MODELS:
            raw = _deepseek_chat(_ds_m, _repair_gh_messages, "repair",
                                 temperature=0.0, max_tokens=20000)
            if not raw: continue
            repaired = sanitize_generated_code(_extract_code(raw))
            valid, _ = validate_generated_script(repaired)
            if valid:
                logger.info("[REPAIR] deepseek/%s succeeded", _ds_m)
                return repaired

    for _attempt in range(DEFAULT_MAX_RETRIES):
        for model in OPENROUTER_REPAIR_MODELS:
            raw = _openrouter_chat(model, messages, "repair", temperature=0.0, max_tokens=16000)
            if not raw:
                continue
            repaired = sanitize_generated_code(_extract_code(raw))
            valid, _reason = validate_generated_script(repaired)
            if valid:
                return repaired
    for _attempt in range(DEFAULT_MAX_RETRIES):
        for model in GROQ_REPAIR_MODELS:
            raw = _groq_chat(model, messages, "repair", temperature=0.0, max_tokens=16000)
            if not raw:
                continue
            repaired = sanitize_generated_code(_extract_code(raw))
            valid, _reason = validate_generated_script(repaired)
            if valid:
                logger.info("[REPAIR] Groq fallback succeeded (model=%s)", model)
                return repaired
    return _procedural_blueprint_fallback(part_data)


def _score_visual_checklist(
    preview_image,
    user_prompt: str,
    part_data: dict,
    spec: Optional[dict],
    multiview: bool,
    reference_image: Optional[bytes] = None,
) -> Optional[dict]:
    """Checklist-based critic: answer concrete yes/no questions derived from
    the build spec instead of emitting one noisy holistic score. Returns the
    standard vqa dict, or None so the caller can fall back to holistic mode."""
    from core.checklist import build_checklist, summarize_results

    spec = spec or part_data.get("build_spec")
    checks = build_checklist(part_data, spec)
    if len(checks) < 4:
        return None

    _view_note = (
        "The FIRST image is a 2x2 grid showing the SAME asset from four angles "
        "(front-iso, side, rear, top-down).\n"
        if multiview else "The FIRST image shows the rendered asset.\n"
    )
    _ref_note = (
        "The SECOND image is the intended concept reference — use it to judge "
        "shape and proportions.\n" if reference_image else ""
    )
    questions = "\n".join(f"  \"{c['id']}\": {c['question']}" for c in checks)
    vision_prompt = (
        "You are inspecting a 3D asset render.\n"
        + _view_note + _ref_note
        + f"\nUser prompt: \"{user_prompt}\"\n\n"
        "Answer EVERY question below with true or false, strictly about what is "
        "visible in the render. Be honest — false answers are how problems get "
        "fixed.\n\nQuestions:\n" + questions + "\n\n"
        "Return strict JSON only:\n"
        "{\"checks\": {\"<id>\": true|false, ...}, "
        "\"feedback\": \"<one sentence on the worst problem, max 20 words>\"}"
    )

    images = [preview_image] + ([reference_image] if reference_image else [])
    try:
        votes = max(1, min(3, int(os.environ.get("ARGUS_CRITIC_VOTES", "1"))))
    except ValueError:
        votes = 1

    results: list[dict] = []
    extra_feedback = ""
    for _v in range(votes):
        raw = None
        for model in GEMINI_VISION_MODELS:
            raw = _gemini_generate_multimodal(model, vision_prompt, images,
                                              "visual_qa", temperature=0.0,
                                              json_mode=True)
            if raw:
                break
        if not raw:
            break
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("checks"), dict):
            continue
        summary = summarize_results(checks, parsed["checks"])
        if summary:
            results.append(summary)
            extra_feedback = str(parsed.get("feedback", "")).strip() or extra_feedback

    if not results:
        return None
    results.sort(key=lambda r: r["visual_score"])
    median = results[len(results) // 2]
    if extra_feedback:
        median["feedback"] = (extra_feedback + " | " + median["feedback"])[:400]
    logger.info("[VISUAL_QA] checklist: %d/%d checks passed → %d/10%s",
                median.get("checklist_passed", 0), median.get("checklist_total", 0),
                median["visual_score"],
                f" (median of {len(results)} votes)" if len(results) > 1 else "")
    return median


def score_visual_quality(
    preview_image: bytes,
    user_prompt: str,
    part_data: dict,
    multiview: bool = False,
    spec: Optional[dict] = None,
    reference_image: Optional[bytes] = None,
) -> dict:
    """
    Use Gemini vision to score the rendered asset against the original prompt.

    Checklist mode (ARGUS_CHECKLIST=1, default): concrete spec-derived yes/no
    questions, score = fraction passed. Falls back to the holistic 0-10 rating
    when no spec/checks are available or the model output is unusable.

    Returns a dict with:
        visual_score   int   0-10 (10 = fully matches prompt)
        visible_parts  list  parts confirmed visible in the render
        missing_parts  list  requested parts that appear absent
        feedback       str   one-sentence assessment
        skipped        bool  True when no key or model was available
    """
    if os.environ.get("ARGUS_CHECKLIST", "1") != "0":
        try:
            result = _score_visual_checklist(
                preview_image, user_prompt, part_data, spec,
                multiview, reference_image,
            )
            if result:
                return result
        except Exception as _cl_exc:  # noqa: BLE001 — never break scoring
            logger.warning("[VISUAL_QA] checklist mode failed: %s", _cl_exc)
        logger.info("[VISUAL_QA] checklist unavailable — holistic fallback")

    parts = part_data.get("parts") or []
    constraints = (part_data.get("critical_constraints") or [])[:4]

    _view_note = (
        "The image is a 2x2 grid showing the SAME asset from four angles "
        "(front-iso, side, rear, top-down). Judge it as one object seen from "
        "multiple views — missing or floating parts are easier to spot here.\n\n"
        if multiview else ""
    )
    vision_prompt = (
        f"You are a 3D asset quality reviewer. A Blender render of the following asset "
        f"is shown in the image.\n\n"
        f"{_view_note}"
        f"User prompt: \"{user_prompt}\"\n"
        f"Expected parts: {', '.join(str(p) for p in parts[:16])}\n"
        f"Critical constraints: {'; '.join(str(c) for c in constraints)}\n\n"
        "Rate the render on a scale from 0 to 10 (10 = the asset perfectly matches the "
        "prompt with all major parts visible and correctly assembled).\n\n"
        "Return strict JSON only with these keys:\n"
        "  visual_score  (integer 0-10)\n"
        "  visible_parts (array of strings — parts clearly present in the render)\n"
        "  missing_parts (array of strings — requested parts that seem absent or wrong)\n"
        "  feedback      (string — one concise sentence, max 20 words)\n"
        "Return ONLY the JSON object, no other text."
    )

    for model in GEMINI_VISION_MODELS:
        raw = _gemini_generate_multimodal(model, vision_prompt, preview_image, "visual_qa")
        if not raw:
            continue
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            continue
        try:
            return {
                "visual_score":   int(parsed.get("visual_score", 0)),
                "visible_parts":  list(parsed.get("visible_parts") or []),
                "missing_parts":  list(parsed.get("missing_parts") or []),
                "feedback":       str(parsed.get("feedback", "")),
                "skipped":        False,
            }
        except (TypeError, ValueError):
            continue

    for _gh_model in ("gpt-4o", "openai/gpt-4.1"):
        raw = _github_vision_chat(vision_prompt, preview_image, "visual_qa", model=_gh_model)
        if raw:
            break
    if raw:
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            try:
                return {
                    "visual_score":  int(parsed.get("visual_score", 0)),
                    "visible_parts": list(parsed.get("visible_parts") or []),
                    "missing_parts": list(parsed.get("missing_parts") or []),
                    "feedback":      str(parsed.get("feedback", "")),
                    "skipped":       False,
                }
            except (TypeError, ValueError):
                pass

    logger.warning("[VISUAL_QA] All vision models failed — skipping visual score")
    return {"visual_score": -1, "visible_parts": [], "missing_parts": [], "feedback": "", "skipped": True}


def generate_visual_repair(
    script: str,
    user_prompt: str,
    part_data: dict,
    visual_feedback: dict,
    render_image: "Optional[bytes]" = None,
    reference_image: "Optional[bytes]" = None,
) -> str:
    """T1.1 — regenerate an improved script driven by visual-QA feedback.

    VISION-GROUNDED: when render_image is supplied, the model SEES the broken
    render (and optionally the reference) and rewrites the code to fix the
    specific geometric errors it can observe — far more effective than reading
    a text score. Falls back to text-only repair when no image / vision is up.
    """
    missing = ", ".join(str(p) for p in (visual_feedback.get("missing_parts") or [])[:10])
    feedback = str(visual_feedback.get("feedback", "")).strip()
    score = visual_feedback.get("visual_score", "?")
    constraints = "; ".join(str(c) for c in (part_data.get("critical_constraints") or [])[:6])

    # Rewrites used to silently swap image-texture materials for flat RGB, because
    # nothing in this prompt mentioned materials. A shape fix then out-scored the
    # textured original (the vision rubric grades geometry, not materials) and the
    # asset shipped untextured. Re-state the resolved maps so they survive a rewrite.
    _tex_paths = (part_data or {}).get("poly_haven_texture_paths") or {}
    _tex_lines: list[str] = []
    if _tex_paths:
        _tex_lines.append(
            "- MATERIALS ARE ALREADY CORRECT — do NOT replace image-texture materials "
            "with plain Base Color values. Keep every ShaderNodeTexImage setup intact."
        )
        _tex_lines.append(
            "  These real PBR textures are downloaded for this asset; if you rebuild a "
            "material, re-load them via ShaderNodeTexImage (wrap each load in "
            "try/except so a missing file can never crash the build):"
        )
        for _mat_name, _maps in _tex_paths.items():
            _tex_lines.append(f"    Material purpose: {_mat_name}")
            for _map_key, _path in (_maps or {}).items():
                _tex_lines.append(f"      {_map_key}: r\"{_path}\"")

    instructions = [
        f"A render of this asset scored {score}/10 against the prompt.",
        f"Reviewer feedback: {feedback}" if feedback else "",
        f"Parts that appear MISSING or wrong: {missing}" if missing else "",
        f"Must satisfy: {constraints}" if constraints else "",
        "",
        "Improve the script so the rendered asset better matches the prompt:",
        "- ADD the missing parts as properly placed, named, parented mesh objects.",
        "- Fix any blocky/flat silhouette by bevelling, tapering, or rounding.",
        "- Re-attach any part that looks floating so it visibly touches its parent.",
        "- Keep everything that already works; do not regress correct parts.",
        *_tex_lines,
        "- Preserve the exact import order, version assert, scene-init, seed, and",
        "  the verbatim export block. Return ONLY executable Python, no markdown.",
    ]
    guidance = "\n".join(s for s in instructions if s != "")

    repair_prompt = (
        "You are a Blender 4.1+ Python expert improving an existing asset script "
        "based on visual feedback from a render.\n\n"
        f"User prompt: {user_prompt}\n\n"
        f"{guidance}\n\n"
        "=== CURRENT SCRIPT ===\n"
        f"{script}\n"
    )

    if render_image:
        images = [render_image]
        vision_prompt = (
            "You are a Blender 4.1+ Python expert. The FIRST image is a 2x2 grid "
            "showing FOUR angles of the 3D asset your current script produced. "
            + ("The SECOND image is the intended reference for what it SHOULD look like.\n\n"
               if reference_image else "\n\n")
            + f"User wanted: {user_prompt}\n\n"
            "STUDY THE RENDER CAREFULLY. Identify exactly what is wrong with the "
            "geometry you see — wrong proportions, missing parts, floating/scattered "
            "pieces, parts that are flat boxes where they should be shaped, wrong "
            "scale, inverted/black faces. Then REWRITE the entire script to fix "
            "those specific visible problems.\n\n"
            f"{guidance}\n\n"
            "=== CURRENT SCRIPT (produced the render you see) ===\n"
            f"{script}\n"
        )
        if reference_image:
            images.append(reference_image)
        raw = _gemini_multimodal_code(vision_prompt, images, "visual_repair_vision",
                                      thinking_budget=6000, max_output_tokens=32000)
        if raw:
            improved = sanitize_generated_code(_extract_code(raw))
            valid, _reason = validate_generated_script(improved)
            if valid:
                logger.info("[VISUAL_REPAIR] VISION-GROUNDED Gemini fix succeeded")
                return improved
            logger.warning("[VISUAL_REPAIR] vision fix failed validation: %s", _reason)
        raw = _github_vision_chat(vision_prompt, render_image, "visual_repair_vision", model="gpt-4o")
        if raw:
            improved = sanitize_generated_code(_extract_code(raw))
            valid, _reason = validate_generated_script(improved)
            if valid:
                logger.info("[VISUAL_REPAIR] VISION-GROUNDED GPT-4o fix succeeded")
                return improved

    _vr_messages = [
        {"role": "system", "content": "Return only executable Blender Python code. No markdown."},
        {"role": "user", "content": repair_prompt},
    ]
    for _gh_m in ("gpt-4.1", "gpt-4o"):
        raw = _github_chat(_gh_m, _vr_messages, "visual_repair",
                           temperature=0.2, max_tokens=20000)
        if not raw: continue
        improved = sanitize_generated_code(_extract_code(raw))
        valid, _ = validate_generated_script(improved)
        if valid:
            logger.info("[VISUAL_REPAIR] github/%s text fix succeeded", _gh_m)
            return improved
    if _deepseek_key():
        raw = _deepseek_chat("deepseek-coder", _vr_messages, "visual_repair",
                             temperature=0.2, max_tokens=20000)
        if raw:
            improved = sanitize_generated_code(_extract_code(raw))
            valid, _ = validate_generated_script(improved)
            if valid:
                logger.info("[VISUAL_REPAIR] deepseek succeeded")
                return improved

    for _attempt in range(2):
        raw = _gemini_generate_code(repair_prompt, "visual_repair")
        if not raw:
            break
        improved = sanitize_generated_code(_extract_code(raw))
        valid, _reason = validate_generated_script(improved)
        if valid:
            logger.info("[VISUAL_REPAIR] Gemini text fix succeeded")
            return improved

    _messages = [
        {"role": "system", "content": "Return only executable Blender Python code. No markdown."},
        {"role": "user", "content": repair_prompt},
    ]

    for model in GROQ_GENERATION_MODELS:
        raw = _groq_chat(model, _messages, "visual_repair", temperature=0.2, max_tokens=16000)
        if not raw:
            continue
        improved = sanitize_generated_code(_extract_code(raw))
        valid, _reason = validate_generated_script(improved)
        if valid:
            logger.info("[VISUAL_REPAIR] Groq produced improved script (model=%s)", model)
            return improved

    return ""


def generate_spec_repair(
    spec: dict,
    user_prompt: str,
    part_data: dict,
    visual_feedback: dict,
    render_image: "Optional[bytes]" = None,
    reference_image: "Optional[bytes]" = None,
) -> "Optional[dict]":
    """JSON-patch visual repair for compiler-built assets.

    Instead of asking a model to rewrite a whole Blender script (the hardest
    possible repair task), the vision model emits a SMALL edit list against the
    build spec; core.spec applies + re-validates it and the deterministic
    compiler rebuilds. The output can never contain a syntax error.

    Returns the new validated spec dict, or None when no usable edits came back.
    """
    from core.spec import apply_spec_edits

    missing = ", ".join(str(p) for p in (visual_feedback.get("missing_parts") or [])[:10])
    feedback = str(visual_feedback.get("feedback", "")).strip()
    score = visual_feedback.get("visual_score", "?")

    spec_json = json.dumps(
        {"name": spec.get("name"), "materials": spec.get("materials"),
         "parts": spec.get("parts")},
        indent=1,
    )

    edit_contract = (
        "Return strict JSON only: {\"edits\": [...]} using ONLY these edit forms:\n"
        "  {\"id\": \"<part id>\", \"set\": {\"pos\": [x,y,z], \"radius\": r, ...}}  — change part fields\n"
        "  {\"id\": \"<part id>\", \"delete\": true}                              — remove a wrong part\n"
        "  {\"add\": {\"id\": \"new_part\", \"primitive\": \"box|cylinder|sphere|wheel|bolt|panel|lathe|tube|ring\", "
        "\"pos\": [x,y,z], <dimensions>, \"material\": \"<palette key>\", \"attach_to\": \"<parent id>\"}}\n"
        "  {\"material\": \"<name>\", \"set\": {\"color\": [r,g,b], \"roughness\": 0-1, \"metallic\": 0-1}}\n"
        "Settable part fields: pos, rot, size, radius, depth, height, width, length, "
        "r_top, axis, segments, profile, path, thickness, bevel, smooth, crease, "
        "taper, taper_axis, material, attach_to, mirror, array.\n"
        "  smooth (box only, 1-3) + crease (0-1): subdivision smoothing — if a part "
        "should look soft/cushioned/rounded set smooth 2 with crease 0-0.2; crease "
        "near 1 keeps edges sharp. bevel (box only, metres): small chamfer for "
        "crisp hard-surface edges.\n"
        "  If the feedback says the shape is blocky, boxy, flat or slab-like, the "
        "fix is usually taper (box, 0.8-0.95) or r_top (cylinder, 0.6-0.9 x radius) "
        "rather than moving parts around — a perfect prism almost never matches a "
        "real object's silhouette.\n"
        "Note: 'params' is internal — set the dimension fields directly (e.g. "
        "{\"set\": {\"size\": [1,2,3]}}, never {\"set\": {\"params\": ...}}).\n"
        "Keep the edit list SMALL and targeted (1-8 edits). Distances in metres, Z up. "
        "Return ONLY the JSON object."
    )

    guidance = (
        f"The rendered asset scored {score}/10 against the prompt.\n"
        + (f"Reviewer feedback: {feedback}\n" if feedback else "")
        + (f"Parts that appear MISSING or wrong: {missing}\n" if missing else "")
        + "\nFix ONLY what is visibly wrong: resize/move misproportioned parts, "
          "delete stray geometry, add genuinely missing parts, adjust materials "
          "that read as the wrong colour/finish. Attached parts are auto-snapped "
          "into contact after your edits, so approximate positions are fine.\n\n"
    )

    repair_prompt = (
        "You are fixing a parametric 3D asset by editing its build spec (JSON), "
        "not by writing code.\n\n"
        f"User wanted: {user_prompt}\n\n"
        + guidance
        + "=== CURRENT BUILD SPEC ===\n"
        + spec_json
        + "\n\n"
        + edit_contract
    )

    def _try_edits(raw: str, source: str) -> "Optional[dict]":
        parsed = _extract_json(raw)
        if not isinstance(parsed, dict):
            return None
        edits = parsed.get("edits")
        if not isinstance(edits, list) or not edits:
            return None
        report = apply_spec_edits(spec, edits)
        if report.ok:
            logger.info("[SPEC_REPAIR] %s applied %d edit(s)", source, len(edits))
            for w in report.warnings[:6]:
                logger.info("[SPEC_REPAIR]   %s", w)
            return report.spec
        logger.warning("[SPEC_REPAIR] %s edits rejected: %s",
                       source, "; ".join(report.errors))
        return None

    if render_image:
        vision_prompt = (
            "The FIRST image is a 2x2 grid showing FOUR angles of the 3D asset "
            "built from the spec below. "
            + ("The SECOND image is the intended reference look.\n\n"
               if reference_image else "\n\n")
            + repair_prompt
        )
        _repair_images = [render_image] + ([reference_image] if reference_image else [])
        for model in GEMINI_VISION_MODELS:
            raw = _gemini_generate_multimodal(model, vision_prompt, _repair_images,
                                              "spec_repair", temperature=0.2,
                                              json_mode=True)
            if not raw:
                continue
            result = _try_edits(raw, f"gemini/{model}")
            if result:
                return result
        raw = _github_vision_chat(vision_prompt, render_image, "spec_repair",
                                  model="gpt-4o")
        if raw:
            result = _try_edits(raw, "github/gpt-4o-vision")
            if result:
                return result

    _messages = [
        {"role": "system", "content": "Return only valid JSON. No markdown."},
        {"role": "user", "content": repair_prompt},
    ]
    for _gh_m in ("gpt-4.1", "gpt-4o"):
        raw = _github_chat(_gh_m, _messages, "spec_repair",
                           temperature=0.2, max_tokens=8000)
        if not raw:
            continue
        result = _try_edits(raw, f"github/{_gh_m}")
        if result:
            return result

    logger.warning("[SPEC_REPAIR] no usable edit list produced")
    return None


def generate_fresh_from_critique(
    user_prompt: str,
    part_data: dict,
    generation_prompt: str,
    visual_feedback: dict,
    render_image: "Optional[bytes]" = None,
    reference_image: "Optional[bytes]" = None,
) -> str:
    """FULL REGENERATION driven by visual critique.

    Unlike generate_visual_repair (which patches an existing script),
    this STARTS FRESH — throws away the broken script entirely and
    writes a new one with the visual failures injected as hard constraints.
    Used when repair loops have exhausted without reaching the target score.
    """
    missing  = ", ".join(str(p) for p in (visual_feedback.get("missing_parts") or [])[:12])
    feedback = str(visual_feedback.get("feedback", "")).strip()
    score    = visual_feedback.get("visual_score", "?")

    critique_prefix = (
        "CRITICAL VISUAL FEEDBACK FROM PREVIOUS ATTEMPT:\n"
        f"The last generated asset scored only {score}/10. DO NOT repeat the same mistakes.\n"
    )
    if feedback:
        critique_prefix += f"Reviewer saw: {feedback}\n"
    if missing:
        critique_prefix += f"These parts were MISSING or wrong: {missing}\n"
    critique_prefix += (
        "\nStart completely fresh. Do NOT copy the old script's approach.\n"
        "Use SubdivisionSurface, Mirror, Array, or from_pydata if primitives alone "
        "cannot produce the correct shape. The render must be recognisable.\n\n"
    )

    augmented_prompt = critique_prefix + generation_prompt

    if render_image:
        vision_regen_prompt = (
            "You are a Blender 4.1+ Python expert writing a FRESH script.\n\n"
            "The FIRST image shows what a PREVIOUS attempt built — it was WRONG.\n"
            + ("The SECOND image shows the REFERENCE of what it should look like.\n\n"
               if reference_image else "\n")
            + critique_prefix
            + "Write a completely new script that fixes these problems.\n"
            "Return ONLY executable Python code. No markdown.\n\n"
            + generation_prompt
        )
        images = [render_image] + ([reference_image] if reference_image else [])
        raw = _gemini_multimodal_code(vision_regen_prompt, images, "fresh_regen",
                                      thinking_budget=8000, max_output_tokens=32000)
        if raw:
            code = sanitize_generated_code(_extract_code(raw))
            valid, _ = validate_generated_script(code)
            if valid:
                logger.info("[FRESH_REGEN] Gemini vision-grounded fresh script OK")
                return code

    _msgs = [
        {"role": "system", "content": "Return only executable Blender Python code. No markdown."},
        {"role": "user", "content": augmented_prompt},
    ]
    for _gh_m in ("gpt-4.1", "gpt-4o"):
        raw = _github_chat(_gh_m, _msgs, "fresh_regen",
                           temperature=0.25, max_tokens=20000)
        if not raw: continue
        code = sanitize_generated_code(_extract_code(raw))
        valid, _ = validate_generated_script(code)
        if valid:
            logger.info("[FRESH_REGEN] github/%s fresh script OK", _gh_m)
            return code

    raw = _gemini_generate_code(augmented_prompt, "fresh_regen")
    if raw:
        code = sanitize_generated_code(_extract_code(raw))
        valid, _ = validate_generated_script(code)
        if valid:
            logger.info("[FRESH_REGEN] Gemini text fresh script OK")
            return code
    return ""


def generate_3d_from_image(image_bytes: bytes) -> "Optional[bytes]":
    """Try HuggingFace image-to-3D models and return GLB bytes, or None."""
    hf_key = _provider_secret("huggingface", "HF_API_KEY")
    if not hf_key or not image_bytes:
        return None

    _I2_3D_MODELS = [
        "stabilityai/TripoSR",
        "JeffreyXiang/TRELLIS-image-large",
    ]
    headers = {"Authorization": f"Bearer {hf_key}"}

    for model_id in _I2_3D_MODELS:
        url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
        t0 = time.monotonic()
        try:
            res = requests.post(
                url, headers=headers,
                data=image_bytes,
                timeout=120,
            )
            if res.status_code == 503:
                logger.debug("[I2_3D] %s loading", model_id)
                continue
            if res.status_code == 403:
                logger.warning("[I2_3D] %s — HF token lacks Inference permission", model_id)
                continue
            if res.status_code != 200:
                logger.debug("[I2_3D] %s HTTP %d", model_id, res.status_code)
                continue
            content = res.content
            if len(content) < 1024:
                continue
            logger.info("[I2_3D] %s → %.1f KB in %.1fs",
                        model_id, len(content)/1024, time.monotonic()-t0)
            return content
        except Exception as exc:
            logger.debug("[I2_3D] %s failed: %s", model_id, exc)
    return None
