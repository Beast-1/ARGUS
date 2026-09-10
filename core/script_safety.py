"""Safety checks for LLM-generated Blender Python before it's executed.

This is the ONLY thing standing between a language model's output and a real
`exec()` call — see core/blender.py:run_blender() (Blender subprocess) and
core/mcp_server.py:execute_blender_code() (a live, persistent Blender GUI
process). It was previously a two-substring text scan for "subprocess." and
"__import__(" — `import subprocess as sp` (no trailing dot after the module
name at the point being matched) or `__import__ ("os")` (a space before the
paren) both defeated it silently, and it never looked for os.system, eval,
socket, or shutil at all.

check_script_safety() walks the actual AST instead of scanning source text, so
formatting/aliasing/whitespace tricks can't hide a call: Import/ImportFrom nodes
are checked against a module denylist, Call nodes against both a bare-name
denylist and an (object, attribute) denylist for the dangerous os.* functions,
and Attribute access against os.environ specifically. This is a denylist, not a
sandbox — it stops the categories of danger a compromised or manipulated LLM
response could realistically emit (process execution, network egress,
filesystem deletion, credential/env access), not a hostile human adversary with
unlimited creativity. Treat it accordingly in any threat model that includes the
latter.
"""
from __future__ import annotations

import ast
import re
import textwrap

_DENIED_MODULES = {
    "subprocess", "socket", "shutil", "ctypes", "urllib", "http",
    "requests", "ftplib", "smtplib", "telnetlib", "webbrowser",
    "multiprocessing", "asyncio",
}

_DENIED_CALLS = {"eval", "exec", "compile", "__import__"}

_DENIED_ATTR_CALLS = {
    ("os", "system"), ("os", "popen"), ("os", "popen2"), ("os", "popen3"),
    ("os", "popen4"), ("os", "remove"), ("os", "rmdir"), ("os", "removedirs"),
    ("os", "unlink"), ("os", "rename"), ("os", "replace"), ("os", "truncate"),
    ("os", "execv"), ("os", "execve"), ("os", "execl"), ("os", "execle"),
    ("os", "execlp"), ("os", "execlpe"), ("os", "execvp"), ("os", "execvpe"),
    ("os", "spawnl"), ("os", "spawnv"), ("os", "spawnle"), ("os", "spawnve"),
    ("os", "fork"), ("os", "forkpty"), ("os", "kill"), ("os", "killpg"),
}

_DENIED_ATTR_ACCESS = {("os", "environ")}


def sanitize_generated_code(code: str) -> str:
    """Strip stray markdown fences and a legacy export kwarg the model
    sometimes hallucinates. Byte-identical to the two independent
    implementations this replaces (core/llm.py's and core/blender.py's, which
    used different regexes for the same job)."""
    lines = code.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if line.lstrip().startswith("```"):
            continue
        line = re.sub(r",?\s*export_selected\s*=\s*(True|False)\s*", "", line)
        cleaned.append(line.rstrip())
    body = textwrap.dedent("\n".join(cleaned)).lstrip("\n").strip()
    return (body + "\n") if body else ""


def validate_python_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        compile(code, "<argus_generated>", "exec")
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc.__class__.__name__}: {exc}"


def check_script_safety(code: str) -> tuple[bool, str]:
    """AST-based denylist. Returns (safe, reason). Assumes code already parses
    (call validate_python_syntax first) — a syntax error here is reported as
    unsafe rather than raising, so callers don't need to order the checks."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc.msg} at line {exc.lineno}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _DENIED_MODULES:
                    return False, f"Forbidden import: {alias.name} (line {node.lineno})"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in _DENIED_MODULES:
                    return False, f"Forbidden import: {node.module} (line {node.lineno})"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DENIED_CALLS:
                return False, f"Forbidden call: {func.id}() (line {node.lineno})"
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if (func.value.id, func.attr) in _DENIED_ATTR_CALLS:
                    return False, (
                        f"Forbidden call: {func.value.id}.{func.attr}() "
                        f"(line {node.lineno})"
                    )
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if (node.value.id, node.attr) in _DENIED_ATTR_ACCESS:
                return False, (
                    f"Forbidden access: {node.value.id}.{node.attr} "
                    f"(line {node.lineno})"
                )

    return True, ""
