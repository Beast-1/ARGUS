"""ARGUS desktop app entry point for the packaged (PyInstaller) build.

Frozen builds resolve their working directory before any core import so
.env, out/ and logs/ land in the right place:
  1. ARGUS_HOME env var, if set
  2. an `argus.home` file next to the exe containing a path
  3. the exe's own folder (portable mode)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_home() -> Path:
    if os.environ.get("ARGUS_HOME"):
        return Path(os.environ["ARGUS_HOME"])
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
        marker = base / "argus.home"
        if marker.exists():
            target = Path(marker.read_text(encoding="utf-8").strip())
            if target.is_dir():
                return target
        return base
    return Path(__file__).parent


home = _resolve_home()
os.chdir(home)

from desktop_app import main  # noqa: E402  (cwd must be set before core imports)

if __name__ == "__main__":
    main()
