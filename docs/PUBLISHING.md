# Publishing ARGUS

This checklist tracks the gap between a local prototype and a publishable release.

## Minimum Release Bar

- Project installs with `pip install -e ".[dev]"`.
- `python -m pytest` passes without Blender or network access.
- `run_desktop.bat` starts the app from a fresh virtual environment.
- A smoke prompt exports a non-empty `.glb` into `out/final/<run_id>/`.
- `.env.example` documents every required runtime setting.
- Generated files, logs, caches, and binaries are ignored by Git.

## Recommended Release Flow

1. Create a clean virtual environment.
2. Install with `pip install -e ".[dev]"`.
3. Run tests.
4. Run one CLI smoke generation.
5. Run one desktop smoke generation.
6. Build a Windows executable with PyInstaller.
7. Test the executable on a machine with Blender installed.

## Known Hardening Work

- Split `core/llm.py` into provider, planner, codegen, repair, and visual QA modules.
- Split `desktop_app.py` into UI, pipeline worker, log parser, and renderer modules.
- Replace UI stdout parsing with structured pipeline events.
- Add Blender integration tests that run only when `BLENDER_PATH` is available.
- Sandbox generated Blender Python before accepting untrusted users or prompts.

