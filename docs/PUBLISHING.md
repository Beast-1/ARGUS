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
  Still ~6k lines, and it holds the handcrafted fallback scripts (see below).
- Consolidate the handcrafted fallback scripts onto `core/components.py`. They each
  redefine their own `make_mat`/`box`/`cyl` helpers, and because they hardcode flat
  colours they used to ship untextured assets — worked around with a texture-upgrade
  epilogue rather than fixed at the source.
- Add Blender integration tests that run only when `BLENDER_PATH` is available.
- Sandbox generated Blender Python before accepting untrusted users or prompts.
- Teach the visual scorer to judge material realism. It grades silhouette only;
  selection compensates with a deterministic texture/topology adjustment
  (`_accept_candidate` in `main.py`), but the rubric itself is still material-blind.
- Package the API as a PyInstaller sidecar so the app launches as one process
  instead of `run_app.bat` starting uvicorn and the Tauri shell separately.

### Done

- ~~Split `desktop_app.py`~~ — replaced by the Tauri + React app in `app/`.
- ~~Replace UI stdout parsing with structured pipeline events~~ — `service/pipeline_events.py`
  now emits typed SSE events; the two duplicated stdout scrapers are gone.

