# ARGUS

ARGUS turns a text prompt into a game-ready 3D asset. An LLM plans the object as a
structured part list, generates the Blender Python needed to build it, renders the
result from multiple angles, scores it against a visual quality bar, and repairs or
regenerates the parts that fail — autonomously, without a human in the loop.

## How it works

1. **Plan** — an LLM (Gemini / DeepSeek, with Groq as a fast fallback) breaks the
   prompt into a structured part list and scene graph.
2. **Build** — a deterministic scene-graph compiler turns that spec into a Blender
   scene; LLM-generated Blender Python is the fallback path for cases the compiler
   can't cover.
3. **Render & score** — a multi-view render grid is scored for visual quality.
4. **Repair** — failing parts get a targeted JSON-patch repair or a fresh regenerate,
   looped until the asset clears the quality bar.
5. **Finish** — UV unwrap, weighted normals, poly-budget decimation, and material
   texturing/export (KHR-compliant `.glb`) run as a final pass.

## Stack

- **Core pipeline**: Python, Blender Python API, LLM APIs (Gemini / DeepSeek / Groq)
- **Desktop app**: Tauri + React + Three.js (`app/`)
- **Backend service**: FastAPI, streaming pipeline events over SSE (`service/`)
- **Eval harness**: benchmark, ablation runner, and quality-metric reporting (`eval/`)

## Requirements

- Python 3.11+
- [Blender](https://www.blender.org/) on your `PATH` (or set `BLENDER_PATH`)
- API keys for at least one of Gemini, DeepSeek, or Groq

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` in the project root with your LLM API key(s) — see `core/config.py`
for the exact variable names ARGUS reads.

### Run the desktop app

```bash
run_app.bat
```

This starts the FastAPI backend and the Tauri + React desktop UI.

### Run from the CLI

```bash
python main.py "a weathered wooden crate with rope handles"
```

## Project layout

```
core/       generation pipeline: planning, scene-graph compiler, LLM codegen,
            visual QA, repair loop
service/    FastAPI backend + structured SSE pipeline events for the desktop UI
app/        Tauri + React + Three.js desktop frontend
eval/       benchmark / ablation / quality-report harness
tests/      test suite (runs without Blender or network access)
docs/       release checklist and design notes
```

## Status

Actively developed. See `docs/PUBLISHING.md` for the release checklist.
