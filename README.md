# ARGUS

ARGUS turns a text prompt into a game-ready 3D asset. An LLM plans the object as a
structured part list, generates the Blender Python needed to build it, renders the
result from multiple angles, scores it against a visual quality bar, and repairs or
regenerates the parts that fail — autonomously, without a human in the loop.

## How it works

1. **Plan** — an LLM (Gemini first, falling through DeepSeek, Groq and
   OpenRouter) breaks the prompt into a structured part list and scene graph.
2. **Build** — a deterministic scene-graph compiler turns that spec into a Blender
   scene; LLM-generated Blender Python is the fallback path for cases the compiler
   can't cover.
3. **Render & score** — a multi-view render grid is scored for visual quality.
4. **Repair** — failing parts get a targeted JSON-patch repair or a fresh regenerate,
   looped until the asset clears the quality bar.
5. **Finish** — UV unwrap, weighted normals, material texturing, and export
   (KHR-compliant `.glb`) run as a final pass. The export is held to its poly
   budget: coplanar detail is dissolved first, then collapsed to the ceiling if
   still over. The saved `.blend` keeps full detail — only the shipped `.glb` is
   budgeted. Disable with `ARGUS_POLY_ENFORCE=0`.

## Stack

- **Core pipeline**: Python, Blender Python API, LLM APIs (Gemini / DeepSeek /
  Groq / OpenRouter, tried in that order)
- **Desktop app**: Tauri + React + Three.js (`app/`)
- **Backend service**: FastAPI, streaming pipeline events over SSE (`service/`)
- **Eval harness**: benchmark, ablation runner, and quality-metric reporting (`eval/`)

## Requirements

- Python 3.11+
- [Blender](https://www.blender.org/) on your `PATH` (or set `BLENDER_PATH`)
- An API key for at least one of Gemini, DeepSeek, Groq, or OpenRouter

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in at least one generation provider's key
(Gemini, DeepSeek, Groq, or OpenRouter):

```bash
cp .env.example .env
```

Every variable ARGUS reads is declared in `core/config.py`, grouped and
documented. To see them all with their defaults and whichever values are
currently active:

```bash
python -m core.config
```

### Run the desktop app

```bash
run_app.bat
```

This starts the FastAPI backend and the Tauri + React desktop UI.

### Run from the CLI

```bash
python main.py "a weathered wooden crate with rope handles"
```

### Host the UI on the web (Cloudflare Pages)

The same React app runs either as the Tauri desktop shell or as a hosted page in
a browser. Generation itself cannot be hosted — it spawns Blender as a
subprocess and a single asset takes minutes, so it stays on your machine and the
browser talks to it over a tunnel.

```bash
cd app && npm run deploy          # builds and pushes to Cloudflare Pages
```

Then, on the machine that runs Blender:

```bash
cloudflared tunnel --url http://127.0.0.1:8420
```

Start the backend with the Pages origin allowed, so the browser's CORS
preflight passes:

```bash
set ARGUS_ALLOWED_ORIGINS=https://argus.pages.dev
run_app.bat
```

The backend prints its API token on startup. Open the Pages URL, enter the
tunnel URL and that token once, and it's remembered in `localStorage`.

> **Anyone with the tunnel URL and token can run generations on your machine** —
> which means LLM-authored Python executing locally. The bearer token is a shared
> secret, not identity. Put [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/)
> in front of the tunnel before leaving it up.

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
