# StoryForge

Interactive fiction you act in, illustrated as you read. An LLM narrates the story turn by turn and
decides when a scene is worth a picture. ComfyUI draws it on a local GPU, keeping characters'
faces consistent across images.

Built and benchmarked on a single RTX 3060 (12 GB) under WSL2.

## How it works

```
browser ── WebSocket ──> FastAPI backend ──> narrator LLM (Ollama local, or Ollama Cloud)
                              │                 │ emits [[IMG ...]] directives in its prose
                              │                 ▼
                              ├── turn_parser ─> image_prompt ─> ComfyUI workflow ─> image
                              └── SQLite: stories, sessions, turns, images, ratings
```

- **Narrator** (`backend/narrator.py`, `context.py`): builds the prompt from the story premise,
  known entities and recent turns. It enforces player agency (never decides what the player does)
  and an illustration format the model can actually follow.
- **Turn parser** (`turn_parser.py`): pulls image directives and entity references out of the prose
  and tolerates the ways models get the format wrong.
- **Images** (`image_prompt.py`, `prompt_style.py`, `workflow.py`, `comfy_client.py`): turn a
  directive into a prompt in the right dialect for the loaded checkpoint (danbooru tags vs.
  natural language), then build and run the ComfyUI graph. Model profiles live in `prompting.toml`.
- **GPU broker** (`gpu_broker.py`): shares 12 GB between ComfyUI and a local LLM by measuring free
  VRAM, evicting the idle side and spotting foreign jobs.
- **Ratings** (`ratings.py`): each story declares a rating that sets narration limits and image
  negative prompts.
- **Frontend** (`frontend/`): reader, story composer, and an image lab for trying models and
  prompts.

## Benchmarks

`bench/` holds the Phase 0 measurements behind every default in `config.json`: image model and
quantisation, resolution, step count, LLM context size, and GPU sharing. Results and decisions are
in [PHASE0.md](PHASE0.md). The prompt-engineering findings are in
[IMAGE_PROMPTING.md](IMAGE_PROMPTING.md).

## Run

Needs a running ComfyUI (`COMFY_DIR`, default `~/ComfyUI`) on :8188, and Ollama locally or an
`OLLAMA_API_KEY` in `.env` for the cloud narrator.

```bash
pip install -r requirements.txt
./start.sh          # starts ComfyUI if needed, then the app on http://127.0.0.1:8100
./stop.sh
```

Stories are JSON files in `stories/`. `stories/lighthouse.json` is a small example showing the
format: premise, world detail, image style, opening scene and entities (characters and places).

`verify_phase*.py` are end-to-end checks against a running stack.
