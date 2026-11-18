"""Which checkpoint draws the pictures, and the settings that differ per model.

Three things live here:

* **The catalog** — what ComfyUI can actually load, joined with what
  `prompting.toml` knows about each one. A checkpoint with no `[models."…"]`
  table still appears and still works; it just runs on family defaults.
* **The active choice** — persisted in `data/active_model.json` so a restart
  keeps the model the reader picked, while `config.json` stays the shipped
  default rather than something the UI rewrites.
* **The merge** — `image_cfg()` layers the model's table over `config.image`,
  so steps, CFG, sampler, prediction type and resolution follow the
  checkpoint. Switching from NoobAI v1.1 to its v-pred sibling without moving
  `sampling` to `v_prediction` renders uniform grey, which is exactly the kind
  of mistake a per-model table exists to prevent.

Nothing here caches the catalog: ComfyUI is the authority on what is
installed, and a model dropped into the folder should appear without a
restart.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend import prompt_book as book

STATE = Path(__file__).resolve().parent.parent / "data" / "active_model.json"

# Settings a model table may push into the image config.
_CFG_KEYS = ("arch", "steps", "cfg", "sampler", "scheduler", "sampling",
             "width", "height",
             # UNET-loader architectures (Anima, FLUX) name their own encoder
             # and VAE; a checkpoint carries all three in one file.
             "clip", "vae", "clip_type", "weight_dtype", "latent_node",
             "hires_scale", "hires_steps", "hires_denoise", "hires_method",
             "hires_model")


def default(cfg: dict) -> str:
    return cfg["image"].get("checkpoint") or cfg["image"].get("unet") or ""


def active(cfg: dict) -> str:
    """The chosen checkpoint, or config.json's default if none was chosen."""
    try:
        name = json.loads(STATE.read_text()).get("checkpoint", "")
    except Exception:  # noqa: BLE001 — no state file yet, or it is corrupt
        name = ""
    return name or default(cfg)


def set_active(name: str) -> str:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"checkpoint": name}, indent=1))
    return name


def image_cfg(cfg: dict, checkpoint: str | None = None) -> dict:
    """`config.image` with the model's own settings layered on top."""
    name = checkpoint or active(cfg)
    out = dict(cfg["image"])
    out["checkpoint"] = name
    for k, v in book.model(name).items():
        if k in _CFG_KEYS:
            out[k] = v
    # A UNET-loader architecture takes the chosen name as its `unet`; only a
    # single-file checkpoint is addressed as `checkpoint`. Setting both would
    # leave whichever the graph ignores pointing at the wrong model.
    if out.get("arch") not in ("sdxl", "sd15"):
        out["unet"] = name
    return out


def describe(cfg: dict, name: str) -> dict:
    """One row for the switcher."""
    m = book.model(name)
    merged = image_cfg(cfg, name)
    return {
        "name": name,
        "label": m.get("label") or name.rsplit(".", 1)[0],
        "arch": merged.get("arch", ""),
        "family": book.family(merged.get("arch", ""), name),
        "steps": merged.get("steps"),
        "cfg": merged.get("cfg"),
        "sampler": merged.get("sampler"),
        "sampling": merged.get("sampling", "eps"),
        "width": merged.get("width"),
        "height": merged.get("height"),
        "tuned": bool(m),
        "notes": m.get("notes", ""),
    }


async def catalog(cfg: dict, comfy) -> dict:
    """Everything installed, plus which one is active.

    Falls back to the names `prompting.toml` knows if ComfyUI is down, so the
    switcher still renders and still says what it cannot reach.
    """
    try:
        names = await comfy.checkpoints()
        reachable = True
    except Exception:  # noqa: BLE001
        names, reachable = [], False

    if reachable:
        # Diffusion models are offered only when prompting.toml describes them.
        # They cannot run without a declared encoder and VAE anyway, and the
        # folder also holds video models (WAN) that have no business in a
        # story's model switcher.
        try:
            described = set(book.models())
            names += [n for n in await comfy.diffusion_models()
                      if n in described and n not in names]
        except Exception:  # noqa: BLE001 — an older ComfyUI without the node
            pass

    if not names:
        names = sorted(book.models())

    cur = active(cfg)
    if cur and cur not in names:
        names = [cur] + names
    return {
        "active": cur,
        "default": default(cfg),
        "comfy_reachable": reachable,
        "models": [describe(cfg, n) for n in names],
    }
