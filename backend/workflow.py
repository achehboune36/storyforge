"""API-format ComfyUI workflow loading and sentinel patching.

game-poc patched workflows by sentinel value rather than node id
(`comfyui_service.py:56`), which is the right instinct — it survives
re-exporting the workflow from ComfyUI as long as the sentinels are re-added,
whereas hardcoded node ids silently target the wrong node.

Two things it did not do, both of which matter here:

1. **Types.** It only ever substituted strings. `KSampler.inputs.seed` must
   remain an int and `cfg` a float, or ComfyUI's `validate_prompt` rejects the
   job with a 400 before anything runs. Each sentinel therefore declares its
   type and the value is coerced.

2. **Completeness.** It silently ignored sentinels it didn't fill, so a typo
   shipped a literal "__SEED__" to the sampler. `patch()` fails loudly instead:
   unknown sentinels, unfilled sentinels and unused values are all errors, and
   they surface at call time rather than as an opaque ComfyUI rejection.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# Sentinel -> the type its value must have in the API payload.
SENTINELS: dict[str, type] = {
    "__CKPT__": str,
    "__PROMPT__": str,
    "__NEGATIVE__": str,
    "__UNET__": str,
    "__CLIP__": str,
    "__VAE__": str,
    "__SAMPLER__": str,
    "__SCHEDULER__": str,
    "__PREFIX__": str,
    "__SEED__": int,
    "__WIDTH__": int,
    "__HEIGHT__": int,
    "__STEPS__": int,
    "__BATCH__": int,
    "__CFG__": float,
    "__DENOISE__": float,
    "__IMAGE__": str,
    "__HIRES_SCALE__": float,
}


class WorkflowError(ValueError):
    """Raised for any sentinel mismatch. Always a programming error, never input."""


def load(path: str | Path) -> dict:
    """Load an API-format workflow (flat {node_id: {class_type, inputs}})."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "nodes" in data:
        raise WorkflowError(
            f"{path} looks like a UI-format workflow (has 'nodes'/'links'). "
            "The API expects the flat {node_id: {class_type, inputs}} form — "
            "export it with 'Save (API Format)' or build it in code."
        )
    for nid, node in data.items():
        if "class_type" not in node or "inputs" not in node:
            raise WorkflowError(f"{path}: node {nid!r} missing class_type/inputs")
    return data


def find_sentinels(workflow: dict) -> set[str]:
    found: set[str] = set()
    for node in workflow.values():
        for val in node.get("inputs", {}).values():
            if isinstance(val, str) and val in SENTINELS:
                found.add(val)
    return found


def patch(workflow: dict, values: dict[str, Any]) -> dict:
    """Return a copy of `workflow` with every sentinel replaced by its value.

    `values` is keyed by sentinel name, e.g. {"__SEED__": 1234}. Every sentinel
    present in the workflow must be supplied, and every supplied value must be
    used — both directions are checked so a rename can't silently no-op.
    """
    unknown = set(values) - set(SENTINELS)
    if unknown:
        raise WorkflowError(f"unknown sentinel(s): {sorted(unknown)}")

    present = find_sentinels(workflow)
    missing = present - set(values)
    if missing:
        raise WorkflowError(f"workflow needs value(s) for: {sorted(missing)}")
    unused = set(values) - present
    if unused:
        raise WorkflowError(
            f"value(s) supplied for sentinel(s) not in the workflow: {sorted(unused)}"
        )

    out = copy.deepcopy(workflow)
    for node in out.values():
        inputs = node.get("inputs", {})
        for key, val in list(inputs.items()):
            if isinstance(val, str) and val in SENTINELS:
                want = SENTINELS[val]
                raw = values[val]
                try:
                    inputs[key] = want(raw)
                except (TypeError, ValueError) as e:
                    raise WorkflowError(
                        f"{val} -> {node.get('class_type')}.{key}: "
                        f"cannot coerce {raw!r} to {want.__name__}"
                    ) from e
    return out


def build(cfg: dict) -> dict:
    """Build the workflow for whichever architecture config.json selects."""
    arch = cfg.get("arch", "klein")
    if arch in ("sdxl", "sd15"):
        # Same node graph: one checkpoint, CLIP text encode, KSampler. SD1.5
        # differs only in what resolution it tolerates, which the model's own
        # table in prompting.toml sets.
        return build_sdxl(cfg)
    if arch == "anima":
        return build_anima(cfg)
    return build_klein(cfg)


def values(cfg: dict, **kw) -> dict[str, Any]:
    """Sentinel values for build(), defaulting from config.json."""
    if cfg.get("arch") in ("sdxl", "sd15"):
        return sdxl_values(cfg, **kw)
    base = klein_values(cfg, **kw)
    if cfg.get("arch") == "anima":
        # Anima has no separate negative-prompt weighting and no batch knob in
        # the graph; drop sentinels the graph does not contain.
        base = {k: v for k, v in base.items()
                if k not in ("__DENOISE__",)}
    return base


def sdxl_values(cfg: dict, *, prompt: str, seed: int, negative: str = "",
                width: int | None = None, height: int | None = None,
                steps: int | None = None, prefix: str = "sf") -> dict[str, Any]:
    """SDXL loads a single checkpoint, so there is no unet/clip/vae triple."""
    return {
        "__CKPT__": cfg["checkpoint"],
        "__PROMPT__": prompt,
        "__NEGATIVE__": negative,
        "__WIDTH__": width if width is not None else cfg["width"],
        "__HEIGHT__": height if height is not None else cfg["height"],
        "__BATCH__": 1,
        "__SEED__": seed,
        "__STEPS__": steps if steps is not None else cfg["steps"],
        "__CFG__": cfg["cfg"],
        "__SAMPLER__": cfg["sampler"],
        "__SCHEDULER__": cfg["scheduler"],
        "__PREFIX__": prefix,
    }


def build_sdxl(cfg: dict, *, reference: str | None = None) -> dict:
    """NoobAI-XL (SDXL). With `reference`, PuLID locks the face to that portrait.

    Character consistency is the point. A fixed seed plus a frozen appearance
    string gets you roughly the same person; PuLID against an actual reference
    portrait gets you *that* person in a new pose, which is what stops the
    reader meeting a stranger every scene.

    The chain follows the user's own working graph:
      PulidModelLoader -> PulidEvaClipLoader -> PulidInsightFaceLoader -> ApplyPulid
    """
    g: dict = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "__CKPT__"}},

        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "__PROMPT__", "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "__NEGATIVE__", "clip": ["1", 1]}},
        "6": {"class_type": "EmptyLatentImage",
              "inputs": {"width": "__WIDTH__", "height": "__HEIGHT__",
                         "batch_size": "__BATCH__"}},
        "7": {"class_type": "KSampler",
              "inputs": {"seed": "__SEED__", "steps": "__STEPS__",
                         "cfg": "__CFG__", "sampler_name": "__SAMPLER__",
                         "scheduler": "__SCHEDULER__", "denoise": 1.0,
                         "model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "__PREFIX__", "images": ["8", 0]}},
    }

    # Prediction type must match the checkpoint. Measured on this machine:
    #   NoobAI-XL-v1.1              + eps           -> correct
    #   NoobAI-XL-v1.1              + v_prediction  -> uniform grey
    #   noobaiXLNAIXL_vPred10Version + v_prediction  -> correct, deeper contrast, ~50% slower
    # The user's saved workflows set v_prediction on v1.1, which is wrong; it
    # was copied from the vPred build. "eps" means omit the node entirely.
    sampling = cfg.get("sampling", "eps")
    model_src = ["1", 0]
    if sampling and sampling != "eps":
        g["2"] = {"class_type": "ModelSamplingDiscrete",
                  "inputs": {"model": ["1", 0], "sampling": sampling, "zsnr": True}}
        model_src = ["2", 0]
    g["7"]["inputs"]["model"] = model_src

    pulid = cfg.get("pulid") or {}
    if reference and pulid.get("enabled"):
        g["10"] = {"class_type": "PulidModelLoader",
                   "inputs": {"pulid_file": pulid["model"]}}
        g["11"] = {"class_type": "PulidEvaClipLoader", "inputs": {}}
        g["12"] = {"class_type": "PulidInsightFaceLoader",
                   "inputs": {"provider": "CUDA"}}
        g["13"] = {"class_type": "LoadImage", "inputs": {"image": reference}}
        g["14"] = {"class_type": "ApplyPulid",
                   "inputs": {"model": model_src, "pulid": ["10", 0],
                              "eva_clip": ["11", 0], "face_analysis": ["12", 0],
                              "image": ["13", 0],
                              "method": pulid.get("method", "fidelity"),
                              "weight": float(pulid.get("weight", 0.9)),
                              "start_at": float(pulid.get("start_at", 0.0)),
                              "end_at": float(pulid.get("end_at", 1.0))}}
        # Sampler now draws from the identity-conditioned model.
        g["7"]["inputs"]["model"] = ["14", 0]
    return g


def build_anima(cfg: dict) -> dict:
    """Anima 2B (anime). Architecture differs from FLUX in three ways:

      * plain `UNETLoader` (safetensors, bf16) — fp8 and GGUF both fail on this
        arch, see config.json image._weight_dtype;
      * the Qwen3-0.6B encoder is detected from its weights, so the CLIPLoader
        `type` is nominal (sd.py:1339 routes it to text_encoders.anima);
      * Wan21 latent format, 16 channels — EmptySD3LatentImage, not
        EmptyFlux2LatentImage.
    """
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "__UNET__",
                         "weight_dtype": cfg.get("weight_dtype", "default")}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "__CLIP__",
                         "type": cfg.get("clip_type", "lumina2")}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "__VAE__"}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "__PROMPT__", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "__NEGATIVE__", "clip": ["2", 0]}},
        "6": {"class_type": cfg.get("latent_node", "EmptySD3LatentImage"),
              "inputs": {"width": "__WIDTH__", "height": "__HEIGHT__",
                         "batch_size": "__BATCH__"}},
        "7": {"class_type": "KSampler",
              "inputs": {"seed": "__SEED__", "steps": "__STEPS__",
                         "cfg": "__CFG__", "sampler_name": "__SAMPLER__",
                         "scheduler": "__SCHEDULER__", "denoise": 1.0,
                         "model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "__PREFIX__", "images": ["8", 0]}},
    }


def build_klein(cfg: dict) -> dict:
    """Build the FLUX.2 Klein workflow in API format, fully sentinelled.

    Kept in code rather than as a checked-in JSON file so the node graph and the
    sentinel set can never drift apart. `write_api_workflow()` dumps it for
    inspection or for loading into ComfyUI's UI.

    Encoder pairing is a hard constraint: Klein 4B needs Qwen3-4B, Klein 9B
    needs Qwen3-8B. A mismatch passes every loader and then fails at KSampler
    with "mat1 and mat2 shapes cannot be multiplied".
    """
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": "__UNET__"}},
        "2": {"class_type": "CLIPLoaderGGUF",
              "inputs": {"clip_name": "__CLIP__", "type": cfg["clip_type"]}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "__VAE__"}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "__PROMPT__", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "__NEGATIVE__", "clip": ["2", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": "__WIDTH__", "height": "__HEIGHT__",
                         "batch_size": "__BATCH__"}},
        "7": {"class_type": "KSampler",
              "inputs": {"seed": "__SEED__", "steps": "__STEPS__",
                         "cfg": "__CFG__", "sampler_name": "__SAMPLER__",
                         "scheduler": "__SCHEDULER__", "denoise": "__DENOISE__",
                         "model": ["1", 0], "positive": ["4", 0],
                         "negative": ["5", 0], "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "__PREFIX__", "images": ["8", 0]}},
    }


def klein_values(cfg: dict, *, prompt: str, seed: int, negative: str = "",
                 width: int | None = None, height: int | None = None,
                 steps: int | None = None, prefix: str = "sf") -> dict[str, Any]:
    """Sentinel values for build_klein(), defaulting from config.json."""
    img = cfg
    return {
        "__UNET__": img["unet"],
        "__CLIP__": img["clip"],
        "__VAE__": img["vae"],
        "__PROMPT__": prompt,
        "__NEGATIVE__": negative,
        "__WIDTH__": width if width is not None else img["width"],
        "__HEIGHT__": height if height is not None else img["height"],
        "__BATCH__": 1,
        "__SEED__": seed,
        "__STEPS__": steps if steps is not None else img["steps"],
        "__CFG__": img["cfg"],
        "__SAMPLER__": img["sampler"],
        "__SCHEDULER__": img["scheduler"],
        "__DENOISE__": 1.0,
        "__PREFIX__": prefix,
    }


def build_hires(cfg: dict) -> dict:
    """The second pass: re-sample an existing image at a larger size.

    Derived from `build(cfg)` rather than written out again, so it inherits
    whichever loaders the architecture needs and can never drift from the
    first pass — the refined image has to come out of the same model, encoder
    and VAE as the one it is replacing.

    The reference workflow upscaled the first pass's *latent*. A second
    ComfyUI job cannot reach that latent, so this re-encodes the saved PNG
    instead: one VAE round trip, invisible next to a 0.48-denoise resample,
    and it means the pass can run minutes later, after a restart, or never.
    """
    g = copy.deepcopy(build(cfg))
    # Whoever supplies the VAE in the base graph supplies it here too: a
    # checkpoint carries its own (slot 2), a UNET graph has a VAELoader.
    vae = (["3", 0] if g.get("3", {}).get("class_type") == "VAELoader"
           else ["1", 2])
    sampler = g["7"]["inputs"]
    g.pop("6", None)                       # the empty latent is not needed
    g["60"] = {"class_type": "LoadImage", "inputs": {"image": "__IMAGE__"}}
    g["61"] = {"class_type": "VAEEncode",
               "inputs": {"pixels": ["60", 0], "vae": vae}}
    g["62"] = {"class_type": "LatentUpscaleBy",
               "inputs": {"samples": ["61", 0], "scale_by": "__HIRES_SCALE__",
                          "upscale_method": cfg.get("hires_method", "nearest-exact")}}
    sampler["latent_image"] = ["62", 0]
    sampler["denoise"] = "__DENOISE__"     # partial: keep the composition
    return g


def hires_values(cfg: dict, *, image: str, prompt: str, seed: int,
                 negative: str = "", prefix: str = "sf_hr") -> dict[str, Any]:
    """Sentinel values for build_hires(). Width/height come from the source
    image, so the base graph's size sentinels are not part of this one."""
    base = values(cfg, prompt=prompt, seed=seed, negative=negative, prefix=prefix,
                  steps=int(cfg.get("hires_steps") or cfg["steps"]))
    base.pop("__WIDTH__", None)
    base.pop("__HEIGHT__", None)
    base.pop("__BATCH__", None)
    base["__IMAGE__"] = image
    base["__HIRES_SCALE__"] = float(cfg.get("hires_scale") or 1.5)
    base["__DENOISE__"] = float(cfg.get("hires_denoise") or 0.48)
    return base


def write_api_workflow(cfg: dict, path: str | Path) -> Path:
    """Dump the sentinelled graph, for inspection and for loading into ComfyUI."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(build_klein(cfg), indent=2) + "\n")
    return p
