"""The one file that decides what text reaches the image model.

`prompting.toml` at the project root holds every prompt fragment, the tag
order, the shot pools, the negatives and the portrait template. This module
loads it, validates it, and hands it out. Nothing here invents a fragment of
its own — if a string is missing from the file, that is an error the operator
should see, not something to paper over with a default.

Two properties matter for the way it gets used, which is a person editing the
file while watching images come out:

* **Per-checkpoint overrides.** `[models."file.safetensors"]` may override the
  family, slot order, quality stack, negatives and the generation settings, so
  switching models is a data change rather than a code change.
* **Hot reload.** The file is re-read whenever its mtime changes, so the next
  image uses the new settings. No restart, no lost session.
* **A bad edit must not end a story.** A TOML syntax error or a missing key
  keeps the last good version in force; the error is recorded and surfaced at
  `/api/debug` and in the image lab, rather than raised into a running turn.
"""

from __future__ import annotations

import threading
import time
import tomllib
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "prompting.toml"

# Slot names `order` may contain. Anything else is a typo, and a typo that
# silently dropped half the prompt would be very hard to see in an image.
KNOWN_SLOTS = ("quality", "count", "characters", "keywords", "shot", "style",
               "safety")
RATING_KEYS = ("all", "13+", "18+")

# Keys a [models."file.safetensors"] table may set. Anything else is a typo,
# and a typo here would silently keep the old setting on every image.
MODEL_KEYS = ("label", "arch", "family", "order", "quality", "negative_extra",
              "steps", "cfg", "sampler", "scheduler", "sampling",
              "width", "height", "resolutions", "notes",
              # UNET-loader architectures (Anima, FLUX) name their own text
              # encoder and VAE; a checkpoint ships all three in one file.
              "clip", "vae", "clip_type", "weight_dtype", "latent_node",
              # Optional second pass. hires_scale > 1 turns it on.
              "hires_scale", "hires_steps", "hires_denoise", "hires_method",
              "hires_model", "max_subjects")

_lock = threading.Lock()
_cache: dict | None = None
_mtime: float = 0.0
_loaded_at: float = 0.0
_error: str = ""


class BookError(ValueError):
    """The file parsed but does not describe a usable prompt."""


def _validate(d: dict) -> None:
    for sect in ("family", "booru", "prose", "negative", "safety", "portrait",
                 "sanitize"):
        if sect not in d:
            raise BookError(f"missing [{sect}]")

    mode = d["family"].get("mode", "auto")
    if mode not in ("auto", "booru", "prose"):
        raise BookError(f"family.mode must be auto|booru|prose, got {mode!r}")

    for fam in ("booru", "prose"):
        order = d[fam].get("order")
        if not isinstance(order, list) or not order:
            raise BookError(f"[{fam}].order must be a non-empty list")
        unknown = [s for s in order if s not in KNOWN_SLOTS]
        if unknown:
            raise BookError(f"[{fam}].order has unknown slot(s) {unknown}; "
                            f"known: {list(KNOWN_SLOTS)}")
        if int(d[fam].get("max_subjects", 0)) < 1:
            raise BookError(f"[{fam}].max_subjects must be >= 1")
        if not isinstance(d[fam].get("shots"), list) or not d[fam]["shots"]:
            raise BookError(f"[{fam}].shots must be a non-empty list")

    if "quality" not in d["booru"]:
        raise BookError("[booru].quality is required (use \"\" for none)")
    if not isinstance(d["prose"].get("quality_by_variant"), dict):
        raise BookError("[prose.quality_by_variant] must be a table")

    for sect in ("negative", "safety"):
        missing = [r for r in RATING_KEYS if r not in d[sect]]
        if missing:
            raise BookError(f"[{sect}] missing rating(s) {missing}")
    if "baseline" not in d["negative"]:
        raise BookError("[negative].baseline is required (use \"\" for none)")

    def _check_sizes(where: str, sizes: dict) -> None:
        for shape, wh in sizes.items():
            if (not isinstance(wh, list) or len(wh) != 2
                    or not all(isinstance(v, int) and v > 0 for v in wh)):
                raise BookError(f"{where}.{shape} must be [width, height]")

    res = d.get("resolutions") or {}
    _check_sizes("[resolutions.sizes]", res.get("sizes") or {})
    if res.get("default") and res["default"] not in (res.get("sizes") or {}):
        raise BookError(f'[resolutions].default = {res["default"]!r} is not a '
                        f'listed shape')

    for name, m in (d.get("models") or {}).items():
        if not isinstance(m, dict):
            raise BookError(f'[models."{name}"] must be a table')
        unknown = [k for k in m if k not in MODEL_KEYS]
        if unknown:
            raise BookError(f'[models."{name}"] has unknown key(s) {unknown}; '
                            f"known: {list(MODEL_KEYS)}")
        if "family" in m and m["family"] not in ("booru", "prose"):
            raise BookError(f'[models."{name}"].family must be booru|prose')
        bad = [s for s in m.get("order", []) if s not in KNOWN_SLOTS]
        if bad:
            raise BookError(f'[models."{name}"].order has unknown slot(s) {bad}')
        _check_sizes(f'[models."{name}".resolutions]', m.get("resolutions") or {})

    tmpl = d["portrait"].get("template", "")
    for ph in ("{appearance}", "{style}"):
        if ph not in tmpl:
            raise BookError(f"[portrait].template must contain {ph}")


def load(force: bool = False) -> dict:
    """Current settings, re-read when the file changes on disk."""
    global _cache, _mtime, _loaded_at, _error
    with _lock:
        try:
            m = PATH.stat().st_mtime
        except OSError as e:
            if _cache is None:
                raise BookError(f"{PATH} is missing: {e}") from e
            _error = f"{PATH.name} disappeared; using the copy loaded earlier"
            return _cache

        if _cache is not None and not force and m == _mtime:
            return _cache

        try:
            parsed = tomllib.loads(PATH.read_text())
            _validate(parsed)
        except Exception as e:  # noqa: BLE001 — a bad edit must not end a turn
            _error = f"{type(e).__name__}: {e}"
            if _cache is None:
                raise
            print(f"  prompting.toml rejected ({_error}) — keeping the last "
                  f"good version")
            _mtime = m          # do not re-report the same bad edit every image
            return _cache

        _cache, _mtime, _loaded_at, _error = parsed, m, time.time(), ""
        return _cache


def status() -> dict:
    """For /api/debug and the lab page: is the file live, and is it valid?"""
    try:
        load()
    except Exception as e:  # noqa: BLE001
        return {"path": str(PATH), "ok": False, "error": str(e)}
    return {
        "path": str(PATH),
        "ok": not _error,
        "error": _error,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_mtime)),
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_loaded_at)),
    }


# ------------------------------------------------------------------ accessors

def model(checkpoint: str = "") -> dict:
    """The [models."<file>"] table for this checkpoint, or {}."""
    return dict((load().get("models") or {}).get(checkpoint or "", {}))


def models() -> dict:
    """Every checkpoint this file has an opinion about."""
    return dict(load().get("models") or {})


def family(arch: str = "", checkpoint: str = "") -> str:
    """'booru' or 'prose': the model's own table first, then [family]."""
    d = load()["family"]
    mode = d.get("mode", "auto")
    if mode != "auto":
        return mode
    declared = model(checkpoint).get("family")
    if declared:
        return declared
    low = f"{arch} {checkpoint}".lower()
    if arch in d.get("booru_arch", []):
        return "booru"
    return "booru" if any(k in low for k in d.get("booru_keywords", [])) else "prose"


def section(fam: str) -> dict:
    return load()[fam]


def shots(fam: str) -> list[str]:
    return list(load()[fam]["shots"])


def max_subjects(fam: str, checkpoint: str = "") -> int:
    """How many entities may contribute their appearance — the model's own
    budget if it states one, else the family's."""
    own = model(checkpoint).get("max_subjects")
    return int(own if own is not None else load()[fam]["max_subjects"])


def order(fam: str, checkpoint: str = "") -> list[str]:
    """Slot order — the model's own, else the family's."""
    return list(model(checkpoint).get("order") or load()[fam]["order"])


def quality(fam: str, checkpoint: str = "") -> str:
    """The quality stack: the model's own, else the family's.

    Worth knowing per checkpoint: Hassaku was not trained on danbooru metadata,
    so "highres"/"absurdres" are noise to it, while a Pony-lineage model wants
    the score ladder instead. That is exactly what a model table is for.
    """
    own = model(checkpoint).get("quality")
    if own is not None:
        return own
    if fam == "booru":
        return load()["booru"].get("quality", "")
    low = (checkpoint or "").lower()
    for key, val in load()["prose"]["quality_by_variant"].items():
        if key in low:
            return val
    return ""


def count_tags(fam: str) -> dict:
    return dict(load()[fam].get("count") or {})


def negative(rating: str, extra: str = "", checkpoint: str = "") -> str:
    """The rating's line + the story's extra + the model's extra + baseline.

    `rating` must already be normalised (ratings.normalise); this module stays
    independent of the rating vocabulary so there is no import cycle.
    """
    d = load()["negative"]
    parts = [d.get(rating, "")]
    if d.get("use_story_extra", True) and extra.strip():
        parts.append(extra.strip())
    parts.append(model(checkpoint).get("negative_extra", ""))
    parts.append(d.get("baseline", ""))
    return ", ".join(p.strip().strip(",") for p in parts if p and p.strip())


def safety(rating: str) -> str:
    return load()["safety"].get(rating, "")


def portrait_template() -> str:
    return load()["portrait"]["template"]


def sanitize(fam: str = "") -> dict:
    """Sanitising flags, with the family's own override applied.

    Weight syntax is the one that genuinely differs: CLIP reads `(tag:1.2)` as
    attention, a Qwen-3 encoder reads it as punctuation.
    """
    d = load()["sanitize"]
    out = {"strip_weights": bool(d.get("strip_weights", True)),
           "dedupe": bool(d.get("dedupe", True)),
           "dedupe_negative": bool(d.get("dedupe_negative", False))}
    if fam and "strip_weights" in (load().get(fam) or {}):
        out["strip_weights"] = bool(load()[fam]["strip_weights"])
    return out


def resolutions(checkpoint: str = "") -> dict[str, tuple[int, int]]:
    """Frame shapes available to a directive, for this checkpoint."""
    sizes = model(checkpoint).get("resolutions")
    if not sizes:
        sizes = (load().get("resolutions") or {}).get("sizes") or {}
    return {k: (int(v[0]), int(v[1])) for k, v in sizes.items()}


def default_shape() -> str:
    return (load().get("resolutions") or {}).get("default", "")
