"""Assembles the image prompt from the slots named in `prompting.toml`.

Every fragment — the quality stack, the shot pool, the content clauses, the
order the slots are emitted in — comes from that file, which is the one place
a person tunes while watching images come out. This module only knows *how* to
assemble, never *what* to say.

Two families, because the encoder decides the dialect:

  * **booru** — NoobAI-XL / Illustrious / Pony. CLIP encoder, danbooru tags,
    comma-separated, earlier is stronger. This is what the project runs.
  * **prose** — Anima and friends. A Qwen-3 LLM encoder that reads sentences,
    so the prompt is a hybrid and ends in a full stop.

The narrator only ever produces the middle layer — what is visible in this
shot. Everything structural is decided here, so visual continuity and content
rating are enforced by code rather than by asking a model to remember.
"""

from __future__ import annotations

import re

from backend import prompt_book as book
from backend import ratings

# Weight syntax never reaches a model: a CLIP encoder spends tokens on it and
# an LLM encoder reads it as literal punctuation. The narrator emits it now and
# then regardless of being told not to.
_WEIGHT_RE = re.compile(r"[\(\)\[\]]|:\s*\d+(\.\d+)?(?=\s*[,\)])")

# Entity rows carry no gender column — a story JSON may declare one, but the
# registry drops it. The appearance string does say, though: these are
# danbooru tag strings and they start with their own count tag. Reading it
# back is what stops "1girl, solo, 1boy, black hair…" on a male character.
_MALE_TAG = re.compile(r"\b(\d*boys?|male focus|otoko no ko)\b")
_FEMALE_TAG = re.compile(r"\b(\d*girls?|female focus)\b")


def _gender(e: dict) -> str:
    g = (e.get("gender") or "").strip().lower()
    if g in ("male", "female"):
        return g
    app = (e.get("appearance_tags") or e.get("appearance") or "").lower()
    if _MALE_TAG.search(app):
        return "male"
    if _FEMALE_TAG.search(app):
        return "female"
    return ""


def is_booru(arch: str, checkpoint: str = "") -> bool:
    return book.family(arch, checkpoint) == "booru"


def split_tags(prompt: str) -> list[str]:
    """Split on commas that are NOT inside a weight group.

    `(worst quality, low quality, logo)` is one emphasis group; splitting it on
    its commas would leave unbalanced parentheses behind as soon as dedupe
    dropped one of the pieces.
    """
    out, depth, cur = [], 0, ""
    for ch in prompt:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [s.strip() for s in out if s.strip()]


def strip_weights(text: str, fam: str = "") -> str:
    """Remove SDXL-style emphasis, if this family asks for it."""
    if not book.sanitize(fam)["strip_weights"]:
        return text.strip()
    cleaned = _WEIGHT_RE.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip(" ,")


def dedupe_tags(prompt: str) -> str:
    """Drop repeated tags, keeping first (strongest) position.

    Entity appearance strings start with their own count tag ("1boy, black
    hair, ..."), which would otherwise collide with the computed count and
    emit "1boy, solo, 1boy". Repeats also waste the CLIP token budget, which
    is only 75 tokens per chunk.
    """
    seen: set[str] = set()
    out: list[str] = []
    for tag in split_tags(prompt):
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return ", ".join(out)


def count_tag(entities: list[dict]) -> str:
    """Danbooru subject-count tag — 1girl / 1boy / 2girls / 1boy, 1girl."""
    tags = book.count_tags("booru")
    chars = [e for e in entities if e.get("kind") == "character"]
    boys = sum(1 for e in chars if _gender(e) == "male")
    # Unknown falls in with the girls, which is the historical default and the
    # commoner case in these stories; declare `gender` or lead the appearance
    # with 1boy/1girl to be sure.
    girls = len(chars) - boys
    parts = []
    if boys:
        parts.append("1boy" if boys == 1 else f"{boys}boys")
    if girls:
        parts.append("1girl" if girls == 1 else f"{girls}girls")
    if not parts:
        return tags.get("none", "")
    if boys + girls == 1 and tags.get("solo"):
        parts.append(tags["solo"])
    return ", ".join(parts)


def subject_count(entities: list[dict]) -> str:
    """The same idea for a model that reads language rather than tags."""
    tags = book.count_tags("prose")
    n = sum(1 for e in entities if e.get("kind") == "character")
    if n == 0:
        return ""
    if n == 1:
        return tags.get("solo", "")
    return (tags.get("few", "").format(n=n) if n < 5 else tags.get("crowd", ""))


def quality_anchor(unet_name: str) -> str:
    """The prose family's per-checkpoint anchor (turbo at CFG 1 wants none)."""
    return book.quality("prose", unet_name)


_SOLO_RE = re.compile(r"\bsolo\b\s*,?\s*", re.I)


def _appearance(e: dict, *, alone: bool = True) -> str:
    """A character's frozen appearance, as it should enter THIS prompt.

    Appearances are authored as standalone tag strings and usually open with
    "1boy, solo, …". In a two-hander that "solo" contradicts the computed
    count and the model splits the difference — it was landing as
    "1boy, 1girl, solo" with two people at a table.
    """
    app = (e.get("appearance_tags") or e.get("appearance") or "").strip()
    if not alone:
        app = _SOLO_RE.sub("", app).strip().strip(",")
    return app


def build(*, keywords: str, entities: list[dict], style: str, rating: str,
          unet_name: str = "", shot: str | None = None,
          max_subjects: int | None = None, arch: str = "") -> str:
    """Assemble the positive prompt in the order `prompting.toml` asks for.

    `entities` are rows from the registry; only their frozen appearance is
    used, never the narrator's description of them.
    """
    fam = book.family(arch, unet_name)
    cfg = book.section(fam)
    limit = (max_subjects if max_subjects is not None
             else book.max_subjects(fam, unet_name))
    terminator = cfg.get("terminator", "")

    def slot(name: str) -> list[str]:
        if name == "quality":
            return [book.quality(fam, unet_name)]
        if name == "count":
            return [count_tag(entities) if fam == "booru"
                    else subject_count(entities)]
        if name == "characters":
            alone = sum(1 for e in entities
                        if e.get("kind") == "character") <= 1
            # Budget the slots by what they can actually contribute. An entity
            # with an empty appearance adds nothing, and two of them once ate
            # both subject slots, so a three-hander went out with no character
            # tags at all and the model drew strangers. People before places:
            # a room described twice costs less than a character described
            # not at all.
            described = [e for e in entities if _appearance(e).strip()]
            described.sort(key=lambda e: e.get("kind") != "character")
            picked = described[:limit]
            tmpl = cfg.get("character_template", "")
            out, n = [], 0
            for e in picked:
                app = strip_weights(_appearance(e, alone=alone), fam).rstrip(", ")
                # A flat comma list gives an LLM text encoder no way to tell
                # whose armour is whose — everyone ends up in the first
                # person's clothes. Naming each one separates them.
                if tmpl and e.get("kind") == "character":
                    n += 1
                    out.append(tmpl.format(n=n, tags=app))
                else:
                    out.append(app)
            return out
        if name == "keywords":
            return [strip_weights(keywords or "", fam).rstrip(", ")]
        if name == "shot":
            return [shot or ""]
        if name == "style":
            return [strip_weights(style or "", fam).rstrip(", ")]
        if name == "safety":
            return [ratings.positive_safety(rating)]
        return []

    parts: list[str] = []
    for name in book.order(fam, unet_name):
        parts.extend(p for p in slot(name) if p and p.strip())

    if terminator:
        # The prose family reads sentences: no part keeps its own full stop.
        parts = [p.strip().rstrip(terminator) for p in parts]

    out = ", ".join(p for p in parts if p.strip())
    if book.sanitize(fam)["dedupe"]:
        out = dedupe_tags(out)
    return out + terminator


def build_booru(*, keywords: str, entities: list[dict], style: str, rating: str,
                shot: str | None = None, max_subjects: int = 2) -> str:
    """Force the booru dialect regardless of the configured checkpoint."""
    return build(keywords=keywords, entities=entities, style=style,
                 rating=rating, shot=shot, max_subjects=max_subjects,
                 arch="sdxl")


def describe(unet_name: str, arch: str = "") -> dict:
    """What the builder will do right now — for the lab and /api/debug."""
    fam = book.family(arch, unet_name)
    return {
        "family": fam,
        "order": book.order(fam, unet_name),
        "quality_anchor": book.quality(fam, unet_name) or "(none)",
        "model_overrides": book.model(unet_name),
        "max_subjects": book.max_subjects(fam, unet_name),
        "shots": book.shots(fam),
        "sanitize": book.sanitize(fam),
        "shapes": sorted(book.resolutions(unet_name)),
        "book": book.status(),
    }
