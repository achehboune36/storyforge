"""Visual consistency: entity registry, locked seeds, prompt assembly.

The LLM is never trusted with visual continuity. It is trusted only to say
*which* entity is on screen and *what is happening around them*. Everything
about how a character looks lives here, server-side, and never enters the
narrative context — if the model sees an appearance string it paraphrases it
into the prose, and the reader gets a character described head-to-toe every
time they appear.

Three mechanisms, in increasing order of how much they matter:

1. **Style token** — a per-story suffix appended byte-identically to every
   prompt, so ComfyUI's conditioning cache can hit and the look stays coherent.

2. **Frozen appearance** — a 15-30 word canonical description per entity,
   pre-declared in the story JSON. Written once, never updated. This is the
   real reason premade story openings matter.

3. **Locked seed** — derived once from (session, entity) and persisted on the
   image row. With a fixed model, a byte-identical style suffix and a frozen
   appearance clause, a fixed seed gives strongly consistent faces at zero VRAM
   cost. It is not identity preservation — a reference adapter is unaffordable
   alongside Klein on 12 GB — but it is most of the way there for free.

   Verified in Phase 1: the same seed reproduces pixel-identical output across
   a full model reload, so a re-run after a preemption returns the same image.

The variety guard jitters the *keywords* (camera angle), never the seed. The
seed is what consistency rests on; rotating it would defeat the whole scheme,
while two near-identical prompts would otherwise produce two near-identical
pictures of the same room.
"""

from __future__ import annotations

import hashlib

from backend import prompt_book as book
from backend import prompt_style

# How many entity appearances may enter one prompt, and which framing tags are
# available, are both tuned in `prompting.toml` (`max_subjects`, `shots`).


def stable_hash(*parts: str) -> int:
    h = hashlib.sha256("\x1f".join(parts).encode()).digest()
    return int.from_bytes(h[:4], "big")


def entity_seed(session_id: str, key: str) -> int:
    """Deterministic per (session, entity). Assigned once, then persisted."""
    return stable_hash("entity", session_id, key) % (2 ** 31 - 1)


def scene_seed(session_id: str, turn_idx: int, slot: int) -> int:
    return stable_hash("scene", session_id, str(turn_idx), str(slot)) % (2 ** 31 - 1)


def split_framing(framing: str, checkpoint: str = "") -> tuple[str, str]:
    """The narrator's third field -> (shot tags, frame shape).

    The shape is whichever word matches a bucket in prompting.toml; everything
    else is camera direction and goes into the prompt as tags. A shape the
    file does not define is left in the camera tags rather than dropped —
    "panoramic" is still useful to the model even if it names no bucket.
    """
    shapes = book.resolutions(checkpoint)
    tags, shape = [], ""
    for tag in (s.strip() for s in (framing or "").split(",")):
        if not tag:
            continue
        key = tag.lower().replace(" ", "")
        if not shape and key in shapes:
            shape = key
        else:
            tags.append(tag)
    return ", ".join(tags), shape


def pick_camera(session_id: str, turn_idx: int, slot: int,
                arch: str = "") -> str:
    """Deterministic rotation — same turn always renders the same way.

    Booru models want danbooru framing tags ("cowboy shot"), not the prose
    phrases an LLM-encoder model understands.
    """
    pool = book.shots(book.family(arch))
    i = stable_hash("cam", session_id, str(turn_idx), str(slot)) % len(pool)
    return pool[i]


def resolve_seed(entities: list[dict], *, session_id: str, turn_idx: int,
                 slot: int) -> tuple[int, str]:
    """Seed comes from the first *character* present, else the first place,
    else the scene itself. Returns (seed, why) — `why` is for the debug page.
    """
    chars = [e for e in entities if e.get("kind") == "character"]
    places = [e for e in entities if e.get("kind") != "character"]
    if chars:
        return int(chars[0]["seed"]), f"character:{chars[0]['key']}"
    if places:
        return int(places[0]["seed"]), f"place:{places[0]['key']}"
    return scene_seed(session_id, turn_idx, slot), "scene"


def build_prompt(*, keywords: str, entities: list[dict], style: str,
                 camera: str | None = None, rating: str = "13+",
                 unet_name: str = "", arch: str = "") -> str:
    """Delegates to prompt_style, which encodes the model-specific ordering.

    Kept as a thin wrapper so callers do not need to know which image
    architecture is configured.
    """
    return prompt_style.build(keywords=keywords, entities=entities, style=style,
                              rating=rating, unet_name=unet_name, shot=camera,
                              arch=arch)


def plan_image(*, session_id: str, turn_idx: int, slot: int, keywords: str,
               entity_keys: list[str], known: dict[str, dict], style: str,
               rating: str = "13+", unet_name: str = "", arch: str = "",
               framing: str = "", width: int = 0, height: int = 0) -> dict:
    """Everything needed to create an image row, resolved in one place.

    `known` maps entity key -> entity row. Keys the model invented that we have
    never seen are dropped here; the caller is responsible for registering them
    first if it wants them to count.

    `framing` is the narrator's own camera and frame shape. It wins over the
    deterministic rotation, because the rotation is a guess made with no idea
    where anyone is standing: it gave a portrait frame to two people facing
    each other across a table and the model duly stacked them vertically.
    """
    ents = [known[k] for k in entity_keys if k in known]
    seed, why = resolve_seed(ents, session_id=session_id, turn_idx=turn_idx,
                             slot=slot)

    asked, shape = split_framing(framing, unet_name)
    camera = asked or pick_camera(session_id, turn_idx, slot, arch)
    size = book.resolutions(unet_name).get(shape)
    w, h = size if size else (width, height)

    prompt = build_prompt(keywords=keywords, entities=ents, style=style,
                          camera=camera, rating=rating, unet_name=unet_name,
                          arch=arch)
    return {
        "prompt": prompt,
        "seed": seed,
        "seed_reason": why,
        "camera": camera,
        "shape": shape,
        "width": w,
        "height": h,
        "entities": [e["key"] for e in ents],
        "unknown_entities": [k for k in entity_keys if k not in known],
    }
