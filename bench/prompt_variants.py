"""Does the directive protocol work — and is a failure the prompt or the model?

Baseline (verify_phase2_live.py) got 0 directives in 3 turns from llama3.2:3b,
including on a scene change and a dramatic reveal. Two candidate causes:

  a) the prompt suppressed it — it said "Most replies need NO illustration"
     and 2 of its 3 examples had none;
  b) a 3B model cannot hold a formatting rule among several other rules.

This runs prompt variants x models over the same fixed turns and prints the
directive cadence for each, so (a) and (b) can be told apart.

Ideal cadence for the 4 turns below is roughly [1, 0, 1, 0]: illustrate the
arrival and the reveal, leave dialogue and the pocket-check alone.

    /home/one/ComfyUI/venv/bin/python -m bench.prompt_variants
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.llm_ollama import OllamaClient, build_messages, stream_turn  # noqa: E402
from backend.turn_parser import segments_from_events  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parent.parent / "config.json").read_text())

RULES = """You are the narrator of a 1940s noir detective story. Write in second person, present tense. 2-4 short paragraphs. Atmospheric and terse.

PLAYER AGENCY - Never decide what the player does, says, or feels.
DIALOGUE - When the player addresses someone, that character must actually speak, in quotation marks.
KNOWN ENTITIES: bar (The Blue Parrot), sal (the bartender), backroom"""

# A: the baseline that produced zero. Kept so the comparison is honest.
A = RULES + """

ILLUSTRATIONS - You may mark at most ONE moment per reply worth illustrating:
[[IMG|entity_ids|visual keywords]]
Use one ONLY when the player arrives somewhere new, meets someone for the first time, or something visually dramatic happens. Most replies need NO illustration.

Example with one:
The door gives.
[[IMG|bar|smoky jazz bar interior, amber lamps, cigarette haze]]
Inside, the air is thick.

Example with none:
"You got the wrong man," he says.

Example with none:
You check your pockets. Keys, matches, four dollars."""

# B: positive framing, explicit triggers, majority of examples DO illustrate.
B = RULES + """

ILLUSTRATIONS - This story is illustrated. When the player arrives somewhere new, sees someone for the first time, or something visually striking is revealed, you MUST emit exactly one illustration line, on its own line, in this exact format:

[[IMG|entity_ids|visual keywords]]

entity_ids = comma-separated ids from KNOWN ENTITIES that are visible, or empty.
visual keywords = 6-12 comma-separated words naming ONLY what a camera would see: place, objects, light, weather, colours. Never actions. Never sentences.

Emit it for: arriving somewhere, first sight of a person, a revealed space, a dramatic visual change.
Skip it for: conversation, thinking, checking pockets, small movements in a place already shown.

ARRIVAL - illustrate:
The door gives way.
[[IMG|bar|smoky jazz bar interior, amber lamps, cigarette haze, wet coats, crowded booths]]
Inside, the air is thick enough to lean on.

REVEAL - illustrate:
The curtain pulls back.
[[IMG|backroom|cramped back room, single flickering bulb, card table, scattered chips, peeling wallpaper]]
Nobody moves.

CONVERSATION - do not illustrate:
"You got the wrong man," Sal says, not looking up from the glass.

SMALL ACTION - do not illustrate:
You check your pockets. Keys, a book of matches, four dollars."""

# C: as B, but the format line is the very first thing in the prompt.
C = """ILLUSTRATION FORMAT (most important rule): when the scene changes to a place or person not yet shown, output exactly one line of the form
[[IMG|entity_ids|6-12 comma-separated visual keywords]]
placed in the story where the image belongs. Keywords name only what is visible - place, objects, light, colour. Never actions or sentences. Do not output this line during conversation or small actions.

""" + RULES + """

ARRIVAL - illustrate:
The door gives way.
[[IMG|bar|smoky jazz bar interior, amber lamps, cigarette haze, wet coats, crowded booths]]
Inside, the air is thick enough to lean on.

REVEAL - illustrate:
The curtain pulls back.
[[IMG|backroom|cramped back room, single flickering bulb, card table, scattered chips]]
Nobody moves.

CONVERSATION - do not illustrate:
"You got the wrong man," Sal says, not looking up."""

VARIANTS = {"A_baseline": A, "B_positive": B, "C_format_first": C}

TURNS = [
    ("arrive", "I push open the door of the jazz bar and step inside."),
    ("dialogue", 'I ask Sal: "Who was the man in the grey hat?"'),
    ("reveal", "I pull back the curtain at the back of the bar."),
    ("small action", "I check my pockets."),
]
IDEAL = [1, 0, 1, 0]


async def run(client: OllamaClient, system: str) -> tuple[list[int], list[str], int]:
    history: list[dict] = []
    counts: list[int] = []
    kws: list[str] = []
    failures = 0
    prev = None
    for _, action in TURNS:
        res = await stream_turn(client, build_messages(system, history, action),
                                max_images=2, previous_keywords=prev)
        if not res.ok:
            return counts + [-1] * (len(TURNS) - len(counts)), kws, failures + 1
        segs = segments_from_events(res.events)
        imgs = [s for s in segs if s["kind"] == "image"]
        prose = " ".join(s["text"] for s in segs if s["kind"] == "prose")
        if "[[IMG" in prose:
            failures += 1
        counts.append(len(imgs))
        for s in imgs:
            kws.append(s["keywords"])
            prev = s["keywords"]
        history.append({"role": "user", "content": action})
        history.append({"role": "assistant", "content": res.llm.text})
    return counts, kws, failures


async def main() -> int:
    models = ["llama3.2:3b", "llama3.1:8b"]
    print(f"turns:  {[t[0] for t in TURNS]}")
    print(f"ideal:  {IDEAL}\n")

    best: tuple[int, str, str] | None = None
    for model in models:
        cfg = json.loads(json.dumps(CFG))
        cfg["llm"]["model"] = model
        client = OllamaClient(cfg)
        if not await client.is_up():
            print("Ollama not reachable"); return 1
        if model not in await client.available_models():
            print(f"-- {model}: not pulled, skipping\n"); continue

        print(f"== {model}")
        for name, system in VARIANTS.items():
            counts, kws, failures = await run(client, system)
            dist = sum(abs(a - b) for a, b in zip(counts, IDEAL + [0] * 9))
            flag = "  <-- ideal" if counts == IDEAL else ""
            print(f"  {name:<16} {counts}  distance={dist} "
                  f"leaks={failures}{flag}")
            for k in kws[:2]:
                print(f"       kw: {k[:88]}")
            if best is None or dist < best[0]:
                best = (dist, model, name)
        print()
        await client.evict()

    if best:
        print(f"best: {best[1]} + {best[2]} (distance {best[0]} from ideal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
