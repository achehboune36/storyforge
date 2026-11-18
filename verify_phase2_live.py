"""Phase 2 live check — can llama3.2:3b actually follow the directive protocol?

    /home/one/ComfyUI/venv/bin/python verify_phase2_live.py

The inline `[[IMG|entities|keywords]]` design is only worth keeping if a 3B
model emits it reliably: right syntax, right cadence (not every turn), and
keywords that describe what is *visible* rather than restating the narration.

Three turns are run against one story premise:
  1. a scene change      -> expect ~1 directive
  2. pure dialogue       -> expect 0 (this is the hard one; small models love
                            to illustrate everything)
  3. a dramatic reveal   -> expect ~1

Nothing here asserts a hard pass/fail on model behaviour — it prints what the
model did so a human can judge whether the prompt needs work. The mechanical
parts (syntax parsed, no leakage) are asserted.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.llm_ollama import OllamaClient, build_messages, stream_turn  # noqa: E402
from backend.turn_parser import ImageSlot, ProseChunk, segments_from_events  # noqa: E402

CFG = json.loads((Path(__file__).resolve().parent / "config.json").read_text())

SYSTEM = """You are the narrator of a 1940s noir detective story. Write in second person, present tense. 2-4 short paragraphs per reply. Atmospheric and terse.

PLAYER AGENCY - Never decide what the player does, says, or feels. Begin after their stated action, describing only the world's reaction.

DIALOGUE - When the player addresses someone, that character must actually speak, in quotation marks on its own line.

ILLUSTRATIONS - You may mark at most ONE moment per reply worth illustrating, using exactly this form on its own line:
[[IMG|entity_ids|visual keywords]]

- entity_ids: comma-separated ids of characters/places present, or empty
- visual keywords: ONLY what is visible. Comma-separated. No actions, no narration, no sentences.
- Use one ONLY when the player arrives somewhere new, meets someone for the first time, or something visually dramatic happens.
- Most replies need NO illustration. Dialogue and small actions never do.

Example with one:
The door gives.
[[IMG|bar|smoky jazz bar interior, amber lamps, cigarette haze, wet coats on hooks]]
Inside, the air is thick enough to lean on.

Example with none:
"You got the wrong man," he says, not looking up from the glass.
The clock behind him ticks on regardless.

Example with none:
You check your pockets. Keys, a book of matches, four dollars.

KNOWN ENTITIES: bar (The Blue Parrot), bartender (Sal), player"""

TURNS = [
    ("scene change", "I push open the door of the jazz bar and step inside."),
    ("pure dialogue", 'I ask Sal: "Who was the man in the grey hat?"'),
    ("dramatic reveal", "I pull back the curtain at the back of the bar."),
]


async def main() -> int:
    client = OllamaClient(CFG)
    if not await client.is_up():
        print("Ollama is not reachable at", client.host)
        print("Start it on Windows (tray app, or `ollama serve`).")
        return 1
    print(f"model={client.model}  num_ctx={client.num_ctx}\n")

    history: list[dict] = []
    prev_keywords: str | None = None
    total_directives = 0
    failures = 0

    for label, action in TURNS:
        print("=" * 64)
        print(f"TURN: {label}\n  player: {action}\n")

        live: list[str] = []

        async def on_event(ev):
            if isinstance(ev, ImageSlot):
                live.append(f"\n  [SLOT {ev.index} opens here | entities={ev.entities}]\n")
            else:
                live.append(ev.text)

        msgs = build_messages(SYSTEM, history, action)
        res = await stream_turn(client, msgs, on_event=on_event,
                                max_images=CFG["image"]["max_images_per_turn"],
                                previous_keywords=prev_keywords)

        if not res.ok:
            print("  LLM FAILED:", res.llm.error)
            return 1

        print("".join(live).strip())

        segs = segments_from_events(res.events)
        imgs = [s for s in segs if s["kind"] == "image"]
        prose = " ".join(s["text"] for s in segs if s["kind"] == "prose")
        total_directives += len(imgs)

        print(f"\n  --- {len(imgs)} image(s), {res.llm.chunks} chunks, "
              f"{res.llm.chunks_per_s:.0f} chunk/s, "
              f"{res.llm.prompt_eval_count} prompt tokens")
        for s in imgs:
            print(f"      entities={s['entities']}")
            print(f"      keywords={s['keywords']}")
            prev_keywords = s["keywords"]
        if res.stats.parse_failures:
            print(f"      parse_failures={res.stats.parse_failures}")
        if res.stats.dropped_over_cap:
            print(f"      dropped_over_cap={res.stats.dropped_over_cap}")
        if res.stats.dropped_repetitive:
            print(f"      dropped_repetitive={res.stats.dropped_repetitive}")

        # Mechanical assertions — these are about our code, not the model.
        if "[[IMG" in prose or "]]" in prose:
            print("      *** FAIL: directive text leaked into prose")
            failures += 1
        if res.llm.truncated:
            print("      *** note: hit num_predict limit (truncated)")
        if any(not s["keywords"].strip() for s in imgs):
            print("      *** FAIL: empty keywords emitted as a slot")
            failures += 1

        history.append({"role": "user", "content": action})
        history.append({"role": "assistant", "content": res.llm.text})
        print()

    print("=" * 64)
    print(f"  {total_directives} directive(s) across {len(TURNS)} turns")
    print(f"  mechanical failures: {failures}")
    print("\n  Judge by eye: did it illustrate the scene change and the reveal,")
    print("  and leave the pure-dialogue turn alone? Are the keywords visual")
    print("  rather than narrative restatements?")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
