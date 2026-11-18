"""Phase 3 verification — persistence, entities, context budgeting.

    python3 verify_phase3.py

Uses a scratch database so it never touches real sessions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend import database as db  # noqa: E402

# Scratch DB before anything opens a connection.
db.DB_PATH = ROOT / "data" / "verify_phase3.db"
if db.DB_PATH.exists():
    db.DB_PATH.unlink()

from backend import context, image_prompt, story_loader  # noqa: E402

CFG = json.loads((ROOT / "config.json").read_text())

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def test_schema_and_story() -> dict:
    print("\nD1  schema + story loading")
    db.init()
    stories = story_loader.load_all(verbose=False)
    check("story loaded", len(stories) >= 1, f"{len(stories)} found")
    sess = story_loader.start_session("lighthouse")
    check("session created", bool(sess["id"]))
    check("style copied to session", "anime" in sess["style"].lower(),
          sess["style"][:40])
    ents = db.get_entities(sess["id"])
    check("entities seeded from story", len(ents) == 17, f"{len(ents)}")
    check("appearances stored",
          all(e["appearance"] for e in ents))
    seeds = {e["seed"] for e in ents}
    check("seeds distinct per entity", len(seeds) == len(ents),
          f"{len(seeds)} distinct of {len(ents)}")
    return sess


def test_seed_stability(sess: dict) -> None:
    print("\nD2  seed determinism")
    a = image_prompt.entity_seed(sess["id"], "alex")
    b = image_prompt.entity_seed(sess["id"], "alex")
    check("same session+key -> same seed", a == b, f"{a}")
    c = image_prompt.entity_seed("other-session", "alex")
    check("different session -> different seed", a != c)
    stored = db.get_entity(sess["id"], "alex")
    check("stored seed matches derivation", int(stored["seed"]) == a,
          f"{stored['seed']} vs {a}")


def test_prompt_building(sess: dict) -> None:
    print("\nD3  image prompt assembly")
    known = {e["key"]: e for e in db.get_entities(sess["id"])}
    plan = image_prompt.plan_image(
        session_id=sess["id"], turn_idx=3, slot=0,
        keywords="towering bookshelves, gas lantern, dust in the light",
        entity_keys=["alex", "library"], known=known, style=sess["style"])

    # What matters is that appearance precedes the narrator's keywords, so
    # identity gets the stronger early-token position.
    pr = plan["prompt"]
    check("appearance precedes keywords",
          "messy short hair" in pr and pr.index("messy") < pr.index("gas lantern"),
          pr[:60])
    check("style appended", "anime screencap" in pr)
    check("safety displacement present for 13+", "safe for work" in pr)
    check("keywords present", "gas lantern" in plan["prompt"])
    check("camera angle injected", plan["camera"] in plan["prompt"])
    check("seed from the character, not the place",
          plan["seed_reason"] == "character:alex", plan["seed_reason"])

    # Framing: the narrator's third field wins over the rotation, and a shape
    # word changes the canvas.
    framed = image_prompt.plan_image(
        session_id=sess["id"], turn_idx=3, slot=0, keywords="two at a table",
        entity_keys=["alex", "mirabel"], known=known, style="",
        framing="from side, landscape", width=832, height=1216)
    check("narrator framing overrides the rotation",
          framed["camera"] == "from side", framed["camera"])
    check("frame shape resizes the canvas",
          framed["width"] > framed["height"],
          f"{framed['width']}x{framed['height']}")
    check("solo dropped from a two-hander", "solo" not in framed["prompt"])

    cap = image_prompt.plan_image(
        session_id=sess["id"], turn_idx=3, slot=0, keywords="x",
        entity_keys=["alex", "mirabel", "sera"], known=known, style="")
    check("at most 2 appearances used",
          cap["prompt"].count("side ponytail") == 0)

    unknown = image_prompt.plan_image(
        session_id=sess["id"], turn_idx=1, slot=0, keywords="a rooftop",
        entity_keys=["rooftop"], known=known, style="")
    check("unknown entity reported, not crashed",
          unknown["unknown_entities"] == ["rooftop"], str(unknown["unknown_entities"]))
    check("unknown falls back to scene seed",
          unknown["seed_reason"] == "scene", unknown["seed_reason"])

    # Determinism of the whole plan — a preempted image must regenerate identically.
    again = image_prompt.plan_image(
        session_id=sess["id"], turn_idx=3, slot=0,
        keywords="towering bookshelves, gas lantern, dust in the light",
        entity_keys=["alex", "library"], known=known, style=sess["style"])
    check("plan is reproducible", again == plan)


def test_turns_and_segments(sess: dict) -> None:
    print("\nD4  turns, segments, images")
    sid = sess["id"]
    t0 = db.add_turn(sid, "user", "I push open the door.")
    t1 = db.add_turn(sid, "narrator", "The door gives way. Inside, smoke.",
                     raw_text="The door gives way.[[IMG|bar|smoky]]Inside, smoke.")
    img = db.create_image(sid, t1, slot_pos=1, keywords="smoky bar",
                          entities=["bar"], prompt="p", seed=42,
                          width=768, height=768, steps=12, model="klein4b")
    db.add_segments(t1, [
        {"kind": "prose", "text": "The door gives way."},
        {"kind": "image", "image_id": img},
        {"kind": "prose", "text": "Inside, smoke."},
    ])
    segs = db.get_segments(sid)
    kinds = [s["kind"] for s in segs if s["kind"]]
    check("segments persisted in order", kinds == ["prose", "image", "prose"],
          str(kinds))
    check("image linked to its segment",
          any(s["image_id"] == img for s in segs))

    db.update_image(img, status="ready", file_path="/tmp/x.png")
    check("image status updated", db.get_image(img)["status"] == "ready")
    try:
        db.update_image(img, nonsense="x")
        check("bad image field rejected", False, "no error")
    except ValueError:
        check("bad image field rejected", True)

    jid = db.enqueue_job(sid, img)
    claimed = db.claim_next_job()
    check("job claimed", claimed and claimed["id"] == jid)
    db.connect().execute("UPDATE jobs SET state='running' WHERE id=?", (jid,))
    db.connect().commit()
    n = db.reconcile_startup()
    check("running jobs requeued on startup", n >= 1, f"{n} requeued")


def test_memories(sess: dict) -> None:
    print("\nD5  core memories")
    sid = sess["id"]
    a = db.add_memory(sid, "Sal knows who wore the grey hat.", importance=8)
    b = db.add_memory(sid, "sal knows who wore the grey hat", importance=8)
    check("duplicate memory deduped", a is not None and b is None)
    db.add_memory(sid, "The player owes Marko forty dollars.", importance=6)
    mems = db.get_memories(sid)
    check("2 distinct memories", len(mems) == 2, str(len(mems)))
    check("ordered by importance", mems[0]["importance"] >= mems[1]["importance"])


def test_context_budget(sess: dict) -> None:
    print("\nD6  context assembly within budget")
    sid = sess["id"]
    # Pile on history until it must be dropped.
    for i in range(40):
        db.add_turn(sid, "user", f"I do thing number {i}. " * 8)
        db.add_turn(sid, "narrator", f"The world reacts to thing {i}. " * 20)

    num_ctx = CFG["llm"]["num_ctx"]
    a = context.assemble(sid, "I look around.", num_ctx=num_ctx)
    budget = context.Budget(total=num_ctx).usable
    check("estimate within usable budget", a.est_total <= budget,
          f"{a.est_total} est vs {budget} usable")
    check("old turns dropped", a.dropped_turns > 0, f"{a.dropped_turns} dropped")
    check("recent turns kept", len(a.history) >= context.VERBATIM_MIN_PAIRS * 2,
          f"{len(a.history)} messages")
    check("entity roster in system", "WHO AND WHAT EXISTS" in a.system)
    # The roster is the cast, not decoration: a budget too small to hold the
    # people silently trimmed it to one entity and the narrator invented a
    # headmaster. Assert a named character actually survives.
    check("roster keeps characters", "alex (" in a.system)
    check("memories in system", "ESTABLISHED FACTS" in a.system)
    check("appearance NEVER in context",
          "heavyset man" not in a.system and
          not any("heavyset man" in m["content"] for m in a.history),
          "appearance correctly absent from narrator context")

    raw = "Prose here.\n[[IMG|bar|smoky, amber]]\nMore prose."
    check("directives stripped from history",
          "[[IMG" not in context.strip_directives(raw),
          context.strip_directives(raw))

    print(f"      parts: {a.parts}")


def main() -> int:
    sess = test_schema_and_story()
    test_seed_stability(sess)
    test_prompt_building(sess)
    test_turns_and_segments(sess)
    test_memories(sess)
    test_context_budget(sess)
    print(f"\n{'=' * 58}")
    print(f"  {len(passed)} passed, {len(failed)} failed")
    for f in failed:
        print(f"    FAILED: {f}")
    print("=" * 58)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
