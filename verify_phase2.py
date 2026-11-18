"""Phase 2 verification — the turn parser.

    python3 verify_phase2.py          (pure logic, no server needed)

The property test is the important one: for ANY chunking of a fixed response,
the splitter must yield an identical segment list. Streaming parsers fail
exactly at chunk boundaries — a directive arriving as "[[IM" + "G|a|b]]" is
the realistic case, not a contrived one, because token boundaries never align
with markup.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.turn_parser import (  # noqa: E402
    ImageSlot, ProseChunk, ProseSplitter, segments_from_events, split_all,
)

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


REF = (
    "The door gives way with a groan.\n\n"
    "[[IMG|elias,bar|smoky jazz bar interior, amber lamps, cigarette haze, wet coats]]\n\n"
    "Inside, the air is thick enough to lean on. A saxophone worries at a note "
    "somewhere behind the crowd.\n\n"
    '"You look lost, friend," the bartender says without looking up.\n\n'
    "[[IMG||close-up of a whiskey glass on a scarred wooden counter, low light]]\n\n"
    "He slides a glass across anyway."
)


def feed_in_chunks(text: str, sizes: list[int]) -> list[dict]:
    sp = ProseSplitter()
    events: list = []
    i = 0
    for n in sizes:
        if i >= len(text):
            break
        events += sp.feed(text[i:i + n])
        i += n
    if i < len(text):
        events += sp.feed(text[i:])
    events += sp.finish()
    return segments_from_events(events)


def test_baseline() -> None:
    print("\nP0  baseline parse")
    segs = segments_from_events(split_all(REF))
    kinds = [s["kind"] for s in segs]
    check("segment order prose/image/prose/image/prose",
          kinds == ["prose", "image", "prose", "image", "prose"], str(kinds))
    imgs = [s for s in segs if s["kind"] == "image"]
    check("2 images", len(imgs) == 2, str(len(imgs)))
    check("entities parsed", imgs[0]["entities"] == ["elias", "bar"],
          str(imgs[0]["entities"]))
    check("empty entity field ok", imgs[1]["entities"] == [], str(imgs[1]["entities"]))
    check("keywords carried", imgs[0]["keywords"].startswith("smoky jazz bar"),
          imgs[0]["keywords"][:40])
    prose = " ".join(s["text"] for s in segs if s["kind"] == "prose")
    check("no directive text leaks into prose",
          "[[IMG" not in prose and "]]" not in prose)
    check("dialogue preserved", '"You look lost, friend,"' in prose)


def test_chunking_property() -> None:
    print("\nP1  property: any chunking yields the same segments")
    expected = segments_from_events(split_all(REF))

    # every single-byte boundary
    bad_single = []
    for cut in range(1, len(REF)):
        got = feed_in_chunks(REF, [cut])
        if got != expected:
            bad_single.append(cut)
    check("all single-cut splits identical", not bad_single,
          f"{len(bad_single)} bad cuts, first at {bad_single[:5]}")

    # one byte at a time — the worst case
    check("1-byte chunks identical", feed_in_chunks(REF, [1] * len(REF)) == expected)

    # random chunkings
    rng = random.Random(1234)
    bad_random = 0
    for _ in range(300):
        sizes = [rng.randint(1, 12) for _ in range(len(REF))]
        if feed_in_chunks(REF, sizes) != expected:
            bad_random += 1
    check("300 random chunkings identical", bad_random == 0, f"{bad_random} differed")

    # the specific nasty case: split inside the opening marker
    for k in range(1, len("[[IMG|")):
        text = "before " + "[[IMG|" + "a|kw]]" + " after"
        pre = text.index("[[IMG|")
        got = feed_in_chunks(text, [pre + k])
        want = segments_from_events(split_all(text))
        if got != want:
            check(f"split inside opening marker at +{k}", False, str(got))
            return
    check("split inside opening marker at every offset", True)


def test_counts_and_caps() -> None:
    print("\nP2  zero / one / many images")
    check("zero directives -> one prose segment",
          [s["kind"] for s in segments_from_events(split_all("Just prose, nothing else."))]
          == ["prose"])

    three = ("a [[IMG||one thing here]] b [[IMG||two other thing]] "
             "c [[IMG||three different thing]] d")
    sp = ProseSplitter(max_images=2)
    evs = sp.feed(three) + sp.finish()
    imgs = [e for e in evs if isinstance(e, ImageSlot)]
    check("cap of 2 enforced", len(imgs) == 2, f"{len(imgs)} emitted")
    check("over-cap counted", sp.stats.dropped_over_cap == 1,
          str(sp.stats.dropped_over_cap))
    check("slot indices sequential", [i.index for i in imgs] == [0, 1],
          str([i.index for i in imgs]))


def test_repetition_cooldown() -> None:
    print("\nP3  repetition cooldown")
    sp = ProseSplitter(previous_keywords="smoky jazz bar interior, amber lamps, haze")
    evs = sp.feed("x [[IMG||smoky jazz bar interior, amber lamps, haze]] y") + sp.finish()
    check("near-identical repeat dropped",
          not [e for e in evs if isinstance(e, ImageSlot)],
          f"dropped={sp.stats.dropped_repetitive}")

    sp2 = ProseSplitter(previous_keywords="smoky jazz bar interior, amber lamps")
    evs2 = sp2.feed("x [[IMG||rain-slick rooftop at dawn, pigeons, cold light]] y") + sp2.finish()
    check("different scene kept",
          len([e for e in evs2 if isinstance(e, ImageSlot)]) == 1)


def test_malformed() -> None:
    print("\nP4  malformed input never corrupts the prose")
    cases = {
        "unterminated": "story text [[IMG||never closed",
        "no separator": "story [[IMG no pipes here]] more",
        "empty keywords": "story [[IMG|elias|]] more",
        "runaway": "story [[IMG||" + ("x" * 500) + " and on",
    }
    for name, text in cases.items():
        segs = segments_from_events(split_all(text))
        prose = " ".join(s["text"] for s in segs if s["kind"] == "prose")
        leaked = "[[IMG" in prose
        check(f"{name}: no marker leaks to prose", not leaked, prose[:60])

    sp = ProseSplitter()
    sp.feed(cases["unterminated"])
    sp.finish()
    check("unterminated counted as parse failure", sp.stats.parse_failures >= 1)


def test_think_stripping() -> None:
    print("\nP5  <think> folded into the same machine")
    text = "Before. <think>internal reasoning the reader must not see</think> After."
    segs = segments_from_events(split_all(text))
    prose = " ".join(s["text"] for s in segs if s["kind"] == "prose")
    check("think content removed", "internal reasoning" not in prose, prose)
    check("surrounding prose intact", "Before." in prose and "After." in prose, prose)
    check("think survives 1-byte chunking",
          feed_in_chunks(text, [1] * len(text)) == segs)


def main() -> int:
    test_baseline()
    test_chunking_property()
    test_counts_and_caps()
    test_repetition_cooldown()
    test_malformed()
    test_think_stripping()
    print(f"\n{'=' * 58}")
    print(f"  {len(passed)} passed, {len(failed)} failed")
    for f in failed:
        print(f"    FAILED: {f}")
    print("=" * 58)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
