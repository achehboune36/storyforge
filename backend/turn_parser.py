"""Streaming splitter for narrative prose with inline image directives.

## The protocol

    [[IMG|elias,alley|rain-slick cobblestones, a single gas lantern|from side, landscape]]

Field 1: comma-separated entity ids (may be empty). Field 2: visual keywords.
Field 3 (optional): camera and frame shape — the narrator knows whether two
people are facing each other across a table or walking away down a street,
and a front-on portrait frame is the wrong answer for both.

## Why not JSON

game-poc had the model append `{"img": "..."}` at the end of its reply and
parsed it by counting braces backwards from the end of the string
(`game_engine.py:16-40`). That works until it doesn't:

  * it depends on the model never writing prose after the block;
  * malformed JSON from a small model loses the image entirely;
  * and the raw JSON streamed visibly into the chat bubble, to be stripped
    later on the client — the user watched it appear and vanish.

More fundamentally, a trailing block says *that* there is an image, not
*where* it belongs. An inline directive is self-anchoring: the point in the
token stream where it appears IS the point in the prose where the image goes.
No "after paragraph 3" bookkeeping for an 8B model to get wrong.

The syntax has no nesting, quoting or escaping — precisely the JSON failure
modes small models exhibit. Ollama's `format: json` is rejected as the primary
path because it degrades prose quality and cannot apply to part of a response.

## Streaming

`feed()` takes arbitrary chunks — token boundaries never align with anything.
While in PROSE we hold back the last few characters, because a chunk may end
mid-way through an opening marker. That is the same trick game-poc used for
its `<think>` filter, and `<think>` stripping is folded into this one machine
rather than run as a second pass.

A directive that does not close within MAX_DIRECTIVE_LEN is abandoned and
counted as a parse failure: emitting a stray `[[IMG` into the story reads far
worse than silently missing an illustration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto

OPEN = "[[IMG"   # pipe is optional here so a malformed directive is still caught
CLOSE = "]]"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
MAX_DIRECTIVE_LEN = 400

# Longest prefix we must hold back in PROSE so a marker split across chunks is
# not emitted as text.
_HOLDBACK = max(len(OPEN), len(THINK_OPEN)) - 1


class _S(Enum):
    PROSE = auto()
    DIRECTIVE = auto()
    THINK = auto()


@dataclass
class ProseChunk:
    text: str


@dataclass
class ImageSlot:
    index: int                      # nth image in this turn, 0-based
    entities: list[str]
    keywords: str
    # Optional third field: camera and frame shape, e.g. "from side, wide
    # shot, landscape". Empty means the deterministic rotation picks a shot
    # and the model's default size is used.
    framing: str = ""

    @property
    def is_usable(self) -> bool:
        return bool(self.keywords.strip())


Event = ProseChunk | ImageSlot


@dataclass
class Stats:
    directives: int = 0
    parse_failures: int = 0
    dropped_over_cap: int = 0
    dropped_repetitive: int = 0
    think_blocks: int = 0


def _norm_tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2}


def jaccard(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class ProseSplitter:
    """Feed streamed text in, get prose and image slots out, in order.

    Not thread-safe; one instance per turn.
    """

    def __init__(self, *, max_images: int = 2, repeat_threshold: float = 0.55,
                 previous_keywords: str | list[str] | None = None):
        self.max_images = max_images
        self.repeat_threshold = repeat_threshold
        # Compared against EVERY image already drawn in this story, not just
        # the last one. Checking only the previous image let the same bar be
        # re-illustrated every third turn, which is exactly what it looked
        # like in play. Threshold lowered from 0.8 to 0.55 for the same reason.
        if isinstance(previous_keywords, str):
            self._seen = [previous_keywords] if previous_keywords else []
        else:
            self._seen = list(previous_keywords or [])
        self._buf = ""
        self._state = _S.PROSE
        self._pending = ""          # accumulating directive/think body
        self._emitted = 0
        self.stats = Stats()
        self.raw = ""               # full un-stripped output, for debugging

    # ------------------------------------------------------------------ public

    def feed(self, chunk: str) -> list[Event]:
        self.raw += chunk
        self._buf += chunk
        return self._drain(final=False)

    def finish(self) -> list[Event]:
        """Flush. An unterminated directive is dropped, not emitted as prose."""
        events = self._drain(final=True)
        if self._state is _S.DIRECTIVE:
            self.stats.parse_failures += 1
            self._pending = ""
            self._state = _S.PROSE
        elif self._state is _S.THINK:
            self._pending = ""
            self._state = _S.PROSE
        if self._buf:
            text, self._buf = self._buf, ""
            if text:
                events.append(ProseChunk(text))
        return [e for e in events if not (isinstance(e, ProseChunk) and not e.text)]

    # ----------------------------------------------------------------- internal

    def _drain(self, *, final: bool) -> list[Event]:
        out: list[Event] = []
        while True:
            if self._state is _S.PROSE:
                event, progressed = self._step_prose(final)
            elif self._state is _S.DIRECTIVE:
                event, progressed = self._step_directive()
            else:
                event, progressed = self._step_think()
            if event is not None:
                out.append(event)
            if not progressed:
                break
        return [e for e in out if not (isinstance(e, ProseChunk) and not e.text)]

    def _step_prose(self, final: bool) -> tuple[Event | None, bool]:
        i_img = self._buf.find(OPEN)
        i_think = self._buf.find(THINK_OPEN)
        candidates = [i for i in (i_img, i_think) if i != -1]

        if candidates:
            i = min(candidates)
            text, self._buf = self._buf[:i], self._buf[i:]
            if i == i_img and (i_think == -1 or i_img < i_think):
                self._buf = self._buf[len(OPEN):]
                self._state = _S.DIRECTIVE
            else:
                self._buf = self._buf[len(THINK_OPEN):]
                self._state = _S.THINK
                self.stats.think_blocks += 1
            return (ProseChunk(text) if text else None), True

        # No marker. Emit everything except a possible split marker prefix.
        keep = 0 if final else _HOLDBACK
        if len(self._buf) > keep:
            cut = len(self._buf) - keep
            text, self._buf = self._buf[:cut], self._buf[cut:]
            return ProseChunk(text), False
        return None, False

    def _step_directive(self) -> tuple[Event | None, bool]:
        # Search the accumulated body, not just the newest chunk: with small
        # chunks the closing "]]" straddles a boundary and would never be seen.
        self._pending += self._buf
        self._buf = ""
        i = self._pending.find(CLOSE)
        if i == -1:
            if len(self._pending) > MAX_DIRECTIVE_LEN:
                # Runaway: treat as a parse failure and resume prose.
                self.stats.parse_failures += 1
                self._pending = ""
                self._state = _S.PROSE
                return None, True
            return None, False

        body = self._pending[:i]
        self._buf = self._pending[i + len(CLOSE):]
        self._pending = ""
        self._state = _S.PROSE
        return self._make_slot(body), True

    def _step_think(self) -> tuple[Event | None, bool]:
        # Same accumulation fix as _step_directive: "</think>" split across
        # chunks was silently swallowing the rest of the response.
        self._pending += self._buf
        self._buf = ""
        i = self._pending.find(THINK_CLOSE)
        if i == -1:
            return None, False
        self._buf = self._pending[i + len(THINK_CLOSE):]
        self._pending = ""
        self._state = _S.PROSE
        return None, True

    def _make_slot(self, body: str) -> Event | None:
        # The opening marker is "[[IMG" so that a model omitting the pipe still
        # gets caught and dropped rather than leaking "[[IMG ..." into prose.
        body = body.lstrip()
        if body.startswith("|"):
            body = body[1:]
        parts = body.split("|")
        if len(parts) < 2:
            self.stats.parse_failures += 1
            return None
        entities = [e.strip() for e in parts[0].split(",") if e.strip()]
        keywords = parts[1].strip()
        # Anything after the third field is folded into framing rather than
        # dropped: a model that writes one pipe too many should still get its
        # camera honoured.
        framing = ", ".join(s.strip() for s in parts[2:] if s.strip())

        if not keywords:
            self.stats.parse_failures += 1
            return None

        self.stats.directives += 1

        if self._emitted >= self.max_images:
            self.stats.dropped_over_cap += 1
            return None

        # Small models re-describe the same room every turn; drop near-repeats
        # against anything already drawn in this story.
        if any(jaccard(keywords, prev) > self.repeat_threshold
               for prev in self._seen):
            self.stats.dropped_repetitive += 1
            return None

        slot = ImageSlot(index=self._emitted, entities=entities,
                         keywords=keywords, framing=framing)
        self._emitted += 1
        self._seen.append(keywords)
        return slot


def split_all(text: str, **kw) -> list[Event]:
    """Convenience for non-streaming use and for tests."""
    sp = ProseSplitter(**kw)
    events = sp.feed(text)
    return events + sp.finish()


def segments_from_events(events: list[Event]) -> list[dict]:
    """Collapse an event stream into the persisted/rendered segment list.

    Adjacent prose is merged; this shape is what goes in `turn_segments`, what
    reconnect replays, and what the frontend renders. Position is a segment
    index, never a character offset.
    """
    segs: list[dict] = []
    for e in events:
        if isinstance(e, ProseChunk):
            if segs and segs[-1]["kind"] == "prose":
                segs[-1]["text"] += e.text
            else:
                segs.append({"kind": "prose", "text": e.text})
        else:
            segs.append({"kind": "image", "index": e.index,
                         "entities": e.entities, "keywords": e.keywords})
    for s in segs:
        if s["kind"] == "prose":
            s["text"] = s["text"]
    return [s for s in segs if s["kind"] != "prose" or s["text"].strip()]
