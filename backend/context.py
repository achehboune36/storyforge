"""Prompt assembly, rolling summary, and durable core memories.

The constraint is hard: `num_ctx` is 4096, chosen in Phase 0 because the
533 MiB it saves over 8192 is what makes the asymmetric GPU strategy fit. With
~700 reserved for generation, roughly 3400 tokens are left for everything else,
and a 30-turn story does not fit in that by replaying raw history.

So context is assembled in priority order, each part with its own budget:

  1. system prompt        (the story's own, directive spec included)
  2. entity roster        ids and roles ONLY
  3. pinned core memories durable facts: a name learned, an oath sworn
  4. rolling summary      everything older than the verbatim window
  5. recent turns         verbatim, directives stripped, fills the remainder
  6. the user's action

**Entity appearances are deliberately excluded.** If the model can see them it
paraphrases them into the prose, and the reader gets a head-to-toe description
of the same character every single time they walk into a room. Appearance is
the image pipeline's business (see image_prompt.py), not the narrator's.

Token counts are estimates — Ollama exposes no tokenizer — so every budget
carries slack. `prompt_eval_count` comes back on every response, which lets
`observed_ratio()` correct the estimator against reality over time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend import database as db
from backend import narrator, ratings
from backend.turn_parser import OPEN, CLOSE

# Empirically ~3.6 chars/token for English prose; deliberately pessimistic.
CHARS_PER_TOKEN = 3.6
SLACK = 0.75            # use only this fraction of the nominal budget

SUMMARISE_AFTER = 12    # turns behind before a summary pass is due

# The directive rule is stated FIRST in the story prompt (measured: burying it
# cost every directive). But assembly appends roster/memories/summary after it,
# which pushes the examples out of the recency position — and that alone was
# enough for llama3.2:3b to stop emitting directives entirely. So the rule is
# bookended: first and last.
FORMAT_REMINDER = (
    "REMINDER - start with the scene header line "
    "(D<day> | IY <date> | <Weekday> | <HH:MM> | <weather> | <Location>), "
    "keep paragraphs to one or two sentences, and if this reply shows "
    "somewhere or someone new, include a line of the form\n"
    "[[IMG|entity_ids|visual keywords|camera and frame shape]]\n"
    "where the image belongs, naming every character visible in it and "
    "choosing the camera from where the scene actually is."
)
VERBATIM_MIN_PAIRS = 2  # never drop below this many recent exchanges


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def observed_ratio(chars: int, prompt_eval_count: int) -> float:
    """Real chars-per-token from a completed call, for calibration."""
    return chars / prompt_eval_count if prompt_eval_count else CHARS_PER_TOKEN


_DIRECTIVE_RE = re.compile(re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.S)


def strip_directives(text: str) -> str:
    """Remove [[IMG|...]] from history.

    Replayed directives teach the model to emit them constantly — it pattern
    matches on its own past output. History carries prose only.
    """
    return re.sub(r"\n{3,}", "\n\n", _DIRECTIVE_RE.sub("", text)).strip()


@dataclass
class Budget:
    total: int
    reserve_output: int = 700
    # 150 was sized for "key (role)" lines and a one-line header. A cast of
    # eighteen with identities needs ~400, and when it did not fit the roster
    # was silently trimmed to a SINGLE entity — which is how the narrator
    # ended up inventing a bearded headmaster while Emily held the role.
    entities: int = 700
    memories: int = 300
    summary: int = 500

    @property
    def usable(self) -> int:
        return int((self.total - self.reserve_output) * SLACK)


@dataclass
class Assembled:
    system: str
    history: list[dict]
    user: str
    parts: dict[str, int] = field(default_factory=dict)   # token estimates
    dropped_turns: int = 0

    @property
    def est_total(self) -> int:
        return sum(self.parts.values())


# Words that carry identity rather than costume detail. A roster line gets the
# gender word plus two or three marks, never the whole tag string.
_IDENTITY_SKIP = re.compile(r"^(solo|looking at viewer|smile|blush|expressionless|"
                            r"grin|smirk|stern expression|calm expression)$")


def _identity(e: dict) -> str:
    """Three or four words: who this is, not what they are wearing in detail.

    Full appearance stays out of the narrator's context — given the whole tag
    string it writes a head-to-toe description every time someone enters a
    room. But withholding *everything* was worse: with only
    "emily (Headmaster, …)" to go on, the narrator gave the headmaster a grey
    beard and folded HIS hands, when Emily is one of the Four Celestials and
    a silver-haired woman. Gender and a couple of marks are identity.
    """
    app = (e.get("appearance") or "").strip()
    if not app or e.get("kind") != "character":
        return ""
    tags = [x.strip() for x in app.split(",") if x.strip()]
    words, out = [], []
    for tag in tags:
        low = tag.lower()
        if low in ("1girl", "2girls", "1boy", "solo"):
            continue
        if low in ("mature female", "old woman", "young woman"):
            words.append(low)
            continue
        if _IDENTITY_SKIP.match(low):
            continue
        out.append(tag)
    gender = ""
    low = app.lower()
    if "1boy" in low or "1male" in low:
        gender = "man"
    elif "1girl" in low or "female" in low or "woman" in low:
        gender = "woman"
    if words and "old woman" in words:
        gender = "old woman"
    elif words and "mature female" in words and gender == "woman":
        gender = "grown woman"
    bits = ([gender] if gender else []) + out[:2]
    return ", ".join(b for b in bits if b)


def entity_roster(session_id: str, budget: int) -> str:
    """Ids and roles, plus a few words of identity — never the full appearance."""
    ents = db.get_entities(session_id)
    if not ents:
        return ""
    lines: list[tuple[str, str, bool]] = []
    for e in ents:
        base = f"{e['key']} ({e['role']})" if e["role"] else e["key"]
        who = _identity(e)
        lines.append((base, f" [{who}]" if who else "", e.get("kind") == "character"))
    # Label matters. "KNOWN ENTITIES" was read by llama3.2:3b as "already
    # shown", which directly contradicts the rule to illustrate places not yet
    # shown — the arrival scene stopped being illustrated because the bar was
    # in this list. The label names an id vocabulary, nothing more.
    # Framing matters as much as content here. Labelled "ENTITY IDS you may
    # use in the entity_ids field", this block read as metadata for the image
    # line — so the narrator never consulted it while writing prose, and
    # invented a bearded "Headmaster Vaelin" while Emily sat in the list
    # holding exactly that role. It is the cast first and an id vocabulary
    # second. The "not yet shown" caveat stays: labelled "KNOWN ENTITIES" it
    # was read as "already shown" and arrival scenes stopped being drawn.
    # The header stays SHORT and the rules about it live in narrator.RULES,
    # which is cached and does not compete with the cast for this budget.
    head = ("WHO AND WHAT EXISTS IN THIS STORY - cast and places. The id is what goes "
            "in entity_ids; [brackets] are who that person is. Being listed does NOT "
            "mean a place has been shown yet: ")
    # Trim in order of what can be lost: places go before people, and a
    # character's identity goes before the character does. Popping blindly
    # off the end dropped the cast and kept the furniture.
    def render(ls: list[tuple[str, str, bool]]) -> str:
        return head + "; ".join(a + b for a, b, _ in ls)

    rows = [(base, extra, is_char) for base, extra, is_char in lines]
    while est_tokens(render(rows)) > budget:
        places = [i for i, (_, _, ch) in enumerate(rows) if not ch]
        if places:
            rows.pop(places[-1])
            continue
        with_extra = [i for i, (_, ex, _) in enumerate(rows) if ex]
        if with_extra:
            i = with_extra[-1]
            rows[i] = (rows[i][0], "", rows[i][2])
            continue
        if len(rows) <= 1:
            break
        rows.pop()
    return render(rows)


def core_memories(session_id: str, budget: int) -> str:
    mems = db.get_memories(session_id, pinned_only=True)
    if not mems:
        return ""
    kept: list[str] = []
    used = 0
    for m in mems:                       # already ordered by importance desc
        line = f"- {m['text']}"
        t = est_tokens(line)
        if used + t > budget:
            continue
        kept.append(line)
        used += t
    return ("ESTABLISHED FACTS (true, do not contradict):\n" + "\n".join(kept)
            if kept else "")


def assemble(session_id: str, user_action: str, *, num_ctx: int,
             reserve_output: int = 700, passing: bool = False) -> Assembled:
    sess = db.get_session(session_id)
    if not sess:
        raise ValueError(f"unknown session {session_id}")
    story = db.get_story(sess["story_id"]) or {}

    b = Budget(total=num_ctx, reserve_output=reserve_output)
    parts: dict[str, int] = {}

    rating = (sess["rating"] if "rating" in sess.keys() else story.get("rating"))
    clause = ratings.narrator_clause(rating)
    # Legacy stories may still carry a full system_prompt; new ones supply a
    # premise and let narrator.py provide the measured rule set.
    if story.get("premise"):
        system = narrator.build_system_prompt(story, clause, passing=passing)
    else:
        system = story.get("system_prompt", "") + "\n\n" + clause
    parts["system"] = est_tokens(system)

    roster = entity_roster(session_id, b.entities)
    if roster:
        system += "\n\n" + roster
        parts["entities"] = est_tokens(roster)

    mems = core_memories(session_id, b.memories)
    if mems:
        system += "\n\n" + mems
        parts["memories"] = est_tokens(mems)

    summary = (sess["summary"] or "").strip()
    if summary:
        if est_tokens(summary) > b.summary:
            summary = summary[: int(b.summary * CHARS_PER_TOKEN)].rsplit(" ", 1)[0]
        system += "\n\nSTORY SO FAR:\n" + summary
        parts["summary"] = est_tokens(summary)

    system += "\n\n" + FORMAT_REMINDER
    parts["reminder"] = est_tokens(FORMAT_REMINDER)

    parts["user"] = est_tokens(user_action)

    # Whatever is left goes to verbatim recent turns, newest first.
    remaining = b.usable - sum(parts.values())
    turns = db.get_turns(session_id)
    after = int(sess["summary_through_turn"])
    candidates = [t for t in turns if t["idx"] > after]

    history: list[dict] = []
    used = 0
    dropped = 0
    for t in reversed(candidates):
        text = strip_directives(t["text"])
        if not text:
            continue
        msg = {"role": "user" if t["role"] == "user" else "assistant",
               "content": text}
        cost = est_tokens(text) + 4
        if used + cost > remaining and len(history) >= VERBATIM_MIN_PAIRS * 2:
            dropped += 1
            continue
        history.insert(0, msg)
        used += cost
    parts["history"] = used

    return Assembled(system=system, history=history, user=user_action,
                     parts=parts, dropped_turns=dropped)


# ------------------------------------------------------------------ summary

SUMMARY_INSTRUCTION = (
    "Summarise the story so far in at most 150 words, in second person, "
    "present tense. Keep only what still matters: where the player is, who "
    "they have met, what they have learned, what is unresolved. No dialogue, "
    "no scene description, no speculation."
)

FACTS_INSTRUCTION = (
    "List up to 3 durable facts established in this passage that must never be "
    "contradicted later — a name learned, an object taken, a promise made, an "
    "injury sustained. One per line, each under 15 words, no numbering. If "
    "there are none, reply with the single word NONE."
)


def summary_due(session_id: str) -> bool:
    sess = db.get_session(session_id)
    if not sess:
        return False
    last = db.next_turn_idx(session_id) - 1
    return (last - int(sess["summary_through_turn"])) > SUMMARISE_AFTER


def build_summary_messages(session_id: str) -> tuple[list[dict], int]:
    """Messages for the summarisation pass, plus the turn index it covers."""
    sess = db.get_session(session_id)
    turns = db.get_turns(session_id)
    after = int(sess["summary_through_turn"])
    # Summarise everything up to the last few turns, which stay verbatim.
    cutoff = max(after, db.next_turn_idx(session_id) - 1 - VERBATIM_MIN_PAIRS * 2)
    body = "\n\n".join(
        f"{'PLAYER' if t['role'] == 'user' else 'NARRATOR'}: {strip_directives(t['text'])}"
        for t in turns if after < t["idx"] <= cutoff)
    prior = (sess["summary"] or "").strip()
    if prior:
        body = f"SUMMARY SO FAR:\n{prior}\n\nNEW EVENTS:\n{body}"
    return ([{"role": "user", "content": f"{SUMMARY_INSTRUCTION}\n\n{body}"}],
            cutoff)


def build_facts_messages(passage: str) -> list[dict]:
    return [{"role": "user", "content": f"{FACTS_INSTRUCTION}\n\n{passage}"}]


def parse_facts(text: str) -> list[str]:
    if text.strip().upper().startswith("NONE"):
        return []
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•0123456789. ").strip()
        if 3 < len(line) < 140:
            out.append(line)
    return out[:3]
