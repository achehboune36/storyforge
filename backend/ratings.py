"""Per-story content rating, applied to BOTH pipelines.

A rating is only meaningful if it reaches every generator. Here it does two
separate jobs:

  * the **narrator** gets a clause in its system prompt describing what the
    story may contain, in the language of what to write rather than what to
    avoid — telling a model "no violence" makes violence salient;
  * the **image model** gets a negative prompt and a positive content clause.
    A rating that only constrained the prose would be worthless the moment the
    illustrator drew something the text never said.

The narrator clauses live here because they are story language. Everything the
image model sees — negatives, positive content clauses — lives in
`prompting.toml`, which is the one file tuned by hand against real output.

Ratings are deliberately coarse. Three buckets a person can reason about beat
a dozen nobody sets correctly.
"""

from __future__ import annotations

from backend import prompt_book as book

RATINGS = ("all", "13+", "18+")
DEFAULT = "13+"

# What the narrator is told. Phrased as permission, not prohibition.
_NARRATOR: dict[str, str] = {
    "all": ("CONTENT - this story is for all ages. Keep danger bloodless and "
            "tension light. Characters may be frightened or hurt, but never "
            "graphically. No sex, no romance beyond a blush, no drinking or "
            "smoking on the page."),
    "13+": ("CONTENT - this story is rated 13+. Peril, menace, violence and its "
            "consequences are allowed, and so are attraction and flirtation. "
            "Keep injury non-graphic and keep intimacy off the page - a closed "
            "door, a morning after. Strong language is rare and mild."),
    "18+": ("CONTENT - this story is rated 18+ and may go where the story goes: "
            "graphic violence, its aftermath, frank language, and sexuality "
            "written with intent rather than for its own sake. Nothing is "
            "gratuitous; everything serves the scene. Follow the player's lead "
            "on how far to take any of it."),
}


def normalise(rating: str | None) -> str:
    r = (rating or "").strip().lower().replace(" ", "")
    aliases = {"everyone": "all", "e": "all", "g": "all", "0+": "all",
               "pg": "all", "pg13": "13+", "13": "13+", "teen": "13+", "t": "13+",
               "16+": "13+", "mature": "18+", "m": "18+", "18": "18+",
               "adult": "18+", "r18": "18+", "nsfw": "18+"}
    r = aliases.get(r, r)
    return r if r in RATINGS else DEFAULT


def positive_safety(rating: str | None) -> str:
    """Content guidance stated positively, for the POSITIVE prompt.

    Text lives in `prompting.toml` under [safety]. This is the primary content
    control at CFG 1, where the negative prompt is nearly inert. Empty for 18+.
    """
    return book.safety(normalise(rating))


def negative_prompt(rating: str | None, extra: str = "",
                    checkpoint: str = "") -> str:
    """What the image model must not draw, for this rating.

    Text lives in `prompting.toml` under [negative]; `extra` is the story's own
    `negative` field, and the active checkpoint may add its own
    (`negative_extra`) — Hassaku, for instance, wants "signature". Re-read on every call, so tuning the file takes effect on
    the next image without a restart.
    """
    out = book.negative(normalise(rating), extra, checkpoint)
    if book.sanitize()["dedupe_negative"]:
        from backend import prompt_style
        out = prompt_style.dedupe_tags(out)
    return out


def narrator_clause(rating: str | None) -> str:
    return _NARRATOR[normalise(rating)]


def label(rating: str | None) -> str:
    return {"all": "All ages", "13+": "13+", "18+": "18+"}[normalise(rating)]


def allows_nsfw(rating: str | None) -> bool:
    return normalise(rating) == "18+"
