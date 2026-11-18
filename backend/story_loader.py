"""Load stories/*.json into the database.

Disk is the source of truth; the `stories` table is a cache keyed by content
hash so editing a story file and restarting picks the change up. Sessions
already in flight keep the prompt they started with, because the story JSON is
copied into the session's entity rows at creation time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend import database as db
from backend import ratings

STORIES_DIR = Path(__file__).resolve().parent.parent / "stories"

REQUIRED = ("id", "title", "style", "opening", "rating")


class StoryError(ValueError):
    pass


def _validate(raw: dict, path: Path) -> None:
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise StoryError(f"{path.name}: missing {missing}")
    if "text" not in raw["opening"]:
        raise StoryError(f"{path.name}: opening needs 'text'")
    if not raw.get("premise") and not raw.get("system_prompt"):
        raise StoryError(f"{path.name}: needs either 'premise' (preferred) or "
                         "a full legacy 'system_prompt'")
    if raw.get("system_prompt") and "[[IMG" not in raw["system_prompt"][:400]:
        print(f"  WARNING {path.name}: legacy system_prompt without the [[IMG| "
              "rule near the top. Measured to cost ~all directives.")


def load_all(*, verbose: bool = True) -> list[dict]:
    out = []
    if not STORIES_DIR.exists():
        return out
    for path in sorted(STORIES_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        _validate(raw, path)
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        raw["rating"] = ratings.normalise(raw.get("rating"))
        db.upsert_story(raw["id"], raw["title"], raw.get("blurb", ""), h, raw,
                        rating=raw["rating"])
        out.append(raw)
        if verbose:
            n = len(raw.get("entities", {}))
            print(f"  loaded story {raw['id']!r} ({raw['title']}) — "
                  f"{n} pre-declared entit{'y' if n == 1 else 'ies'}")
    return out


def start_session(story_id: str, *, title: str | None = None) -> dict:
    """Create a session and seed its entity registry from the story."""
    story = db.get_story(story_id)
    if not story:
        raise StoryError(f"unknown story {story_id!r}")

    rating = ratings.normalise(story.get("rating"))
    sess = db.create_session(
        story_id, style=story["style"], title=title or story["title"],
        rating=rating,
        negative=ratings.negative_prompt(rating, story.get("negative", "")))

    from backend.image_prompt import entity_seed
    for key, ent in (story.get("entities") or {}).items():
        db.upsert_entity(
            sess["id"], key,
            kind=ent.get("kind", "character"),
            role=ent.get("role", ""),
            appearance=ent.get("appearance", ""),
            seed=entity_seed(sess["id"], key),
        )
    return sess
