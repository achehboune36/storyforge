"""SQLite persistence.

Connection handling is lifted from gameshow/backend/database.py:9-19 — WAL,
thread-local connections, foreign keys on — which has been running in
production there. Two changes for this app:

  * Writes go through `asyncio.to_thread` at the call sites that matter. The
    original blocks the event loop on every write; that is fine for one row per
    question, less fine when a turn writes a row per segment while prose is
    streaming.
  * `turn_segments` is the load-bearing table. It IS the persistence format,
    the reconnect replay payload, and what the frontend renders. Position is a
    segment index, never a character offset, so a re-render can never drift
    from where the image actually belongs.

game-poc kept sessions in a process-local dict and lost every story on
restart. That is the thing this module exists to prevent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "storyforge.db"
_local = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS stories (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    blurb         TEXT,
    rating        TEXT NOT NULL DEFAULT '13+',
    content_hash  TEXT NOT NULL,
    json          TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id                   TEXT PRIMARY KEY,
    story_id             TEXT NOT NULL REFERENCES stories(id),
    token                TEXT NOT NULL UNIQUE,
    title                TEXT,
    created_at           TEXT NOT NULL,
    last_active_at       TEXT NOT NULL,
    state                TEXT NOT NULL DEFAULT 'active',
    style                TEXT NOT NULL DEFAULT '',
    rating               TEXT NOT NULL DEFAULT '13+',
    negative             TEXT NOT NULL DEFAULT '',
    summary              TEXT NOT NULL DEFAULT '',
    summary_through_turn INTEGER NOT NULL DEFAULT -1
);

CREATE TABLE IF NOT EXISTS turns (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx        INTEGER NOT NULL,
    role       TEXT NOT NULL,                 -- 'user' | 'narrator'
    created_at TEXT NOT NULL,
    text       TEXT NOT NULL DEFAULT '',      -- user text, or narrator prose
    raw_text   TEXT NOT NULL DEFAULT '',      -- un-stripped LLM output, for debugging
    UNIQUE(session_id, idx)
);

CREATE TABLE IF NOT EXISTS turn_segments (
    id       TEXT PRIMARY KEY,
    turn_id  TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    pos      INTEGER NOT NULL,
    kind     TEXT NOT NULL,                   -- 'prose' | 'image'
    text     TEXT NOT NULL DEFAULT '',
    image_id TEXT,
    UNIQUE(turn_id, pos)
);

CREATE TABLE IF NOT EXISTS images (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn_id         TEXT REFERENCES turns(id) ON DELETE CASCADE,
    slot_pos        INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,            -- queued|running|ready|failed|preempted|cancelled
    keywords        TEXT NOT NULL DEFAULT '',
    entities        TEXT NOT NULL DEFAULT '[]',
    prompt          TEXT NOT NULL DEFAULT '',
    negative        TEXT NOT NULL DEFAULT '',
    seed            INTEGER NOT NULL DEFAULT 0,
    width           INTEGER NOT NULL DEFAULT 0,
    height          INTEGER NOT NULL DEFAULT 0,
    steps           INTEGER NOT NULL DEFAULT 0,
    model           TEXT NOT NULL DEFAULT '',
    comfy_prompt_id TEXT,
    file_path       TEXT,
    error           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    hires           TEXT NOT NULL DEFAULT ''   -- ''|queued|running|done|failed|skip
);

CREATE TABLE IF NOT EXISTS entities (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    key           TEXT NOT NULL,
    kind          TEXT NOT NULL DEFAULT 'character',
    role          TEXT NOT NULL DEFAULT '',
    appearance    TEXT NOT NULL DEFAULT '',
    seed          INTEGER NOT NULL,
    ref_image_id  TEXT,
    first_turn_id TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE(session_id, key)
);

CREATE TABLE IF NOT EXISTS memories (
    id         TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL DEFAULT 'fact',
    text       TEXT NOT NULL,
    importance INTEGER NOT NULL DEFAULT 5,
    pinned     INTEGER NOT NULL DEFAULT 1,
    turn_id    TEXT,
    norm_hash  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, norm_hash)
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    image_id    TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    state       TEXT NOT NULL,                -- queued|running|done|failed|cancelled
    priority    INTEGER NOT NULL DEFAULT 5,
    attempts    INTEGER NOT NULL DEFAULT 0,
    enqueued_at TEXT NOT NULL,
    started_at  TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_session   ON turns(session_id, idx);
CREATE INDEX IF NOT EXISTS idx_segments_turn   ON turn_segments(turn_id, pos);
CREATE INDEX IF NOT EXISTS idx_images_session  ON images(session_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_state      ON jobs(state, priority, enqueued_at);
CREATE INDEX IF NOT EXISTS idx_entities_sess   ON entities(session_id, key);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_id() -> str:
    return uuid.uuid4().hex


def connect() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return _local.conn


def init() -> None:
    conn = connect()
    conn.executescript(SCHEMA)
    # Additive migration: CREATE TABLE IF NOT EXISTS leaves an existing table
    # alone, so a column added to SCHEMA never reaches a database that already
    # has playthroughs in it. Those are the ones that matter.
    have = {r[1] for r in conn.execute("PRAGMA table_info(images)")}
    if "hires" not in have:
        conn.execute("ALTER TABLE images ADD COLUMN hires TEXT NOT NULL DEFAULT ''")
    conn.commit()


def reconcile_startup() -> int:
    """Requeue jobs that were running when the process died.

    ComfyUI's own queue does not survive our restart, so a 'running' row is
    always stale on boot. Returns how many were requeued.
    """
    conn = connect()
    cur = conn.execute(
        "UPDATE jobs SET state='queued', started_at=NULL WHERE state='running'")
    conn.execute(
        "UPDATE images SET status='queued', started_at=NULL WHERE status='running'")
    conn.commit()
    return cur.rowcount


# ------------------------------------------------------------------- stories

def upsert_story(sid: str, title: str, blurb: str, content_hash: str,
                 raw: dict, rating: str = "13+") -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO stories (id,title,blurb,rating,content_hash,json,updated_at) "
        "VALUES (?,?,?,?,?,?,?) "
        "ON CONFLICT(id) DO UPDATE SET title=excluded.title, blurb=excluded.blurb,"
        " rating=excluded.rating, content_hash=excluded.content_hash,"
        " json=excluded.json, updated_at=excluded.updated_at",
        (sid, title, blurb, rating, content_hash, json.dumps(raw), now()))
    conn.commit()


def get_story(sid: str) -> dict | None:
    row = connect().execute("SELECT * FROM stories WHERE id=?", (sid,)).fetchone()
    return json.loads(row["json"]) if row else None


def list_stories() -> list[dict]:
    rows = connect().execute(
        "SELECT id,title,blurb,rating FROM stories ORDER BY title").fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ sessions

def create_session(story_id: str, *, style: str, title: str | None = None,
                   rating: str = "13+", negative: str = "") -> dict:
    conn = connect()
    sid, token, t = new_id(), new_id(), now()
    conn.execute(
        "INSERT INTO sessions (id,story_id,token,title,created_at,last_active_at,"
        "state,style,rating,negative) VALUES (?,?,?,?,?,?, 'active', ?,?,?)",
        (sid, story_id, token, title, t, t, style, rating, negative))
    conn.commit()
    return get_session(sid)  # type: ignore[return-value]


def get_session(sid: str) -> dict | None:
    row = connect().execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None


def get_session_by_token(token: str) -> dict | None:
    row = connect().execute("SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
    return dict(row) if row else None


def touch_session(sid: str) -> None:
    conn = connect()
    conn.execute("UPDATE sessions SET last_active_at=? WHERE id=?", (now(), sid))
    conn.commit()


def set_summary(sid: str, summary: str, through_turn: int) -> None:
    conn = connect()
    conn.execute(
        "UPDATE sessions SET summary=?, summary_through_turn=? WHERE id=?",
        (summary, through_turn, sid))
    conn.commit()


def delete_session(session_id: str) -> dict:
    """Delete a playthrough and everything belonging to it.

    turns, turn_segments, images, entities, memories and jobs all carry
    ON DELETE CASCADE from sessions, and `PRAGMA foreign_keys=ON` is set in
    connect(), so removing the session row removes the lot. The counts are
    read first so the caller can report what went, and so a silent
    foreign-key misconfiguration would show up as zeros rather than passing
    unnoticed.
    """
    conn = connect()
    counts = {}
    for table, col in (("turns", "session_id"), ("images", "session_id"),
                       ("entities", "session_id"), ("memories", "session_id"),
                       ("jobs", "session_id")):
        counts[table] = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col}=?", (session_id,)).fetchone()[0]
    counts["turn_segments"] = conn.execute(
        "SELECT COUNT(*) FROM turn_segments ts JOIN turns t ON t.id=ts.turn_id "
        "WHERE t.session_id=?", (session_id,)).fetchone()[0]

    cur = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    counts["deleted"] = cur.rowcount > 0

    left = conn.execute("SELECT COUNT(*) FROM turns WHERE session_id=?",
                        (session_id,)).fetchone()[0]
    counts["orphans_left"] = left        # must be 0; non-zero means FKs are off
    return counts


def list_sessions(story_id: str | None = None) -> list[dict]:
    q = ("SELECT s.*, st.title AS story_title FROM sessions s "
         "JOIN stories st ON st.id = s.story_id")
    args: tuple = ()
    if story_id:
        q += " WHERE s.story_id=?"
        args = (story_id,)
    q += " ORDER BY s.last_active_at DESC"
    return [dict(r) for r in connect().execute(q, args).fetchall()]


# --------------------------------------------------------------------- turns

def next_turn_idx(session_id: str) -> int:
    row = connect().execute(
        "SELECT COALESCE(MAX(idx), -1) AS m FROM turns WHERE session_id=?",
        (session_id,)).fetchone()
    return int(row["m"]) + 1


def add_turn(session_id: str, role: str, text: str, *, raw_text: str = "",
             idx: int | None = None) -> str:
    conn = connect()
    tid = new_id()
    if idx is None:
        idx = next_turn_idx(session_id)
    conn.execute(
        "INSERT INTO turns (id,session_id,idx,role,created_at,text,raw_text) "
        "VALUES (?,?,?,?,?,?,?)",
        (tid, session_id, idx, role, now(), text, raw_text))
    conn.commit()
    return tid


def add_segments(turn_id: str, segments: Iterable[dict]) -> list[str]:
    """Persist the ordered segment list for a narrator turn."""
    conn = connect()
    ids = []
    for pos, seg in enumerate(segments):
        sid = new_id()
        ids.append(sid)
        conn.execute(
            "INSERT INTO turn_segments (id,turn_id,pos,kind,text,image_id) "
            "VALUES (?,?,?,?,?,?)",
            (sid, turn_id, pos, seg["kind"], seg.get("text", ""),
             seg.get("image_id")))
    conn.commit()
    return ids


def get_turns(session_id: str, *, limit: int | None = None) -> list[dict]:
    q = "SELECT * FROM turns WHERE session_id=? ORDER BY idx"
    rows = connect().execute(q, (session_id,)).fetchall()
    out = [dict(r) for r in rows]
    return out[-limit:] if limit else out


def rollback_last(session_id: str, *, pairs: int = 1) -> dict:
    """Remove the last narrator turn, and the player turn that prompted it.

    Letting the reader rewind and redirect is the whole point of the feature,
    so this deletes rather than hides: a hidden turn would still be replayed
    into the model's context and would keep steering the story the reader is
    trying to steer away from.
    """
    conn = connect()
    removed = {"turns": 0, "images": 0, "files": []}
    for _ in range(max(pairs, 1)):
        rows = conn.execute(
            "SELECT id, role FROM turns WHERE session_id=? ORDER BY idx DESC LIMIT 2",
            (session_id,)).fetchall()
        if not rows:
            break
        # Drop the trailing narrator turn plus the user turn under it.
        targets = [rows[0]["id"]]
        if len(rows) > 1 and rows[0]["role"] == "narrator" and rows[1]["role"] == "user":
            targets.append(rows[1]["id"])
        for tid in targets:
            for img in conn.execute(
                    "SELECT id, file_path FROM images WHERE turn_id=?", (tid,)):
                removed["images"] += 1
                if img["file_path"]:
                    removed["files"].append(img["file_path"])
            conn.execute("DELETE FROM turns WHERE id=?", (tid,))
            removed["turns"] += 1
    # Any summary may now cover turns that no longer exist; recompute from scratch.
    conn.execute("UPDATE sessions SET summary='', summary_through_turn=-1 WHERE id=?",
                 (session_id,))
    conn.commit()
    return removed


def get_segments(session_id: str) -> list[dict]:
    """Full render/replay payload: every segment of every turn, in order."""
    rows = connect().execute(
        "SELECT t.idx AS turn_idx, t.role, t.id AS turn_id, "
        "       s.pos, s.kind, s.text, s.image_id "
        "FROM turns t LEFT JOIN turn_segments s ON s.turn_id = t.id "
        "WHERE t.session_id=? ORDER BY t.idx, s.pos", (session_id,)).fetchall()
    return [dict(r) for r in rows]


# -------------------------------------------------------------------- images

def create_image(session_id: str, turn_id: str | None, *, slot_pos: int,
                 keywords: str, entities: list[str], prompt: str, seed: int,
                 width: int, height: int, steps: int, model: str,
                 negative: str = "") -> str:
    conn = connect()
    iid = new_id()
    conn.execute(
        "INSERT INTO images (id,session_id,turn_id,slot_pos,status,keywords,"
        "entities,prompt,negative,seed,width,height,steps,model,created_at) "
        "VALUES (?,?,?,?, 'queued', ?,?,?,?,?,?,?,?,?,?)",
        (iid, session_id, turn_id, slot_pos, keywords, json.dumps(entities),
         prompt, negative, seed, width, height, steps, model, now()))
    conn.commit()
    return iid


def image_file_name(img: Any) -> str:
    """The file to serve for this image row: the refined pass once it exists.

    URLs are built from the image id, not from file_path, so this is the one
    place that knows a second pass happened. Reading it off the row means a
    reload after the refine lands shows the better picture with no event.
    """
    hr = ""
    try:
        hr = img["hires"] or ""
    except (KeyError, IndexError, TypeError):
        hr = ""
    return f"{img['id']}_hr.png" if hr == "done" else f"{img['id']}.png"


def update_image(image_id: str, **fields: Any) -> None:
    if not fields:
        return
    allowed = {"status", "comfy_prompt_id", "file_path", "error", "attempts",
               "started_at", "finished_at", "seed", "prompt", "hires"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"cannot update image field(s): {sorted(bad)}")
    conn = connect()
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE images SET {sets} WHERE id=?",
                 (*fields.values(), image_id))
    conn.commit()


def get_image(image_id: str) -> dict | None:
    row = connect().execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["entities"] = json.loads(d["entities"])
    return d


def get_images(session_id: str) -> list[dict]:
    rows = connect().execute(
        "SELECT * FROM images WHERE session_id=? ORDER BY created_at",
        (session_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["entities"] = json.loads(d["entities"])
        out.append(d)
    return out


# ------------------------------------------------------------------ entities

def upsert_entity(session_id: str, key: str, *, kind: str, role: str,
                  appearance: str, seed: int) -> dict:
    """Insert if absent. Appearance is frozen on first write — never updated."""
    conn = connect()
    conn.execute(
        "INSERT INTO entities (id,session_id,key,kind,role,appearance,seed,created_at) "
        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(session_id,key) DO NOTHING",
        (new_id(), session_id, key, kind, role, appearance, seed, now()))
    conn.commit()
    return get_entity(session_id, key)  # type: ignore[return-value]


def get_entity(session_id: str, key: str) -> dict | None:
    row = connect().execute(
        "SELECT * FROM entities WHERE session_id=? AND key=?",
        (session_id, key)).fetchone()
    return dict(row) if row else None


def set_entity_ref(session_id: str, key: str, filename: str) -> None:
    """Freeze this character's reference portrait. Written once."""
    conn = connect()
    conn.execute("UPDATE entities SET ref_image_id=? WHERE session_id=? AND key=?",
                 (filename, session_id, key))
    conn.commit()


def get_entities(session_id: str) -> list[dict]:
    rows = connect().execute(
        "SELECT * FROM entities WHERE session_id=? ORDER BY key",
        (session_id,)).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ memories

def add_memory(session_id: str, text: str, *, kind: str = "fact",
               importance: int = 5, pinned: bool = True,
               turn_id: str | None = None) -> str | None:
    """Deduped by normalised text; returns None if already known."""
    import hashlib
    import re
    norm = re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()
    norm = re.sub(r"\s+", " ", norm)
    h = hashlib.sha256(norm.encode()).hexdigest()[:32]
    conn = connect()
    mid = new_id()
    cur = conn.execute(
        "INSERT INTO memories (id,session_id,kind,text,importance,pinned,turn_id,"
        "norm_hash,created_at) VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(session_id,norm_hash) DO NOTHING",
        (mid, session_id, kind, text.strip(), importance, int(pinned), turn_id,
         h, now()))
    conn.commit()
    return mid if cur.rowcount else None


def get_memories(session_id: str, *, pinned_only: bool = True) -> list[dict]:
    q = "SELECT * FROM memories WHERE session_id=?"
    if pinned_only:
        q += " AND pinned=1"
    q += " ORDER BY importance DESC, created_at"
    return [dict(r) for r in connect().execute(q, (session_id,)).fetchall()]


# ---------------------------------------------------------------------- jobs

def enqueue_job(session_id: str, image_id: str, *, priority: int = 5) -> str:
    conn = connect()
    jid = new_id()
    conn.execute(
        "INSERT INTO jobs (id,session_id,image_id,state,priority,enqueued_at) "
        "VALUES (?,?,?, 'queued', ?, ?)",
        (jid, session_id, image_id, priority, now()))
    conn.commit()
    return jid


def claim_next_job() -> dict | None:
    conn = connect()
    row = conn.execute(
        "SELECT * FROM jobs WHERE state='queued' "
        "ORDER BY priority, enqueued_at LIMIT 1").fetchone()
    if not row:
        return None
    conn.execute(
        "UPDATE jobs SET state='running', started_at=?, attempts=attempts+1 "
        "WHERE id=?", (now(), row["id"]))
    conn.commit()
    return dict(row)


def finish_job(job_id: str, state: str) -> None:
    conn = connect()
    conn.execute("UPDATE jobs SET state=? WHERE id=?", (state, job_id))
    conn.commit()


def pending_jobs(session_id: str | None = None) -> list[dict]:
    q = "SELECT * FROM jobs WHERE state IN ('queued','running')"
    args: tuple = ()
    if session_id:
        q += " AND session_id=?"
        args = (session_id,)
    q += " ORDER BY priority, enqueued_at"
    return [dict(r) for r in connect().execute(q, args).fetchall()]
